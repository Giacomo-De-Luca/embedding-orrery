"""Hook-based SAE attachment to PyTorch model layers."""

import contextlib
import warnings
from collections.abc import Generator

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from interpret.sae.activation_store import ActivationStore
from interpret.sae.loading import load_sae
from interpret.sae.sae_config import (
    GemmaScopeSAEConfig,
    HookType,
    QwenScopeSAEConfig,
)
from interpret.sae.sae_model import SAEBase
from interpret.sae.steering import (
    ResolvedSteeringOp,
    SteeringOp,
    apply_steering,
    resolve_op,
)

SAEConfigT = GemmaScopeSAEConfig | QwenScopeSAEConfig


class HookManager:
    """Manages attaching/detaching SAEs as forward hooks on model layers.

    Supports both activation capture (via SAE encode) and steering
    interventions (additive / orthogonal / ablation / projection cap).
    Steering and capture coexist on the same layer through a single
    forward hook: steering is applied before SAE encoding, so captured
    feature activations reflect the post-intervention hidden state.

    Usage::

        wrapper = GemmaPytorchInference("google/gemma-3-4b-it")
        manager = HookManager()
        manager.add_sae(SAEConfig(layer_index=9))
        manager.add_sae(SAEConfig(layer_index=29))
        manager.add_steering(SteeringOp(
            layer_index=9, mode=SteeringMode.ADDITIVE,
            feature_index=4287, strength=6.0, normalise=True,
        ))

        with manager.session(wrapper.model.model.layers) as store:
            result = wrapper.generate("What colour is the sky?")
            prefill_acts = store.prefill(layer=29).feature_acts
    """

    def __init__(self) -> None:
        self._saes: dict[tuple[int, HookType], SAEBase] = {}
        self._configs: dict[tuple[int, HookType], SAEConfigT] = {}
        self._handles: list[RemovableHandle] = []
        self.store = ActivationStore()
        self._prefill_done: bool = False
        self._steering_ops: list[SteeringOp] = []
        self._resolved_by_layer: dict[tuple[int, HookType], list[ResolvedSteeringOp]] = {}
        self.strength_multiplier: float = 1.0

    def add_sae(self, config: SAEConfigT) -> None:
        """Load an SAE and register it for hook attachment.

        Args:
            config: SAE configuration. The SAE weights are downloaded
                    from HuggingFace on first call (cached thereafter).
        """
        sae = load_sae(config)
        key = (config.layer_index, config.hook_type)
        self._saes[key] = sae
        self._configs[key] = config

    def add_steering(self, op: SteeringOp | list[SteeringOp]) -> None:
        """Register one or more steering ops to apply when hooks are attached.

        Resolution against SAE weights is deferred until ``attach()`` so
        the device + dtype of the target layer are known. Multiple ops on
        the same layer compose in insertion order.

        Args:
            op: A single SteeringOp or list of them.
        """
        if isinstance(op, list):
            self._steering_ops.extend(op)
        else:
            self._steering_ops.append(op)

    def clear_steering(self) -> None:
        """Remove all registered steering ops."""
        self._steering_ops.clear()
        self._resolved_by_layer.clear()

    def set_strength_multiplier(self, multiplier: float) -> None:
        """Set a global multiplier applied to every op's strength.

        Has no effect on PROJECTION_CAP ops (their bounds are absolute).
        """
        self.strength_multiplier = multiplier

    def attach(self, layers: nn.ModuleList) -> None:
        """Attach forward hooks to the specified model layers.

        Hooks are routed to the sub-module matching ``hook_type``:
        ``RESID_POST`` -> ``layers[i]`` (full decoder layer output),
        ``ATTN_OUT``   -> ``layers[i].self_attn``,
        ``MLP_OUT``    -> ``layers[i].mlp``.

        If prefill_only is set on a config, that SAE's hook auto-removes
        after the first forward pass (the prefill).

        Steering ops with no SAE registered at the same
        ``(layer_index, hook_type)`` get a lightweight steering-only hook.
        When an SAE and steering ops share a site, they run inside a single
        combined hook (steering before SAE capture).

        Args:
            layers: The model's decoder layer ModuleList
                    (e.g. wrapper.model.model.layers).

        Raises:
            ValueError: If a layer_index is out of range.
        """
        self.detach()
        self.store.clear()
        self._prefill_done = False

        self._resolve_steering(layers)
        sae_keys = set(self._saes.keys())

        for (layer_idx, hook_type), sae in self._saes.items():
            if layer_idx >= len(layers):
                raise ValueError(
                    f"layer_index {layer_idx} out of range (model has {len(layers)} layers)"
                )
            config = self._configs[(layer_idx, hook_type)]
            target = self._resolve_hook_target(layers[layer_idx], hook_type)
            steer_ops = self._resolved_by_layer.get((layer_idx, hook_type), [])

            if steer_ops and not config.read_only:
                warnings.warn(
                    f"Layer {layer_idx} ({hook_type.value}) has both steering "
                    "ops and an SAE with read_only=False. The steered hidden "
                    "state will be replaced by its (lossy) SAE reconstruction.",
                    stacklevel=2,
                )

            handle = target.register_forward_hook(
                self._make_sae_hook(sae, config, (layer_idx, hook_type), steer_ops)
            )
            self._handles.append(handle)

        for (layer_idx, hook_type), steer_ops in self._resolved_by_layer.items():
            if (layer_idx, hook_type) in sae_keys:
                continue
            if layer_idx >= len(layers):
                raise ValueError(
                    f"steering layer_index {layer_idx} out of range "
                    f"(model has {len(layers)} layers)"
                )
            target = self._resolve_hook_target(layers[layer_idx], hook_type)
            handle = target.register_forward_hook(self._make_steering_only_hook(steer_ops))
            self._handles.append(handle)

        # Attach a lightweight sentinel hook to the last layer to detect
        # when prefill ends (first forward pass completes for all layers).
        def _prefill_sentinel(module: nn.Module, inputs: tuple, output: torch.Tensor):
            self._prefill_done = True
            return None

        sentinel = layers[-1].register_forward_hook(_prefill_sentinel)
        self._handles.append(sentinel)

    def _resolve_steering(self, layers: nn.ModuleList) -> None:
        """Materialise all registered SteeringOps against their target layers."""
        self._resolved_by_layer.clear()
        for op in self._steering_ops:
            if op.layer_index >= len(layers):
                raise ValueError(
                    f"steering layer_index {op.layer_index} out of range "
                    f"(model has {len(layers)} layers)"
                )
            layer = layers[op.layer_index]
            param = next(layer.parameters())
            sae_key = op.sae_key or (op.layer_index, op.hook_type)
            sae = self._saes.get(sae_key) if op.feature_index is not None else None
            resolved = resolve_op(op, sae, device=param.device, dtype=param.dtype)
            bucket_key = (op.layer_index, op.hook_type)
            self._resolved_by_layer.setdefault(bucket_key, []).append(resolved)

    @staticmethod
    def _resolve_hook_target(layer: nn.Module, hook_type: HookType) -> nn.Module:
        """Pick the sub-module to attach a forward hook to for ``hook_type``."""
        if hook_type is HookType.RESID_POST:
            return layer
        if hook_type is HookType.ATTN_OUT:
            # Qwen3.5 hybrid layers expose `linear_attn` (Gated DeltaNet) instead
            # of `self_attn`; mirror the wrapper's resolver (qwen3_transformers.py).
            target = getattr(layer, "self_attn", None) or getattr(layer, "linear_attn", None)
        elif hook_type is HookType.MLP_OUT:
            target = layer.mlp
        else:
            raise ValueError(f"unsupported hook_type: {hook_type}")
        assert isinstance(target, nn.Module), (
            f"{hook_type.value} target on layer is not an nn.Module: {type(target)}"
        )
        return target

    def _make_sae_hook(
        self,
        sae_ref: SAEBase,
        cfg: SAEConfigT,
        key: tuple[int, HookType],
        steer_ops: list[ResolvedSteeringOp],
    ):
        def hook_fn(
            module: nn.Module,
            inputs: tuple,
            output: torch.Tensor,
        ) -> torch.Tensor | tuple | None:
            # Skip if prefill_only and we already captured the prefill
            if cfg.prefill_only and self._prefill_done:
                return None

            with torch.no_grad():
                hidden_states = output[0] if isinstance(output, tuple) else output
                if steer_ops:
                    hidden_states = apply_steering(
                        hidden_states, steer_ops, self.strength_multiplier
                    )

                feature_acts, reconstruction = sae_ref(hidden_states)
                self.store.record(
                    key,
                    feature_acts.detach(),
                    reconstruction=reconstruction.detach(),
                    collect_last_only=cfg.collect_last_only,
                )

                if not cfg.read_only:
                    if isinstance(output, tuple):
                        return (reconstruction,) + output[1:]
                    return reconstruction
                if steer_ops:
                    if isinstance(output, tuple):
                        return (hidden_states,) + output[1:]
                    return hidden_states
            return None

        return hook_fn

    def _make_steering_only_hook(self, steer_ops: list[ResolvedSteeringOp]):
        def hook_fn(
            module: nn.Module,
            inputs: tuple,
            output: torch.Tensor,
        ) -> torch.Tensor | tuple | None:
            with torch.no_grad():
                hidden_states = output[0] if isinstance(output, tuple) else output
                hidden_states = apply_steering(hidden_states, steer_ops, self.strength_multiplier)
                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                return hidden_states

        return hook_fn

    def detach(self) -> None:
        """Remove all attached hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def reset(self) -> None:
        """Clear stored activations but keep hooks (and steering ops) attached."""
        self.store.clear()
        self._prefill_done = False

    @contextlib.contextmanager
    def session(self, layers: nn.ModuleList) -> Generator[ActivationStore, None, None]:
        """Context manager: attach hooks, yield store, detach on exit.

        Args:
            layers: The model's decoder layer ModuleList.

        Yields:
            The ActivationStore collecting feature activations.
        """
        self.attach(layers)
        try:
            yield self.store
        finally:
            self.detach()
