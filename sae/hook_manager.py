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
        # SAEs are keyed by (layer_index, hook_type, sae_id) where sae_id
        # = config.identity() (e.g. "w16k_l0_medium"). The third slot lets
        # two SAEs at the same (layer, hook_type) site coexist (e.g.
        # Gemma L29 W16K + L29 W65K in one forward pass).
        self._saes: dict[tuple[int, HookType, str], SAEBase] = {}
        self._configs: dict[tuple[int, HookType, str], SAEConfigT] = {}
        self._handles: list[RemovableHandle] = []
        self.store = ActivationStore()
        self._prefill_done: bool = False
        self._steering_ops: list[SteeringOp] = []
        self._resolved_by_layer: dict[tuple[int, HookType], list[ResolvedSteeringOp]] = {}
        self.strength_multiplier: float = 1.0

    def add_sae(self, config: SAEConfigT) -> None:
        """Load an SAE and register it for hook attachment.

        Two SAEs at the same ``(layer, hook_type)`` site are allowed when
        both have ``read_only=True`` — they share one forward hook and
        each produces an independent ``ActivationRecord``. Co-attaching a
        non-read-only SAE at an already-occupied site is rejected because
        the post-hook hidden state would be ambiguous.

        Args:
            config: SAE configuration. The SAE weights are downloaded
                    from HuggingFace on first call (cached thereafter).
        """
        sae_id = config.identity()
        key = (config.layer_index, config.hook_type, sae_id)
        if key in self._saes:
            raise ValueError(
                f"SAE already registered at site (L{config.layer_index}, "
                f"{config.hook_type.value}, {sae_id}). Use a different "
                "width / l0_size or remove the duplicate."
            )
        site = (config.layer_index, config.hook_type)
        existing_at_site = [
            (sid, c) for (li, ht, sid), c in self._configs.items()
            if (li, ht) == site
        ]
        if existing_at_site:
            offenders = [sid for sid, c in existing_at_site if not c.read_only]
            if not config.read_only:
                offenders.append(f"new:{sae_id}")
            if offenders:
                raise ValueError(
                    f"Cannot co-attach SAE {sae_id!r} at site "
                    f"(L{config.layer_index}, {config.hook_type.value}): "
                    f"every SAE sharing a site must have read_only=True "
                    f"(offenders: {offenders})."
                )
        self._saes[key] = load_sae(config)
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

        # Group SAEs by (layer, hook_type) site so co-attached SAEs
        # share a single forward hook (one forward pass through the
        # layer, N SAE encodes).
        sites: dict[
            tuple[int, HookType], list[tuple[SAEBase, SAEConfigT, str]]
        ] = {}
        for (layer_idx, hook_type, sae_id), sae in self._saes.items():
            cfg = self._configs[(layer_idx, hook_type, sae_id)]
            sites.setdefault((layer_idx, hook_type), []).append(
                (sae, cfg, sae_id),
            )
        site_keys = set(sites.keys())

        for site_key, saes_at_site in sites.items():
            layer_idx, hook_type = site_key
            if layer_idx >= len(layers):
                raise ValueError(
                    f"layer_index {layer_idx} out of range "
                    f"(model has {len(layers)} layers)"
                )
            target = self._resolve_hook_target(layers[layer_idx], hook_type)
            steer_ops = self._resolved_by_layer.get(site_key, [])

            # When a single non-read-only SAE shares a site with steering
            # ops, the steered hidden state will be replaced by the SAE's
            # reconstruction. add_sae() forbids the multi-SAE variant of
            # this — read_only is required at shared sites.
            primary_cfg = saes_at_site[0][1]
            if steer_ops and not primary_cfg.read_only and len(saes_at_site) == 1:
                warnings.warn(
                    f"Layer {layer_idx} ({hook_type.value}) has both steering "
                    "ops and an SAE with read_only=False. The steered hidden "
                    "state will be replaced by its (lossy) SAE reconstruction.",
                    stacklevel=2,
                )

            handle = target.register_forward_hook(
                self._make_sae_site_hook(site_key, saes_at_site, steer_ops),
            )
            self._handles.append(handle)

        for (layer_idx, hook_type), steer_ops in self._resolved_by_layer.items():
            if (layer_idx, hook_type) in site_keys:
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
            # SteeringOp.sae_key is a 2-tuple (layer, hook_type). Find any
            # registered SAE at that site; warn if more than one matches
            # (steering against a co-attached site is unusual — pick the
            # first by registration order).
            site = op.sae_key or (op.layer_index, op.hook_type)
            sae = None
            if op.feature_index is not None:
                matches = [
                    (k, s) for k, s in self._saes.items()
                    if (k[0], k[1]) == site
                ]
                if matches:
                    if len(matches) > 1:
                        ids = [k[2] for k, _ in matches]
                        warnings.warn(
                            f"Steering op at (L{site[0]}, {site[1].value}) "
                            f"matches {len(matches)} co-attached SAEs ({ids}); "
                            "resolving against the first by registration order.",
                            stacklevel=2,
                        )
                    sae = matches[0][1]
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

    def _make_sae_site_hook(
        self,
        site_key: tuple[int, HookType],
        saes_at_site: list[tuple[SAEBase, SAEConfigT, str]],
        steer_ops: list[ResolvedSteeringOp],
    ):
        """Combined forward hook that runs every SAE registered at ``site_key``.

        When only one SAE is attached at the site, records are stored with
        ``sae_id=""`` — backwards-compatible with the legacy single-SAE
        callers that read via ``store.prefill(layer, hook_type)`` without
        an ``sae_id`` argument. With two or more SAEs, each SAE's records
        use its own ``identity()`` slug so they're individually addressable.
        """
        layer_idx, hook_type = site_key
        # add_sae() enforces read_only=True at shared sites, so when
        # len(saes_at_site) > 1 every config has read_only=True. The
        # single-SAE branch still supports write-back (e.g. steering +
        # SAE reconstruction replacing the output).
        single_sae = len(saes_at_site) == 1

        def hook_fn(
            module: nn.Module,
            inputs: tuple,
            output: torch.Tensor,
        ) -> torch.Tensor | tuple | None:
            with torch.no_grad():
                hidden_states = output[0] if isinstance(output, tuple) else output
                if steer_ops:
                    hidden_states = apply_steering(
                        hidden_states, steer_ops, self.strength_multiplier,
                    )

                last_reconstruction = None
                any_write = False
                for sae_ref, cfg, sae_id in saes_at_site:
                    if cfg.prefill_only and self._prefill_done:
                        continue
                    feature_acts, reconstruction = sae_ref(hidden_states)
                    record_sae_id = "" if single_sae else sae_id
                    self.store.record(
                        (layer_idx, hook_type, record_sae_id),
                        feature_acts.detach(),
                        reconstruction=reconstruction.detach(),
                        collect_last_only=cfg.collect_last_only,
                    )
                    if not cfg.read_only:
                        any_write = True
                        last_reconstruction = reconstruction

                if any_write:
                    if isinstance(output, tuple):
                        return (last_reconstruction,) + output[1:]
                    return last_reconstruction
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
