"""Service wrapping the interpret/ toolkit for SAE inference via GraphQL.

Manages the Gemma3 model lifecycle (load on demand, stay resident, explicit
unload) and exposes three use cases:

1. **Prompt activations** — run a prompt through the model with SAE hooks,
   return per-token top-k feature activations with Neuronpedia labels.
2. **Steered generation** — apply additive steering on an SAE feature
   direction and generate baseline vs steered text.
3. **Prompt highlight** — run a prompt, max-pool SAE activations across
   tokens, return nonzero (feature_index, activation) pairs for scatter
   plot highlighting.

All public methods are **synchronous** (blocking).  The GraphQL mutation
layer wraps them with ``asyncio.to_thread()`` and acquires ``self._lock``
to serialise GPU access.
"""

import asyncio
import gc
import logging
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .token_emitter import emit_token

# The interpret/ toolkit uses `interpret.*` absolute imports internally.
# Ensure its parent directory is on sys.path so those imports resolve
# when the backend is started from the project root.
_INTERPRET_PARENT = str(Path(__file__).resolve().parents[2])
if _INTERPRET_PARENT not in sys.path:
    sys.path.insert(0, _INTERPRET_PARENT)

from interpret.inference.gemma_pytorch import GemmaPytorchInference  # noqa: E402, I001
from interpret.sae.exploration.prompt_explorer import (  # noqa: E402
    PromptExplorer,
    PromptExplorerConfig,
)
from interpret.sae.hook_manager import HookManager  # noqa: E402
from interpret.sae.sae_config import (  # noqa: E402
    HOOK_TYPE_FROM_STR,
    GemmaScopeSAEConfig,
    HookType,
    WIDTH_TO_D_SAE,
)
from interpret.sae.steering import SteeringMode, SteeringOp  # noqa: E402
from interpret.sae import paths as sae_paths  # noqa: E402

logger = logging.getLogger("star_map." + __name__)

_DEFAULT_LAYERS = [9, 17, 22, 29]


# ---------------------------------------------------------------------------
# Service result dataclasses (plain Python, not Strawberry)
# ---------------------------------------------------------------------------


@dataclass
class ModelStatusResult:
    loaded: bool
    model_name: str | None = None
    device: str | None = None


@dataclass
class ActiveFeatureResult:
    index: int
    activation: float
    label: str
    density: float | None = None


@dataclass
class TokenFeaturesResult:
    token: str
    position: int
    features: list[ActiveFeatureResult] = field(default_factory=list)


@dataclass
class LayerActivationsResult:
    layer: int
    width: str
    tokens: list[TokenFeaturesResult] = field(default_factory=list)


@dataclass
class PromptActivationsResult:
    prompt: str
    token_strings: list[str] = field(default_factory=list)
    layers: list[LayerActivationsResult] = field(default_factory=list)


@dataclass
class SteeringSpec:
    """A single steering feature specification (service-internal)."""

    feature_index: int
    layer: int
    hook_type: str
    width: str
    strength: float


@dataclass
class SteeredGenerationResult:
    baseline_text: str
    steered_text: str
    steering: list[SteeringSpec]


@dataclass
class FeatureActivation:
    feature_index: int
    activation: float


# ---------------------------------------------------------------------------
# InterpretService
# ---------------------------------------------------------------------------


class InterpretService:
    """Manages Gemma3 model lifecycle and SAE inference operations.

    The ``_lock`` attribute is an :class:`asyncio.Lock` intended to be
    acquired by the GraphQL mutation layer (not inside service methods)
    to serialise GPU access across concurrent requests.
    """

    def __init__(self) -> None:
        self._wrapper: GemmaPytorchInference | None = None
        self._prompt_explorer: PromptExplorer | None = None
        self._model_name: str | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def get_status(self) -> ModelStatusResult:
        """Return the current model status."""
        if self._wrapper is None:
            return ModelStatusResult(loaded=False)
        return ModelStatusResult(
            loaded=True,
            model_name=self._model_name,
            device=str(self._wrapper.device),
        )

    def load_model(
        self,
        checkpoint: str = "google/gemma-3-4b-it",
    ) -> ModelStatusResult:
        """Load the Gemma model into GPU memory.

        Raises:
            RuntimeError: If a model is already loaded.
        """
        if self._wrapper is not None:
            raise RuntimeError(
                f"Model already loaded ({self._model_name}). Call unloadModel first."
            )
        logger.info("Loading model %s ...", checkpoint)
        self._wrapper = GemmaPytorchInference(checkpoint)
        self._model_name = checkpoint
        self._prompt_explorer = None  # rebuilt lazily
        logger.info("Model loaded on %s", self._wrapper.device)
        return self.get_status()

    def unload_model(self) -> ModelStatusResult:
        """Unload the model and free GPU memory."""
        if self._wrapper is None:
            return self.get_status()

        logger.info("Unloading model %s ...", self._model_name)
        del self._wrapper
        self._wrapper = None
        self._prompt_explorer = None
        self._model_name = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded, GPU memory released.")
        return self.get_status()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_model(self) -> GemmaPytorchInference:
        if self._wrapper is None:
            raise RuntimeError("Model not loaded. Call loadModel first.")
        return self._wrapper

    def _get_prompt_explorer(
        self,
        layers: list[int],
        width: str,
        top_k: int,
    ) -> PromptExplorer:
        """Return a PromptExplorer, creating one lazily.

        A new explorer is created whenever the requested config differs
        from the cached one, or when no explorer exists yet.
        """
        wrapper = self._require_model()

        need_rebuild = (
            self._prompt_explorer is None
            or self._prompt_explorer.config.layers != layers
            or self._prompt_explorer.config.width != width
            or self._prompt_explorer.config.top_k != top_k
        )
        if need_rebuild:
            config = PromptExplorerConfig(
                wrapper=wrapper,
                layers=layers,
                width=width,
                top_k=top_k,
            )
            # Derive labels dir from a default SAE config (only model ID matters)
            resolved_labels_dir = sae_paths.labels_dir(GemmaScopeSAEConfig(layer_index=0))
            if resolved_labels_dir.is_dir():
                config.labels_dir = resolved_labels_dir
            else:
                logger.warning(
                    "Neuronpedia labels directory not found at %s — "
                    "feature labels will be unavailable.",
                    resolved_labels_dir,
                )
            self._prompt_explorer = PromptExplorer(config)

        return self._prompt_explorer

    @staticmethod
    def _parse_hook_type(hook_type: str) -> HookType:
        ht = HOOK_TYPE_FROM_STR.get(hook_type)
        if ht is None:
            raise ValueError(
                f"Unknown hook_type '{hook_type}'. Valid: {list(HOOK_TYPE_FROM_STR.keys())}"
            )
        return ht

    # ------------------------------------------------------------------
    # UC1: Prompt activations
    # ------------------------------------------------------------------

    def run_prompt_activations(
        self,
        prompt: str,
        layers: list[int] | None,
        width: str,
        top_k: int,
    ) -> PromptActivationsResult:
        """Run a prompt through the model with SAE hooks.

        Returns per-token top-k feature activations with Neuronpedia labels
        for every requested layer.
        """
        self._require_model()
        effective_layers = layers if layers else _DEFAULT_LAYERS

        explorer = self._get_prompt_explorer(effective_layers, width, top_k)
        prompt_result = explorer.run_prompt(prompt, output_len=1, top_k=top_k)

        # Convert the toolkit's PromptResult → service dataclasses.
        layer_results: list[LayerActivationsResult] = []
        for layer_idx in sorted(prompt_result.layers.keys()):
            lr = prompt_result.layers[layer_idx]
            token_results: list[TokenFeaturesResult] = []
            for tf in lr.tokens:
                features = [
                    ActiveFeatureResult(
                        index=f.index,
                        activation=f.activation,
                        label=f.label,
                        density=f.density,
                    )
                    for f in tf.features
                ]
                token_results.append(
                    TokenFeaturesResult(
                        token=tf.token,
                        position=tf.position,
                        features=features,
                    )
                )
            layer_results.append(
                LayerActivationsResult(
                    layer=lr.layer,
                    width=lr.width,
                    tokens=token_results,
                )
            )

        return PromptActivationsResult(
            prompt=prompt,
            token_strings=list(prompt_result.token_strings),
            layers=layer_results,
        )

    # ------------------------------------------------------------------
    # UC2: Steered generation
    # ------------------------------------------------------------------

    def generate_steered(
        self,
        prompt: str,
        steering_specs: list[SteeringSpec],
        output_len: int,
        temperature: float | None,
    ) -> SteeredGenerationResult:
        """Generate baseline and steered text with one or more features.

        Each steering spec contributes an additive intervention resolved
        from the SAE decoder matrix ``w_dec[feature_index]``.  Multiple
        features can target the same or different layers.
        """
        wrapper = self._require_model()
        device = str(wrapper.device)

        if not steering_specs:
            raise ValueError("At least one steering spec is required.")

        # Validate all specs up-front before any inference.
        for spec in steering_specs:
            d_sae = WIDTH_TO_D_SAE.get(spec.width)
            if d_sae is None:
                raise ValueError(f"Unknown SAE width '{spec.width}'")
            if not 0 <= spec.feature_index < d_sae:
                raise ValueError(
                    f"feature_index {spec.feature_index} out of range [0, {d_sae})"
                )
            self._parse_hook_type(spec.hook_type)  # validate early

        # --- Baseline (no steering) ---
        baseline_text = wrapper.generate(
            prompt,
            output_len=output_len,
            temperature=temperature,
        )

        # --- Steered ---
        # Collect unique (layer, hook_type, width) combos for SAE loading.
        manager = HookManager()
        seen_sae_keys: set[tuple[int, str, str]] = set()
        for spec in steering_specs:
            ht = self._parse_hook_type(spec.hook_type)
            sae_key = (spec.layer, spec.hook_type, spec.width)
            if sae_key not in seen_sae_keys:
                seen_sae_keys.add(sae_key)
                manager.add_sae(GemmaScopeSAEConfig(
                    layer_index=spec.layer,
                    hook_type=ht,
                    width=spec.width,
                    device=device,
                    read_only=True,
                ))
            manager.add_steering(SteeringOp(
                layer_index=spec.layer,
                mode=SteeringMode.ADDITIVE,
                feature_index=spec.feature_index,
                strength=spec.strength,
                normalise=False,
                hook_type=ht,
            ))

        with manager.session(wrapper.model.model.layers):
            steered_text = wrapper.generate(
                prompt,
                output_len=output_len,
                temperature=temperature,
            )

        return SteeredGenerationResult(
            baseline_text=baseline_text,
            steered_text=steered_text,
            steering=steering_specs,
        )

    # ------------------------------------------------------------------
    # UC3: Prompt highlight (max-pooled activations for scatter plot)
    # ------------------------------------------------------------------

    def run_prompt_highlight(
        self,
        prompt: str,
        layer: int,
        width: str,
        hook_type: str,
    ) -> list[FeatureActivation]:
        """Run a prompt and return max-pooled SAE feature activations.

        The returned list contains ``(feature_index, activation)`` pairs
        for every feature whose max activation across tokens is nonzero.
        These map directly to ``metadata.index`` in SAE scatter plot
        collections.
        """
        wrapper = self._require_model()
        ht = self._parse_hook_type(hook_type)
        device = str(wrapper.device)

        sae_config = GemmaScopeSAEConfig(
            layer_index=layer,
            hook_type=ht,
            width=width,
            device=device,
            prefill_only=True,
            read_only=True,
        )

        manager = HookManager()
        manager.add_sae(sae_config)

        with manager.session(wrapper.model.model.layers) as store:
            wrapper.generate(prompt, output_len=1)
            record = store.prefill(layer=layer, hook_type=ht)

        if record is None:
            logger.warning("No prefill activations captured for layer %d", layer)
            return []

        # feature_acts shape: (batch=1, seq_len, d_sae) → max-pool → (d_sae,)
        feature_acts = record.feature_acts[0]  # (seq_len, d_sae)
        max_pooled = feature_acts.max(dim=0).values  # (d_sae,)

        # Return nonzero features, sorted by activation descending.
        nonzero_mask = max_pooled > 0
        nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=True)[0]

        if len(nonzero_indices) == 0:
            return []

        values = max_pooled[nonzero_indices]
        order = values.argsort(descending=True)
        sorted_indices = nonzero_indices[order]
        sorted_values = values[order]

        return [
            FeatureActivation(
                feature_index=int(idx),
                activation=float(val),
            )
            for idx, val in zip(sorted_indices, sorted_values, strict=True)
        ]

    # ------------------------------------------------------------------
    # UC3b: Batch prompt highlight (max-pooled activations for many docs)
    # ------------------------------------------------------------------

    def run_batch_highlight(
        self,
        documents: list[tuple[str, str]],
        layer: int,
        width: str,
        hook_type: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[tuple[str, list[FeatureActivation]]]:
        """Run SAE highlight inference on multiple documents.

        Keeps the HookManager session open across all documents so SAE
        weights are loaded only once.

        Parameters
        ----------
        documents:
            List of ``(item_id, text)`` pairs.
        layer:
            Transformer layer index for the SAE.
        width:
            SAE width string (e.g. ``"16k"``).
        hook_type:
            Hook type string (e.g. ``"RESID_POST"``).
        progress_callback:
            Optional ``(done, total)`` callback invoked after each document.

        Returns
        -------
        List of ``(item_id, activations)`` where *activations* is a list
        of :class:`FeatureActivation` for nonzero features, sorted by
        activation descending.
        """
        wrapper = self._require_model()
        ht = self._parse_hook_type(hook_type)
        device = str(wrapper.device)

        sae_config = GemmaScopeSAEConfig(
            layer_index=layer,
            hook_type=ht,
            width=width,
            device=device,
            prefill_only=True,
            read_only=True,
        )

        manager = HookManager()
        manager.add_sae(sae_config)

        total = len(documents)
        results: list[tuple[str, list[FeatureActivation]]] = []

        with manager.session(wrapper.model.model.layers) as store:
            for i, (item_id, text) in enumerate(documents):
                store.clear()
                try:
                    wrapper.generate(text, output_len=1)
                    record = store.prefill(layer=layer, hook_type=ht)
                except Exception:
                    logger.exception(
                        "Failed inference for item %s (%d/%d)", item_id, i + 1, total
                    )
                    results.append((item_id, []))
                    if progress_callback:
                        progress_callback(i + 1, total)
                    continue

                if record is None:
                    results.append((item_id, []))
                else:
                    feature_acts = record.feature_acts[0]  # (seq_len, d_sae)
                    max_pooled = feature_acts.max(dim=0).values  # (d_sae,)

                    nonzero_mask = max_pooled > 0
                    nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=True)[0]

                    if len(nonzero_indices) == 0:
                        activations: list[FeatureActivation] = []
                    else:
                        values = max_pooled[nonzero_indices]
                        order = values.argsort(descending=True)
                        sorted_indices = nonzero_indices[order]
                        sorted_values = values[order]
                        activations = [
                            FeatureActivation(
                                feature_index=int(idx),
                                activation=float(val),
                            )
                            for idx, val in zip(
                                sorted_indices, sorted_values, strict=True
                            )
                        ]
                    results.append((item_id, activations))

                if progress_callback:
                    progress_callback(i + 1, total)

        return results

    # ------------------------------------------------------------------
    # UC4: Streaming chat generation
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        turns: list[tuple[str, str]],
        stream_id: str,
        output_len: int = 256,
        temperature: float | None = None,
        top_p: float = 0.95,
        top_k: int = 64,
        cancel_event: threading.Event | None = None,
        steering_specs: list[SteeringSpec] | None = None,
    ) -> None:
        """Run streaming chat generation, emitting tokens via token_emitter.

        This is a blocking method intended to run in a thread via
        ``asyncio.to_thread()``. The caller (subscription resolver) must
        acquire ``self._lock`` before spawning the thread.

        Optional steering_specs activate SAE-based additive steering
        on one or more features during generation (same mechanism as
        ``generate_steered``).
        """
        wrapper = self._require_model()

        try:
            # Build optional HookManager for steering
            manager: HookManager | None = None
            if steering_specs:
                device = str(wrapper.device)
                manager = HookManager()
                seen_sae_keys: set[tuple[int, str, str]] = set()

                for spec in steering_specs:
                    ht = self._parse_hook_type(spec.hook_type)
                    width = spec.width
                    d_sae = WIDTH_TO_D_SAE.get(width)
                    if d_sae is None:
                        raise ValueError(f"Unknown SAE width '{width}'")
                    if not 0 <= spec.feature_index < d_sae:
                        raise ValueError(
                            f"feature_index {spec.feature_index} out of range [0, {d_sae})"
                        )

                    sae_key = (spec.layer, spec.hook_type, width)
                    if sae_key not in seen_sae_keys:
                        seen_sae_keys.add(sae_key)
                        manager.add_sae(GemmaScopeSAEConfig(
                            layer_index=spec.layer,
                            hook_type=ht,
                            width=width,
                            device=device,
                            read_only=True,
                        ))

                    manager.add_steering(
                        SteeringOp(
                            layer_index=spec.layer,
                            mode=SteeringMode.ADDITIVE,
                            feature_index=spec.feature_index,
                            strength=spec.strength,
                            normalise=False,
                            hook_type=ht,
                        )
                    )

            def _run_generation():
                for event in wrapper.generate_chat_stream(
                    turns,
                    output_len,
                    temperature,
                    top_p,
                    top_k,
                    cancel_event=cancel_event,
                ):
                    emit_token(
                        stream_id,
                        event.token_index,
                        event.token_id,
                        event.text_delta,
                        event.is_done,
                    )

            if manager is not None:
                with manager.session(wrapper.model.model.layers):
                    _run_generation()
            else:
                _run_generation()

        except Exception as e:
            logger.exception("Streaming generation failed")
            emit_token(stream_id, 0, 0, "", done=True, error=str(e))
