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

from ..utils.resource_paths import DIRECTIONS_DIR
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
from interpret.sae.loading import clear_sae_cache  # noqa: E402
from interpret.sae.sae_config import (  # noqa: E402
    HOOK_TYPE_FROM_STR,
    GemmaScopeSAEConfig,
    HookType,
    WIDTH_TO_D_SAE,
)
from interpret.sae.source_ids import neuronpedia_source_id  # noqa: E402
from interpret.sae.steering import SteeringMode, SteeringOp  # noqa: E402
from interpret.sae import paths as sae_paths  # noqa: E402

logger = logging.getLogger("orrery." + __name__)

# Default layers per model size — must match the published GemmaScope-2 residual
# SAE layers in google/gemma-scope-2-{size}-{variant} (identical for pt and it).
# Only layers for which SAE weights exist are listed; verified 2026-07 against the
# repo `resid_post/layer_{N}_...` folders (and mirrored on the Neuronpedia catalog).
_DEFAULT_LAYERS_BY_SIZE: dict[str, list[int]] = {
    "1b": [7, 13, 17, 22],
    "4b": [9, 17, 22, 29],
    "12b": [12, 24, 31, 41],
    "27b": [16, 31, 40, 53],
}
_DEFAULT_LAYERS = [9, 17, 22, 29]  # fallback for unknown sizes


# ---------------------------------------------------------------------------
# Direction-vector presets
# ---------------------------------------------------------------------------
#
# Activation-additive steering directions stored as pre-extracted 1-D ``.pt``
# files (model-dim vectors). Unlike SAE-feature steering, the vector is
# loaded directly rather than read from ``sae.w_dec``. The runtime applies
# them at ``RESID_POST`` of the configured layer. The metadata JSON files
# co-located with each ``.pt`` are provenance only; layer + model binding
# live here as the runtime source of truth.


@dataclass(frozen=True)
class DirectionPreset:
    name: str
    file: str  # filename inside DIRECTIONS_DIR
    layer: int  # applied at RESID_POST of this layer
    model_id: str  # gated to this model only


DIRECTION_REGISTRY: dict[str, DirectionPreset] = {
    "refusal": DirectionPreset(
        name="refusal",
        file="refusal_direction.pt",
        layer=14,
        model_id="gemma-3-4b-it",
    ),
    "poetry": DirectionPreset(
        name="poetry",
        file="poetry_direction.pt",
        layer=11,
        model_id="gemma-3-4b-it",
    ),
}


# ---------------------------------------------------------------------------
# Service result dataclasses (plain Python, not Strawberry)
# ---------------------------------------------------------------------------


@dataclass
class ModelStatusResult:
    loaded: bool
    model_name: str | None = None
    device: str | None = None
    variant: str | None = None  # "it" (instruction-tuned) or "pt" (pretrained/base)
    model_size: str | None = None  # "4b", "12b", etc.


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
    """A single steering specification (service-internal).

    Two mutually exclusive flavours:
      - SAE feature: ``feature_index`` + ``layer`` + ``hook_type`` + ``width``
        resolve a direction from ``sae.w_dec``.
      - Pre-extracted direction: ``direction_name`` resolves a 1-D vector
        from ``DIRECTION_REGISTRY``. The other fields are ignored.
    """

    feature_index: int
    layer: int
    hook_type: str
    width: str
    strength: float
    direction_name: str | None = None


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

    # Coverage filter constants
    COVERAGE_THRESHOLD: float = 0.80
    DENSITY_FLOOR: float = 1e-4
    NEURONPEDIA_TOP_K: int = 50

    def __init__(self) -> None:
        self._wrapper: GemmaPytorchInference | None = None
        self._prompt_explorer: PromptExplorer | None = None
        self._model_name: str | None = None
        self._model_size: str = "4b"
        self._variant: str = "it"
        self._lock = asyncio.Lock()
        self._direction_cache: dict[str, torch.Tensor] = {}

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
            variant=self._variant,
            model_size=self._model_size,
        )

    @staticmethod
    def _parse_checkpoint(checkpoint: str) -> tuple[str, str]:
        """Extract (model_size, variant) from a checkpoint string.

        Handles formats like:
            "google/gemma-3-4b-it"  → ("4b", "it")
            "google/gemma-3-1b"     → ("1b", "pt")
            "gemma-3-12b-it"        → ("12b", "it")
        """
        # Strip org prefix
        name = checkpoint.rsplit("/", 1)[-1]  # "gemma-3-4b-it"
        # Remove "gemma-3-" prefix if present
        if name.startswith("gemma-3-"):
            name = name[len("gemma-3-") :]  # "4b-it" or "1b"
        parts = name.split("-", 1)
        model_size = parts[0]  # "4b"
        variant = parts[1] if len(parts) > 1 else "pt"  # "it" or default "pt"
        return model_size, variant

    @staticmethod
    def _normalize_checkpoint(checkpoint: str, model_size: str, variant: str) -> str:
        """Ensure the checkpoint string has the variant suffix.

        HuggingFace repos require explicit variant: google/gemma-3-1b-pt
        (google/gemma-3-1b alone returns 404).
        """
        name = checkpoint.rsplit("/", 1)[-1]
        # If it already ends with the variant, leave it alone
        if name.endswith(f"-{variant}"):
            return checkpoint
        # Reconstruct with the variant suffix
        org = checkpoint.rsplit("/", 1)[0] if "/" in checkpoint else ""
        canonical = f"gemma-3-{model_size}-{variant}"
        return f"{org}/{canonical}" if org else canonical

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
        self._model_size, self._variant = self._parse_checkpoint(checkpoint)
        checkpoint = self._normalize_checkpoint(checkpoint, self._model_size, self._variant)
        logger.info("Loading model %s ...", checkpoint)
        self._wrapper = GemmaPytorchInference(
            checkpoint, model_size=self._model_size, precision="bfloat16"
        )
        self._model_name = checkpoint
        self._prompt_explorer = None  # rebuilt lazily
        logger.info("Model loaded on %s", self._wrapper.device)
        return self.get_status()

    @property
    def _neuronpedia_model_id(self) -> str:
        """Derive Neuronpedia model ID from the loaded model's size and variant."""
        base = f"gemma-3-{self._model_size}"
        return base if self._variant == "pt" else f"{base}-{self._variant}"

    def unload_model(self) -> ModelStatusResult:
        """Unload the model and free GPU memory."""
        if self._wrapper is None:
            return self.get_status()

        logger.info("Unloading model %s ...", self._model_name)
        del self._wrapper
        self._wrapper = None
        self._prompt_explorer = None
        self._model_name = None
        self._model_size = "4b"
        self._variant = "it"
        self._direction_cache.clear()
        clear_sae_cache()
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

    def _load_direction(self, name: str) -> torch.Tensor:
        """Resolve a 1-D direction vector by registry name, with per-instance caching.

        The cached tensor lives on the model's device with ``float32`` dtype.
        Cleared on ``unload_model`` so a fresh device is picked up on reload.
        """
        if name in self._direction_cache:
            return self._direction_cache[name]
        preset = DIRECTION_REGISTRY.get(name)
        if preset is None:
            raise KeyError(f"unknown direction: {name!r}")
        wrapper = self._require_model()
        if preset.model_id != self._neuronpedia_model_id:
            raise ValueError(
                f"direction {name!r} is bound to model {preset.model_id!r}, "
                f"but {self._neuronpedia_model_id!r} is loaded"
            )
        path = DIRECTIONS_DIR / preset.file
        if not path.exists():
            raise FileNotFoundError(f"direction file missing: {path}")
        vec = torch.load(path, map_location=wrapper.device, weights_only=False)
        if not isinstance(vec, torch.Tensor) or vec.ndim != 1:
            raise ValueError(
                f"direction {name!r} must be a 1-D tensor, got "
                f"{type(vec).__name__} shape={getattr(vec, 'shape', None)}"
            )
        vec = vec.to(dtype=torch.float32).contiguous()
        self._direction_cache[name] = vec
        return vec

    def _build_steering_session(
        self, steering_specs: list[SteeringSpec], device: str
    ) -> HookManager:
        """Build a HookManager populated with the given steering ops.

        Branches per-spec on ``direction_name``:
          - Direction specs use ``SteeringOp(vector=...)`` and skip ``add_sae``
            (registry lookup provides the application layer; site is
            ``RESID_POST``).
          - SAE specs validate ``(width, feature_index, hook_type)`` and add
            the matching ``GemmaScopeSAEConfig`` (deduped by key).
        """
        manager = HookManager()
        seen_sae_keys: set[tuple[int, str, str]] = set()
        for spec in steering_specs:
            if spec.direction_name is not None:
                preset = DIRECTION_REGISTRY.get(spec.direction_name)
                if preset is None:
                    raise KeyError(f"unknown direction: {spec.direction_name!r}")
                manager.add_steering(
                    SteeringOp(
                        layer_index=preset.layer,
                        mode=SteeringMode.ADDITIVE,
                        vector=self._load_direction(spec.direction_name),
                        strength=spec.strength,
                        normalise=False,
                        hook_type=HookType.RESID_POST,
                    )
                )
                continue

            ht = self._parse_hook_type(spec.hook_type)
            d_sae = WIDTH_TO_D_SAE.get(spec.width)
            if d_sae is None:
                raise ValueError(f"Unknown SAE width '{spec.width}'")
            if not 0 <= spec.feature_index < d_sae:
                raise ValueError(f"feature_index {spec.feature_index} out of range [0, {d_sae})")

            sae_key = (spec.layer, spec.hook_type, spec.width)
            if sae_key not in seen_sae_keys:
                seen_sae_keys.add(sae_key)
                manager.add_sae(
                    GemmaScopeSAEConfig(
                        layer_index=spec.layer,
                        hook_type=ht,
                        width=spec.width,
                        model_size=self._model_size,
                        variant=self._variant,
                        device=device,
                        read_only=True,
                    )
                )
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
        return manager

    def _get_prompt_explorer(
        self,
        saes: list[tuple[int, str]],
        top_k: int,
    ) -> PromptExplorer:
        """Return a PromptExplorer, creating one lazily.

        A new explorer is created whenever the requested (layer, width)
        specs or top_k differ from the cached one, or when no explorer
        exists yet.
        """
        wrapper = self._require_model()

        need_rebuild = (
            self._prompt_explorer is None
            or self._prompt_explorer.config.effective_saes() != saes
            or self._prompt_explorer.config.top_k != top_k
        )
        if need_rebuild:
            resolved_labels_dir = sae_paths.labels_dir(
                GemmaScopeSAEConfig(
                    layer_index=0,
                    model_size=self._model_size,
                    variant=self._variant,
                )
            )
            config = PromptExplorerConfig(
                wrapper=wrapper,
                saes=saes,
                top_k=top_k,
                skip_labels=True,
                model_size=self._model_size,
                variant=self._variant,
            )
            if resolved_labels_dir.is_dir():
                config.labels_dir = resolved_labels_dir
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

    def _find_prompt_token_range(self, token_strings: list[str], prompt: str) -> tuple[int, int]:
        """Find the start/end indices of the actual user prompt tokens.

        For IT models, the chat template wraps as:
          <bos><start_of_turn>user\\n{prompt}<end_of_turn>\\n<start_of_turn>model

        For base (pt) models, there's no template — just BOS + prompt tokens.

        Returns (start, end) where token_strings[start:end] are the prompt tokens.
        """
        # Base models: no chat template, skip only BOS (position 0)
        if self._variant == "pt":
            return 1, len(token_strings)

        # IT models: strip the chat template prefix/suffix
        wrapper = self._require_model()
        prefix = "<start_of_turn>user\n"
        prefix_ids = wrapper.tokenize(prefix, bos=True)
        start = len(prefix_ids)

        # Find <end_of_turn> token scanning backwards from the end
        end = len(token_strings)
        for i in range(len(token_strings) - 1, start - 1, -1):
            if "<end_of_turn>" in token_strings[i]:
                end = i
                break

        return start, end

    def _compute_coverage_exclusions(
        self,
        feature_acts: torch.Tensor,
        prompt_start: int,
        prompt_end: int,
        include_bos: bool,
        label_map: dict[int, tuple[str, float | None]],
    ) -> set[int]:
        """Return feature indices to exclude based on coverage + density floor.

        Args:
            feature_acts: Raw activation tensor ``(seq_len, d_sae)``.
            prompt_start: Start index of user prompt tokens.
            prompt_end: End index of user prompt tokens.
            include_bos: If True, coverage computed over all positions (incl.
                BOS and template). If False, only prompt token positions.
            label_map: ``{feature_index: (label, density)}`` from DuckDB.
        """
        fired = feature_acts > 0
        if include_bos:
            coverage = fired.sum(dim=0).float() / feature_acts.shape[0]
        else:
            n = max(prompt_end - prompt_start, 1)
            # Edge case: single-token prompt — skip coverage filter
            if n <= 1:
                coverage = torch.zeros(feature_acts.shape[1])
            else:
                coverage = fired[prompt_start:prompt_end].sum(dim=0).float() / n

        exclude = set(torch.nonzero(coverage >= self.COVERAGE_THRESHOLD, as_tuple=True)[0].tolist())

        # Dead/near-dead features below density floor
        for idx, (_, density) in label_map.items():
            if density is not None and density < self.DENSITY_FLOOR:
                exclude.add(idx)

        return exclude

    def run_prompt_activations(
        self,
        prompt: str,
        layers: list[int] | None,
        width: str,
        top_k: int,
        db_model_id: str | None = None,
        db_sae_id: str | None = None,
        skip_chat_template: bool = False,
        filter_mode: str = "neuronpedia",
        saes: list[tuple[int, str]] | None = None,
    ) -> PromptActivationsResult:
        """Run a prompt through the model with SAE hooks.

        Returns per-token feature activations with labels from DuckDB,
        filtered to only include the actual prompt tokens (no chat template).

        Args:
            db_model_id: DuckDB model_id for label lookup (from frontend selector).
            db_sae_id: DuckDB sae_id for label lookup (from frontend selector).
                Only meaningful for single-SAE calls; with multiple specs pass
                ``None`` so each layer derives its own width-aware sae_id.
            skip_chat_template: If True, treat the prompt as raw text even for
                instruction-tuned models (skip BOS only, no chat wrapping).
            filter_mode: One of ``"neuronpedia"``, ``"coverage_bos"``,
                ``"coverage_no_bos"``. Controls how features are filtered.
            saes: Explicit (layer, width) SAE specs — hooks all of them in a
                single forward pass (two widths at one layer co-attach).
                Takes precedence over ``layers`` + ``width``.
        """
        from backend.API.duckdb_instance import get_duckdb_client

        self._require_model()
        if saes:
            specs = [(int(layer), str(w)) for layer, w in saes]
        else:
            effective_layers = (
                layers if layers else _DEFAULT_LAYERS_BY_SIZE.get(self._model_size, _DEFAULT_LAYERS)
            )
            specs = [(layer, width) for layer in effective_layers]
        if len(specs) > 1:
            # A single db_sae_id would mislabel every other spec — force the
            # per-spec width-aware derivation below.
            db_sae_id = None

        # Mode-specific top-k: NEURONPEDIA caps at 50, coverage modes need all
        if filter_mode == "neuronpedia":
            effective_top_k = self.NEURONPEDIA_TOP_K
        else:
            effective_top_k = 0  # all nonzero

        explorer = self._get_prompt_explorer(specs, effective_top_k)
        prompt_result = explorer.run_prompt(prompt, output_len=1, top_k=effective_top_k)

        # Determine which token positions belong to the actual prompt
        all_token_strings = list(prompt_result.token_strings)
        if skip_chat_template:
            prompt_start, prompt_end = 1, len(all_token_strings)
        else:
            prompt_start, prompt_end = self._find_prompt_token_range(all_token_strings, prompt)
        prompt_token_strings = all_token_strings[prompt_start:prompt_end]

        # Use frontend-provided identifiers (authoritative), fall back to derived
        model_id = db_model_id or self._neuronpedia_model_id

        db = get_duckdb_client()

        layer_results: list[LayerActivationsResult] = []
        for key in sorted(prompt_result.layers.keys()):
            lr = prompt_result.layers[key]
            layer_idx, layer_width = key

            # Filter to only prompt token positions
            prompt_tokens = [tf for tf in lr.tokens if prompt_start <= tf.position < prompt_end]

            # Gather all unique feature indices for this layer
            all_indices: list[int] = list({f.index for tf in prompt_tokens for f in tf.features})

            # Batch fetch labels + densities from DuckDB
            if db_sae_id:
                sae_id = db_sae_id
            else:
                sae_id = neuronpedia_source_id(
                    GemmaScopeSAEConfig(
                        layer_index=layer_idx,
                        width=layer_width,
                        model_size=self._model_size,
                        variant=self._variant,
                    )
                )
            label_map: dict[int, tuple[str, float | None]] = {}
            if all_indices:
                label_map = db.get_sae_feature_labels_batch(model_id, sae_id, all_indices)

            # Compute exclusion set for coverage modes
            if filter_mode in ("coverage_bos", "coverage_no_bos"):
                exclude = self._compute_coverage_exclusions(
                    lr.feature_acts,
                    prompt_start,
                    prompt_end,
                    include_bos=(filter_mode == "coverage_bos"),
                    label_map=label_map,
                )
            else:
                exclude = set()

            token_results: list[TokenFeaturesResult] = []
            for tf in prompt_tokens:
                features = []
                for f in tf.features:
                    if f.index in exclude:
                        continue
                    db_label, db_density = label_map.get(f.index, ("", None))
                    density = db_density if db_density is not None else f.density
                    features.append(
                        ActiveFeatureResult(
                            index=f.index,
                            activation=f.activation,
                            label=db_label or f.label,
                            density=density,
                        )
                    )
                token_results.append(
                    TokenFeaturesResult(
                        token=tf.token,
                        position=tf.position - prompt_start,
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
            token_strings=prompt_token_strings,
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

        # Build (and validate) the steering session up-front so a malformed
        # spec fails before we burn cycles on the baseline generation.
        manager = self._build_steering_session(steering_specs, device)

        # --- Baseline (no steering) ---
        if self._variant == "pt":
            baseline_text = wrapper.generate_from_template(
                prompt,
                output_len=output_len,
                temperature=temperature,
            )
        else:
            baseline_text = wrapper.generate(
                prompt,
                output_len=output_len,
                temperature=temperature,
            )

        # --- Steered ---

        with manager.session(wrapper.model.model.layers):
            if self._variant == "pt":
                steered_text = wrapper.generate_from_template(
                    prompt,
                    output_len=output_len,
                    temperature=temperature,
                )
            else:
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
    # Shared activation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _max_pool_activations(record) -> list[FeatureActivation]:
        """Max-pool activation record across tokens, return nonzero features sorted desc."""
        feature_acts = record.feature_acts[0]  # (seq_len, d_sae)
        max_pooled = feature_acts.max(dim=0).values  # (d_sae,)

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
            model_size=self._model_size,
            variant=self._variant,
            device=device,
            prefill_only=True,
            read_only=True,
        )

        manager = HookManager()
        manager.add_sae(sae_config)

        with manager.session(wrapper.model.model.layers) as store:
            if self._variant == "pt":
                wrapper.generate_from_template(prompt, output_len=1)
            else:
                wrapper.generate(prompt, output_len=1)
            record = store.prefill(layer=layer, hook_type=ht)

        if record is None:
            logger.warning("No prefill activations captured for layer %d", layer)
            return []

        return self._max_pool_activations(record)

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
        result_callback: Callable[[str, list[FeatureActivation]], None] | None = None,
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
        result_callback:
            Optional ``(item_id, activations)`` callback invoked after each
            document so callers can persist results incrementally.

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
            model_size=self._model_size,
            variant=self._variant,
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
                manager.reset()
                try:
                    if self._variant == "pt":
                        wrapper.generate_from_template(text, output_len=1)
                    else:
                        wrapper.generate(text, output_len=1)
                    record = store.prefill(layer=layer, hook_type=ht)
                except Exception:
                    logger.exception("Failed inference for item %s (%d/%d)", item_id, i + 1, total)
                    activations: list[FeatureActivation] = []
                    results.append((item_id, activations))
                    if result_callback:
                        result_callback(item_id, activations)
                    if progress_callback:
                        progress_callback(i + 1, total)
                    continue

                activations = self._max_pool_activations(record) if record else []
                results.append((item_id, activations))
                if result_callback:
                    result_callback(item_id, activations)

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
        seed: int | None = None,
    ) -> None:
        """Run streaming chat generation, emitting tokens via token_emitter.

        This is a blocking method intended to run in a thread via
        ``asyncio.to_thread()``. The caller (subscription resolver) must
        acquire ``self._lock`` before spawning the thread.

        Optional steering_specs activate SAE-based additive steering
        on one or more features during generation (same mechanism as
        ``generate_steered``).

        When ``seed`` is set, the global PyTorch RNG is seeded immediately
        before sampling so output is reproducible. The GPU lock serialises
        streams, so a steered and a baseline call sharing one seed each get
        an identical fresh RNG state — making their differences attributable
        to steering rather than sampling noise.
        """
        wrapper = self._require_model()

        try:
            # Build optional HookManager for steering
            manager: HookManager | None = None
            if steering_specs:
                manager = self._build_steering_session(steering_specs, str(wrapper.device))

            def _run_generation():
                if self._variant == "pt":
                    # Base model: no chat template. Use the last user turn as
                    # a plain prompt via generate_from_template (non-streaming).
                    # NOTE: cancel_event is not checked — generate_from_template
                    # is a single blocking call. The user's Stop button will only
                    # take effect after generation completes.
                    last_user_content = next(
                        (content for role, content in reversed(turns) if role == "user"),
                        "",
                    )
                    text = wrapper.generate_from_template(
                        last_user_content,
                        output_len=output_len,
                        temperature=temperature,
                    )
                    # Emit the full response as a single token event
                    emit_token(stream_id, 0, 0, text, done=False)
                    emit_token(stream_id, 1, 0, "", done=True)
                else:
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

            # Seed immediately before sampling so both code paths (pt/it) are
            # covered and nothing consumes RNG between here and torch.multinomial.
            if seed is not None:
                torch.manual_seed(seed)

            if manager is not None:
                with manager.session(wrapper.model.model.layers):
                    _run_generation()
            else:
                _run_generation()

        except Exception as e:
            logger.exception("Streaming generation failed")
            emit_token(stream_id, 0, 0, "", done=True, error=str(e))
