"""Configuration dataclasses + YAML loader for the SAE autointerpreter pipeline.

All stages are configured from a single YAML (or JSON) file so an entire
experiment is reproducible from a manifest. A top-level file may contain
either a single experiment (keys: ``run_slug``, ``collect``, ``extract``,
``agents``, ``score``, ``stages``) or a list under the ``experiments`` key
for sweep runs.

Example YAML::

    run_slug: debug_L29_16k_last
    collect:
      layer_index: 29
      width: "16k"
      aggregation: last_token
      limit: 5000
    extract:
      top_k: 50
    agents:
      label_workers: 5
      show_zero_fraction_to_evaluator: ab
    score:
      min_pearson: 0.3
"""

from __future__ import annotations

import json
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from interpret.sae.sae_config import (
    GemmaScopeSAEConfig,
    HookType,
    QwenScopeSAEConfig,
    SAEConfig,
)

Aggregation = Literal["last_token", "mean_prefill", "max_prefill"]
ZeroHintMode = Literal["on", "off", "ab"]
Family = Literal["gemma", "qwen"]
SourceKind = Literal["sae", "embedding", "residual"]
# Dense (embedding-dimension) extraction modes:
#   "signed" — one feature per dim; both poles shown; signed eval scores.
#   "split"  — two features per dim (pos = max(0,x), neg = max(0,-x)); each
#              non-negative, reusing the SAE-style 0-10 rubric.
DimMode = Literal["signed", "split"]
DimSelect = Literal["all", "top_variance"]

# Repository root used to anchor every project-relative path produced by
# this config (output directories, WordNet XML, label store, …). Computed
# from this file's location so the answer is stable no matter where the
# Python process was invoked from.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

# Toolkit root (the `interpret/` package in the parent repo; the repo root in
# a standalone astrolabe checkout). Anchors toolkit-internal resources — the
# agent_system job queue/launcher and the autointerpret YAML configs — so the
# agent stages keep working when the toolkit is extracted on its own.
TOOLKIT_ROOT: Path = Path(__file__).resolve().parents[2]
AGENT_SYSTEM_DIR: Path = TOOLKIT_ROOT / "agent_system"


@dataclass
class SAESpec:
    """One SAE to capture during a collect pass.

    ``family`` discriminates between Gemma-scope and Qwen-scope; the
    family-specific fields below are then consumed by :meth:`to_sae_config`.
    Defaults match the historical single-Gemma path so a yaml that omits
    family-specific fields still works for Gemma.
    """

    family: Family = "gemma"
    layer_index: int = 29
    hook_type: str = "resid_post"
    # Shared (Gemma "16k"/"65k"/"262k", Qwen "32k"/"64k"/"80k").
    width: str = "16k"
    # Gemma-specific.
    l0_size: str = "medium"
    model_size: str = "4b"
    variant: str = "it"
    # Qwen-specific (``model_size`` is reused — for Qwen e.g. ``"1.7B"``).
    k: int = 50

    def to_sae_config(self):
        """Build the right ``SAEConfig`` subclass for this spec."""
        if self.family == "gemma":
            return GemmaScopeSAEConfig(
                layer_index=self.layer_index,
                hook_type=HookType(self.hook_type),
                width=self.width,
                l0_size=self.l0_size,
                model_size=self.model_size,
                variant=self.variant,
            )
        if self.family == "qwen":
            return QwenScopeSAEConfig(
                layer_index=self.layer_index,
                hook_type=HookType(self.hook_type),
                model_size=self.model_size,
                width=self.width,
                k=self.k,
            )
        raise ValueError(f"Unsupported SAE family: {self.family!r}")


@dataclass
class BaseModelSpec:
    """Base-LM identification — selects the wrapper used by the collector.

    Defaults to the historical Gemma-3-4b-it path. Switch ``family`` to
    ``"qwen"`` and update ``checkpoint`` (e.g. ``"Qwen/Qwen3-1.7B-Base"``)
    to run a Qwen pass.
    """

    family: Family = "gemma"
    checkpoint: str = "google/gemma-3-4b-it"
    device: str = "mps"
    dtype: str = "bfloat16"
    add_bos: bool = True
    use_chat_template: bool = False


@dataclass
class EmbeddingSourceSpec:
    """Sentence-transformer embedding source for the autointerpreter.

    Produces one pooled (dense, signed) vector per WordNet prompt via the
    project's embedding factory (``scripts/utils/embedding_database``). A
    sentence-transformer pools internally, so no per-token ``aggregation``
    applies on this path.
    """

    provider: str = "sentence_transformers"
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "mps"
    normalize: bool = False
    # Prompt preset/string forwarded to the encoder. For EmbeddingGemma use the
    # sentence-similarity preset ``"STS"``; leave ``None`` for plain encoders
    # (e.g. all-MiniLM-L6-v2). Known preset names are passed as ``prompt_name``.
    prompt: str | None = None
    activation_dtype: Literal["float16", "float32"] = "float32"
    embed_batch_size: int = 64


# Capture points exposed by the forked gemma_pytorch per-layer activation
# cache, plus the top-level post-final-RMSNorm output.
RESIDUAL_INTERMEDIATES = {"pre_attn", "post_attn", "mlp_out", "post_mlp", "final_norm"}


@dataclass
class ResidualSiteSpec:
    """One residual-stream capture point for a raw-dimension collect pass.

    ``intermediate`` names a point in the per-layer cache (``pre_attn`` /
    ``post_attn`` / ``mlp_out`` / ``post_mlp``) or the top-level
    ``final_norm`` (after the final RMSNorm — ``layer_index`` is ignored
    there). ``post_mlp`` at layer L is the layer's output, i.e. the same
    ``resid_post`` site the Gemma-scope SAEs read, so raw dims collected
    there are directly comparable to that layer's SAE features.
    """

    layer_index: int = 29
    intermediate: str = "post_mlp"


@dataclass
class ResidualSourceSpec:
    """Raw residual-stream source (no SAE): dense, signed hidden-state dims.

    The base LM comes from the usual ``base_model`` / flat checkpoint fields
    and ``aggregation`` applies across prefill tokens exactly as on the SAE
    path. Each site lands in its own :class:`DenseActivationStore`, so the
    downstream stages treat it like an embedding run (``signed``/``split``
    dim modes, embed-axis/embed-dim agents).
    """

    sites: list[ResidualSiteSpec] = field(
        default_factory=lambda: [ResidualSiteSpec()],
    )
    # float32 by default: Gemma-3 late-layer residual components exceed the
    # float16 max (65504) — a float16 store silently saturates to inf.
    activation_dtype: Literal["float16", "float32"] = "float32"
    # Aggregation override for the residual capture: a single mode, a list
    # of modes, or ``None``. ``None`` inherits the collect-level
    # ``aggregation`` (the full list on the SAE side-capture path; the
    # single resolved mode on the standalone residual path). A list writes
    # one store per (site, aggregation) — e.g. last_token + max_prefill raw
    # residuals captured during one SAE pass. Multi-mode is honoured only on
    # the SAE side-capture path; the standalone residual collector is
    # single-store and rejects a list.
    aggregation: Aggregation | list[Aggregation] | None = None


def residual_subdir(site: ResidualSiteSpec) -> str:
    """Directory name for one residual site inside a multi-site umbrella run."""
    if site.intermediate == "final_norm":
        return "final_norm"
    return f"L{site.layer_index}_{site.intermediate}"


def normalize_aggregations(
    value: str | list[str] | tuple[str, ...] | None,
    default: list[str],
) -> list[str]:
    """Coerce a str/list/None aggregation field to a validated, unique list.

    ``None`` returns ``default`` (already validated). A str becomes a
    one-element list. Raises on empty, unknown, or duplicate modes — the
    same rules as :meth:`AutoInterpretCollectConfig.resolve_aggregations`.
    """
    if value is None:
        return list(default)
    aggs = list(value) if isinstance(value, (list, tuple)) else [value]
    if not aggs:
        raise ValueError("aggregation must name at least one mode")
    valid = {"last_token", "mean_prefill", "max_prefill"}
    for agg in aggs:
        if agg not in valid:
            raise ValueError(f"aggregation invalid: {agg!r}; valid: {sorted(valid)}")
    if len(set(aggs)) != len(aggs):
        raise ValueError(f"aggregation contains duplicates: {aggs}")
    return aggs


def residual_unit_layout(
    sites: list[ResidualSiteSpec], aggregations: list[str],
) -> list[tuple[ResidualSiteSpec, str, str]]:
    """On-disk layout for every (residual site, aggregation) store.

    Returns ``(site, aggregation, subdir_name)`` triples in site-major
    order. With one aggregation the subdir is the bare ``residual_subdir``
    (so a single-mode side-capture keeps the historical ``final_norm`` /
    ``L9_post_mlp`` names); with several it is suffixed ``_<aggregation>``.
    Unlike the SAE layout there is no root-level (``None``) case — residual
    side-capture always lands in a named subdir, since ``run_dir`` belongs
    to the SAE layout.
    """
    multi_agg = len(aggregations) > 1
    layout: list[tuple[ResidualSiteSpec, str, str]] = []
    for site in sites:
        for agg in aggregations:
            base = residual_subdir(site)
            layout.append((site, agg, f"{base}_{agg}" if multi_agg else base))
    return layout


def sae_subdir(spec: SAESpec) -> str:
    """Directory name for an SAE's outputs inside a multi-SAE umbrella run.

    Includes layer + width + hook_type to be unique across every
    combination we expect to mix in one pass (e.g. Gemma L9 W16K +
    L29 W16K + L29 W65K all resid_post).
    """
    return f"L{spec.layer_index}_w{spec.width}_{spec.hook_type}"


def sae_unit_layout(
    specs: list[SAESpec], aggregations: list[str],
) -> list[tuple[SAESpec, str, str | None]]:
    """On-disk layout for every (SAE, aggregation) store of a collect pass.

    Returns ``(spec, aggregation, subdir_name)`` triples in spec-major
    order; ``subdir_name`` is ``None`` for the legacy single-store layout
    (everything directly under ``run_dir``). Shared by the collector and
    the runner so the store writer and the downstream stages can never
    disagree about where a store lives:

    - 1 SAE × 1 aggregation  → legacy root layout (``None``).
    - N SAEs × 1 aggregation → ``L{n}_w{w}_{hook}`` (historical multi-SAE).
    - any × M aggregations   → ``L{n}_w{w}_{hook}_{aggregation}``.
    """
    multi = len(specs) * len(aggregations) > 1
    multi_agg = len(aggregations) > 1
    layout: list[tuple[SAESpec, str, str | None]] = []
    for spec in specs:
        for agg in aggregations:
            if not multi:
                sub = None
            elif multi_agg:
                sub = f"{sae_subdir(spec)}_{agg}"
            else:
                sub = sae_subdir(spec)
            layout.append((spec, agg, sub))
    return layout


@dataclass
class AutoInterpretCollectConfig:
    """Stage 1 — collect SAE feature activations over a WordNet slice."""

    # SAE identification (all configurable, never hardcoded downstream)
    layer_index: int = 29
    hook_type: str = "resid_post"                 # maps to HookType enum
    width: str = "16k"                            # "16k" | "65k" | "262k"
    l0_size: str = "medium"
    model_size: str = "4b"
    variant: str = "it"
    checkpoint: str = "google/gemma-3-4b-it"
    device: str = "mps"
    dtype: str = "bfloat16"

    # Aggregation across prefill tokens. A single mode (str) or a list of
    # modes: with a list, one forward pass writes one store per
    # (SAE, aggregation) combination — the forward dominates collect cost,
    # so capturing several conventions in one pass is nearly free.
    aggregation: Aggregation | list[Aggregation] = "last_token"

    # WordNet iteration
    limit: int | None = 5000                      # debug slice by default
    pos_filter: list[str] | None = None           # None = all POS
    prompt_template: str = "{word}: {definition}."
    add_bos: bool = True
    use_chat_template: bool = False
    # Path to the WordNet XML. Relative paths are resolved against
    # :data:`PROJECT_ROOT`, not the WordNetParser's script dir nor the
    # process cwd, so the existing repository cache is honoured even when
    # the pipeline is launched from a sibling directory. Set to ``None``
    # to let the parser fall back to its built-in default (and re-download
    # if the file is missing).
    wordnet_xml_path: str | None = "resources/dictionaries/english-wordnet-2024.xml"

    # Storage
    output_root: Path = Path("resources/sae_autointerpret")
    activation_dtype: Literal["float16", "float32"] = "float16"
    sparse_threshold: float = 1.0e-6
    flush_every: int = 5000
    resume: bool = True

    # Multi-SAE / multi-family extensions. Both default to ``None`` so
    # existing single-Gemma yamls (which set the flat fields above) load
    # unchanged. When set, they supersede the flat fields entirely.
    base_model: BaseModelSpec | None = None
    saes: list[SAESpec] | None = None

    # Source discriminator. ``"sae"`` (default) keeps every existing YAML
    # working unchanged; ``"embedding"`` switches the collector to
    # :class:`EmbeddingCollector` + a dense store, and the SAE-specific fields
    # above are ignored.
    source_kind: SourceKind = "sae"
    embedding: EmbeddingSourceSpec | None = None
    residual: ResidualSourceSpec | None = None

    def resolve_embedding(self) -> EmbeddingSourceSpec:
        """Return the embedding spec, defaulting a bare one when omitted."""
        return self.embedding if self.embedding is not None else EmbeddingSourceSpec()

    def resolve_residual(self) -> ResidualSourceSpec:
        """Return the residual spec, defaulting a bare one when omitted."""
        return self.residual if self.residual is not None else ResidualSourceSpec()

    def resolve_base_model(self) -> BaseModelSpec:
        """Return the new-style :class:`BaseModelSpec`, building from flat fields if absent."""
        if self.base_model is not None:
            return self.base_model
        return BaseModelSpec(
            family="gemma",
            checkpoint=self.checkpoint,
            device=self.device,
            dtype=self.dtype,
            add_bos=self.add_bos,
            use_chat_template=self.use_chat_template,
        )

    def resolve_aggregations(self) -> list[str]:
        """Return the aggregation modes as a validated, duplicate-free list."""
        if self.aggregation is None:
            raise ValueError("collect.aggregation must name at least one mode")
        return normalize_aggregations(self.aggregation, [])

    def resolve_residual_aggregations(self) -> list[str]:
        """Aggregation modes for the residual side-capture (inherits collect-level)."""
        return normalize_aggregations(
            self.resolve_residual().aggregation, self.resolve_aggregations(),
        )

    def resolve_saes(self) -> list[SAESpec]:
        """Return the SAE list — wraps the flat fields into a single Gemma spec when ``saes`` is None."""
        if self.saes is not None:
            return list(self.saes)
        return [
            SAESpec(
                family="gemma",
                layer_index=self.layer_index,
                hook_type=self.hook_type,
                width=self.width,
                l0_size=self.l0_size,
                model_size=self.model_size,
                variant=self.variant,
            )
        ]

    def to_sae_config(self) -> SAEConfig:
        """Build the SAEConfig for the *first* resolved spec.

        Kept for backwards compat with single-SAE callers (extract,
        score). For multi-SAE iteration, use ``resolve_saes()`` and
        ``spec.to_sae_config()`` directly.
        """
        first = self.resolve_saes()[0]
        cfg = first.to_sae_config()
        # Honour the collect-time hook policy that single-SAE callers
        # historically expected on the returned config.
        cfg.dtype = self.dtype
        cfg.device = self.device
        cfg.collect_last_only = True
        cfg.prefill_only = True
        cfg.read_only = True
        return cfg

    def run_slug(self) -> str:
        """Deterministic directory name for this collection run.

        Single-SAE Gemma runs use the historical ``<np_model>_L<n>_<hook>_w<w>_<agg>``
        slug for backwards compat. Multi-SAE runs derive a compact umbrella
        slug from the spec list (callers that care override via the
        top-level ``run_slug:`` field on :class:`AutoInterpretConfig`).
        Embedding runs derive ``emb_<model>`` from the model name; residual
        runs derive ``resid_<checkpoint>_<sites>_<agg>``.
        """
        if self.source_kind == "embedding":
            model_slug = self.resolve_embedding().model_name.split("/")[-1]
            return f"emb_{model_slug.replace('-', '_')}"
        if self.source_kind == "residual":
            ckpt = self.resolve_base_model().checkpoint.split("/")[-1]
            sites = "_".join(
                residual_subdir(s) for s in self.resolve_residual().sites
            )
            return f"resid_{ckpt.replace('-', '_')}_{sites}_{self.resolve_aggregations()[0]}"
        agg_slug = "_".join(self.resolve_aggregations())
        specs = self.resolve_saes()
        if len(specs) == 1 and specs[0].family == "gemma" and self.saes is None:
            sae = self.to_sae_config()
            return (
                f"{sae.neuronpedia_model_id}_L{self.layer_index}"
                f"_{self.hook_type}_w{self.width}_{agg_slug}"
            )
        bm = self.resolve_base_model()
        parts = [sae_subdir(spec) for spec in specs]
        return f"{bm.family}_{'_'.join(parts)}_{agg_slug}"

    def run_dir(self) -> Path:
        return Path(self.output_root) / self.run_slug()


@dataclass
class TopKExtractConfig:
    """Stage 2 — top-k + np.linspace sampling over collected activations."""

    top_k: int = 50
    eval_sample_count: int = 50
    eval_shuffle_seed: int = 42

    density_min: float = 1.0e-5
    density_max: float = 0.3
    require_min_nonzero: int = 50

    feature_indices: list[int] | None = None       # None = all that pass filter

    # ── Dense (embedding-dimension) extraction only — ignored on the SAE path ──
    # How to treat each signed embedding dimension (see :data:`DimMode`).
    dim_mode: DimMode = "signed"
    # Dimension selection: "all" labels every dim (~384-1024, no filter);
    # "top_variance" keeps the highest-variance dims (capped by max_features).
    select: DimSelect = "all"
    min_variance: float = 0.0                       # drop dims below this variance
    max_features: int | None = None                 # cap when select="top_variance"
    # Read the activation matrix from another run_dir (e.g. to run a second
    # dim_mode over an already-collected embedding pass without re-embedding).
    # None = read from this run's own directory.
    activations_run_dir: str | None = None


@dataclass
class AgentStageConfig:
    """Stage 3+4 — launch interpreter and evaluator agents via AgentSystem."""

    label_task: str = "autointerpret-label"
    eval_task: str = "autointerpret-eval"
    label_workers: int = 5
    label_reps_per_worker: int = 100
    eval_workers: int = 5
    eval_reps_per_worker: int = 100
    show_zero_fraction_to_evaluator: ZeroHintMode = "ab"
    poll_interval_seconds: int = 10
    # Per-run agent model override (e.g. "sonnet"). When set, the launcher's
    # ``-m`` flag overrides the model in the task JSON for both stages. None
    # keeps each task JSON's declared model (haiku for the autointerpret tasks).
    model_override: str | None = None
    # When True (default), the orchestrator raises ``RuntimeError`` if the
    # queue settles with any ``failed`` items rather than letting downstream
    # stages run against truncated input. Flip to ``False`` only for
    # exploratory smoke runs where partial completion is acceptable.
    fail_on_queue_errors: bool = True

    # Where input files live (matches the task JSONs' input_folder)
    label_input_dir: Path = Path("resources/jobs/autointerpret-label/input")
    label_results_dir: Path = Path("resources/jobs/autointerpret-label/queue/results")
    eval_input_dir: Path = Path("resources/jobs/autointerpret-eval/input")
    eval_results_dir: Path = Path("resources/jobs/autointerpret-eval/queue/results")


@dataclass
class AutoInterpretScoreConfig:
    """Stage 5 — score correlations and optionally push back to FeatureLabelStore."""

    write_to_label_store: bool = True
    label_store_dir: Path = Path("resources/sae_labels/neuronpedia_gemma-3-4b-it")
    method_name: str = "autointerpret"
    min_pearson: float = 0.3
    report_ab_split: bool = True


@dataclass
class StageFlags:
    """Skip flags so a run can resume partway through."""

    skip_collect: bool = False
    skip_topk: bool = False
    skip_label: bool = False
    skip_eval: bool = False
    skip_score: bool = False


@dataclass
class AutoInterpretConfig:
    """Full configuration for one autointerpreter experiment."""

    run_slug: str | None = None
    collect: AutoInterpretCollectConfig = field(default_factory=AutoInterpretCollectConfig)
    extract: TopKExtractConfig = field(default_factory=TopKExtractConfig)
    agents: AgentStageConfig = field(default_factory=AgentStageConfig)
    score: AutoInterpretScoreConfig = field(default_factory=AutoInterpretScoreConfig)
    stages: StageFlags = field(default_factory=StageFlags)

    def resolved_slug(self) -> str:
        return self.run_slug or self.collect.run_slug()


# ── Loader ───────────────────────────────────────────────────────────────────

def _unwrap_optional(ftype):
    """Strip ``T | None`` / ``Optional[T]`` down to ``T``; return ftype unchanged otherwise.

    Handles both the ``typing.Union`` form (``Optional[T]``) and the
    PEP-604 ``T | None`` form, whose origin is ``types.UnionType``.
    """
    import types as _types

    origin = typing.get_origin(ftype)
    if origin is typing.Union or origin is _types.UnionType:
        non_none = [a for a in typing.get_args(ftype) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return ftype


def _coerce_value(ftype, value):
    """Coerce one yaml value into the declared dataclass field type."""
    if value is None:
        return None
    ftype = _unwrap_optional(ftype)
    origin = typing.get_origin(ftype)
    if origin is list and typing.get_args(ftype):
        inner = typing.get_args(ftype)[0]
        if is_dataclass(inner):
            return [_coerce(inner, item) for item in value]
        return value
    if ftype is Path:
        return Path(value)
    if is_dataclass(ftype) and isinstance(value, dict):
        return _coerce(ftype, value)
    return value


def _coerce(cls, data: dict[str, Any]):
    """Recursively build a dataclass from a plain dict.

    Handles nested dataclasses, ``list[Dataclass]`` (so the new
    ``saes: list[SAESpec]`` shape coerces correctly), ``Optional[T]``
    unwrapping, and ``Path`` casting.
    """
    if not is_dataclass(cls) or data is None:
        return data
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = hints.get(f.name, f.type)
        kwargs[f.name] = _coerce_value(ftype, value)
    return cls(**kwargs)


def _load_raw(path: Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith((".yaml", ".yml")):
        return yaml.safe_load(text)
    if str(path).endswith(".json"):
        return json.loads(text)
    raise ValueError(f"Unsupported config extension: {path}")


def load_experiments(path: Path | str) -> list[AutoInterpretConfig]:
    """Load one or more experiments from a YAML/JSON file.

    Returns a list even for a single-experiment file (for uniform iteration).
    """
    raw = _load_raw(Path(path))
    if raw is None:
        raise ValueError(f"Empty config file: {path}")
    if isinstance(raw, dict) and "experiments" in raw:
        items = raw["experiments"]
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    configs = [_coerce(AutoInterpretConfig, item) for item in items]
    for cfg in configs:
        _validate(cfg)
    return configs


def _validate(cfg: AutoInterpretConfig) -> None:
    c = cfg.collect
    if cfg.agents.show_zero_fraction_to_evaluator not in {"on", "off", "ab"}:
        raise ValueError(
            f"show_zero_fraction_to_evaluator must be on/off/ab, "
            f"got {cfg.agents.show_zero_fraction_to_evaluator!r}"
        )

    if c.source_kind == "embedding":
        _validate_embedding(c)
        return
    if c.source_kind == "residual":
        _validate_residual(c)
        return
    if c.source_kind != "sae":
        raise ValueError(
            f"source_kind must be 'sae', 'embedding' or 'residual', "
            f"got {c.source_kind!r}"
        )

    c.resolve_aggregations()  # raises on empty / unknown / duplicate modes

    # SAE-list validation. ``resolve_saes()`` produces the spec list either
    # from the explicit ``saes:`` key or by wrapping the flat fields.
    specs = c.resolve_saes()
    base = c.resolve_base_model()
    seen_keys: set[tuple[int, str, str]] = set()
    for i, spec in enumerate(specs):
        if spec.family not in {"gemma", "qwen"}:
            raise ValueError(
                f"saes[{i}].family must be 'gemma' or 'qwen', got {spec.family!r}"
            )
        if spec.family != base.family:
            raise ValueError(
                f"saes[{i}].family={spec.family!r} does not match "
                f"base_model.family={base.family!r}; one collect pass uses "
                "one base LM, so every SAE must belong to the same family."
            )
        if spec.hook_type not in {"resid_post", "mlp_out", "attn_out"}:
            raise ValueError(f"saes[{i}].hook_type invalid: {spec.hook_type!r}")
        # Constructing the SAEConfig validates family-specific fields
        # (e.g. Qwen registry lookups, Gemma width whitelist).
        sae_cfg = spec.to_sae_config()
        key = (spec.layer_index, spec.hook_type, sae_cfg.identity())
        if key in seen_keys:
            raise ValueError(
                f"duplicate SAE spec in saes[]: "
                f"(layer={spec.layer_index}, hook={spec.hook_type}, "
                f"id={sae_cfg.identity()}) appears more than once."
            )
        seen_keys.add(key)

    # Optional side-capture: a `residual:` block on the SAE path collects
    # raw base-model embeddings (e.g. final_norm last_token) during the
    # same forward pass. Gemma-only — capture reads the forked
    # gemma_pytorch activation cache.
    if c.residual is not None:
        if base.family != "gemma":
            raise ValueError(
                "residual side-capture during an SAE collect is implemented "
                "for the Gemma wrapper only (forked gemma_pytorch activation "
                f"cache); got base_model.family={base.family!r}"
            )
        # Side-capture may fan out over several aggregations (one store per
        # site × mode), so multi-mode is allowed here.
        _validate_residual_sites(c.residual, allow_multi_agg=True)


def _validate_embedding(c: AutoInterpretCollectConfig) -> None:
    """Validate the embedding source block (family/hook checks don't apply).

    Only the ``sentence_transformers`` provider is supported by
    :class:`EmbeddingCollector` so far, so the provider is a plain string
    check — we deliberately don't import the embedding stack here, keeping this
    config module decoupled from ``scripts/`` (and chromadb).
    """
    if c.embedding is None:
        raise ValueError(
            "source_kind='embedding' requires an `embedding:` block"
        )
    if c.embedding.provider != "sentence_transformers":
        raise ValueError(
            f"embedding.provider {c.embedding.provider!r} is not supported; "
            "only 'sentence_transformers' is implemented for the embedding "
            "autointerpreter."
        )


def _validate_residual(c: AutoInterpretCollectConfig) -> None:
    """Validate the residual source block.

    The collector reads the forked ``gemma_pytorch`` internal activation
    cache, so only the Gemma family is accepted (Qwen capture would go
    through ``Qwen3Inference.cache_activations`` — not wired yet). Layer
    bounds are checked at collect time against the loaded model's depth.
    """
    if c.residual is None:
        raise ValueError("source_kind='residual' requires a `residual:` block")
    base = c.resolve_base_model()
    if base.family != "gemma":
        raise ValueError(
            "source_kind='residual' is implemented for the Gemma wrapper only "
            f"(forked gemma_pytorch activation cache); got "
            f"base_model.family={base.family!r}"
        )
    aggs = c.resolve_aggregations()
    if len(aggs) > 1:
        raise ValueError(
            "source_kind='residual' supports a single aggregation mode per "
            f"pass (the residual collector is single-store); got {aggs}"
        )
    # Standalone residual path is single-store: reject a multi-mode override.
    _validate_residual_sites(c.residual, allow_multi_agg=False)


def _validate_residual_sites(
    spec: ResidualSourceSpec, allow_multi_agg: bool,
) -> None:
    """Site-list + aggregation checks shared by the residual source and side-capture."""
    if not spec.sites:
        raise ValueError("residual.sites must list at least one capture site")
    seen: set[str] = set()
    for i, site in enumerate(spec.sites):
        if site.intermediate not in RESIDUAL_INTERMEDIATES:
            raise ValueError(
                f"residual.sites[{i}].intermediate invalid: "
                f"{site.intermediate!r}; valid: {sorted(RESIDUAL_INTERMEDIATES)}"
            )
        key = residual_subdir(site)
        if key in seen:
            raise ValueError(f"duplicate residual site: {key}")
        seen.add(key)
    # Validate the override list (None inherits, so skip — the collect-level
    # field is validated separately). normalize_aggregations raises on
    # unknown/duplicate modes.
    if spec.aggregation is not None:
        agg_list = normalize_aggregations(spec.aggregation, [])
        if not allow_multi_agg and len(agg_list) > 1:
            raise ValueError(
                "residual.aggregation must be a single mode on the standalone "
                f"residual path (single-store); got {agg_list}"
            )


def dump_yaml(cfg: AutoInterpretConfig, path: Path) -> None:
    """Write a config back to YAML for reproducibility (experiment.yaml)."""

    def _as_dict(obj):
        if is_dataclass(obj):
            return {f.name: _as_dict(getattr(obj, f.name)) for f in fields(obj)}
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, list):
            return [_as_dict(x) for x in obj]
        return obj

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(_as_dict(cfg), sort_keys=False), encoding="utf-8")
