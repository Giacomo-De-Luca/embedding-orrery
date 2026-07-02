"""Config dataclasses + YAML loader for the steering-based autointerpreter.

This is the intervention counterpart to the read-the-activations autointerpreter
in the parent package. Instead of labelling a feature from the WordNet samples
that activate it, we steer the feature into Gemma-3's residual stream during
generation, collect the model's answers to a handful of fixed questions across a
strength sweep, and have an LLM judge name and rate the resulting behaviour.

Everything is driven from one YAML file (single experiment, or a list under an
``experiments:`` key) so a run is reproducible from a manifest. The loader reuses
the parent package's generic dataclass coercion (`_coerce` / `_load_raw`) and the
`BaseModelSpec` / `SAESpec` specs, and writes ``experiment.yaml`` via the parent's
`dump_yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from interpret.sae.autointerpreter.config import (
    PROJECT_ROOT,
    BaseModelSpec,
    SAESpec,
    _coerce,
    _load_raw,
)

DirectionKind = Literal["w_dec_parquet", "sae"]
SteeringModeName = Literal["additive"]
FeatureSortBy = Literal["pearson", "spearman", "feature_idx"]

# The four generic probe questions. Short, open-ended prompts where a steered
# behaviour is easy to spot against the baseline answer.
DEFAULT_QUESTIONS: list[str] = [
    "What is your favourite job?",
    "How do you feel?",
    "Tell me a story.",
    "What are the values in life?",
]

# Known-good Gemma-3 steering coefficients (residual norms run into the
# thousands; the model collapses above ~2000 — see poetry_directions config).
DEFAULT_STRENGTHS: list[float] = [800.0, 1000.0, 1200.0, 1400.0]


def resolve_path(p: str | Path) -> Path:
    """Anchor a project-relative path against PROJECT_ROOT (absolute paths pass through)."""
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass
class SteeringDirectionSpec:
    """Where the steering direction (one per feature) comes from.

    ``w_dec_parquet`` (default) reads decoder rows from an extracted parquet in
    ``resources/sae_vectors/`` — no model/SAE download. ``sae`` loads the SAE via
    ``HookManager.add_sae`` and steers by ``feature_index`` (downloads gemma-scope
    weights on first use).
    """

    kind: DirectionKind = "w_dec_parquet"
    w_dec_parquet_path: str | None = (
        "resources/sae_vectors/w_dec_layer9_resid_post_w16384.parquet"
    )


@dataclass
class SteeringFeatureSelection:
    """Which features to interpret, and where their activation-labels come from.

    Three mutually-exclusive ways to pick features (checked in this order):

    1. ``random_sample`` (non-Pearson) — seeded-random ``random_sample`` features
       from the decoder parquet (``direction.w_dec_parquet_path``), optionally
       restricted to a ``density_min``/``density_max`` band so dead and ubiquitous
       features are dropped. The comparison label is the parquet's Neuronpedia
       ``label``. Use this to escape the format-detector bias at high Pearson.
    2. ``from_scores_parquet`` (+ ``top_n``/``min_pearson``/``sort_by``) — select
       from a prior autointerpret ``scores.parquet``; its
       ``short_name``/``explanation``/``pearson`` become the comparison label.
    3. ``feature_indices`` — an explicit list (labels pulled from
       ``from_scores_parquet`` if also given, else absent).
    """

    feature_indices: list[int] | None = None
    from_scores_parquet: str | None = None
    top_n: int | None = 100
    min_pearson: float | None = None
    sort_by: FeatureSortBy = "pearson"
    # Non-Pearson random selection (reads density + Neuronpedia label from the
    # decoder parquet). When set, supersedes the two options above.
    random_sample: int | None = None
    random_seed: int = 42
    density_min: float | None = None
    density_max: float | None = None


@dataclass
class SteeringGenerationConfig:
    """The steer-and-generate sweep."""

    questions: list[str] = field(default_factory=lambda: list(DEFAULT_QUESTIONS))
    strengths: list[float] = field(default_factory=lambda: list(DEFAULT_STRENGTHS))
    include_baseline: bool = True  # strength-0 answers, computed once and shared
    max_tokens: int = 128  # -> GemmaPytorchInference.generate(output_len=...)
    temperature: float | None = None  # None = greedy (deterministic baseline)
    mode: SteeringModeName = "additive"
    normalise: bool = False  # match feature_index steering (raw decoder row)
    # Reject any |strength| above this — Gemma-3 collapses near ~2000.
    max_abs_strength: float = 2000.0


@dataclass
class SteeringJudgeConfig:
    """The LLM-judge stage, driven through the AgentSystem job queue."""

    task: str = "steering-judge"
    workers: int = 4
    reps_per_worker: int = 40
    model_override: str | None = "sonnet"
    poll_interval_seconds: int = 15
    # Overnight default: don't abort aggregation if a few judges fail/stale.
    fail_on_queue_errors: bool = False
    # Must match the task JSON's {variant}->empty collapsed paths.
    input_dir: str = "resources/jobs/steering-judge/input"
    results_dir: str = "resources/jobs/steering-judge/queue/results"


@dataclass
class SteeringOutputConfig:
    output_root: str = "resources/sae_autointerpret_steering"
    run_slug: str | None = None


@dataclass
class SteeringStageFlags:
    skip_generate: bool = False
    skip_judge: bool = False
    skip_aggregate: bool = False


@dataclass
class SteeringInterpretConfig:
    """Full configuration for one steering-autointerpret experiment."""

    base_model: BaseModelSpec = field(default_factory=BaseModelSpec)
    sae: SAESpec = field(default_factory=lambda: SAESpec(layer_index=9, width="16k"))
    direction: SteeringDirectionSpec = field(default_factory=SteeringDirectionSpec)
    features: SteeringFeatureSelection = field(default_factory=SteeringFeatureSelection)
    generation: SteeringGenerationConfig = field(default_factory=SteeringGenerationConfig)
    judge: SteeringJudgeConfig = field(default_factory=SteeringJudgeConfig)
    output: SteeringOutputConfig = field(default_factory=SteeringOutputConfig)
    stages: SteeringStageFlags = field(default_factory=SteeringStageFlags)

    def resolved_slug(self) -> str:
        return self.output.run_slug or f"steering_L{self.sae.layer_index}_w{self.sae.width}"

    def run_dir(self) -> Path:
        return resolve_path(self.output.output_root) / self.resolved_slug()

    # ── Feature + activation-label resolution ──────────────────────────────

    def resolve_features(self) -> list[dict]:
        """Return ``[{feature_index, activation_label}]`` for every feature to steer.

        ``activation_label`` is a dict (short_name/explanation/pearson/polarity)
        when a scores parquet is available for that index, else ``None``.
        """
        sel = self.features
        if sel.random_sample:
            return self._resolve_random_features()
        label_lookup: dict[int, dict] = {}

        if sel.from_scores_parquet:
            import pandas as pd

            df = pd.read_parquet(resolve_path(sel.from_scores_parquet))
            for r in df.itertuples(index=False):
                label_lookup[int(r.feature_idx)] = {
                    "short_name": getattr(r, "short_name", None),
                    "explanation": getattr(r, "explanation", None),
                    "pearson": _to_float(getattr(r, "pearson", None)),
                    "polarity": getattr(r, "polarity", None),
                }
            if sel.feature_indices is None:
                if sel.min_pearson is not None:
                    df = df[df["pearson"] >= sel.min_pearson]
                ascending = sel.sort_by == "feature_idx"
                df = df.sort_values(sel.sort_by, ascending=ascending)
                if sel.top_n:
                    df = df.head(sel.top_n)
                indices = [int(x) for x in df["feature_idx"].tolist()]
            else:
                indices = [int(i) for i in sel.feature_indices]
        else:
            if not sel.feature_indices:
                raise ValueError(
                    "features needs either `feature_indices` or `from_scores_parquet`"
                )
            indices = [int(i) for i in sel.feature_indices]

        return [
            {"feature_index": i, "activation_label": label_lookup.get(i)}
            for i in indices
        ]

    def _resolve_random_features(self) -> list[dict]:
        """Seeded-random features from the decoder parquet within a density band.

        The comparison label is the parquet's Neuronpedia ``label``. Overlap with
        any prior Pearson-selected run is left to chance (negligible at 16k).
        """
        import numpy as np
        import pandas as pd

        sel = self.features
        if not self.direction.w_dec_parquet_path:
            raise ValueError(
                "random_sample selection reads density/label from the decoder "
                "parquet, but direction.w_dec_parquet_path is unset."
            )
        df = pd.read_parquet(
            resolve_path(self.direction.w_dec_parquet_path),
            columns=["index", "density", "label"],
        )
        density = df["density"].astype(float)
        mask = pd.Series(True, index=df.index)
        if sel.density_min is not None:
            mask &= density >= sel.density_min
        if sel.density_max is not None:
            mask &= density <= sel.density_max
        pool = df[mask]
        if pool.empty:
            raise ValueError(
                f"no features in density band "
                f"[{sel.density_min}, {sel.density_max}]"
            )
        n = min(sel.random_sample, len(pool))
        rng = np.random.default_rng(sel.random_seed)
        picks = np.sort(rng.choice(len(pool), size=n, replace=False))
        rows = pool.iloc[picks]
        return [
            {
                "feature_index": int(idx),
                "activation_label": {
                    "short_name": None if label is None else str(label),
                    "explanation": None,
                    "pearson": None,
                    "polarity": None,
                    "source": "neuronpedia",
                },
            }
            for idx, label in zip(rows["index"].to_numpy(), rows["label"].to_numpy())
        ]


def _to_float(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


# ── Loader ─────────────────────────────────────────────────────────────────

def _validate(cfg: SteeringInterpretConfig) -> None:
    g = cfg.generation
    if g.mode != "additive":
        raise ValueError(f"generation.mode must be 'additive', got {g.mode!r}")
    if not g.questions:
        raise ValueError("generation.questions must be non-empty")
    if not g.strengths:
        raise ValueError("generation.strengths must be non-empty")
    over = [s for s in g.strengths if abs(s) > g.max_abs_strength]
    if over:
        raise ValueError(
            f"generation.strengths {over} exceed max_abs_strength="
            f"{g.max_abs_strength} (Gemma-3 collapses near ~2000)."
        )
    if cfg.direction.kind == "w_dec_parquet" and not cfg.direction.w_dec_parquet_path:
        raise ValueError("direction.kind='w_dec_parquet' requires w_dec_parquet_path")
    if cfg.direction.kind not in ("w_dec_parquet", "sae"):
        raise ValueError(f"direction.kind invalid: {cfg.direction.kind!r}")


def load_steering_experiments(path: Path | str) -> list[SteeringInterpretConfig]:
    """Load one or more steering experiments from a YAML/JSON file."""
    raw = _load_raw(Path(path))
    if raw is None:
        raise ValueError(f"Empty config file: {path}")
    if isinstance(raw, dict) and "experiments" in raw:
        items = raw["experiments"]
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    configs = [_coerce(SteeringInterpretConfig, item) for item in items]
    for cfg in configs:
        _validate(cfg)
    return configs
