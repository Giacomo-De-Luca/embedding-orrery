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

from interpret.sae.sae_config import HookType, SAEConfig

Aggregation = Literal["last_token", "mean_prefill", "max_prefill"]
ZeroHintMode = Literal["on", "off", "ab"]


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

    # Aggregation across prefill tokens
    aggregation: Aggregation = "last_token"

    # WordNet iteration
    limit: int | None = 5000                      # debug slice by default
    pos_filter: list[str] | None = None           # None = all POS
    prompt_template: str = "{word}: {definition}."
    add_bos: bool = True
    use_chat_template: bool = False

    # Storage
    output_root: Path = Path("resources/sae_autointerpret")
    activation_dtype: Literal["float16", "float32"] = "float16"
    sparse_threshold: float = 1.0e-6
    flush_every: int = 5000
    resume: bool = True

    def to_sae_config(self) -> SAEConfig:
        """Build the SAEConfig the HookManager wants."""
        return SAEConfig(
            layer_index=self.layer_index,
            hook_type=HookType(self.hook_type),
            model_size=self.model_size,
            variant=self.variant,
            width=self.width,
            l0_size=self.l0_size,
            dtype=self.dtype,
            device=self.device,
            collect_last_only=True,
            prefill_only=True,
            read_only=True,
        )

    def run_slug(self) -> str:
        """Deterministic directory name for this collection run."""
        sae = self.to_sae_config()
        return (
            f"{sae.neuronpedia_model_id}_L{self.layer_index}"
            f"_{self.hook_type}_w{self.width}_{self.aggregation}"
        )

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

def _coerce(cls, data: dict[str, Any]):
    """Recursively build a dataclass from a plain dict, casting Path fields."""
    if not is_dataclass(cls) or data is None:
        return data
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _coerce(ftype, value)
        elif ftype is Path:
            kwargs[f.name] = Path(value) if value is not None else None
        else:
            kwargs[f.name] = value
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
    if not (0 <= c.layer_index < 34):
        raise ValueError(f"layer_index {c.layer_index} out of range (0-33)")
    if c.width not in {"16k", "65k", "262k"}:
        raise ValueError(f"width must be one of 16k/65k/262k, got {c.width!r}")
    if c.aggregation not in {"last_token", "mean_prefill", "max_prefill"}:
        raise ValueError(f"aggregation invalid: {c.aggregation!r}")
    if c.hook_type not in {"resid_post", "mlp_out", "attn_out"}:
        raise ValueError(f"hook_type invalid: {c.hook_type!r}")
    if cfg.agents.show_zero_fraction_to_evaluator not in {"on", "off", "ab"}:
        raise ValueError(
            f"show_zero_fraction_to_evaluator must be on/off/ab, "
            f"got {cfg.agents.show_zero_fraction_to_evaluator!r}"
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
