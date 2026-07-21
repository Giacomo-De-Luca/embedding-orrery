"""Top-level experiment configuration.

An experiment defines a manifest + a list of targets + a list of
extractions + a list of probes (+ optional SAE analyses). The
orchestrator runs every probe against every (extraction, target) pair.

SAE is a first-class extraction type — it references another extraction
by `source_extraction: <name>`. The orchestrator topologically sorts
extractions so dependencies run first.

Loaded from YAML via OmegaConf; tagged unions are dispatched on `type`.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from interpret.probing.configs.concat_extraction import (
    ConcatExtractionConfig,
)
from interpret.probing.configs.csv_features_extraction import (
    CSVFeaturesExtractionConfig,
)
from interpret.probing.configs.delta_extraction import (
    DeltaExtractionConfig,
)
from interpret.probing.configs.extraction import (
    EncoderExtractionConfig,
    GemmaExtractionConfig,
)
from interpret.probing.configs.probe import (
    MLPProbeSpec,
    ProbeSpec,
    SklearnProbeSpec,
)
from interpret.probing.configs.residual_pooled_extraction import (
    ResidualPooledExtractionConfig,
)
from interpret.probing.configs.sae_analysis import (
    CorrelationMapConfig,
    FeatureSweepConfig,
    LassoAlphaSweepConfig,
    SAEAnalysisConfig,
    TopFeaturesConfig,
)
from interpret.probing.configs.sae_extraction import (
    SAEExtractionConfig,
)
from interpret.probing.configs.sae_pooled_extraction import (
    SAEPooledExtractionConfig,
)
from interpret.probing.configs.token_extraction import (
    TokenLevelExtractionConfig,
)
from interpret.probing.utils.enums import TaskType

ExtractionConfig = (
    EncoderExtractionConfig
    | GemmaExtractionConfig
    | SAEExtractionConfig
    | DeltaExtractionConfig
    | CSVFeaturesExtractionConfig
    | TokenLevelExtractionConfig
    | SAEPooledExtractionConfig
    | ResidualPooledExtractionConfig
    | ConcatExtractionConfig
)

# Extractions that reference sibling extractions (single `source_extraction`
# or plural `source_extractions`) — resolved after their sources.
_DERIVED_EXTRACTION_TYPES = (
    SAEExtractionConfig,
    DeltaExtractionConfig,
    SAEPooledExtractionConfig,
    ResidualPooledExtractionConfig,
    ConcatExtractionConfig,
)

_EXTRACTION_TYPES = {
    "encoder": EncoderExtractionConfig,
    "gemma": GemmaExtractionConfig,
    "sae": SAEExtractionConfig,
    "delta": DeltaExtractionConfig,
    "csv_features": CSVFeaturesExtractionConfig,
    "token_residuals": TokenLevelExtractionConfig,
    "sae_pooled": SAEPooledExtractionConfig,
    "residual_pooled": ResidualPooledExtractionConfig,
    "concat": ConcatExtractionConfig,
}


def _dependency_names(extraction) -> list[str]:
    """Names of sibling extractions this one consumes (empty if independent)."""
    if hasattr(extraction, "source_extractions"):
        return list(extraction.source_extractions)
    if hasattr(extraction, "source_extraction"):
        return [extraction.source_extraction]
    return []


_PROBE_TYPES = {
    "mlp": MLPProbeSpec,
    "sklearn": SklearnProbeSpec,
}
_SAE_ANALYSIS_TYPES = {
    "correlation_map": CorrelationMapConfig,
    "top_features": TopFeaturesConfig,
    "feature_sweep": FeatureSweepConfig,
    "lasso_alpha_sweep": LassoAlphaSweepConfig,
}


@dataclass
class ManifestSpec:
    """Reference to a manifest builder by Python path + kwargs."""

    # "module.path:ClassName"
    path: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def resolve(self) -> type:
        """Import + return the manifest class."""
        if ":" not in self.path:
            raise ValueError(
                f"ManifestSpec.path must be 'module.path:ClassName', got {self.path!r}",
            )
        module_path, class_name = self.path.split(":", 1)
        module = importlib.import_module(module_path)
        try:
            return getattr(module, class_name)
        except AttributeError as e:
            raise ImportError(
                f"Module {module_path!r} has no attribute {class_name!r}",
            ) from e


@dataclass
class TargetSpec:
    """One target column to probe."""

    source: str
    column: str
    task_type: TaskType = TaskType.REGRESSION
    name: str | None = None  # folder name; defaults to column
    num_classes: int | None = None  # required for classification

    def __post_init__(self) -> None:
        if isinstance(self.task_type, str):
            self.task_type = TaskType(self.task_type)
        if self.name is None:
            self.name = self.column
        if self.task_type is TaskType.CLASSIFICATION and self.num_classes is None:
            raise ValueError(
                f"TargetSpec(source={self.source!r}, column={self.column!r}): "
                f"num_classes required for classification.",
            )


@dataclass
class ExperimentConfig:
    """End-to-end experiment configuration."""

    name: str
    output_dir: str
    manifest: ManifestSpec
    extractions: list[ExtractionConfig] = field(default_factory=list)
    targets: list[TargetSpec] = field(default_factory=list)
    probes: list[ProbeSpec] = field(default_factory=list)
    sae_analysis: list[SAEAnalysisConfig] = field(default_factory=list)
    # Manifest column whose values group rows that must NOT be split across
    # train/val (e.g. `source_image` for THINGS-coloured paired probes —
    # without this, all tints of one object end up on both sides of the
    # split and inflate val_r2 via shape leakage). When None, probes use
    # the historical random permutation split.
    group_column: str | None = None
    cache_enabled: bool = True

    # MLP ablation runner: list of seeds to average over. When set, the
    # runner trains every variant once per seed and reports mean + std
    # of accuracy_drop, exposing how much of a per-feature drop is real
    # signal vs. training noise. None → run once with each spec's own seed
    # (current behaviour).
    ablation_seeds: list[int] | None = None

    # Engine-level auto figures (layer curves, probe×target heatmap,
    # best-metric bars) emitted by `ExperimentVisualiser` after the probe
    # stage. Off by default because the auto plots are degenerate for
    # single-layer extractions (csv_features). Multi-layer experiments
    # that want them should set this to true in their YAML.
    automatic_visualisations: bool = False

    def __post_init__(self) -> None:
        # Eager manifest import so typos fail at config load, not stage run.
        self.manifest.resolve()

        # Per-extraction name uniqueness — both for routing and for the
        # shared activations cache.
        ext_names = [e.name for e in self.extractions]
        if len(ext_names) != len(set(ext_names)):
            raise ValueError(
                f"Duplicate extraction names: {ext_names}. Each extraction "
                f"must have a unique name (drives the cache filename).",
            )

        # Derived extractions reference sibling extractions by name.
        for ext in self.extractions:
            for dep in _dependency_names(ext):
                if dep not in ext_names:
                    raise ValueError(
                        f"{type(ext).__name__} {ext.name!r}: "
                        f"source extraction {dep!r} not "
                        f"in extractions: {ext_names}",
                    )

        # Probe skip-lists must reference real extraction names —
        # a typo would silently skip nothing.
        for probe in self.probes:
            for skip in getattr(probe, "skip_extractions", None) or ():
                if skip not in ext_names:
                    raise ValueError(
                        f"Probe {probe.name!r}: skip_extractions entry "
                        f"{skip!r} not in extractions: {ext_names}",
                    )

        # Per-target name uniqueness — folder collisions overwrite results.
        # Case-insensitive: macOS APFS / Windows NTFS treat "B" and "b" as
        # the same path, so we must reject those even though they're distinct
        # Python strings. Targets like RGB `B` and LAB `b` are the typical
        # offenders — use explicit names (e.g. `rgb_B`, `lab_b`).
        names = [t.name for t in self.targets]
        lowered = [n.lower() for n in names]
        if len(lowered) != len(set(lowered)):
            raise ValueError(
                f"Duplicate target names (case-insensitive): {names}. "
                f"Set TargetSpec.name explicitly to disambiguate.",
            )

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    def topo_sorted_extractions(self) -> list[ExtractionConfig]:
        """Return extractions in dependency order (sources before dependents).

        Independent extractions (encoder, gemma) come first. Derived
        extractions (sae, delta) are placed after their `source_extraction`.
        Delta-on-sae works because the sae extraction is fully resolved
        before any derived extraction that names it.
        """
        independent: list[ExtractionConfig] = [
            e for e in self.extractions if not isinstance(e, _DERIVED_EXTRACTION_TYPES)
        ]
        pending: list[ExtractionConfig] = [
            e for e in self.extractions if isinstance(e, _DERIVED_EXTRACTION_TYPES)
        ]
        resolved_names: set[str] = {e.name for e in independent}
        result: list[ExtractionConfig] = list(independent)

        while pending:
            next_round: list[ExtractionConfig] = []
            progress = False
            for e in pending:
                if all(dep in resolved_names for dep in _dependency_names(e)):
                    result.append(e)
                    resolved_names.add(e.name)
                    progress = True
                else:
                    next_round.append(e)
            if not progress:
                cyclic = [e.name for e in next_round]
                raise ValueError(
                    f"Cyclic or unresolved extraction dependencies: "
                    f"{cyclic}. Each derived extraction's source_extraction "
                    f"must be defined and not depend transitively on itself.",
                )
            pending = next_round
        return result

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExperimentConfig:
        """Load an experiment config from a YAML file."""
        raw = OmegaConf.load(str(path))
        d = OmegaConf.to_container(raw, resolve=True)
        if not isinstance(d, dict):
            raise ValueError(f"Top-level YAML must be a mapping; got {type(d)}")
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        """Build an ExperimentConfig from a plain dict (dispatching unions)."""
        d = dict(d)

        manifest_d = d.pop("manifest")
        manifest = ManifestSpec(**manifest_d)

        extraction_dicts = d.pop("extractions", []) or []
        extractions = [
            _build_tagged(ed, _EXTRACTION_TYPES, "extraction") for ed in extraction_dicts
        ]

        target_dicts = d.pop("targets", []) or []
        targets = [TargetSpec(**td) for td in target_dicts]

        probe_dicts = d.pop("probes", []) or []
        probes = [_build_tagged(pd, _PROBE_TYPES, "probe") for pd in probe_dicts]

        analysis_dicts = d.pop("sae_analysis", []) or []
        sae_analysis = [
            _build_tagged(ad, _SAE_ANALYSIS_TYPES, "sae_analysis") for ad in analysis_dicts
        ]

        return cls(
            manifest=manifest,
            extractions=extractions,
            targets=targets,
            probes=probes,
            sae_analysis=sae_analysis,
            **d,  # name, output_dir, cache_enabled
        )


def _build_tagged(
    d: dict[str, Any],
    type_map: dict[str, type],
    label: str,
):
    """Dispatch on the `type` field to instantiate the right dataclass."""
    if "type" not in d:
        raise ValueError(
            f"{label}: missing 'type' discriminator. Valid: {list(type_map)}",
        )
    type_value = d["type"]
    if type_value not in type_map:
        raise ValueError(
            f"{label}: unknown type {type_value!r}. Valid: {list(type_map)}",
        )
    return type_map[type_value](**d)
