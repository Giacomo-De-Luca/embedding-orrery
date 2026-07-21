"""End-to-end pipeline driver: YAML config -> populated experiment dir.

Stages, in order:
  1. Resolve manifest -> save manifest.csv.
  2. For each extraction (topologically sorted, residuals before SAEs):
       extract or load from `resources/extracted_activations/<experiment_name>/`.
  3. For each (extraction, target, probe): train + write to
       `<output>/probes/<extraction.name>/<target.name>/<probe.name>/`.
  4. For each (sae_extraction, target, sae_analysis): run analysis.
  5. Per-experiment figures via `ExperimentVisualiser`.

The activation cache is **namespaced per experiment** under
`resources/extracted_activations/<experiment_name>/`. Each extraction's
user-chosen `name` is the cache filename within that subfolder. Cache
hits are validated by sidecar dict-equality. Different experiments do
not share cache files, even when their extraction configs are identical
— manifests typically differ in their sample list, which the cached
activations depend on but the extraction config does not capture.
"""

from __future__ import annotations

import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.caching import StageCache, _normalise
from interpret.probing.configs.concat_extraction import (
    ConcatExtractionConfig,
)
from interpret.probing.configs.csv_features_extraction import (
    CSVFeaturesExtractionConfig,
)
from interpret.probing.configs.delta_extraction import (
    DeltaExtractionConfig,
)
from interpret.probing.configs.experiment import (
    ExperimentConfig,
    ExtractionConfig,
    TargetSpec,
)
from interpret.probing.configs.extraction import (
    EncoderExtractionConfig,
    GemmaExtractionConfig,
)
from interpret.probing.configs.probe import (
    MLPProbeSpec,
    SklearnProbeSpec,
)
from interpret.probing.configs.residual_pooled_extraction import (
    ResidualPooledExtractionConfig,
)
from interpret.probing.configs.sae_analysis import (
    CorrelationMapConfig,
    FeatureSweepConfig,
    LassoAlphaSweepConfig,
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
from interpret.probing.extraction.extract_concat_activations import (
    extract_concat_activations,
)
from interpret.probing.extraction.extract_csv_features import (
    extract_csv_features,
)
from interpret.probing.extraction.extract_delta_activations import (
    extract_delta_activations,
)
from interpret.probing.extraction.extract_encoder_activations import (
    extract_encoder_activations,
)
from interpret.probing.extraction.extract_gemma_activations import (
    extract_gemma_activations,
    extract_gemma_activations_from_dataframe,
)
from interpret.probing.extraction.extract_residual_pooled import (
    extract_residual_pooled,
)
from interpret.probing.extraction.extract_sae_activations import (
    extract_sae_activations,
)
from interpret.probing.extraction.extract_sae_pooled import (
    extract_sae_pooled,
)
from interpret.probing.extraction.extract_token_residuals import (
    extract_token_residuals,
)
from interpret.probing.manifests.manifest_base import ManifestBuilder
from interpret.probing.probes.mlp_probe import train_mlp_probes
from interpret.probing.probes.sklearn_probes import (
    train_sklearn_probe,
)
from interpret.probing.sae_analysis.correlation_map import (
    run_correlation_map,
)
from interpret.probing.sae_analysis.feature_sweep import (
    run_feature_sweep,
)
from interpret.probing.sae_analysis.lasso_alpha_sweep import (
    run_lasso_alpha_sweep,
)
from interpret.probing.sae_analysis.top_features import (
    run_top_features,
)
from interpret.probing.utils.enums import TaskType
from interpret.probing.visualisations import ExperimentVisualiser

CACHE_ROOT = Path("resources/extracted_activations")


def run_experiment(config: ExperimentConfig) -> Path:
    """Run one experiment end-to-end. Returns the output directory."""
    output_dir = config.output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = CACHE_ROOT / config.name
    cache = StageCache(cache_dir) if config.cache_enabled else None

    print(f"\n=== Experiment: {config.name} ===")
    print(f"output_dir: {output_dir}")
    print(f"cache_dir: {cache_dir}")

    _save_experiment_yaml(config, output_dir / "experiment.yaml")
    errors: list[dict] = []

    # 1. Manifest
    manifest = _build_manifest(config)
    manifest_path = output_dir / "manifest.csv"
    manifest.build_dataframe().to_csv(manifest_path, index=False)
    print(f"manifest: {len(manifest.samples)} samples -> {manifest_path}")

    # 2. Extractions (topo-sorted; residuals first, SAEs after)
    datasets: dict[str, ActivationDataset] = {}
    extractions_by_name = {e.name: e for e in config.extractions}
    for extraction in config.topo_sorted_extractions():
        try:
            datasets[extraction.name] = _resolve_extraction(
                extraction,
                manifest,
                datasets,
                extractions_by_name,
                cache,
            )
            ds = datasets[extraction.name]
            print(
                f"extraction '{extraction.name}': {len(ds)} samples, "
                f"keys={ds.layer_intermediate_keys()}",
            )
        except Exception as exc:  # noqa: BLE001 — per-extraction isolation
            errors.append(
                {
                    "stage": "extraction",
                    "extraction": extraction.name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            print(
                f"  [error] extraction {extraction.name}: {exc}",
                file=sys.stderr,
            )

    # 3. Probes per (extraction, target, probe)
    for ext_name, dataset in datasets.items():
        if getattr(extractions_by_name.get(ext_name), "skip_probes", False):
            print(f"probes: skipping token-level extraction '{ext_name}'")
            continue
        for target in config.targets:
            try:
                _run_target_probes(
                    extraction_name=ext_name,
                    target=target,
                    config=config,
                    manifest=manifest,
                    probe_dataset=dataset,
                    output_dir=output_dir,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "stage": "probes",
                        "extraction": ext_name,
                        "target": target.name,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                )
                print(
                    f"  [error] probes {ext_name}/{target.name}: {exc}",
                    file=sys.stderr,
                )

    # 4. SAE analysis: only for SAE extractions (classic + pooled)
    sae_extractions = [
        e
        for e in config.extractions
        if isinstance(e, (SAEExtractionConfig, SAEPooledExtractionConfig))
    ]
    if config.sae_analysis and sae_extractions:
        for sae_ext in sae_extractions:
            sae_dataset = datasets.get(sae_ext.name)
            if sae_dataset is None:
                continue  # extraction failed earlier
            for target in config.targets:
                try:
                    _run_target_sae_analysis(
                        sae_extraction=sae_ext,
                        sae_dataset=sae_dataset,
                        target=target,
                        config=config,
                        manifest=manifest,
                        output_dir=output_dir,
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        {
                            "stage": "sae_analysis",
                            "extraction": sae_ext.name,
                            "target": target.name,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        },
                    )
                    print(
                        f"  [error] sae_analysis {sae_ext.name}/{target.name}: {exc}",
                        file=sys.stderr,
                    )

    # 5. Visualisations.
    if config.automatic_visualisations:
        try:
            ExperimentVisualiser(output_dir).render()
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "stage": "visualisations",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"  [error] visualisations: {exc}", file=sys.stderr)

    # 6. Errors
    if errors:
        with open(output_dir / "errors.json", "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, default=str)
        print(f"\nWrote {len(errors)} errors to errors.json")

    print(f"Done. -> {output_dir}")
    return output_dir


# ── Manifest ────────────────────────────────────────────────────────────────


def _build_manifest(config: ExperimentConfig) -> ManifestBuilder:
    cls = config.manifest.resolve()
    return cls(**config.manifest.kwargs)


# ── Extraction dispatch ────────────────────────────────────────────────────


def _resolve_extraction(
    extraction: ExtractionConfig,
    manifest: ManifestBuilder,
    datasets: dict[str, ActivationDataset],
    extractions_by_name: dict[str, ExtractionConfig],
    cache: StageCache | None,
) -> ActivationDataset:
    """Compute or load one extraction; SAEs read from `datasets`."""
    stem = extraction.cache_filename()

    if cache is not None:
        cached = cache.get(stem, extraction)
        if cached is not None:
            print(f"extraction '{extraction.name}': cache hit -> {cached.name}")
            return ActivationDataset.load(cached)

    print(f"extraction '{extraction.name}': cache miss, computing...")

    if isinstance(extraction, CSVFeaturesExtractionConfig):
        dataset = extract_csv_features(extraction, manifest)
    elif isinstance(extraction, EncoderExtractionConfig):
        dataset = extract_encoder_activations(extraction, manifest.samples)
    elif isinstance(extraction, GemmaExtractionConfig):
        model = _load_gemma(extraction.checkpoint_path)
        try:
            if extraction.image_column is None:
                # Text mode: extract over manifest.samples so activations
                # align with manifest.get_sample_indices(...) downstream.
                dataset = extract_gemma_activations(
                    extraction,
                    manifest.samples,
                    model,
                )
            else:
                # Image mode: needs full DataFrame for image_column lookups.
                dataset = extract_gemma_activations_from_dataframe(
                    extraction,
                    manifest.build_dataframe(),
                    model,
                )
        finally:
            model = None
            _release_accelerator_memory()
    elif isinstance(extraction, TokenLevelExtractionConfig):
        if extraction.family == "gemma":
            model = _load_gemma(extraction.checkpoint)
        else:
            model = _load_qwen(extraction.checkpoint, extraction.device)
        try:
            dataset = extract_token_residuals(
                extraction,
                manifest.samples,
                model,
            )
        finally:
            model = None
            _release_accelerator_memory()
    elif isinstance(extraction, SAEPooledExtractionConfig):
        source = _require_source(
            extraction.source_extraction,
            extraction.name,
            datasets,
        )
        dataset = extract_sae_pooled(source, extraction)
    elif isinstance(extraction, ResidualPooledExtractionConfig):
        source = _require_source(
            extraction.source_extraction,
            extraction.name,
            datasets,
        )
        dataset = extract_residual_pooled(source, extraction)
    elif isinstance(extraction, ConcatExtractionConfig):
        sources = [
            (name, _require_source(name, extraction.name, datasets))
            for name in extraction.source_extractions
        ]
        dataset = extract_concat_activations(sources, extraction)
    elif isinstance(extraction, SAEExtractionConfig):
        source = datasets.get(extraction.source_extraction)
        if source is None:
            raise RuntimeError(
                f"SAE extraction '{extraction.name}': source "
                f"'{extraction.source_extraction}' has no dataset (extraction "
                f"failed or missing).",
            )
        _validate_sae_source_token_position(extraction, extractions_by_name)
        dataset = extract_sae_activations(source, extraction)
    elif isinstance(extraction, DeltaExtractionConfig):
        source = datasets.get(extraction.source_extraction)
        if source is None:
            raise RuntimeError(
                f"Delta extraction '{extraction.name}': source "
                f"'{extraction.source_extraction}' has no dataset "
                f"(extraction failed or missing).",
            )
        dataset = extract_delta_activations(source, manifest, extraction)
    else:
        raise TypeError(f"Unknown extraction type: {type(extraction)}")

    if cache is not None:
        cache.put(stem, extraction, dataset.save)
    return dataset


def _require_source(
    source_name: str,
    extraction_name: str,
    datasets: dict[str, ActivationDataset],
) -> ActivationDataset:
    """Fetch a derived extraction's resolved source or raise."""
    source = datasets.get(source_name)
    if source is None:
        raise RuntimeError(
            f"Extraction '{extraction_name}': source '{source_name}' has no "
            f"dataset (extraction failed or missing).",
        )
    return source


def _release_accelerator_memory() -> None:
    """Reclaim device memory after a model is dropped.

    The caller must clear its own reference first (`model = None`) —
    otherwise the weights stay alive through that binding.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _load_qwen(checkpoint: str, device: str | None):
    """Lazy import + construct Qwen3Inference (bf16; device auto when None)."""
    from interpret.inference.qwen3_transformers import Qwen3Inference

    print(f"  loading Qwen from {checkpoint}")
    return Qwen3Inference(checkpoint, dtype="bfloat16", device=device)


def _validate_sae_source_token_position(
    sae: SAEExtractionConfig,
    extractions_by_name: dict[str, ExtractionConfig],
) -> None:
    """Raise if the SAE's token_position doesn't match its source's."""
    source = extractions_by_name.get(sae.source_extraction)
    if source is None:
        return  # caught earlier
    source_tp = getattr(source, "token_position", None)
    if source_tp is None:
        # Encoder source — token_position is N/A; skip validation.
        return
    if source_tp != sae.token_position:
        raise ValueError(
            f"SAE extraction '{sae.name}' declares token_position="
            f"{sae.token_position.value!r} but source "
            f"'{source.name}' uses {source_tp.value!r}. "
            f"They must match — SAE encoding consumes the source's "
            f"pre-pooled vector and cannot re-pool.",
        )


def _load_gemma(checkpoint_path: str | None):
    """Lazy import + checkpoint resolve for GemmaPytorchInference."""
    from interpret.inference.gemma_pytorch import (
        GemmaPytorchInference,
        _resolve_default_checkpoint,
    )

    if checkpoint_path is None:
        checkpoint_path = _resolve_default_checkpoint()
        if checkpoint_path is None:
            raise RuntimeError(
                "No Gemma3 4b checkpoint found in HF cache. "
                "Run: huggingface-cli download google/gemma-3-4b-it",
            )
    print(f"  loading Gemma from {checkpoint_path}")
    return GemmaPytorchInference(checkpoint_path)


# ── Probe stage ─────────────────────────────────────────────────────────────


def _run_target_probes(
    *,
    extraction_name: str,
    target: TargetSpec,
    config: ExperimentConfig,
    manifest: ManifestBuilder,
    probe_dataset: ActivationDataset,
    output_dir: Path,
) -> None:
    """Train every probe spec on one (extraction, target) pair."""
    rated_words, ratings = manifest.get_rated_samples(
        target.source,
        target.column,
    )
    # subset() looks up by sample_id, so it works for full extractions
    # AND derived extractions (delta) whose sample_ids are a proper
    # subset of the manifest.
    filtered = probe_dataset.subset(rated_words)
    groups = _resolve_groups(manifest, rated_words, config.group_column)
    target_dir = output_dir / "probes" / extraction_name / target.name

    print(
        f"\n--- extraction={extraction_name} target={target.name} "
        f"(source={target.source}, column={target.column}, "
        f"n={len(rated_words)}) ---",
    )
    for spec in config.probes:
        if isinstance(spec, MLPProbeSpec):
            probe_dir = target_dir / spec.name
            ratings_arr = np.asarray(ratings)
            if target.task_type is TaskType.CLASSIFICATION:
                # CrossEntropyLoss expects LongTensor[N] of class indices.
                y = torch.from_numpy(ratings_arr).long()
            else:
                y = torch.from_numpy(ratings_arr).float()
                if y.ndim == 1:
                    y = y.unsqueeze(1)
            train_mlp_probes(
                filtered,
                spec,
                y,
                probe_dir,
                task_type=target.task_type,
                num_classes=target.num_classes,
                target_columns=[target.column],
                groups=groups,
            )
        elif isinstance(spec, SklearnProbeSpec):
            probe_dir = target_dir / spec.name
            # Concat datasets carry their own per-column names; fall back
            # to the manifest's (csv_features) otherwise.
            feature_names = filtered.metadata.get("feature_names") or getattr(
                manifest,
                "feature_columns",
                None,
            )
            train_sklearn_probe(
                filtered,
                spec,
                np.asarray(ratings),
                probe_dir,
                groups=groups,
                feature_names=list(feature_names) if feature_names else None,
            )
        else:
            raise TypeError(f"Unknown probe spec: {type(spec)}")


def _resolve_groups(
    manifest: ManifestBuilder,
    rated_words: list[str],
    group_column: str | None,
) -> np.ndarray | None:
    """Return the group label for each rated sample, or None if disabled."""
    if group_column is None:
        return None
    df = manifest.build_dataframe()
    if group_column not in df.columns:
        raise ValueError(
            f"group_column={group_column!r} not in manifest columns {df.columns.tolist()}",
        )
    indices = manifest.get_sample_indices(rated_words)
    return df.iloc[indices][group_column].to_numpy()


# ── SAE analysis stage ──────────────────────────────────────────────────────


def _run_target_sae_analysis(
    *,
    sae_extraction: SAEExtractionConfig | SAEPooledExtractionConfig,
    sae_dataset: ActivationDataset,
    target: TargetSpec,
    config: ExperimentConfig,
    manifest: ManifestBuilder,
    output_dir: Path,
) -> None:
    rated_words, ratings = manifest.get_rated_samples(
        target.source,
        target.column,
    )
    filtered_sae = sae_dataset.subset(rated_words)
    analysis_dir = output_dir / "sae_analysis" / sae_extraction.name / target.name

    print(
        f"\n--- sae_analysis extraction={sae_extraction.name} target={target.name} ---",
    )
    for analysis in config.sae_analysis:
        if isinstance(analysis, CorrelationMapConfig):
            run_correlation_map(
                filtered_sae,
                targets={target.column: np.asarray(ratings)},
                config=analysis,
                output_dir=analysis_dir / "correlation_map",
                width=sae_extraction.width,
            )
        elif isinstance(analysis, TopFeaturesConfig):
            directions_dir = (
                output_dir
                / "probes"
                / sae_extraction.name
                / target.name
                / analysis.source_probe
                / "directions"
            )
            if not directions_dir.exists():
                print(
                    f"  top_features: source_probe "
                    f"{analysis.source_probe!r} has no directions at "
                    f"{directions_dir} — run that probe with "
                    f"save_directions=true first",
                )
                continue
            run_top_features(
                filtered_sae,
                directions_dir=directions_dir,
                config=analysis,
                output_dir=analysis_dir / "top_features",
                width=sae_extraction.width,
            )
        elif isinstance(analysis, FeatureSweepConfig):
            sweep_directions_dir: Path | None = None
            if analysis.ranking == "lasso":
                sweep_directions_dir = (
                    output_dir
                    / "probes"
                    / sae_extraction.name
                    / target.name
                    / analysis.source_probe
                    / "directions"
                )
                if not sweep_directions_dir.exists():
                    print(
                        f"  feature_sweep: ranking='lasso' but "
                        f"{analysis.source_probe!r} has no directions at "
                        f"{sweep_directions_dir} — skipping",
                    )
                    continue
            sweep_subdir = f"feature_sweep_{analysis.ranking}" + (
                "_pooled" if analysis.pool_layers else ""
            )
            run_feature_sweep(
                filtered_sae,
                target_values=np.asarray(ratings),
                config=analysis,
                output_dir=analysis_dir / sweep_subdir,
                width=sae_extraction.width,
                target_name=target.name or target.column,
                directions_dir=sweep_directions_dir,
            )
        elif isinstance(analysis, LassoAlphaSweepConfig):
            run_lasso_alpha_sweep(
                filtered_sae,
                target_values=np.asarray(ratings),
                config=analysis,
                output_dir=analysis_dir / "lasso_alpha_sweep",
                target_name=target.name or target.column,
            )
        else:
            raise TypeError(f"Unknown sae_analysis config: {type(analysis)}")


# ── Helpers ─────────────────────────────────────────────────────────────────


def _save_experiment_yaml(config: ExperimentConfig, path: Path) -> None:
    """Serialise the resolved config so the run is reproducible from disk."""
    payload: dict[str, Any] = _normalise(config)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)


def run_from_yaml(yaml_path: Path | str) -> Path:
    """Load a YAML experiment config and run it."""
    config = ExperimentConfig.from_yaml(yaml_path)
    return run_experiment(config)


def main() -> None:
    """CLI: `uv run python -m interpret.probing.orchestrator <yaml>`."""
    if len(sys.argv) != 2:
        print(
            "Usage: python -m interpret.probing.orchestrator <path/to/experiment.yaml>",
            file=sys.stderr,
        )
        sys.exit(2)
    run_from_yaml(sys.argv[1])


if __name__ == "__main__":
    main()
