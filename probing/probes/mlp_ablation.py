"""Leave-one-feature(-or-group)-out ablation for MLP + sklearn classification probes.

Runs *after* the main orchestrator. For every
``(csv_features extraction, target, probe)`` combination in the
experiment YAML — where ``probe`` is either an ``MLPProbeSpec`` or a
classification-capable ``SklearnProbeSpec`` (``logreg`` / ``svc``) —
trains:

  * one baseline MLP on the full feature tensor;
  * one variant MLP per feature column dropped (per-feature importance);
  * one variant MLP per *group* dropped along each configured feature
    axis (default: ``category``, ``context``, ``section``), where group
    membership is parsed from feature names of the form
    ``c{ctx}_p{section}_{CATEGORY}``.

Outputs (per ``(extraction, target, probe)``):

  * ``feature_importance.csv`` + ``.png`` — per-feature drops (existing).
  * ``per_feature_aggregated.csv`` — per-feature drops aggregated by axis
    (mean, sum, max, count) — fast pandas summary, no extra training.
  * ``group_<axis>_importance.csv`` + ``.png`` — leave-one-group-out
    drops per axis. With 12 features per category etc., this captures
    the cumulative signal a redundant per-feature ablation can't.

Reuses ``train_mlp_probes`` so the training loop is identical to the
orchestrator's — only the input tensor changes.

CLI: ``uv run python -m interpret.probing.probes.mlp_ablation <experiment.yaml>``
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.probes.ablation_feature_bars import (
    render_single_experiment_pair,
)
from interpret.probing.configs.csv_features_extraction import (
    CSVFeaturesExtractionConfig,
)
from interpret.probing.configs.experiment import (
    ExperimentConfig,
    TargetSpec,
)
from interpret.probing.configs.probe import (
    MLPProbeSpec,
    ProbeSpec,
    SklearnProbeSpec,
)
from interpret.probing.manifests.manifest_base import ManifestBuilder
from interpret.probing.probes.mlp_probe import train_mlp_probes
from interpret.probing.probes.sklearn_probes import (
    train_sklearn_probe,
)
from interpret.probing.utils.cross_validation import resolve_folds
from interpret.probing.utils.enums import TaskType

# sklearn probe kinds that produce class labels and so support the
# leave-one-out workflow. Other kinds (ridge / lasso / svr / massmean)
# are regression-only and skipped at runner-selection time.
_SKLEARN_CLASSIFICATION_KINDS = {"logreg", "svc"}

CACHE_ROOT = Path("resources/extracted_activations")

# Axis name → 0-based index after splitting a feature name on "_" with
# maxsplit=2. Names look like ``c0_p1_HARMFUL_PAYLOAD`` so the third
# field captures multi-token category labels intact.
_AXIS_INDEX = {
    "context": 0,
    "section": 1,
    "category": 2,
}
DEFAULT_GROUP_AXES = ("category", "context", "section")


@dataclass
class _AblationContext:
    extraction: CSVFeaturesExtractionConfig
    dataset: ActivationDataset
    feature_columns: list[str]
    layer_key: tuple[int, str]
    target: TargetSpec
    spec: ProbeSpec
    targets: np.ndarray  # Canonical form; per-trainer code casts as needed.
    output_dir: Path


class ProbeAblationRunner:
    """Train baseline + leave-one-out probes for an experiment.

    The runner consumes an already-loaded ``ExperimentConfig``: it picks
    every ``CSVFeaturesExtractionConfig`` extraction, every supported
    probe spec (``MLPProbeSpec`` or classification-capable
    ``SklearnProbeSpec``), and every target, then writes per-feature
    importance numbers under ``<output_dir>/ablation/<extraction>/<target>/``.

    For sklearn, only ``logreg`` and ``svc`` are ablated — the other
    kinds (ridge / lasso / svr / massmean) are regression-only and so
    don't fit the binary-classification ablation workflow.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.cache_dir = CACHE_ROOT / config.name
        self._manifest: ManifestBuilder | None = None

    def run(self) -> Path:
        ablation_root = self.config.output_path / "ablation"
        ablation_root.mkdir(parents=True, exist_ok=True)

        csv_extractions = [
            e for e in self.config.extractions
            if isinstance(e, CSVFeaturesExtractionConfig)
        ]
        if not csv_extractions:
            print(
                "ProbeAblationRunner: no csv_features extractions in config — "
                "nothing to ablate.",
            )
            return ablation_root

        ablation_specs = [
            s for s in self.config.probes if self._is_ablatable(s)
        ]
        if not ablation_specs:
            print(
                "ProbeAblationRunner: no MLP or sklearn-classification "
                "probes in config — skipping.",
            )
            return ablation_root

        for extraction in csv_extractions:
            dataset = self._load_dataset(extraction)
            feature_columns = self._resolve_feature_columns(extraction, dataset)
            layer_key = self._resolve_single_layer_key(dataset)

            for target in self.config.targets:
                target_values = self._resolve_targets(target)
                for spec in ablation_specs:
                    out_dir = (
                        ablation_root / extraction.name
                        / (target.name or target.column) / spec.name
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)
                    ctx = _AblationContext(
                        extraction=extraction,
                        dataset=dataset,
                        feature_columns=feature_columns,
                        layer_key=layer_key,
                        target=target,
                        spec=spec,
                        targets=target_values,
                        output_dir=out_dir,
                    )
                    self._ablate_one(ctx)

        print(f"\nDone. Ablation outputs -> {ablation_root}")
        return ablation_root

    @staticmethod
    def _is_ablatable(spec: ProbeSpec) -> bool:
        """Return True if this probe spec supports leave-one-out ablation."""
        if isinstance(spec, MLPProbeSpec):
            return True
        if isinstance(spec, SklearnProbeSpec):
            return spec.kind in _SKLEARN_CLASSIFICATION_KINDS
        return False

    # ── Loading helpers ─────────────────────────────────────────────────────

    def _load_dataset(
        self, extraction: CSVFeaturesExtractionConfig,
    ) -> ActivationDataset:
        cache_path = self.cache_dir / f"{extraction.cache_filename()}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"No cached extraction at {cache_path}. Run the "
                f"orchestrator first: `uv run python -m "
                f"interpret.probing.orchestrator <yaml>`",
            )
        return ActivationDataset.load(cache_path)

    @staticmethod
    def _resolve_feature_columns(
        extraction: CSVFeaturesExtractionConfig,
        dataset: ActivationDataset,
    ) -> list[str]:
        if extraction.feature_columns:
            return list(extraction.feature_columns)
        meta_cols = dataset.metadata.get("feature_columns")
        if not meta_cols:
            raise RuntimeError(
                f"Extraction {extraction.name!r}: feature_columns missing "
                f"from both config and dataset metadata; cannot label "
                f"ablation results.",
            )
        return list(meta_cols)

    @staticmethod
    def _resolve_single_layer_key(
        dataset: ActivationDataset,
    ) -> tuple[int, str]:
        keys = dataset.layer_intermediate_keys()
        if len(keys) != 1:
            raise ValueError(
                f"ProbeAblationRunner expects a single (layer, intermediate) "
                f"key for csv_features extractions; got {keys}.",
            )
        return keys[0]

    def _resolve_targets(self, target: TargetSpec) -> torch.Tensor:
        if self._manifest is None:
            cls = self.config.manifest.resolve()
            self._manifest = cls(**self.config.manifest.kwargs)
        rated, values = self._manifest.get_rated_samples(
            target.source, target.column,
        )
        # The cached extraction was saved over `manifest.samples` in order;
        # for csv_features the rated set IS the full sample list with the
        # same ordering, so a direct tensor cast is safe. We assert that
        # to fail loudly if the assumption ever breaks.
        if list(rated) != list(self._manifest.samples):
            raise RuntimeError(
                f"Target {target.column!r}: rated rows do not align with "
                f"manifest.samples — ablation requires full alignment for "
                f"csv_features.",
            )
        # Canonical ndarray form. Per-trainer code in `_train_*_variant`
        # casts to torch.Tensor (LongTensor / Float[N,1]) when needed.
        arr = np.asarray(values)
        if target.task_type is TaskType.CLASSIFICATION:
            return arr.astype(np.int64, copy=False)
        return arr.astype(np.float32, copy=False)

    # ── Ablation loop ──────────────────────────────────────────────────────

    def _resolve_splits(
        self, ctx: _AblationContext,
    ) -> list[tuple[str, int, tuple[np.ndarray, np.ndarray] | None]]:
        """Return (label, seed, indices_override) for every split to run.

        Three modes, in priority order:
          1. ``spec.n_folds`` set: stratified k-fold (KFold for regression).
             Splits are ``[("fold_0", spec.seed, (train, val)), ...]``.
             ``ablation_seeds`` is ignored when n_folds is set on a spec.
          2. ``config.ablation_seeds`` set: one entry per seed, with
             ``indices_override=None`` so each seed produces its own
             internal random split.
          3. Default: a single split with the spec's own seed.

        K-fold construction delegates to
        ``probing/utils/cross_validation.py:resolve_folds`` so the
        splitter behaviour stays in lockstep with the orchestrator's
        probe stage.
        """
        spec = ctx.spec
        if spec.n_folds:
            is_classification = ctx.target.task_type is TaskType.CLASSIFICATION
            folds = resolve_folds(
                n=len(ctx.targets),
                n_folds=spec.n_folds,
                seed=spec.seed,
                is_classification=is_classification,
                stratify_y=(
                    np.asarray(ctx.targets) if is_classification else None
                ),
            )
            return [
                (label, spec.seed, (train, val))
                for label, train, val in folds
            ]

        seeds = self.config.ablation_seeds or [spec.seed]
        return [(f"seed_{s}", s, None) for s in seeds]

    def _ablate_one(self, ctx: _AblationContext) -> None:
        layer, intermediate = ctx.layer_key
        full_X, _ = ctx.dataset.get(layer, intermediate)
        n_features = full_X.shape[1]
        if n_features != len(ctx.feature_columns):
            raise RuntimeError(
                f"feature_columns length {len(ctx.feature_columns)} != "
                f"tensor n_features {n_features}.",
            )

        target_label = ctx.target.name or ctx.target.column
        splits = self._resolve_splits(ctx)
        split_labels = [label for label, _, _ in splits]
        print(
            f"\n=== Ablation: extraction={ctx.extraction.name} "
            f"target={target_label} probe={ctx.spec.name} "
            f"({n_features} features, splits={split_labels}) ===",
        )

        feature_rows: list[dict] = []
        group_rows: dict[str, list[dict]] = {axis: [] for axis in DEFAULT_GROUP_AXES}

        for split_label, seed, indices_override in splits:
            baseline_metrics = self._train_variant(
                ctx, X=full_X, label=f"baseline_{split_label}",
                seed=seed, indices_override=indices_override,
            )
            baseline_acc = baseline_metrics.get("val_accuracy")
            baseline_loss = baseline_metrics.get("val_loss")
            # sklearn classification writes val_loss as NaN; format
            # defensively so the print never crashes the run.
            acc_str = (
                f"{baseline_acc:.4f}" if baseline_acc is not None else "nan"
            )
            loss_str = (
                f"{baseline_loss:.4f}" if baseline_loss is not None else "nan"
            )
            print(
                f"  {split_label} baseline val_accuracy={acc_str} "
                f"val_loss={loss_str}",
            )

            for i in tqdm(
                range(n_features), desc=f"{split_label} per-feature",
            ):
                mask = torch.ones(n_features, dtype=torch.bool)
                mask[i] = False
                X_dropped = full_X[:, mask]
                metrics = self._train_variant(
                    ctx, X=X_dropped, label=f"ablate_{i}_{split_label}",
                    seed=seed, indices_override=indices_override,
                )
                ablated_acc = metrics.get("val_accuracy")
                ablated_loss = metrics.get("val_loss")
                feature_rows.append(
                    {
                        "split": split_label,
                        "feature_index": i,
                        "feature_name": ctx.feature_columns[i],
                        "baseline_val_accuracy": baseline_acc,
                        "ablated_val_accuracy": ablated_acc,
                        "accuracy_drop": (
                            None if baseline_acc is None or ablated_acc is None
                            else baseline_acc - ablated_acc
                        ),
                        "baseline_val_loss": baseline_loss,
                        "ablated_val_loss": ablated_loss,
                    },
                )

            for axis in DEFAULT_GROUP_AXES:
                group_rows[axis].extend(
                    self._collect_group_rows(
                        ctx,
                        full_X=full_X,
                        axis=axis,
                        baseline_acc=baseline_acc,
                        baseline_loss=baseline_loss,
                        split_label=split_label,
                        seed=seed,
                        indices_override=indices_override,
                    ),
                )

        feat_df = pd.DataFrame(feature_rows)
        feat_long_path = ctx.output_dir / "feature_importance.csv"
        feat_df.to_csv(feat_long_path, index=False)
        feat_summary = self._summarise(
            feat_df, group_cols=["feature_index", "feature_name"],
        )
        feat_summary_path = ctx.output_dir / "feature_importance_summary.csv"
        feat_summary.to_csv(feat_summary_path, index=False)
        print(f"  wrote {feat_long_path}")
        print(f"  wrote {feat_summary_path}")
        # Paper-style per-feature bar chart (clean + ±1σ companion). The
        # old single-strip 72-bar plot was illegible; the grouped layout
        # makes the category / context / section structure visible.
        render_single_experiment_pair(
            feat_summary_path,
            out_dir=ctx.output_dir / "figures",
            out_stem="feature_importance",
        )

        # Aggregate the seed-mean per-feature drops by axis (no extra training).
        self._aggregate_per_feature(feat_summary, ctx)

        # Save + summarise per-axis group ablations.
        for axis in DEFAULT_GROUP_AXES:
            self._save_group_results(ctx, axis, group_rows[axis])

    def _collect_group_rows(
        self,
        ctx: _AblationContext,
        *,
        full_X: torch.Tensor,
        axis: str,
        baseline_acc: float | None,
        baseline_loss: float | None,
        split_label: str,
        seed: int,
        indices_override: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> list[dict]:
        """Run one leave-one-group-out ablation pass for a single split.

        Returns the per-group rows so the caller can accumulate across
        splits (seeds or folds) and emit a single long-form CSV at the end.
        """
        groups = self._group_feature_indices(ctx.feature_columns, axis)
        if not groups:
            return []
        n_features = full_X.shape[1]

        rows: list[dict] = []
        for group_name, indices in tqdm(
            sorted(groups.items()),
            desc=f"{split_label} leave-one-{axis}-out",
        ):
            mask = torch.ones(n_features, dtype=torch.bool)
            mask[torch.tensor(indices, dtype=torch.long)] = False
            X_dropped = full_X[:, mask]
            metrics = self._train_variant(
                ctx,
                X=X_dropped,
                label=f"ablate_{axis}_{group_name}_{split_label}",
                seed=seed,
                indices_override=indices_override,
            )
            ablated_acc = metrics.get("val_accuracy")
            ablated_loss = metrics.get("val_loss")
            rows.append(
                {
                    "split": split_label,
                    "axis": axis,
                    "group": group_name,
                    "n_features_dropped": len(indices),
                    "feature_indices": ",".join(str(i) for i in indices),
                    "feature_names": ",".join(
                        ctx.feature_columns[i] for i in indices
                    ),
                    "baseline_val_accuracy": baseline_acc,
                    "ablated_val_accuracy": ablated_acc,
                    "accuracy_drop": (
                        None if baseline_acc is None or ablated_acc is None
                        else baseline_acc - ablated_acc
                    ),
                    "baseline_val_loss": baseline_loss,
                    "ablated_val_loss": ablated_loss,
                },
            )
        return rows

    def _save_group_results(
        self,
        ctx: _AblationContext,
        axis: str,
        rows: list[dict],
    ) -> None:
        """Write the long-form + per-axis summary CSVs."""
        if not rows:
            return
        df = pd.DataFrame(rows)
        long_path = ctx.output_dir / f"group_{axis}_importance.csv"
        df.to_csv(long_path, index=False)
        summary = self._summarise(df, group_cols=["axis", "group"])
        # `n_features_dropped` is constant per (axis, group); join it in.
        sizes = df.drop_duplicates(["axis", "group"])[
            ["axis", "group", "n_features_dropped"]
        ]
        summary = summary.merge(sizes, on=["axis", "group"], how="left")
        summary_path = ctx.output_dir / f"group_{axis}_importance_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"  wrote {long_path}")
        print(f"  wrote {summary_path}")

    @staticmethod
    def _summarise(
        df: pd.DataFrame, group_cols: list[str],
    ) -> pd.DataFrame:
        """Aggregate per-split accuracy_drops to mean / std / min / max / count.

        Splits can be either seeds (different random 80/20 splits) or
        folds (StratifiedKFold partitions); the aggregation is the same.

        Names the mean column ``accuracy_drop`` so downstream code that
        already operates on that column (axis aggregation, bar plots) keeps
        working unchanged when fed the summary instead of a long-form df.
        """
        agg = (
            df.groupby(group_cols, sort=False)["accuracy_drop"].agg(
                accuracy_drop="mean",
                std_accuracy_drop="std",
                min_accuracy_drop="min",
                max_accuracy_drop="max",
                n_splits="count",
            ).reset_index()
        )
        means = (
            df.groupby(group_cols, sort=False)[
                ["baseline_val_accuracy", "ablated_val_accuracy"]
            ].mean().reset_index()
        )
        return agg.merge(means, on=group_cols, how="left")

    def _aggregate_per_feature(
        self, df: pd.DataFrame, ctx: _AblationContext,
    ) -> None:
        """Aggregate per-feature drops by axis without retraining.

        Produces one long-form row per (axis, group) with mean / sum /
        max / count of ``accuracy_drop`` across the features in that
        group. Cheap pandas summary; complements the leave-one-group-out
        ablation by showing how individual contributions distribute
        within a group.
        """
        if df.empty or "feature_name" not in df.columns:
            return
        axis_tokens = pd.DataFrame(
            [self._parse_axes(name) for name in df["feature_name"]],
            columns=list(_AXIS_INDEX),
        )
        merged = pd.concat([df.reset_index(drop=True), axis_tokens], axis=1)
        rows: list[dict] = []
        for axis in _AXIS_INDEX:
            grouped = merged.groupby(axis)["accuracy_drop"].agg(
                ["mean", "sum", "max", "count"],
            )
            for group_name, stats in grouped.iterrows():
                rows.append(
                    {
                        "axis": axis,
                        "group": group_name,
                        "n_features": int(stats["count"]),
                        "mean_accuracy_drop": float(stats["mean"]),
                        "sum_accuracy_drop": float(stats["sum"]),
                        "max_accuracy_drop": float(stats["max"]),
                    },
                )
        agg = pd.DataFrame(rows)
        csv_path = ctx.output_dir / "per_feature_aggregated.csv"
        agg.to_csv(csv_path, index=False)
        print(f"  wrote {csv_path}")

    @staticmethod
    def _parse_axes(feature_name: str) -> tuple[str, str, str]:
        """Split a name like ``c0_p1_HARMFUL_PAYLOAD`` into its three axis tokens.

        Uses ``maxsplit=2`` so multi-token category labels stay intact.
        Falls back to ``("", "", feature_name)`` for names that don't
        match the convention — keeps the aggregator robust to partial
        renaming or unrelated columns.
        """
        parts = feature_name.split("_", 2)
        if len(parts) < 3:
            return ("", "", feature_name)
        return (parts[0], parts[1], parts[2])

    @classmethod
    def _group_feature_indices(
        cls, feature_columns: list[str], axis: str,
    ) -> dict[str, list[int]]:
        if axis not in _AXIS_INDEX:
            raise ValueError(
                f"Unknown axis {axis!r}. Valid: {list(_AXIS_INDEX)}",
            )
        idx = _AXIS_INDEX[axis]
        groups: dict[str, list[int]] = {}
        for i, name in enumerate(feature_columns):
            key = cls._parse_axes(name)[idx]
            groups.setdefault(key, []).append(i)
        return groups

    def _train_variant(
        self,
        ctx: _AblationContext,
        *,
        X: torch.Tensor,
        label: str,
        seed: int | None = None,
        indices_override: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict:
        """Train a single ablation variant and return its summary metrics.

        Dispatches to the MLP- or sklearn-specific path based on
        ``ctx.spec``. ``seed`` overrides ``ctx.spec.seed`` for this call
        without mutating the original spec — used by the multi-split
        loop to vary the train/val split across runs.
        ``indices_override`` lets the caller supply explicit
        (train_idx, val_idx) — used by the k-fold path so every fold
        sees the same prescribed split partition.
        """
        spec = (
            ctx.spec if seed is None
            else dataclasses.replace(ctx.spec, seed=seed)
        )
        if isinstance(spec, MLPProbeSpec):
            return self._train_mlp_variant(
                ctx, X=X, label=label, spec=spec,
                indices_override=indices_override,
            )
        if isinstance(spec, SklearnProbeSpec):
            return self._train_sklearn_variant(
                ctx, X=X, label=label, spec=spec,
                indices_override=indices_override,
            )
        raise TypeError(f"Unsupported probe spec for ablation: {type(spec)}")

    def _train_mlp_variant(
        self,
        ctx: _AblationContext,
        *,
        X: torch.Tensor,
        label: str,
        spec: MLPProbeSpec,
        indices_override: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict:
        if ctx.target.task_type is TaskType.CLASSIFICATION:
            targets = torch.from_numpy(ctx.targets).long()
        else:
            targets = torch.from_numpy(ctx.targets).float()
            if targets.ndim == 1:
                targets = targets.unsqueeze(1)

        variant_dataset = self._build_variant_dataset(ctx, X=X, label=label)
        with tempfile.TemporaryDirectory(prefix="probe_ablation_") as tmp:
            tmp_dir = Path(tmp)
            train_mlp_probes(
                variant_dataset,
                spec,
                targets,
                tmp_dir,
                task_type=ctx.target.task_type,
                num_classes=ctx.target.num_classes,
                target_columns=[ctx.target.column],
                indices_override=indices_override,
            )
            with open(tmp_dir / "summary.json", encoding="utf-8") as f:
                summary = json.load(f)
            results = pd.read_csv(tmp_dir / "probe_results.csv")
            row = results.iloc[0].to_dict()

        # MLP CSV carries val_loss; sklearn CSV doesn't. Pull both with
        # .get() so the merged summary keeps a stable schema regardless
        # of which probe trained this variant.
        best = summary.get("best") or {}
        return {
            "val_accuracy": row.get("val_accuracy"),
            "val_loss": row.get("val_loss"),
            "val_f1_weighted": row.get("val_f1_weighted"),
            "best_metric": best.get("metric"),
            "best_value": best.get("value"),
        }

    def _train_sklearn_variant(
        self,
        ctx: _AblationContext,
        *,
        X: torch.Tensor,
        label: str,
        spec: SklearnProbeSpec,
        indices_override: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict:
        variant_dataset = self._build_variant_dataset(ctx, X=X, label=label)
        with tempfile.TemporaryDirectory(prefix="probe_ablation_") as tmp:
            tmp_dir = Path(tmp)
            train_sklearn_probe(
                variant_dataset,
                spec,
                ctx.targets,
                tmp_dir,
                indices_override=indices_override,
            )
            results = pd.read_csv(tmp_dir / "probe_results.csv")
            row = results.iloc[0].to_dict()

        # sklearn classification CSVs leave val_loss as NaN — keep the
        # key so the downstream summary schema stays stable.
        return {
            "val_accuracy": row.get("val_accuracy"),
            "val_loss": row.get("val_loss"),
            "val_f1_weighted": row.get("val_f1_weighted"),
            "best_metric": "val_accuracy",
            "best_value": row.get("val_accuracy"),
        }

    @staticmethod
    def _build_variant_dataset(
        ctx: _AblationContext, *, X: torch.Tensor, label: str,
    ) -> ActivationDataset:
        layer, intermediate = ctx.layer_key
        return ActivationDataset(
            activations={(layer, intermediate): X},
            sample_ids=list(ctx.dataset.sample_ids),
            metadata={
                **ctx.dataset.metadata,
                "ablation_label": label,
                "n_features_used": int(X.shape[1]),
            },
        )

#: Backward-compat alias for the old class name. The runner now ablates
#: any classification probe, not just MLP — but downstream callers may
#: still import the old name.
MLPAblationRunner = ProbeAblationRunner


def run_from_yaml(yaml_path: Path | str) -> Path:
    """Load an experiment YAML and run the ablation."""
    config = ExperimentConfig.from_yaml(yaml_path)
    return ProbeAblationRunner(config).run()


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python -m interpret.probing.probes."
            "mlp_ablation <path/to/experiment.yaml>",
            file=sys.stderr,
        )
        sys.exit(2)
    run_from_yaml(sys.argv[1])


if __name__ == "__main__":
    main()
