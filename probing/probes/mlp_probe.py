"""MLP probe: one probe per (layer, intermediate) pair.

Wraps the proven training logic from `probing/trainer.py` + `probing/model.py`
with three targeted changes:

1. No `ProbeConfig` dependency — takes an `MLPProbeSpec` + targets directly.
2. `best_metric` is parametrised via the spec (and direction inferred from
   a small map). Default = val_r2 for regression, val_accuracy for
   classification.
3. Optional distance metric (e.g. `val_lab_distance`, mean CIEDE2000) when
   `spec.distance` names an `ExperimentalDistance` whose `target_dim` matches.
"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import warnings

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.probe import MLPProbeSpec
from interpret.probing.utils.cross_validation import resolve_folds
from interpret.probing.utils.enums import TaskType
from interpret.utils.distances import ExperimentalDistance, resolve_distance

# True = maximise, False = minimise. Used by best-epoch selection and summary.
_HIGHER_IS_BETTER = {
    "val_r2": True,
    "val_accuracy": True,
    "val_f1_weighted": True,
    "val_pearson": True,
    "val_spearman": True,
    "val_loss": False,
    "val_mse": False,
    "val_mae": False,
    "val_lab_distance": False,
}


def _classification_loss(
    y_train: torch.Tensor, class_weight: str | None,
) -> nn.Module:
    """Build a CrossEntropyLoss, optionally with sklearn-style balanced weights."""
    if class_weight is None:
        return nn.CrossEntropyLoss()
    if class_weight != "balanced":
        raise ValueError(
            f"MLP class_weight must be None or 'balanced', got "
            f"{class_weight!r}",
        )
    y_np = y_train.cpu().numpy()
    classes = np.unique(y_np)
    weights = compute_class_weight("balanced", classes=classes, y=y_np)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=weight_tensor)


class ProbeModel(nn.Module):
    """Configurable MLP probe.

    Architecture: input -> [Linear + ReLU + Dropout] x N -> Linear -> output.
    Empty `hidden_dims` produces a linear probe.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims if hidden_dims is not None else [512]
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


def train_mlp_probes(
    dataset: ActivationDataset,
    spec: MLPProbeSpec,
    targets: torch.Tensor,
    output_dir: Path,
    *,
    task_type: TaskType = TaskType.REGRESSION,
    num_classes: int | None = None,
    target_columns: list[str] | None = None,
    groups: np.ndarray | None = None,
    indices_override: tuple[np.ndarray, np.ndarray] | None = None,
) -> Path:
    """Train one MLP probe per (layer, intermediate) in `dataset`.

    Args:
        dataset: Activations container. `dataset.targets` is ignored —
            pass `targets` explicitly so a single dataset can serve probes
            against multiple target columns.
        spec: MLP architecture + training hyperparams.
        targets: Tensor[N, target_dim] (regression) or LongTensor[N] (classification).
        output_dir: Where `probe_results.csv`, `summary.json`, and
            `checkpoints/` are written.
        task_type: Regression or classification.
        num_classes: Required for classification.
        target_columns: Names of target columns, recorded in summary.json.
        groups: Optional per-sample group labels. When provided, the
            train/val split uses `GroupShuffleSplit` so members of the
            same group never end up on opposite sides of the split.
        indices_override: Optional explicit (train_idx, val_idx) pair.
            When provided, bypasses internal split logic — used by
            callers that drive their own k-fold or other custom split
            schedule.

    Returns:
        The output directory.
    """
    if task_type is TaskType.CLASSIFICATION and num_classes is None:
        raise ValueError("num_classes required for classification.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    is_classification = task_type is TaskType.CLASSIFICATION
    folds_arr = resolve_folds(
        n=int(targets.shape[0]),
        n_folds=spec.n_folds,
        seed=spec.seed,
        is_classification=is_classification,
        stratify_y=(
            targets.detach().cpu().numpy().reshape(-1)
            if is_classification else None
        ),
        train_split=spec.train_split,
        groups=groups,
        indices_override=indices_override,
    )
    folds = [
        (
            label,
            torch.from_numpy(np.asarray(train)).long(),
            torch.from_numpy(np.asarray(val)).long(),
        )
        for label, train, val in folds_arr
    ]

    metric_name = _resolve_best_metric(spec, task_type)
    # A distance plug-in may surface a metric_key not in the static map; fall
    # back to the distance's own `higher_is_better`.
    higher_is_better = _HIGHER_IS_BETTER.get(
        metric_name,
        resolve_distance(spec.distance).higher_is_better
        if spec.distance else True,
    )

    csv_path = output_dir / "probe_results.csv"
    results: list[dict] = []
    csv_handle = open(csv_path, "w", newline="", encoding="utf-8")
    writer: csv.DictWriter | None = None
    multi_fold = len(folds) > 1
    try:
        keys = dataset.layer_intermediate_keys()
        for layer, inter in tqdm(keys, desc="MLP probes"):
            X, _ = dataset.get(layer, inter)
            for fold_label, train_idx, val_idx in folds:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = targets[train_idx], targets[val_idx]

                metrics, best_state = _train_single_probe(
                    X_train, y_train, X_val, y_val,
                    spec=spec,
                    task_type=task_type,
                    num_classes=num_classes,
                )
                metrics["layer"] = layer
                metrics["intermediate"] = inter
                metrics["fold"] = fold_label

                ckpt_suffix = f"_{fold_label}" if multi_fold else ""
                torch.save(
                    best_state,
                    checkpoints_dir / f"layer_{layer}_{inter}{ckpt_suffix}.pt",
                )
                results.append(metrics)

                # Incremental CSV write: easy resumability + crash-safety.
                if writer is None:
                    writer = csv.DictWriter(csv_handle, fieldnames=list(metrics))
                    writer.writeheader()
                writer.writerow(metrics)
                csv_handle.flush()
    finally:
        csv_handle.close()

    _write_summary(
        output_dir,
        spec=spec,
        results=results,
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        task_type=task_type,
        target_columns=target_columns,
        dataset_metadata=dataset.metadata,
        multi_fold=multi_fold,
    )
    return output_dir


# ── Internals ────────────────────────────────────────────────────────────────


def _resolve_best_metric(
    spec: MLPProbeSpec, task_type: TaskType,
) -> str:
    """Pick the best-epoch metric from spec / distance / task_type."""
    if spec.best_metric is not None:
        if spec.best_metric not in _HIGHER_IS_BETTER:
            raise ValueError(
                f"Unknown best_metric {spec.best_metric!r}. "
                f"Known: {sorted(_HIGHER_IS_BETTER)}",
            )
        return spec.best_metric
    if spec.distance is not None:
        return resolve_distance(spec.distance).metric_key
    return (
        "val_r2"
        if task_type is TaskType.REGRESSION
        else "val_accuracy"
    )


def _train_single_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    spec: MLPProbeSpec,
    task_type: TaskType,
    num_classes: int | None,
) -> tuple[dict, dict]:
    # Seed torch so weight init + dropout RNG advance from a known state.
    # Combined with the seeded per-epoch shuffle below, two runs of the
    # same (spec, X, y, split) now produce bit-identical metrics — which
    # is what the ablation runner needs to attribute drops to features
    # rather than to training noise.
    torch.manual_seed(spec.seed)

    input_dim = X_train.shape[1]
    output_dim = (
        num_classes if task_type is TaskType.CLASSIFICATION
        else y_train.shape[1]
    )
    probe = ProbeModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=spec.hidden_dims,
        dropout=spec.dropout,
    )
    loss_fn: nn.Module = (
        _classification_loss(y_train, spec.class_weight)
        if task_type is TaskType.CLASSIFICATION
        else nn.MSELoss()
    )
    optimiser = torch.optim.Adam(
        probe.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )

    best_val_loss = float("inf")
    best_state = deepcopy(probe.state_dict())
    best_epoch = 0
    patience_counter = 0
    n_train = X_train.shape[0]
    final_train_loss = float("inf")
    final_epoch = 0

    for epoch in range(spec.epochs):
        final_epoch = epoch
        probe.train()
        # Per-epoch generator so the minibatch order is reproducible.
        # Using a large multiplier on spec.seed avoids accidental overlap
        # between (seed=42, epoch=1) and (seed=43, epoch=0) etc.
        shuffle_gen = torch.Generator().manual_seed(
            spec.seed * 100_000 + epoch,
        )
        perm = torch.randperm(n_train, generator=shuffle_gen)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, spec.batch_size):
            idx = perm[start : start + spec.batch_size]
            xb, yb = X_train[idx], y_train[idx]
            optimiser.zero_grad()
            pred = probe(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            n_batches += 1
        final_train_loss = epoch_loss / max(n_batches, 1)

        probe.eval()
        with torch.no_grad():
            val_pred = probe(X_val)
            val_loss = loss_fn(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(probe.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= spec.patience:
                break

    probe.load_state_dict(best_state)
    metrics = _compute_metrics(
        probe, X_val, y_val,
        loss_fn=loss_fn,
        task_type=task_type,
        distance=resolve_distance(spec.distance) if spec.distance else None,
    )
    metrics.update(
        {
            "train_loss_final": final_train_loss,
            "best_epoch": best_epoch,
            "total_epochs": final_epoch + 1,
        },
    )
    return metrics, best_state


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r that returns NaN for constant arrays."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman that returns NaN silently for constant arrays."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = spearmanr(a, b)
    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
        return float("nan")
    return float(rho)  # type: ignore[arg-type]


def _compute_metrics(
    probe: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    loss_fn: nn.Module,
    task_type: TaskType,
    distance: ExperimentalDistance | None,
) -> dict:
    probe.eval()
    with torch.no_grad():
        pred = probe(X)
        val_loss = loss_fn(pred, y).item()

    if task_type is TaskType.CLASSIFICATION:
        labels = pred.argmax(dim=1)
        accuracy = (labels == y).float().mean().item()
        f1 = float(
            f1_score(y.cpu().numpy(), labels.cpu().numpy(), average="weighted"),
        )
        return {
            "val_loss": val_loss,
            "val_accuracy": accuracy,
            "val_f1_weighted": f1,
        }

    # Regression
    mse = val_loss
    mae = (pred - y).abs().mean().item()
    ss_res = ((y - pred) ** 2).sum().item()
    ss_tot = ((y - y.mean(dim=0)) ** 2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    out: dict = {
        "val_loss": val_loss,
        "val_mse": mse,
        "val_mae": mae,
        "val_r2": r2,
    }

    # Per-column Pearson + Spearman (for 1-D targets; multi-dim gets
    # an average across columns).
    pred_np = pred.cpu().numpy()
    y_np = y.cpu().numpy()
    if y_np.shape[-1] == 1:
        p, y_flat = pred_np.ravel(), y_np.ravel()
        out["val_pearson"] = _safe_pearson(y_flat, p)
        out["val_spearman"] = _safe_spearman(y_flat, p)
    else:
        # Multi-dim: average correlation across columns.
        pearsons = [
            _safe_pearson(y_np[:, c], pred_np[:, c])
            for c in range(y_np.shape[1])
        ]
        spearmans = [
            _safe_spearman(y_np[:, c], pred_np[:, c])
            for c in range(y_np.shape[1])
        ]
        out["val_pearson"] = float(np.nanmean(pearsons))
        out["val_spearman"] = float(np.nanmean(spearmans))

    if distance is not None and distance.applies_to(y.shape[-1]):
        pred_np = pred.cpu().numpy().astype(np.float64)
        y_np = y.cpu().numpy().astype(np.float64)
        per_sample = distance.batch(pred_np, y_np)
        out[distance.metric_key] = float(per_sample.mean())

    return out


_AGGREGATABLE_MLP_METRICS = (
    "val_loss", "val_mse", "val_mae", "val_r2",
    "val_pearson", "val_spearman",
    "val_accuracy", "val_f1_weighted",
    "val_lab_distance",
)


def _write_summary(
    output_dir: Path,
    *,
    spec: MLPProbeSpec,
    results: list[dict],
    metric_name: str,
    higher_is_better: bool,
    task_type: TaskType,
    target_columns: list[str] | None,
    dataset_metadata: dict,
    multi_fold: bool,
) -> None:
    summary: dict = {
        "spec": {
            "type": spec.type,
            "name": spec.name,
            "hidden_dims": spec.hidden_dims,
            "dropout": spec.dropout,
            "epochs": spec.epochs,
            "patience": spec.patience,
            "learning_rate": spec.learning_rate,
            "weight_decay": spec.weight_decay,
            "batch_size": spec.batch_size,
            "train_split": spec.train_split,
            "seed": spec.seed,
            "best_metric": metric_name,
            "n_folds": spec.n_folds,
            "distance": spec.distance,
        },
        "task_type": task_type.value,
        "target_columns": list(target_columns) if target_columns else [],
        "num_probes": len(results),
        "dataset_metadata": dataset_metadata,
    }
    if results and metric_name in results[0]:
        if higher_is_better:
            best = max(results, key=lambda r: r[metric_name])
        else:
            best = min(results, key=lambda r: r[metric_name])
        summary["best"] = {
            "metric": metric_name,
            "value": best[metric_name],
            "layer": best["layer"],
            "intermediate": best["intermediate"],
            "fold": best.get("fold", "fold_0"),
        }

    if multi_fold and results:
        # Per (layer, intermediate) aggregation across folds.
        agg_tree: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
        keys = sorted({(r["layer"], r["intermediate"]) for r in results})
        for layer, inter in keys:
            sub = [r for r in results if r["layer"] == layer and r["intermediate"] == inter]
            inter_tree = agg_tree.setdefault(int(layer), {})
            metric_tree = inter_tree.setdefault(str(inter), {})
            for metric in _AGGREGATABLE_MLP_METRICS:
                values = np.asarray(
                    [r[metric] for r in sub if metric in r and r[metric] is not None],
                    dtype=float,
                )
                values = values[~np.isnan(values)]
                if values.size == 0:
                    continue
                metric_tree[metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "n": int(values.size),
                }
        summary["aggregated"] = agg_tree

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
