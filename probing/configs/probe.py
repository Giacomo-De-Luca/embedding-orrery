"""Probe specs: MLP and sklearn-family probes.

Both share a `type` discriminator. SklearnProbeSpec carries every probe
kind's hyperparams as fields with defaults — each kind reads only the
fields it needs.

Both also carry `distance`: a dotted-path (`"module:Class"`) naming an
`interpret.utils.distances.ExperimentalDistance`. When set and the target width
matches the distance's `target_dim`, the trainer additionally reports that
distance's metric (e.g. `val_lab_distance`, mean CIEDE2000 over 3-D LAB targets,
via `interpret.utils.distances:LabCiede2000Distance`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MLPProbeSpec:
    """Train an MLP regression/classification probe per (layer, intermediate)."""

    type: Literal["mlp"] = "mlp"
    name: str = "mlp"  # subdirectory name under <output>/probes/

    hidden_dims: list[int] = field(default_factory=lambda: [512])
    dropout: float = 0.1
    epochs: int = 100
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    train_split: float = 0.8
    seed: int = 42

    # Controls validation-best metric used for early-stopping summary.
    # Overrides the trainer's hardcoded val_r2/val_accuracy default.
    best_metric: str | None = None

    # Dotted-path to an ExperimentalDistance; its metric is reported when the
    # target width matches (e.g. "interpret.utils.distances:LabCiede2000Distance").
    distance: str | None = None

    # Classification only. "balanced" → inverse-frequency CrossEntropy
    # weights computed from y_train (sklearn convention). None → uniform.
    class_weight: Literal["balanced"] | None = None

    # K-fold cross-validation. When set, the ablation runner trains
    # this probe once per fold (StratifiedKFold for classification,
    # KFold otherwise) instead of once per `ablation_seeds` entry.
    # Leaves the orchestrator's probe stage untouched — headline probe
    # accuracy reads the single-seed split.
    n_folds: int | None = None

    def __post_init__(self) -> None:
        if self.type != "mlp":
            raise ValueError(f"MLPProbeSpec.type must be 'mlp', got {self.type!r}")
        if self.n_folds is not None and self.n_folds < 2:
            raise ValueError(
                f"MLPProbeSpec.n_folds must be >= 2 or None; got "
                f"{self.n_folds!r}",
            )


SklearnKind = Literal["ridge", "lasso", "svr", "svc", "logreg", "massmean"]


@dataclass
class SklearnProbeSpec:
    """Train a sklearn-family probe.

    `kind` selects the estimator. Hyperparams that don't apply to the
    selected kind are silently ignored.
    """

    type: Literal["sklearn"] = "sklearn"
    kind: SklearnKind = "ridge"
    name: str | None = None  # defaults to kind; subdir name under <output>/probes/

    # Shared
    train_split: float = 0.8
    seed: int = 42
    save_directions: bool = False  # write probe coef_ + scaler stats as .npz
    center_only: bool = False       # if True, StandardScaler(with_std=False)
    # When False, skip StandardScaler entirely. The fitted coef_ is then
    # in the feature's native units rather than per-feature z-scores —
    # useful when downstream consumers want raw-scale β (e.g. comparing
    # logreg coefficients across experiments without per-fold rescaling).
    # Mutually informative with `center_only`: `standardise=False`
    # supersedes `center_only` (no scaling at all).
    standardise: bool = True

    # Linear (ridge / lasso)
    alpha: float = 1.0
    max_iter: int = 5000  # lasso convergence

    # SVR / LogReg
    C: float = 1.0
    kernel: str = "rbf"  # SVR

    # LogReg classification binning
    classification_bins: int = 5
    logreg_max_iter: int = 1000

    # Dotted-path to an ExperimentalDistance; its metric is reported when the
    # target width matches (e.g. "interpret.utils.distances:LabCiede2000Distance").
    distance: str | None = None

    # Classification only (logreg, svc). "balanced" → sklearn's inverse-
    # frequency reweighting. None → uniform. Ignored by regression kinds.
    class_weight: Literal["balanced"] | None = None

    # K-fold cross-validation. When set, the ablation runner trains
    # this probe once per fold instead of once per `ablation_seeds`
    # entry. See MLPProbeSpec.n_folds for the same field.
    n_folds: int | None = None

    def __post_init__(self) -> None:
        if self.type != "sklearn":
            raise ValueError(
                f"SklearnProbeSpec.type must be 'sklearn', got {self.type!r}",
            )
        if self.name is None:
            self.name = self.kind
        if self.n_folds is not None and self.n_folds < 2:
            raise ValueError(
                f"SklearnProbeSpec.n_folds must be >= 2 or None; got "
                f"{self.n_folds!r}",
            )


# Discriminated-union alias for type checking.
ProbeSpec = MLPProbeSpec | SklearnProbeSpec
