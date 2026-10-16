"""Torch-free probing types shared between the GraphQL layer and probing_service.

Lives in its own module so ``queries.py``/``converters.py`` can import the
probe config and field-name helpers without pulling in ``probing_service``
(and with it torch and the interpret/ toolkit) at schema-build time. The
heavy ``train_probe_for_collection`` import happens lazily inside the
``train_probe`` mutation.
"""

import re
from dataclasses import dataclass

PROBE_KINDS = (
    "ridge",
    "lasso",
    "massmean",
    "massmean_cov",
    "svr",
    "logreg",
    "svc",
    "mlp",
    "mlp_clf",
)

# Classifier kinds: categorical targets with 2..MAX_PROBE_CLASSES classes.
# Binary targets score P(class 1); multiclass targets score the confidence
# (max class probability) and persist the predicted class + all probabilities.
# `mlp_clf` is the MLP trained as a classifier (the UI shows it as MLP with
# its Classifier switch on) — a distinct kind so it never overwrites a
# regression MLP on the same field.
CLASSIFICATION_KINDS = ("logreg", "svc", "mlp_clf")

# Classes kept for training need at least this many labelled items, or
# ceil(min_class_fraction * n_labelled) when that is larger.
MIN_CLASS_COUNT_FLOOR = 5
DEFAULT_MIN_CLASS_FRACTION = 0.01
# Upper bound on kept classes (after the min-class threshold).
MAX_PROBE_CLASSES = 50


class ProbeTask:
    """What a trained probe predicts; persisted in the probe's config snapshot."""

    REGRESSION = "regression"
    BINARY = "binary"
    MULTICLASS = "multiclass"


# Kinds whose scores are predictions in target units (residuals meaningful).
# Excludes the classifier kinds (probabilities) and the mass-mean family (uncalibrated
# projections — their residuals come from the calibrated readout instead).
_PREDICTIVE_KINDS = ("ridge", "lasso", "mlp", "svr")

# Projection kinds that get a univariate calibrated readout (slope/intercept
# fitted on the train split): calibrated predictions -> residuals + R².
_CALIBRATED_KINDS = ("massmean", "massmean_cov")

# MLP hidden-layer nonlinearities — torch-free mirror of
# interpret/probing/configs/probe.py::MLP_ACTIVATIONS (keep in sync).
MLP_ACTIVATIONS = ("relu", "gelu", "tanh", "silu")


@dataclass
class ProbeConfig:
    """Configuration for one probe run."""

    collection_name: str
    target_field: str
    kind: str = "ridge"
    alpha: float = 1.0  # ridge L2 strength
    c: float = 1.0  # SVR / logreg / SVC inverse-regularisation
    kernel: str = "rbf"  # SVR / SVC kernel
    class_weight: str | None = None  # classifier kinds: None | "balanced"
    hidden_dims: list[int] | None = None  # MLP only; None -> [256]
    epochs: int = 100
    patience: int = 10
    # MLP: fraction of the train side held out for early stopping (0 disables).
    dev_split: float = 0.2
    activation: str = "relu"  # MLP hidden-layer nonlinearity (MLP_ACTIVATIONS)
    seed: int = 7
    train_split: float = 0.8
    max_train_samples: int = 50_000
    # Classifier kinds, 3+ class targets: classes below
    # max(MIN_CLASS_COUNT_FLOOR, ceil(fraction * n_labelled)) are excluded.
    min_class_fraction: float = DEFAULT_MIN_CLASS_FRACTION


def sanitize_field_key(name: str) -> str:
    """Make a name safe for metadata keys and path segments."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def score_field_names(target_field: str, kind: str) -> tuple[str, str | None]:
    """Derived metadata field names for a probe's score and residual.

    Residuals exist for kinds whose score is a prediction in target units,
    and for the mass-mean family via the calibrated readout. Classifier
    scores are probabilities — no residual.
    """
    key = sanitize_field_key(target_field)
    score = f"probe_{key}_{kind}_score"
    has_residual = kind in _PREDICTIVE_KINDS or kind in _CALIBRATED_KINDS
    residual = f"probe_{key}_{kind}_residual" if has_residual else None
    return score, residual


def derive_probe_task(kind: str, config: dict | None) -> str:
    """Task of a stored probe; rows trained before multiclass support carry
    no ``task`` in their config snapshot (logreg was binary-only then)."""
    task = (config or {}).get("task")
    if task:
        return task
    return ProbeTask.BINARY if kind in CLASSIFICATION_KINDS else ProbeTask.REGRESSION
