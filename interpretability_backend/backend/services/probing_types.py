"""Torch-free probing types shared between the GraphQL layer and probing_service.

Lives in its own module so ``queries.py``/``converters.py`` can import the
probe config and field-name helpers without pulling in ``probing_service``
(and with it torch and the interpret/ toolkit) at schema-build time. The
heavy ``train_probe_for_collection`` import happens lazily inside the
``train_probe`` mutation.
"""

import re
from dataclasses import dataclass

PROBE_KINDS = ("ridge", "lasso", "massmean", "massmean_cov", "svr", "logreg", "mlp")

# Kinds requiring a binary (two-class) target — scores are P(class 1).
_BINARY_KINDS = ("logreg",)

# Kinds whose scores are predictions in target units (residuals meaningful).
# Excludes logreg (probability) and the mass-mean family (uncalibrated
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
    c: float = 1.0  # SVR / logreg inverse-regularisation
    kernel: str = "rbf"  # SVR kernel
    class_weight: str | None = None  # logreg: None | "balanced"
    hidden_dims: list[int] | None = None  # MLP only; None -> [256]
    epochs: int = 100
    patience: int = 10
    # MLP: fraction of the train side held out for early stopping (0 disables).
    dev_split: float = 0.2
    activation: str = "relu"  # MLP hidden-layer nonlinearity (MLP_ACTIVATIONS)
    seed: int = 7
    train_split: float = 0.8
    max_train_samples: int = 50_000


def sanitize_field_key(name: str) -> str:
    """Make a name safe for metadata keys and path segments."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def binary_target_mapping(values: list[str | None]) -> dict[str, float] | None:
    """Map a binary categorical column to 0/1 targets, or None if not binary.

    Exactly two distinct non-null values are required (case-sensitive). The
    mapping is deterministic: the alphabetically first value becomes 0.0 and
    the second 1.0 (e.g. {"safe": 0.0, "unsafe": 1.0}).
    """
    distinct = sorted({v for v in values if v is not None})
    if len(distinct) != 2:
        return None
    return {distinct[0]: 0.0, distinct[1]: 1.0}


def score_field_names(target_field: str, kind: str) -> tuple[str, str | None]:
    """Derived metadata field names for a probe's score and residual.

    Residuals exist for kinds whose score is a prediction in target units,
    and for the mass-mean family via the calibrated readout. Logreg scores
    are probabilities — no residual.
    """
    key = sanitize_field_key(target_field)
    score = f"probe_{key}_{kind}_score"
    has_residual = kind in _PREDICTIVE_KINDS or kind in _CALIBRATED_KINDS
    residual = f"probe_{key}_{kind}_residual" if has_residual else None
    return score, residual
