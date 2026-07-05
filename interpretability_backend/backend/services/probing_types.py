"""Torch-free probing types shared between the GraphQL layer and probing_service.

Lives in its own module so ``queries.py``/``converters.py`` can import the
probe config and field-name helpers without pulling in ``probing_service``
(and with it torch and the interpret/ toolkit) at schema-build time. The
heavy ``train_probe_for_collection`` import happens lazily inside the
``train_probe`` mutation.
"""

import re
from dataclasses import dataclass

PROBE_KINDS = ("ridge", "massmean", "mlp")

# Kinds whose scores are predictions in target units (residuals meaningful).
_PREDICTIVE_KINDS = ("ridge", "mlp")


@dataclass
class ProbeConfig:
    """Configuration for one probe run."""

    collection_name: str
    target_field: str
    kind: str = "ridge"
    alpha: float = 1.0
    hidden_dims: list[int] | None = None  # MLP only; None -> [256]
    epochs: int = 100
    patience: int = 10
    seed: int = 42
    train_split: float = 0.8
    max_train_samples: int = 50_000


def sanitize_field_key(name: str) -> str:
    """Make a name safe for metadata keys and path segments."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def score_field_names(target_field: str, kind: str) -> tuple[str, str | None]:
    """Derived metadata field names for a probe's score and residual.

    Residuals only exist for kinds whose score is a prediction in target
    units; massmean scores are direction projections on an arbitrary scale.
    """
    key = sanitize_field_key(target_field)
    score = f"probe_{key}_{kind}_score"
    residual = f"probe_{key}_{kind}_residual" if kind in _PREDICTIVE_KINDS else None
    return score, residual
