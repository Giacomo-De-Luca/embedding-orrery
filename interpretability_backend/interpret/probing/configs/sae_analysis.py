"""SAE-feature analysis configs.

Three analysis types: per-feature Spearman correlation, top-K features
ranked by |coef| from a trained Lasso (or other linear) probe, and a
top-K refit sweep that reports CV R² as a function of feature count.
All operate on SAE feature activations + targets and produce ranked
tables with Neuronpedia labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CorrelationMapConfig:
    """Per-feature Spearman correlation against each target column."""

    type: Literal["correlation_map"] = "correlation_map"
    top_k: int = 30
    # Filter features active on more than this fraction of samples
    # (mirrors Neuronpedia's >1% density filter). None = no filter.
    max_density: float | None = None

    # Where to find Neuronpedia label parquet files.
    sae_vectors_dir: str = "resources/sae_vectors"

    def __post_init__(self) -> None:
        if self.type != "correlation_map":
            raise ValueError(
                f"CorrelationMapConfig.type must be 'correlation_map', got {self.type!r}",
            )


@dataclass
class TopFeaturesConfig:
    """Top-K SAE features ranked by |coef| of a trained linear probe."""

    type: Literal["top_features"] = "top_features"
    # Name of the sklearn probe (matches SklearnProbeSpec.name) whose
    # directions .npz files this analysis reads.
    source_probe: str = "lasso"
    top_k: int = 30

    sae_vectors_dir: str = "resources/sae_vectors"

    def __post_init__(self) -> None:
        if self.type != "top_features":
            raise ValueError(
                f"TopFeaturesConfig.type must be 'top_features', got {self.type!r}",
            )


@dataclass
class FeatureSweepConfig:
    """How well do the top-K SAE features predict the target as K varies?

    For each layer, rank features (by |Spearman ρ| or |lasso coef|), then
    refit plain OLS on top-K features for K in `k_values` and report
    cross-validated R²/Spearman per K. Surfaces the prediction-quality
    curve as a function of feature count — answers "how many sparse
    features do I need to get to R² ~ 0.9?".
    """

    type: Literal["feature_sweep"] = "feature_sweep"
    # "pearson" — rank by |Pearson r| with target. Aligned with the OLS
    #     refit; the most principled choice when the downstream model is
    #     linear, but more sensitive to outliers in heavy-tailed feature
    #     activation distributions.
    # "rho" — rank by |Spearman ρ| with target. Robust to non-linear
    #     monotonic relationships; can over-select features whose
    #     monotonic-but-non-linear relationship OLS won't fully exploit.
    # "lasso" — rank by |coef| of a saved sklearn linear probe.
    ranking: Literal["pearson", "rho", "lasso"] = "rho"
    # Probe name to read directions from when ranking == "lasso".
    source_probe: str = "lasso"
    k_values: list[int] = field(
        default_factory=lambda: [1, 3, 5, 10, 20, 50],
    )
    n_splits: int = 5  # K-fold CV for held-out R²

    # When True, concatenate features across ALL SAE layers into one pool,
    # rank globally, and refit OLS on the top-K of the pooled features.
    # Output structure changes from `{layer: ...}` to `{"pooled": ...}` with
    # each result row carrying `feature_layers` alongside `feature_indices`
    # so you can see which layer each chosen feature came from.
    pool_layers: bool = False

    sae_vectors_dir: str = "resources/sae_vectors"

    def __post_init__(self) -> None:
        if self.type != "feature_sweep":
            raise ValueError(
                f"FeatureSweepConfig.type must be 'feature_sweep', got {self.type!r}",
            )
        if self.ranking not in ("pearson", "rho", "lasso"):
            raise ValueError(
                f"FeatureSweepConfig.ranking must be one of 'pearson', "
                f"'rho', 'lasso'; got {self.ranking!r}",
            )
        if not self.k_values or any(k <= 0 for k in self.k_values):
            raise ValueError(
                f"FeatureSweepConfig.k_values must be a non-empty list of "
                f"positive ints, got {self.k_values!r}",
            )
        if self.n_splits < 2:
            raise ValueError(
                f"FeatureSweepConfig.n_splits must be >= 2, got {self.n_splits}",
            )


@dataclass
class LassoAlphaSweepConfig:
    """Sweep Lasso alpha on SAE features; record (alpha, n_nonzero, R²) per layer.

    The existing single-alpha lasso probe answers "what does Lasso find at
    one chosen alpha?". This sweep answers "where is the natural sparsity
    point?" — at low alpha Lasso keeps most features; at high alpha it
    keeps a handful. Each alpha's `coef_` is saved as a separate `.npz`
    direction file so downstream analyses can pick the alpha that gives
    the desired sparsity / R² trade-off.
    """

    type: Literal["lasso_alpha_sweep"] = "lasso_alpha_sweep"
    alphas: list[float] = field(
        default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    )
    n_splits: int = 5
    sae_vectors_dir: str = "resources/sae_vectors"

    def __post_init__(self) -> None:
        if self.type != "lasso_alpha_sweep":
            raise ValueError(
                f"LassoAlphaSweepConfig.type must be 'lasso_alpha_sweep', got {self.type!r}",
            )
        if not self.alphas or any(a <= 0 for a in self.alphas):
            raise ValueError(
                f"LassoAlphaSweepConfig.alphas must be a non-empty list "
                f"of positive floats, got {self.alphas!r}",
            )
        if self.n_splits < 2:
            raise ValueError(
                f"LassoAlphaSweepConfig.n_splits must be >= 2, got {self.n_splits}",
            )


SAEAnalysisConfig = (
    CorrelationMapConfig | TopFeaturesConfig | FeatureSweepConfig | LassoAlphaSweepConfig
)
