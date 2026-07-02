"""Enums shared across the probing pipeline."""

from enum import Enum


class TokenPosition(Enum):
    """Strategy for reducing the sequence dimension of activations.

    Applied at extraction time — the extractor pools each sample's
    `[seq_len, hidden]` token sequence to a single `[hidden]` vector.
    Downstream stages (SAE encoding, probes) consume the pre-pooled
    vector and cannot re-pool it. Future extension: token-level extraction
    + downstream pooling for per-experiment strategy comparison.
    """

    LAST = "last"            # activation[:, -1, :]
    FIRST = "first"          # activation[:, 0, :]
    MEAN = "mean"            # activation.mean(dim=1)
    MAX = "max"              # activation.max(dim=1).values — per-feature max across tokens
    WORD_LAST = "word_last"  # last token of the target word within the prompt


class TaskType(Enum):
    """Whether the probe performs regression or classification."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
