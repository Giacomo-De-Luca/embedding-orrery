"""Shared fold resolution for probes + ablation.

Three sklearn-flavoured callers (`train_sklearn_probe`,
`train_mlp_probes`, `ProbeAblationRunner`) each need to turn a probe
spec + an optional `indices_override` into a list of
``(label, train_idx, val_idx)`` triples. Before this module they each
maintained their own near-identical copy of that logic.

This module owns the splitter construction. Callers wrap the result in
their own preferred shape (torch tensors, ablation-style triples with
extra seed entries, etc.).
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import (
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
)


def resolve_folds(
    *,
    n: int,
    n_folds: int | None,
    seed: int,
    is_classification: bool,
    stratify_y: np.ndarray | None = None,
    train_split: float | None = None,
    groups: np.ndarray | None = None,
    indices_override: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Build the train/val splits for a probe run.

    Resolution order (the first matching branch wins):

      1. ``indices_override`` set → a single-fold list with that
         override, label ``"fold_0"``. ``indices_override`` always
         wins over ``n_folds`` — the ablation runner constructs its
         own k-fold splits and passes them to a probe whose spec
         still carries ``n_folds`` from the experiment YAML, so the
         override-wins semantics is the load-bearing contract.
      2. ``n_folds`` set → ``StratifiedKFold(n_folds)`` for
         classification (using ``stratify_y`` as the class label) or
         ``KFold(n_folds)`` for regression. ``groups`` are not
         supported alongside ``n_folds`` and a paired call raises;
         every existing caller using groups omits ``n_folds`` already.
      3. Default → a single random / group-aware split via
         ``GroupShuffleSplit`` when groups are given, else a uniform
         permutation.

    Args:
        n: Total sample count (used to size the split index space).
        n_folds: When set, drives k-fold CV.
        seed: Master seed for both the splitter and the fallback
            random split.
        is_classification: Selects ``StratifiedKFold`` vs ``KFold``.
        stratify_y: Class-label array for stratification — REQUIRED
            when ``is_classification=True`` and ``n_folds`` is set.
            For continuous targets that are later binned (e.g.
            ``logreg`` with quantile bins), pass the binned labels
            here, not the raw continuous values.
        train_split: Fraction of samples to allocate to train in the
            single-split fallback. Required when neither
            ``indices_override`` nor ``n_folds`` is set.
        groups: Optional per-sample group labels for the single-split
            fallback (forces ``GroupShuffleSplit``).
        indices_override: Pre-computed (train, val) override.

    Returns:
        List of ``(fold_label, train_idx, val_idx)`` triples. Indices
        are ndarray (not torch tensors); each caller wraps as needed.
    """
    if indices_override is not None:
        train, val = (
            np.asarray(indices_override[0]),
            np.asarray(indices_override[1]),
        )
        return [("fold_0", train, val)]

    if n_folds:
        if groups is not None:
            raise NotImplementedError(
                "`n_folds` together with `groups` is not supported. "
                "Drop one or the other.",
            )
        if is_classification:
            if stratify_y is None:
                raise ValueError(
                    "`stratify_y` required for classification k-fold.",
                )
            splitter = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=seed,
            )
            iter_splits = splitter.split(np.zeros(n), np.asarray(stratify_y))
        else:
            splitter = KFold(
                n_splits=n_folds, shuffle=True, random_state=seed,
            )
            iter_splits = splitter.split(np.zeros(n))
        return [
            (f"fold_{i}", np.asarray(train), np.asarray(val))
            for i, (train, val) in enumerate(iter_splits)
        ]

    if train_split is None:
        raise ValueError(
            "`train_split` required when `n_folds` and "
            "`indices_override` are both unset.",
        )

    if groups is None:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        cut = int(n * train_split)
        return [("fold_0", perm[:cut], perm[cut:])]

    if len(groups) != n:
        raise ValueError(
            f"`groups` length {len(groups)} does not match n={n}.",
        )
    n_unique = len(np.unique(groups))
    if n_unique < 2:
        raise ValueError(
            f"Group split needs at least 2 unique groups; got "
            f"{n_unique}.",
        )
    splitter = GroupShuffleSplit(
        n_splits=1, train_size=train_split, random_state=seed,
    )
    train_idx, val_idx = next(splitter.split(np.zeros(n), groups=groups))
    return [("fold_0", np.asarray(train_idx), np.asarray(val_idx))]
