"""Seeded train/validation splits for embedding-space probes.

``ProbeSplitter.split`` returns ``(pool_idx, train_rel, val_rel)``: the rows a
probe trains on (the valid rows, subsampled to the training cap) and the
train/validation positions *within* that pool.

- Without labels (regression and binary probes) it reproduces the original
  inline split bit-for-bit — one RNG for the cap subsample and the
  permutation — so metrics stay comparable across kinds on the same field
  and with probes trained before this module existed.
- With labels (multiclass probes) the split is stratified per class: every
  class keeps at least one validation row and one training row, and the cap
  subsample keeps at least ``min_per_class`` rows of every class, so rare
  classes survive both.
"""

import numpy as np


class ProbeSplitter:
    """One seeded RNG drives both the training-cap subsample and the split."""

    def __init__(self, seed: int, train_split: float, min_per_class: int = 5):
        self.rng = np.random.default_rng(seed)
        self.train_split = train_split
        self.min_per_class = min_per_class

    def split(
        self,
        valid_idx: np.ndarray,
        cap: int,
        labels: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split ``valid_idx`` (row indices); ``labels`` align to ``valid_idx``."""
        if labels is None:
            return self._random_split(valid_idx, cap)
        return self._stratified_split(np.asarray(valid_idx), np.asarray(labels), cap)

    def _random_split(self, valid_idx: np.ndarray, cap: int):
        if len(valid_idx) > cap:
            pool_idx = self.rng.choice(valid_idx, size=cap, replace=False)
            pool_idx.sort()
        else:
            pool_idx = valid_idx
        perm = self.rng.permutation(len(pool_idx))
        cut = int(len(pool_idx) * self.train_split)
        return pool_idx, perm[:cut], perm[cut:]

    def _stratified_split(self, valid_idx: np.ndarray, labels: np.ndarray, cap: int):
        keep = self._stratified_subsample(labels, cap) if len(valid_idx) > cap else None
        if keep is not None:
            valid_idx, labels = valid_idx[keep], labels[keep]

        train_parts, val_parts = [], []
        for cls in np.unique(labels):
            members = self.rng.permutation(np.flatnonzero(labels == cls))
            n_val = max(1, round(len(members) * (1.0 - self.train_split)))
            n_val = min(n_val, len(members) - 1)  # keep >= 1 training row
            val_parts.append(members[:n_val])
            train_parts.append(members[n_val:])
        return valid_idx, np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(val_parts))

    def _stratified_subsample(self, labels: np.ndarray, cap: int) -> np.ndarray:
        """Sorted positions of a cap-sized, per-class proportional subsample.

        Every class first keeps ``min(count, min_per_class)`` rows; the rest
        of the cap is shared in proportion to the remaining class sizes.
        """
        classes, counts = np.unique(labels, return_counts=True)
        base = np.minimum(counts, self.min_per_class)
        spare = counts - base
        remaining = max(cap - int(base.sum()), 0)
        extra = (
            np.floor(remaining * spare / spare.sum()).astype(int)
            if spare.sum()
            else np.zeros_like(spare)
        )
        quotas = base + np.minimum(extra, spare)
        chosen = [
            self.rng.choice(np.flatnonzero(labels == cls), size=int(quota), replace=False)
            for cls, quota in zip(classes, quotas, strict=True)
        ]
        return np.sort(np.concatenate(chosen))
