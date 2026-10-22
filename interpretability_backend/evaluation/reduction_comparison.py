"""Compare topic-reduction strategies on stored collections, storing nothing.

For each collection: its base topics (``BaseTopics``: the original clusters of
the active extraction), then every configured strategy and cut through
``TopicReductionService.plan``, the app's own code path. Each partition is
scored with:

- topic count, largest-topic share (of clustered documents), size entropy
  (1 = equal sizes), noise share, noise absorbed;
- keyword diversity, C_v / U_Mass coherence and the silhouette in the stored
  projection the clustering used (``score_metrics``, the topic-quality scorer);
- NMI / ARI against a ground-truth metadata field, when one is configured;
- ARI between strategies at the same topic count (``agreement``).

Driven by ``run_reduction_comparison.py`` (config ``reduction_comparison.toml``).
"""

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from interpretability_backend.backend.services.topic_quality_service import score_metrics
from interpretability_backend.backend.services.topic_reduction_service import (
    BaseTopics,
    TopicReductionService,
)
from interpretability_backend.backend.services.topic_reduction_types import (
    LEGACY_STRATEGIES,
    ReductionRequest,
)

logger = logging.getLogger("orrery." + __name__)

NOISE = -1
# score_metrics result keys reported per row.
METRIC_KEYS = ("topic_diversity", "coherence_cv", "coherence_umass", "silhouette_cluster_space")


@dataclass(frozen=True)
class ComparisonCase:
    """One strategy and cut to score."""

    strategy: str
    method: str = "fixed_n"
    n_topics: int | None = None
    distance_threshold: float | None = None
    absorb_noise: bool = False

    @property
    def name(self) -> str:
        if self.method == "auto":
            cut = f"d<={self.distance_threshold}" if self.distance_threshold is not None else "auto"
        else:
            cut = f"N={self.n_topics}"
        return f"{self.strategy} {cut}" + (" +noise" if self.absorb_noise else "")

    def request(self) -> ReductionRequest:
        return ReductionRequest(
            strategy=self.strategy,
            method=self.method,
            n_topics=self.n_topics,
            distance_threshold=self.distance_threshold,
            absorb_noise=self.absorb_noise,
        )


class PartitionStats:
    """Shape of a partition, and its agreement with another one (pure numpy/sklearn)."""

    @staticmethod
    def sizes(labels: np.ndarray) -> dict:
        labels = np.asarray(labels)
        clustered = labels[labels != NOISE]
        counts = np.unique(clustered, return_counts=True)[1] if clustered.size else np.array([])
        shares = counts / counts.sum() if counts.size else counts
        entropy = (
            float(-(shares * np.log(shares)).sum() / np.log(len(shares)))
            if len(shares) > 1
            else 0.0
        )
        return {
            "n_topics": len(counts),
            "largest_share": float(shares.max()) if counts.size else 0.0,
            "size_entropy": entropy,
            "noise_share": float(np.mean(labels == NOISE)) if labels.size else 0.0,
        }

    @staticmethod
    def agreement(a: np.ndarray, b: np.ndarray) -> float | None:
        """ARI over the items both partitions cluster (noise excluded)."""
        a, b = np.asarray(a), np.asarray(b)
        both = (a != NOISE) & (b != NOISE)
        if both.sum() < 2:
            return None
        return float(adjusted_rand_score(a[both], b[both]))

    @staticmethod
    def against(labels: np.ndarray, truth: list) -> dict:
        """NMI / ARI against ground-truth values, over clustered items that have one."""
        labels = np.asarray(labels)
        keep = [i for i, value in enumerate(truth) if value is not None and labels[i] != NOISE]
        if len(keep) < 2:
            return {"nmi": None, "ari": None}
        predicted = labels[keep]
        expected = [str(truth[i]) for i in keep]
        return {
            "nmi": float(normalized_mutual_info_score(expected, predicted)),
            "ari": float(adjusted_rand_score(expected, predicted)),
        }


class _Silent:
    """Progress channel for ``score_metrics`` that drops its messages."""

    def send(self, item) -> None:
        pass


class ReductionComparison:
    def __init__(
        self,
        duckdb,
        service: TopicReductionService,
        metrics: set[str] | None = None,
        sample_size: int = 10000,
        n_keywords: int = 10,
    ) -> None:
        self._db = duckdb
        self._service = service
        self._metrics = metrics
        self._sample_size = sample_size
        self._n_keywords = n_keywords

    @staticmethod
    def cases(
        strategies: list[str],
        n_topics: list[int],
        distance_thresholds: list[float],
        legacy_auto: bool = True,
        absorb_noise: bool = True,
    ) -> list[ComparisonCase]:
        """Every configured strategy x cut (density twice when absorbing noise)."""
        cases = []
        for strategy in strategies:
            cases += [ComparisonCase(strategy, n_topics=n) for n in n_topics]
            if strategy == "density" and absorb_noise:
                cases += [ComparisonCase(strategy, n_topics=n, absorb_noise=True) for n in n_topics]
            if strategy == "average":
                cases += [
                    ComparisonCase(strategy, method="auto", distance_threshold=t)
                    for t in distance_thresholds
                ]
            if strategy in LEGACY_STRATEGIES and legacy_auto:
                cases.append(ComparisonCase(strategy, method="auto"))
        return cases

    def compare(
        self,
        collection_name: str,
        cases: list[ComparisonCase],
        ground_truth_field: str | None = None,
    ) -> dict:
        """Score the base topics and every case on one collection."""
        base = BaseTopics.load(self._db, collection_name, self._service.trees)
        coords, truth, scored = self._projection(base, ground_truth_field)
        k = len(base.topic_ids)

        rows = [self._row("base topics", None, base.topics, base, coords, truth, scored)]
        partitions: dict[str, np.ndarray] = {}
        for case in cases:
            if case.n_topics is not None and case.n_topics >= k:
                rows.append(self._skipped(case, f"N >= the {k} base topics"))
                continue
            try:
                plan = self._service.plan(base, case.request())
            except ValueError as exc:
                rows.append(self._skipped(case, str(exc)))
                continue
            labels = plan.result.documents_df["Topic"].to_numpy()  # aligned with base items
            partitions[case.name] = labels
            rows.append(
                self._row(
                    case.name, case, labels, base, coords, truth, scored, plan.result.absorbed_noise
                )
            )
        return {
            "collection": collection_name,
            "base_level": base.level,
            "base_topics": k,
            "items": len(base.item_ids),
            "rows": rows,
            "agreement": self._agreement(cases, partitions),
        }

    def _projection(self, base: BaseTopics, ground_truth_field: str | None):
        """Projection coordinates and ground truth aligned to the base items, and
        which items have coordinates (only those are scored)."""
        projection_type = base.config.get("projection_type") or "umap_2d"
        data = self._db.get_projection_data(base.collection_name, projection_type) or {}
        coords_of = dict(zip(data.get("ids", []), data.get("coordinates", []), strict=True))
        meta_of = dict(zip(data.get("ids", []), data.get("item_metadata", []), strict=True))
        scored = np.array([item in coords_of for item in base.item_ids])
        coords = np.array(
            [coords_of[i] for i, keep in zip(base.item_ids, scored, strict=True) if keep]
        )
        truth = [
            (meta_of.get(item) or {}).get(ground_truth_field) if ground_truth_field else None
            for item in base.item_ids
        ]
        return coords, truth, scored

    def _row(self, name, case, labels, base, coords, truth, scored, absorbed=0) -> dict:
        row = {
            "case": name,
            "strategy": case.strategy if case else "none",
            "method": case.method if case else None,
            "n_topics_requested": case.n_topics if case else None,
            "distance_threshold": case.distance_threshold if case else None,
            "absorb_noise": case.absorb_noise if case else False,
            **PartitionStats.sizes(labels),
            "absorbed_noise": absorbed,
            **PartitionStats.against(labels, truth),
        }
        if scored.any():
            config = base.config
            metrics = score_metrics(
                _Silent(),
                [doc for doc, keep in zip(base.documents, scored, strict=True) if keep],
                np.asarray(labels)[scored],
                coords,
                metrics=self._metrics,
                sample_size=self._sample_size,
                language=base.language,
                n_keywords=self._n_keywords,
                cluster_space=f"{config.get('cluster_on') or 'projection'}/"
                f"{config.get('projection_type') or 'umap_2d'}",
            )
            row.update({key: metrics.get(key) for key in METRIC_KEYS if key in metrics})
        return row

    @staticmethod
    def _skipped(case: ComparisonCase, reason: str) -> dict:
        return {"case": case.name, "strategy": case.strategy, "skipped": reason}

    @staticmethod
    def _agreement(cases: list[ComparisonCase], partitions: dict[str, np.ndarray]) -> list[dict]:
        """ARI between every two strategies cut at the same topic count."""
        by_n: dict[int, list[str]] = {}
        for case in cases:
            if case.n_topics is not None and case.name in partitions:
                by_n.setdefault(case.n_topics, []).append(case.name)
        return [
            {
                "n_topics": n,
                "a": a,
                "b": b,
                "ari": PartitionStats.agreement(partitions[a], partitions[b]),
            }
            for n, names in sorted(by_n.items())
            for a, b in combinations(names, 2)
        ]
