"""Exact co-occurrence with bitsets and bounded-degree mutual neighbor selection."""

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pyarrow as pa

from .contract import EDGE_SCHEMA


@dataclass(frozen=True)
class Sparsification:
    topK: int = 8
    minimumJointCount: int = 20
    minimumScore: float = 0.1
    method: str = "mutual_top_k"

    def __post_init__(self) -> None:
        if (type(self.topK) is not int or self.topK < 1
                or type(self.minimumJointCount) is not int or self.minimumJointCount < 1
                or not np.isfinite(self.minimumScore) or not 0 <= self.minimumScore < 1
                or self.method != "mutual_top_k"):
            raise ValueError("invalid sparsification configuration")


class EventActivity:
    """A full feature dictionary by explicit event universe, packed into uint64 words."""

    def __init__(self, width: int, event_count: int):
        if type(width) is not int or width <= 0 or type(event_count) is not int or event_count < 0:
            raise ValueError("invalid activity shape")
        self.width = width
        self.event_count = event_count
        self.bits = np.zeros((width, (event_count + 63) // 64), dtype=np.uint64)

    def add(self, events: np.ndarray, features: np.ndarray) -> None:
        events, features = np.asarray(events), np.asarray(features)
        if (events.shape != features.shape or events.ndim != 1
                or not np.issubdtype(events.dtype, np.integer)
                or not np.issubdtype(features.dtype, np.integer)
                or np.any(events < 0) or np.any(events >= self.event_count)
                or np.any(features < 0) or np.any(features >= self.width)):
            raise ValueError("invalid feature/event indices")
        # Repeated feature/event rows set the same bit, counting the event once.
        masks = np.left_shift(np.uint64(1), (events % 64).astype(np.uint64))
        np.bitwise_or.at(self.bits, (features, events // 64), masks)

    def active_counts(self) -> np.ndarray:
        return np.bitwise_count(self.bits).sum(axis=1, dtype=np.uint64)

    def joint_counts(self, feature: int) -> np.ndarray:
        return np.bitwise_count(self.bits & self.bits[feature]).sum(axis=1, dtype=np.uint64)

    @classmethod
    def from_token_events(cls, width: int,
                          tokens: Iterable[tuple[str, int, list[int]]], unit: str) -> "EventActivity":
        if unit not in ("document", "token"):
            raise ValueError("invalid event unit")
        events: dict[object, set[int]] = {}
        for document, position, features in tokens:
            key = document if unit == "document" else (document, position)
            events.setdefault(key, set()).update(features)
        activity = cls(width, len(events))
        for index, features in enumerate(events.values()):
            activity.add(np.full(len(features), index, dtype=np.int64),
                         np.array(sorted(features), dtype=np.int64))
        return activity


@dataclass(frozen=True)
class RankedNeighbors:
    indices: np.ndarray
    counts: np.ndarray
    scores: np.ndarray
    minimumJointCount: int
    minimumScore: float


class SparseGraphBuilder:
    def __init__(self, activity: EventActivity):
        self.activity = activity
        self.counts = activity.active_counts()

    def rank(self, top_k: int, minimum_joint: int, minimum_score: float,
             progress: Callable[[int, int], None] | None = None) -> RankedNeighbors:
        Sparsification(top_k, minimum_joint, minimum_score)
        width, total = self.activity.width, self.activity.event_count
        indices = np.full((width, top_k), -1, dtype=np.int32)
        joint_counts = np.zeros((width, top_k), dtype=np.uint64)
        scores = np.zeros((width, top_k), dtype=np.float64)
        for feature in range(width):
            if self.counts[feature] >= minimum_joint:
                joints = self.activity.joint_counts(feature)
                # c=N gives undefined NPMI (constant events); assign zero/exclude.
                candidates = np.flatnonzero((joints >= minimum_joint) & (joints < total))
                candidates = candidates[candidates != feature]
                joint = joints[candidates].astype(np.float64)
                score = np.log(joint * total / (float(self.counts[feature]) * self.counts[candidates]))
                score /= -np.log(joint / total)
                eligible = (score > 0) & (score >= minimum_score)
                candidates, score = candidates[eligible], score[eligible]
                order = np.lexsort((candidates, -joints[candidates].astype(np.int64), -score))[:top_k]
                selected = candidates[order]
                indices[feature, :len(selected)] = selected
                joint_counts[feature, :len(selected)] = joints[selected]
                scores[feature, :len(selected)] = score[order]
            if progress and ((feature + 1) % 512 == 0 or feature + 1 == width):
                progress(feature + 1, width)
        return RankedNeighbors(indices, joint_counts, scores, minimum_joint, minimum_score)

    @staticmethod
    def edges(ranked: RankedNeighbors, policy: Sparsification, ids: list[str]) -> pa.Table:
        if (policy.topK > ranked.indices.shape[1]
                or policy.minimumJointCount != ranked.minimumJointCount
                or policy.minimumScore != ranked.minimumScore):
            raise ValueError("ranking does not match requested policy")
        neighbors = ranked.indices[:, :policy.topK]
        sources, slots = np.nonzero(neighbors > np.arange(len(ids))[:, None])
        targets = neighbors[sources, slots]
        mutual = np.any(neighbors[targets] == sources[:, None], axis=1)
        sources, slots, targets = sources[mutual], slots[mutual], targets[mutual]
        order = np.lexsort((targets, sources))
        sources, slots, targets = sources[order], slots[order], targets[order]
        scores = ranked.scores[sources, slots]
        return pa.Table.from_pydict({
            "source": [ids[i] for i in sources], "target": [ids[i] for i in targets],
            "sourceIndex": sources, "targetIndex": targets,
            "weight": np.clip(scores, 0, 1).astype(np.float32),
            "cooccurrenceCount": ranked.counts[sources, slots].astype(np.float64), "score": scores,
        }, schema=EDGE_SCHEMA)
