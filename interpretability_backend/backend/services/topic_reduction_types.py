"""Topic reduction requests and settings, without the clustering stack.

Imported by the GraphQL layer (converters, types), so it stays stdlib-only:
the reduction itself (``topic_reduction_service``) pulls in hdbscan, sklearn
and scipy and is imported lazily (torch-free / lean import boundary).

Strategies:

- ``average``: average linkage on centred-semantic topic centroids;
- ``ward``: size-weighted Ward linkage on the same centroids;
- ``density``: HDBSCAN's condensed tree (extractions made with HDBSCAN);
- ``legacy_ctfidf`` / ``legacy_semantic``: the original BERTopic-style
  reduction on c-TF-IDF rows or raw centroids, kept only for comparison.

``method`` is ``fixed_n`` (cut at ``n_topics``, noise not counted) or
``auto``: a distance-threshold cut for ``average`` (default from the config),
HDBSCAN run on the topic vectors for the legacy strategies.
"""

from dataclasses import dataclass
from typing import ClassVar

from ..utils.json_settings import JsonFileSettings
from ..utils.resource_paths import PACKAGE_ROOT

TREE_LINKAGES = ("average", "ward", "density")
LEGACY_STRATEGIES = ("legacy_ctfidf", "legacy_semantic")
STRATEGIES = TREE_LINKAGES + LEGACY_STRATEGIES
METHODS = ("fixed_n", "auto")
# Strategies whose "auto" method exists: a threshold cut, or legacy HDBSCAN.
AUTO_STRATEGIES = ("average", *LEGACY_STRATEGIES)


@dataclass(frozen=True)
class TopicReductionSettings(JsonFileSettings):
    """Tunables for topic reduction and merge trees (all optional in the config file)."""

    SETTINGS_LABEL: ClassVar[str] = "topic reduction settings"

    # "auto" average-linkage cut: merge while the centred cosine distance is at most this.
    auto_distance_threshold: float = 0.7
    # Merge trees are built for at most this many base topics.
    max_tree_topics: int = 3000
    # Merge trees (and fallback centroid tables) kept in memory.
    tree_cache_size: int = 32
    # Without stored centroids, compute them from the vectors up to this many items.
    fallback_centroid_max_items: int = 200_000

    def __post_init__(self):
        self.require_number("auto_distance_threshold", minimum=0.0, maximum=2.0)
        self.require_int("max_tree_topics", minimum=3)
        self.require_int("tree_cache_size", minimum=1)
        self.require_int("fallback_centroid_max_items", minimum=0)


SETTINGS = TopicReductionSettings.from_file(PACKAGE_ROOT / "config" / "topic_reduction.json")


@dataclass(frozen=True)
class ReductionRequest:
    """A validated topic reduction: which strategy, where to cut, how to label."""

    strategy: str
    method: str = "fixed_n"
    n_topics: int | None = None
    distance_threshold: float | None = None
    absorb_noise: bool = False
    regenerate_labels: bool = False
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3-flash-preview"

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(f"Unknown reduction strategy {self.strategy!r}; expected {STRATEGIES}")
        if self.method not in METHODS:
            raise ValueError(f"Unknown reduction method {self.method!r}; expected {METHODS}")
        if self.method == "fixed_n":
            if not isinstance(self.n_topics, int) or self.n_topics < 1:
                raise ValueError("A fixed_n reduction needs nTopics >= 1")
            if self.distance_threshold is not None:
                raise ValueError("distanceThreshold applies to the auto method only")
        elif self.strategy not in AUTO_STRATEGIES:
            raise ValueError(f"The {self.strategy} strategy needs a topic count (fixed_n)")
        elif self.distance_threshold is not None and self.strategy != "average":
            raise ValueError("distanceThreshold applies to the average strategy only")
        if self.distance_threshold is not None and not 0.0 <= self.distance_threshold <= 2.0:
            raise ValueError("distanceThreshold is a cosine distance: between 0 and 2")
        if self.absorb_noise and self.strategy != "density":
            raise ValueError("absorbNoise needs the density strategy")

    @classmethod
    def create(
        cls,
        *,
        strategy: str | None,
        method: str,
        n_topics: int | None,
        use_ctfidf: bool = True,
        distance_threshold: float | None = None,
        absorb_noise: bool = False,
        settings: TopicReductionSettings = SETTINGS,
        **labelling,
    ) -> "ReductionRequest":
        """A request from API inputs; a null ``strategy`` is the legacy one
        ``use_ctfidf`` picks, and an auto average cut without a threshold uses
        the configured default."""
        strategy = strategy or ("legacy_ctfidf" if use_ctfidf else "legacy_semantic")
        if method == "auto" and strategy == "average" and distance_threshold is None:
            distance_threshold = settings.auto_distance_threshold
        return cls(
            strategy=strategy,
            method=method,
            n_topics=n_topics if method == "fixed_n" else None,
            distance_threshold=distance_threshold,
            absorb_noise=absorb_noise,
            **labelling,
        )

    @property
    def is_tree(self) -> bool:
        return self.strategy in TREE_LINKAGES

    def snapshot(self) -> dict:
        """The cut, as recorded in the reduced extraction's config snapshot."""
        return {
            "strategy": self.strategy,
            "method": self.method,
            "n_topics": self.n_topics,
            "distance_threshold": self.distance_threshold,
            "absorb_noise": self.absorb_noise,
        }
