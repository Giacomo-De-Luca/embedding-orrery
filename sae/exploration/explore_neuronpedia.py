"""Lookup utility for Neuronpedia feature labels and activation examples.

Wraps the local JSONL files under
``resources/sae_labels/neuronpedia_gemma-3-4b-it/`` to answer two
questions:

1. "Which features have a label matching <pattern>?" — substring or
   regex search across the Neuronpedia autointerpreter labels, with
   density filtering. Can search one layer or many at once.
2. "What are the top-activating documents for feature <idx>?" — returns
   the reconstructed text plus per-token activation values so you can
   see which tokens inside each document drive the feature.

A single :class:`NeuronpediaExplorer` instance is reusable across many
queries and many layers. Features JSONL files are lazy-loaded and cached
per layer on first access, so repeated searches are cheap.

Usage is config-driven: edit the constants in ``main()`` below and run::

    uv run python -m interpret.sae.exploration.explore_neuronpedia
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
LABELS_DIR = _REPO_ROOT / "resources/sae_labels/neuronpedia_gemma-3-4b-it"


@dataclass
class FeatureMatch:
    """A feature label matched by a pattern search."""

    layer: int
    index: int
    label: str
    density: float


@dataclass
class ActivationExample:
    """One top-activating token sequence for a feature.

    ``values`` is aligned 1:1 with ``tokens`` and contains the per-token
    activation of the target feature inside this document.
    """

    layer: int
    feature_index: int
    max_value: float
    max_token_index: int
    tokens: list[str]
    values: list[float]
    text: str  # verbatim token join (same as ''.join(tokens))
    context: str  # window of tokens around the peak

    def top_tokens(self, k: int = 5) -> list[tuple[int, str, float]]:
        """Return the ``k`` highest-activating tokens inside this document.

        Returns:
            List of ``(token_index, token_string, activation)`` tuples
            sorted by activation descending. Empty if ``values`` was
            missing from the source record.
        """
        if not self.values:
            return []
        indexed = sorted(
            enumerate(self.values), key=lambda it: -it[1]
        )
        return [
            (i, self.tokens[i], val) for i, val in indexed[:k]
        ]


@dataclass
class _FeatureRow:
    """Internal cached row from a features JSONL file."""

    index: int
    density: float
    label: str
    top_logits: list
    bottom_logits: list


@dataclass
class FeatureInfo:
    """Metadata about a single feature."""

    layer: int
    index: int
    density: float
    label: str
    top_logits: list
    bottom_logits: list


class NeuronpediaExplorer:
    """Reusable reader for Neuronpedia features + activations JSONL files.

    Construct once, then query many times across one or more layers.
    Feature files (~65 MB each) are lazy-loaded into memory on first
    access and cached for subsequent queries. Activation files (~1.9 GB
    each) are streamed on every call since they're too large to cache.

    Example::

        explorer = NeuronpediaExplorer(layers=[9, 17, 22, 29])
        matches = explorer.search_labels("pirate")
        for layer, hits in matches.items():
            print(layer, len(hits))

        # Narrow the search without reconstructing the explorer:
        explorer.set_layers(29)
        orange_hits = explorer.search_labels("orange")

        examples = explorer.get_top_activations(feature_index=14525, layer=29)
        for ex in examples:
            print(ex.max_value, ex.top_tokens(3))
    """

    def __init__(
        self,
        layers: int | list[int] = [9, 17, 22, 29],
        width: int = 16_384,
        model_id: str = "gemma-3-4b-it",
        labels_dir: Path = LABELS_DIR,
    ) -> None:
        self._layers: list[int] = self._normalise_layers(layers)
        self.width = width
        self.model_id = model_id
        self.labels_dir = Path(labels_dir)
        self._features_cache: dict[int, list[_FeatureRow]] = {}

    # --- layer management ---

    @property
    def layers(self) -> list[int]:
        """Current default layers used when a method doesn't specify any."""
        return list(self._layers)

    def set_layers(self, layers: int | list[int]) -> None:
        """Replace the default layer set for subsequent queries."""
        self._layers = self._normalise_layers(layers)

    @staticmethod
    def _normalise_layers(layers: int | list[int] | tuple[int, ...]) -> list[int]:
        if isinstance(layers, int):
            return [layers]
        return [int(x) for x in layers]

    def _resolve_layers(self, layers: int | list[int] | None) -> list[int]:
        if layers is None:
            return list(self._layers)
        return self._normalise_layers(layers)

    # --- path helpers ---

    @property
    def _width_str(self) -> str:
        return f"{self.width // 1024}k" if self.width >= 1024 else str(self.width)

    def features_path(self, layer: int) -> Path:
        stem = f"{self.model_id}_{layer}-gemmascope-2-res-{self._width_str}"
        return self.labels_dir / f"{stem}_features.jsonl"

    def activations_path(self, layer: int) -> Path:
        stem = f"{self.model_id}_{layer}-gemmascope-2-res-{self._width_str}"
        return self.labels_dir / f"{stem}_activations.jsonl"

    # --- features loading (cached) ---

    def _load_features(self, layer: int) -> list[_FeatureRow]:
        """Load and cache the features JSONL for one layer."""
        if layer in self._features_cache:
            return self._features_cache[layer]

        path = self.features_path(layer)
        if not path.exists():
            raise FileNotFoundError(f"No features file at {path}")

        rows: list[_FeatureRow] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                explanations = entry.get("explanations", [])
                label = explanations[0].get("text", "") if explanations else ""
                rows.append(
                    _FeatureRow(
                        index=int(entry["index"]),
                        density=float(entry.get("density", 0.0)),
                        label=label,
                        top_logits=entry.get("top_logits") or [],
                        bottom_logits=entry.get("bottom_logits") or [],
                    )
                )

        self._features_cache[layer] = rows
        return rows

    def clear_cache(self) -> None:
        """Drop all cached feature rows."""
        self._features_cache.clear()

    # --- label search ---

    def search_labels(
        self,
        pattern: str,
        layers: int | list[int] | None = None,
        regex: bool = False,
        min_density: float = 0.0,
        limit: int = 50,
    ) -> dict[int, list[FeatureMatch]]:
        """Scan feature labels for a substring or regex pattern.

        Args:
            pattern: Substring to match, or regex if ``regex=True``.
            layers: Single layer, list of layers, or ``None`` to use the
                explorer's default layer set.
            regex: If True, ``pattern`` is compiled as a regex (re.I).
            min_density: Drop features below this density threshold.
            limit: Max matches per layer (sorted by density descending).

        Returns:
            Dict mapping layer index to its list of matches.
        """
        if regex:
            rx = re.compile(pattern, re.IGNORECASE)
            predicate = lambda text: rx.search(text) is not None
        else:
            needle = pattern.lower()
            predicate = lambda text: needle in text.lower()

        result: dict[int, list[FeatureMatch]] = {}
        for layer in self._resolve_layers(layers):
            rows = self._load_features(layer)
            matches = [
                FeatureMatch(
                    layer=layer,
                    index=row.index,
                    label=row.label,
                    density=row.density,
                )
                for row in rows
                if row.density >= min_density and predicate(row.label)
            ]
            matches.sort(key=lambda m: -m.density)
            result[layer] = matches[:limit]
        return result

    # --- feature info ---

    def get_feature_info(
        self, feature_index: int, layer: int | None = None
    ) -> FeatureInfo | None:
        """Return density, label, and logits for a single feature."""

        if not layer:
            layer = self.layers[0]

        for row in self._load_features(layer):
            if row.index == feature_index:
                return FeatureInfo(
                    layer=layer,
                    index=row.index,
                    density=row.density,
                    label=row.label,
                    top_logits=row.top_logits,
                    bottom_logits=row.bottom_logits,
                )
        return None

    # --- activation lookup ---

    def get_top_activations(
        self,
        feature_index: int,
        layer: int | None = None,
        k: int = 5,
        context_window: int = 5,
    ) -> list[ActivationExample]:
        """Return the top-``k`` activation documents for a feature.

        Args:
            feature_index: Target feature.
            layer: Layer the feature lives in.
            k: Maximum number of documents to return.
            context_window: Tokens on each side of the peak included in
                ``ActivationExample.context``.

        Returns:
            Documents sorted by ``max_value`` descending. Each example
            carries the full per-token ``values`` array.
        """
        if not layer:
            layer = self.layers[0]

        path = self.activations_path(layer)
        if not path.exists():
            raise FileNotFoundError(f"No activations file at {path}")

        records: list[ActivationExample] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if int(entry.get("index", -1)) != feature_index:
                    continue
                tokens = entry["tokens"]
                raw_values = entry.get("values") or []
                max_idx = int(entry["maxValueTokenIndex"])
                lo = max(0, max_idx - context_window)
                hi = min(len(tokens), max_idx + context_window + 1)
                records.append(
                    ActivationExample(
                        layer=layer,
                        feature_index=feature_index,
                        max_value=float(entry["maxValue"]),
                        max_token_index=max_idx,
                        tokens=list(tokens),
                        values=[float(v) for v in raw_values],
                        text="".join(tokens),
                        context="".join(tokens[lo:hi]),
                    )
                )

        records.sort(key=lambda r: -r.max_value)
        return records[:k]


# ------------------------------ example usage ------------------------------ #


# Edit these constants to run a different exploration. No CLI.

SEARCH_LAYERS = [9, 17, 22, 29]
SEARCH_PATTERN = "orange"
SEARCH_IS_REGEX = False
SEARCH_MIN_DENSITY = 0.0
SEARCH_LIMIT = 10

INSPECT_LAYER = 29
INSPECT_FEATURE = 14525
INSPECT_TOP_K = 5
INSPECT_CONTEXT_WINDOW = 5
INSPECT_TOP_TOKENS_PER_EXAMPLE = 5


def main() -> None:
    explorer = NeuronpediaExplorer(layers=SEARCH_LAYERS)

    # 1. Search feature labels across layers
    print(f"### Label search: {SEARCH_PATTERN!r} "
          f"(min_density={SEARCH_MIN_DENSITY})\n")
    results = explorer.search_labels(
        SEARCH_PATTERN,
        regex=SEARCH_IS_REGEX,
        min_density=SEARCH_MIN_DENSITY,
        limit=SEARCH_LIMIT,
    )
    for layer, matches in results.items():
        print(f"-- layer {layer}: {len(matches)} match(es) --")
        for m in matches:
            print(f"  feat {m.index:6d}  density={m.density:.5f}  {m.label}")
        print()

    # 2. Inspect a specific feature
    print(f"### Top-{INSPECT_TOP_K} activations for "
          f"layer {INSPECT_LAYER} / feature {INSPECT_FEATURE}\n")
    info = explorer.get_feature_info(INSPECT_FEATURE, layer=INSPECT_LAYER)
    if info is not None:
        print(f"label: {info.label!r}")
        print(f"density: {info.density:.5f}")
        if info.top_logits:
            top = info.top_logits[:5]
            print("top logits:", [(tok, round(s, 2)) for tok, s in top])
        print()

    examples = explorer.get_top_activations(
        INSPECT_FEATURE,
        layer=INSPECT_LAYER,
        k=INSPECT_TOP_K,
        context_window=INSPECT_CONTEXT_WINDOW,
    )
    for i, ex in enumerate(examples):
        print(f"-- example {i}: max={ex.max_value:.1f} "
              f"at tok {ex.max_token_index} --")
        print(f"   context: {ex.context!r}")
        top_tokens = ex.top_tokens(INSPECT_TOP_TOKENS_PER_EXAMPLE)
        if top_tokens:
            print("   top tokens in doc:")
            for idx, tok, val in top_tokens:
                print(f"     tok[{idx}]={tok!r:>20s}  act={val:8.1f}")
        print(f"   full text (first 200 chars): {ex.text[:200]!r}")
        print()


if __name__ == "__main__":
    main()
