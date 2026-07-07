"""Config-driven topic-quality evaluation over already-extracted collections.

Reads ``eval_config.toml`` (next to this file by default, or a path in the
``ORRERY_EVAL_CONFIG`` env var), then for each configured collection scores the
**current active** topic extraction via
:func:`backend.services.topic_quality_service.score_topic_quality` (which also
persists the result on the extraction row), prints a report, and writes the
results to a JSON file.

Run:
    uv run python -m interpretability_backend.evaluation.run_evaluation

Note: DBCV requires the live fitted HDBSCAN model, which is not persisted, so it
is ``null`` when scoring stored labels here (it is only available inside a fresh
extraction flow). All other metrics are computed. The silhouette is measured in
the stored-projection space the clustering ran on (``silhouette_cluster_space``);
raw high-dimensional embedding silhouette was removed as non-discriminative.
"""

import logging
from pathlib import Path

from interpretability_backend.backend.services.topic_quality_service import score_topic_quality
from interpretability_backend.evaluation.utils.runner_common import (
    load_config,
    resolve_config_path,
    write_results,
)

logger = logging.getLogger("orrery." + __name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "eval_config.toml"
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "evaluation_results.json"


def _print_report(metrics: dict) -> None:
    """Pretty-print one collection's metrics."""
    print("\n" + "=" * 70)
    print(
        f"TOPIC QUALITY: {metrics['collection_name']}  "
        f"(level={metrics.get('level', 'topic')}, {metrics.get('num_items', '?')} items)"
    )
    print("=" * 70)

    def fmt(value):
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    ordered = [
        ("DBCV (HDBSCAN validity)", "dbcv"),
        ("Silhouette (cluster space)", "silhouette_cluster_space"),
        ("Topic diversity", "topic_diversity"),
        ("Coherence C_v", "coherence_cv"),
        ("Coherence U_Mass", "coherence_umass"),
        ("Clusters evaluated", "num_clusters_evaluated"),
        ("Silhouette sampled", "sampled"),
    ]
    for title, key in ordered:
        if key in metrics:
            print(f"  {title:<36} {fmt(metrics.get(key))}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

    config_path = resolve_config_path("ORRERY_EVAL_CONFIG", DEFAULT_CONFIG_PATH)
    config = load_config(config_path)

    collections = config.get("collections", [])
    sample_size = int(config.get("sample_size", 10000))
    n_keywords = int(config.get("n_keywords", 10))
    level = config.get("level", "topic")
    language = config.get("language", "english")
    metrics_selection = config.get("metrics")  # optional list; None = all
    output_path = Path(config.get("output_path", DEFAULT_OUTPUT_PATH))

    if not collections:
        print(f"No collections listed in {config_path}. Add a `collections = [...]` entry.")
        return

    results = []
    for collection_name in collections:
        metrics = score_topic_quality(
            collection_name=collection_name,
            level=level,
            metrics=set(metrics_selection) if metrics_selection else None,
            sample_size=sample_size,
            language=language,
            n_keywords=n_keywords,
        )
        if metrics.get("error"):
            logger.warning("Skipping %r: %s", collection_name, metrics["error"])
            continue
        results.append(metrics)
        _print_report(metrics)

    write_results(output_path, results)


if __name__ == "__main__":
    main()
