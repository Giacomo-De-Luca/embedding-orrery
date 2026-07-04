"""Config-driven topic-quality evaluation over already-extracted collections.

Reads ``eval_config.toml`` (next to this file by default, or a path in the
``ORRERY_EVAL_CONFIG`` env var), then for each configured collection it loads the
**current active** topic extraction from DuckDB, the projection coordinates, and
the original embeddings from ChromaDB, runs :class:`TopicQualityEvaluator`, prints
a report, and writes the results to a JSON file.

Run:
    uv run python -m interpretability_backend.evaluation.run_evaluation

Note: DBCV requires the live fitted HDBSCAN model, which is not persisted, so it is
``null`` when scoring stored labels here (it is only available inside a fresh
extraction flow). All other metrics are computed.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from interpretability_backend.backend.topic_extraction.cluster_and_label import GenerateTopics
from interpretability_backend.backend.utils.duckdb_sync import _get_db as _get_duckdb
from interpretability_backend.backend.utils.embedding_loader import load_embeddings_for_ids
from interpretability_backend.evaluation.quality_metrics import TopicQualityEvaluator
from interpretability_backend.evaluation.utils import (
    load_config,
    resolve_config_path,
    write_results,
)

logger = logging.getLogger("orrery." + __name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "eval_config.toml"
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "evaluation_results.json"


def _recompute_keywords(documents, labels, n_keywords, language):
    """Recompute per-cluster c-TF-IDF keywords for the given labels.

    Per-subtopic keywords are not persisted (only reduced-topic keywords are), so
    we derive keywords for whichever level is evaluated directly from the documents
    and the chosen label column. Reuses the production c-TF-IDF implementation so
    coherence/diversity are consistent with the actual assignments.
    """
    documents_df = pd.DataFrame(
        {"Document_ID": range(len(documents)), "Document": documents, "Topic": labels}
    )
    generator = GenerateTopics(documents=list(documents), language=language)
    return generator.extract_topics(documents_df, n_words=n_keywords)


def evaluate_collection(
    duckdb,
    collection_name: str,
    projection_type: str = "umap_2d",
    sample_size: int = 10000,
    n_keywords: int = 10,
    level: str = "topic",
    language: str | None = "english",
) -> dict | None:
    """Evaluate the current active extraction for one collection.

    ``level`` selects which assignment to score:
      * ``"topic"`` — the active (possibly reduced) topic ids.
      * ``"subtopic"`` — the pre-reduction subtopic ids (the original HDBSCAN
        density clusters), which is the more meaningful geometric evaluation when
        reduction merged topics in c-TF-IDF space.
    """
    active = duckdb.get_active_topics(collection_name)
    if not active:
        logger.warning("Collection %r has no active topics; skipping", collection_name)
        return None

    projection = duckdb.get_projection_data(collection_name, projection_type)
    if not projection:
        logger.warning(
            "Collection %r has no %s projections; skipping", collection_name, projection_type
        )
        return None

    ids = projection["ids"]
    # `items.document` is nullable; coerce any None to "" so c-TF-IDF's str join is safe.
    documents = [d if isinstance(d, str) else "" for d in (projection["documents"] or [])]
    if not documents:
        documents = [""] * len(ids)
    coords = np.array(projection["coordinates"], dtype=np.float64)

    # Build per-item labels aligned to projection ids, from the chosen level.
    label_col = "subtopic_id" if level == "subtopic" else "topic_id"
    rows = duckdb.get_topic_assignments_raw(active["id"], columns=["item_id", label_col])
    assignment = {item_id: label for item_id, label in rows}
    labels = np.array(
        [assignment.get(item_id) if assignment.get(item_id) is not None else -1 for item_id in ids]
    )

    if level == "subtopic" and len(set(labels.tolist()) - {-1}) < 2:
        logger.warning(
            "Collection %r has no usable subtopics (was reduction applied?); skipping",
            collection_name,
        )
        return None

    # Per-cluster keywords for diversity/coherence (recomputed for the chosen level).
    topics_data = _recompute_keywords(documents, labels, n_keywords, language)

    embeddings = load_embeddings_for_ids(collection_name, ids)
    if embeddings is None:
        logger.warning(
            "Embeddings unavailable for %r; silhouette_embedding will be null", collection_name
        )

    metrics = TopicQualityEvaluator().evaluate(
        labels=labels,
        projection_coords=coords,
        embeddings=embeddings,
        topics_data=topics_data,
        documents=documents,
        language=language,
        sample_size=sample_size,
        n_keywords=n_keywords,
        cluster_space="stored",
    )
    metrics["collection_name"] = collection_name
    metrics["projection_type"] = projection_type
    metrics["level"] = level
    metrics["num_items"] = len(ids)
    return metrics


def _print_report(metrics: dict) -> None:
    """Pretty-print one collection's metrics."""
    print("\n" + "=" * 70)
    print(
        f"TOPIC QUALITY: {metrics['collection_name']}  "
        f"(level={metrics.get('level', 'topic')}, {metrics['num_items']} items)"
    )
    print("=" * 70)

    def fmt(value):
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    ordered = [
        ("DBCV (HDBSCAN validity)", "dbcv"),
        ("Silhouette (embedding, cosine)", "silhouette_embedding"),
        ("Silhouette (projection, euclidean)", "silhouette_projection"),
        ("Topic diversity", "topic_diversity"),
        ("Coherence C_v", "coherence_cv"),
        ("Coherence U_Mass", "coherence_umass"),
        ("Clusters evaluated", "num_clusters_evaluated"),
        ("Silhouette sampled", "sampled"),
    ]
    for title, key in ordered:
        print(f"  {title:<36} {fmt(metrics.get(key))}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

    config_path = resolve_config_path("ORRERY_EVAL_CONFIG", DEFAULT_CONFIG_PATH)
    config = load_config(config_path)

    collections = config.get("collections", [])
    projection_type = config.get("projection_type", "umap_2d")
    sample_size = int(config.get("sample_size", 10000))
    n_keywords = int(config.get("n_keywords", 10))
    level = config.get("level", "topic")
    language = config.get("language", "english")
    output_path = Path(config.get("output_path", DEFAULT_OUTPUT_PATH))

    if not collections:
        print(f"No collections listed in {config_path}. Add a `collections = [...]` entry.")
        return

    duckdb = _get_duckdb()
    if duckdb is None:
        print("DuckDB unavailable; cannot evaluate.")
        return

    results = []
    for collection_name in collections:
        metrics = evaluate_collection(
            duckdb, collection_name, projection_type, sample_size, n_keywords, level, language
        )
        if metrics:
            results.append(metrics)
            _print_report(metrics)

    write_results(output_path, results)


if __name__ == "__main__":
    main()
