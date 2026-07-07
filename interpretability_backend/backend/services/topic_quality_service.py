"""Topic-quality scoring service.

Scores the *current active* topic extraction of a collection with the
``evaluation`` package's :class:`TopicQualityEvaluator` (silhouette in the
stored-projection space, topic diversity, C_v / U_Mass coherence — DBCV is
``None`` here because the fitted HDBSCAN model is not persisted), persists the
result on the ``topic_extractions`` row keyed by level, and emits coarse
progress under job id ``{collection}_evaluate``.

Shared by the GraphQL ``evaluateTopics`` mutation and the config-driven
``evaluation/run_evaluation.py`` runner. This module pulls in the clustering
stack (hdbscan → sklearn) and gensim, so the API layer must import it lazily
(see the torch-free import boundary in the backend CLAUDE.md).
"""

import json
import logging
import time

import numpy as np
import pandas as pd

# The evaluation package is a sibling of backend/. When this module is loaded as
# interpretability_backend.backend.services.* (server, module runners) the
# relative form resolves; when loaded as top-level backend.* (unit tests via
# conftest sys.path) it cannot, so fall back to the top-level package name.
try:
    from ...evaluation.quality_metrics import TopicQualityEvaluator
except ImportError:
    from evaluation.quality_metrics import TopicQualityEvaluator

from ..topic_extraction.cluster_and_label import GenerateTopics
from ..utils.duckdb_sync import _get_db as _get_duckdb
from .progress_emitter import emit_progress

logger = logging.getLogger("orrery." + __name__)

LEVELS = ("topic", "subtopic")


def _recompute_keywords(documents, labels, n_keywords, language):
    """Recompute per-cluster c-TF-IDF keywords for the given labels.

    Per-subtopic keywords are not persisted (only reduced-topic keywords are), so
    keywords for whichever level is evaluated are derived directly from the
    documents and the chosen label column, reusing the production c-TF-IDF
    implementation so coherence/diversity are consistent with the assignments.
    """
    documents_df = pd.DataFrame(
        {"Document_ID": range(len(documents)), "Document": documents, "Topic": labels}
    )
    generator = GenerateTopics(documents=list(documents), language=language)
    return generator.extract_topics(documents_df, n_words=n_keywords)


def _extraction_config(active: dict) -> dict:
    """Parse the config snapshot stored on the extraction row."""
    config = active.get("config")
    if isinstance(config, str):
        try:
            return json.loads(config)
        except (TypeError, ValueError):
            return {}
    return config or {}


def score_topic_quality(
    collection_name: str,
    level: str = "topic",
    metrics: set[str] | None = None,
    sample_size: int = 10000,
    language: str | None = "english",
    n_keywords: int = 10,
    persist: bool = True,
    duckdb=None,
) -> dict:
    """Score the active topic extraction of ``collection_name``.

    Args:
        collection_name: Vector collection with an active topic extraction.
        level: ``"topic"`` (active/possibly-reduced topics) or ``"subtopic"``
            (pre-reduction HDBSCAN density clusters).
        metrics: Metric selection (see ``METRIC_NAMES``); ``None`` = all.
        sample_size: Silhouette subsample cap.
        language: Stop-words language for keyword/coherence tokenization.
        n_keywords: Top-N keywords per topic for diversity/coherence.
        persist: Store the result on the extraction row (keyed by level).
        duckdb: Optional client override (tests); defaults to the shared one.

    Returns:
        Metrics dict (plus meta fields) or ``{"error": ...}``. Never raises.
    """
    start_time = time.time()
    job_id = f"{collection_name}_evaluate"

    def _progress(stage: int, message: str, status: str = "running", error: str | None = None):
        emit_progress(
            job_id=job_id,
            status=status,
            items_processed=stage,
            total_items=4,
            current_batch=stage,
            total_batches=4,
            message=message,
            error=error,
        )

    def _fail(message: str) -> dict:
        logger.warning("Topic quality scoring failed for %r: %s", collection_name, message)
        _progress(0, f"Failed: {message}", status="failed", error=message)
        return {"error": message, "collection_name": collection_name, "level": level}

    try:
        if level not in LEVELS:
            return _fail(f"Invalid level {level!r}; expected one of {LEVELS}")

        duckdb = duckdb or _get_duckdb()
        if duckdb is None:
            return _fail("DuckDB unavailable")

        _progress(1, "Loading topics and projections...")
        active = duckdb.get_active_topics(collection_name)
        if not active:
            return _fail(f"Collection '{collection_name}' has no topics. Run extractTopics first.")

        config = _extraction_config(active)
        projection_type = config.get("projection_type") or "umap_2d"
        cluster_on = config.get("cluster_on") or "projection"

        projection = duckdb.get_projection_data(collection_name, projection_type)
        if not projection:
            return _fail(f"No {projection_type} projections found for '{collection_name}'")

        ids = projection["ids"]
        # `items.document` is nullable; coerce None to "" so c-TF-IDF's str join is safe.
        documents = [d if isinstance(d, str) else "" for d in (projection["documents"] or [])]
        if not documents:
            documents = [""] * len(ids)
        coords = np.array(projection["coordinates"], dtype=np.float64)

        label_col = "subtopic_id" if level == "subtopic" else "topic_id"
        rows = duckdb.get_topic_assignments_raw(active["id"], columns=["item_id", label_col])
        assignment = {item_id: lab for item_id, lab in rows}
        labels = np.array(
            [
                assignment.get(item_id) if assignment.get(item_id) is not None else -1
                for item_id in ids
            ]
        )

        if len(set(labels.tolist()) - {-1}) < 2:
            if level == "subtopic":
                return _fail(
                    f"Collection '{collection_name}' has no usable subtopics "
                    "(was reduction applied?)"
                )
            return _fail(f"Collection '{collection_name}' has fewer than 2 topics")

        # Keywords are only needed for the keyword-based metrics; the c-TF-IDF
        # fit dominates cost on large collections, so skip it otherwise.
        keyword_metrics = {"diversity", "coherence_cv", "coherence_umass"}
        topics_data = None
        if metrics is None or (set(metrics) & keyword_metrics):
            _progress(2, "Recomputing per-topic keywords (c-TF-IDF)...")
            topics_data = _recompute_keywords(documents, labels, n_keywords, language)

        _progress(3, "Computing quality metrics...")
        result = TopicQualityEvaluator().evaluate(
            labels=labels,
            projection_coords=coords,
            topics_data=topics_data,
            documents=documents,
            language=language,
            sample_size=sample_size,
            n_keywords=n_keywords,
            metrics=metrics,
            cluster_space=f"{cluster_on}/{projection_type}",
        )
        result["collection_name"] = collection_name
        result["level"] = level
        result["projection_type"] = projection_type
        result["num_items"] = len(ids)
        result["duration_seconds"] = round(time.time() - start_time, 2)

        if persist:
            duckdb.update_topic_quality_metrics(active["id"], level, result)

        _progress(4, "Complete!", status="completed")
        return result

    except Exception as e:
        logger.error("Topic quality scoring error for %r: %s", collection_name, e)
        return _fail(str(e))
