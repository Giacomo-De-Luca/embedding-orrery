"""
Topic extraction service for clustering embeddings and generating labels.

Orchestrates HDBSCAN clustering on projection coordinates and c-TF-IDF
keyword extraction. Optionally generates human-readable labels via LLM
(Gemini default, OpenAI supported).
"""

import json
import logging
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..topic_extraction.llm_labeling import (
    _create_labeler,
    generate_llm_label_for_topic,
)
from ..utils import cancellable_task
from ..utils.cancellable_task import CancellableTask
from ..utils.duckdb_sync import _get_db as _get_duckdb
from ..utils.embedding_loader import cancellable_embedding_batches, load_embeddings_for_ids
from .cancel_registry import (
    CANCELLED_MESSAGE,
    JobCancelled,
    emit_cancelled,
    raise_if_cancelled,
    report_cancelled,
)
from .job_ids import JobIds
from .progress_emitter import emit_progress
from .topic_centroid_service import Preloaded, TopicCentroidService
from .topic_reduction_types import ReductionRequest

if TYPE_CHECKING:
    from ..topic_extraction.cluster_and_label import ClusteringOutcome
    from ..utils.topic_merge_tree import TopicMergeTree

logger = logging.getLogger("orrery." + __name__)


def _reduce_for_clustering(embeddings, n_components, min_dist, n_neighbors):
    """Reduce raw embeddings with a BERTopic-style UMAP for density clustering.

    Unlike the stored visualization UMAP (min_dist=0.1, 2D/3D), this packs points
    tightly (min_dist≈0) into ~5D so HDBSCAN's density estimation is sharper while
    retaining more structure than 2D. UMAP output is euclidean-clusterable, so no
    post-normalisation is needed. random_state matches compute_projections.py.
    """
    import umap  # local import: optional heavy dep, mirrors compute_projections.py guard

    # UMAP requires n_neighbors < n_samples and n_components < n_samples; clamp so
    # small collections don't raise. Tiny-N failures are still caught at the call site.
    n_samples = len(embeddings)
    n_neighbors = max(2, min(n_neighbors, n_samples - 1))
    n_components = max(2, min(n_components, n_samples - 2))

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=7,
        verbose=False,
    )
    return reducer.fit_transform(embeddings)


def cluster_documents(
    channel,
    config: "TopicExtractionConfig",
    projection_coords: np.ndarray,
    raw: np.ndarray | None,
) -> "ClusteringOutcome":
    """Task target: per-document cluster labels (and HDBSCAN's density merge
    tree) in the configured space.

    Spaces (``config.cluster_on``):
      "projection"   — stored UMAP/PCA viz coords (fast, but tuned for looks)
      "cluster_umap" — BERTopic-style fresh UMAP (5D, min_dist=0) for sharper density (default)
      "embedding"    — raw L2-normalised vectors (euclidean ≈ cosine for HDBSCAN)
    Without raw vectors, or when the clustering UMAP fails (small N), the
    projection coords are used. Runs in a CancellableTask child on large
    collections, so a cancel can stop the UMAP / HDBSCAN fits mid-step.
    """
    # Deferred like extract_topics: hdbscan/sklearn stay out of module scope.
    from ..topic_extraction.cluster_and_label import cluster_points

    points = projection_coords
    if raw is not None and config.cluster_on == "cluster_umap":
        channel.send("Reducing vectors for clustering (UMAP)...")
        try:
            points = _reduce_for_clustering(
                raw,
                n_components=config.cluster_n_components,
                min_dist=config.cluster_min_dist,
                n_neighbors=config.cluster_n_neighbors,
            )
        except Exception as exc:
            # Small-N or any UMAP failure → degrade to projection coords
            # rather than hard-failing the whole extraction.
            logger.warning(
                "Clustering UMAP failed for %s (%s); falling back to projection coords.",
                config.collection_name,
                exc,
            )
    elif raw is not None and config.cluster_on == "embedding":
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        points = raw / np.where(norms == 0, 1.0, norms)

    channel.send(f"Running {config.clustering_method.upper()} clustering...")
    return cluster_points(
        points, config.clustering_method, config.min_topic_size, config.n_clusters
    )


@dataclass
class TopicExtractionConfig:
    """Configuration for topic extraction."""

    collection_name: str
    min_topic_size: int = 10
    n_keywords: int = 10
    use_llm_labels: bool = False
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3-flash-preview"
    projection_type: str = "umap_2d"  # pca_2d, pca_3d, umap_2d, umap_3d
    language: str | None = "english"  # Stop words language for CountVectorizer

    # Clustering config
    clustering_method: str = "hdbscan"  # hdbscan, kmeans, gmm, spectral
    n_clusters: int | None = None  # Required for kmeans, gmm, spectral
    # "projection" (stored UMAP/PCA coords) | "cluster_umap" (BERTopic-style 5D UMAP)
    # | "embedding" (raw L2-normalised vectors)
    cluster_on: str = "cluster_umap"
    # BERTopic-style clustering UMAP params (used only when cluster_on == "cluster_umap")
    cluster_n_components: int = 5
    cluster_min_dist: float = 0.0
    cluster_n_neighbors: int = 15

    # Reduction config (strategies: services/topic_reduction_types.py)
    reduce_topics: bool = False
    reduction_method: str = "auto"  # "auto" or "fixed_n"
    nr_topics: int | None = None  # topics to keep, noise not counted
    use_ctfidf_for_reduction: bool = True  # legacy strategy choice when no strategy is set
    reduction_strategy: str | None = None
    reduction_distance_threshold: float | None = None
    reduction_absorb_noise: bool = False


@dataclass
class TopicInfoResult:
    """Information about a single extracted topic."""

    topic_id: int
    keywords: list[tuple[str, float]]
    label: str | None
    count: int
    subtopics: list[str] | None = None


@dataclass
class TopicExtractionResult:
    """Result of topic extraction."""

    collection_name: str
    num_topics: int
    num_noise_points: int
    topics: list[TopicInfoResult]
    duration_seconds: float
    error: str | None = None

    # Reduction tracking
    num_topics_before_reduction: int | None = None
    reduction_applied: bool = False
    topic_mappings: dict[int, int] | None = None
    absorbed_noise: int = 0  # noise documents a density cut moved into topics


def _cancelled_topics_result(collection_name: str, start_time: float) -> TopicExtractionResult:
    """Result of an extraction or reduction stopped by a cancel (nothing stored)."""
    return TopicExtractionResult(
        collection_name=collection_name,
        num_topics=0,
        num_noise_points=0,
        topics=[],
        duration_seconds=time.time() - start_time,
        error=CANCELLED_MESSAGE,
    )


def _config_snapshot(config: TopicExtractionConfig) -> dict:
    """The extraction settings stored with its topics (``topic_extractions.config``)."""
    return {
        "min_topic_size": config.min_topic_size,
        "n_keywords": config.n_keywords,
        "projection_type": config.projection_type,
        "used_llm": config.use_llm_labels,
        "clustering_method": config.clustering_method,
        "n_clusters": config.n_clusters,
        "cluster_on": config.cluster_on,
        "cluster_n_components": config.cluster_n_components,
        "cluster_min_dist": config.cluster_min_dist,
        "cluster_n_neighbors": config.cluster_n_neighbors,
        "language": config.language,
    }


def _sync_topics_to_duckdb(
    collection_name: str,
    topic_infos: list[TopicInfoResult],
    topic_assignments: dict[str, int],
    topic_labels: dict[int, str],
    config_snapshot: dict | None = None,
    subtopic_assignments: dict[str, int] | None = None,
    subtopic_labels: dict[int, str] | None = None,
    reduction_applied: bool = False,
    num_topics_before_reduction: int | None = None,
    reduction_method: str | None = None,
    reduction_target: int | None = None,
    topic_hierarchy: dict | None = None,
    merge_trees: Sequence["TopicMergeTree"] = (),
) -> str:
    """Store topics in DuckDB as the collection's new active extraction.

    Inserts topic_info, topic_assignments (with subtopic fields after a
    reduction) and merge trees into an inactive extraction, then activates
    it: a failure half-way leaves the previous extraction active, removes
    the partial rows and fails the job. Returns the new extraction id.
    """
    db = _get_duckdb()
    if not db:
        raise RuntimeError("DuckDB is unavailable: the topics were not stored")
    vc = db.get_vector_collection(collection_name)
    if not vc:
        raise ValueError(f"Vector collection {collection_name!r} is not registered")

    ext_id = db.create_topic_extraction(
        collection_name, vc["dataset_name"], config=config_snapshot, activate=False
    )
    try:
        topic_count = len([t for t in topic_infos if t.topic_id != -1])
        if reduction_applied:
            db.update_topic_extraction(
                ext_id,
                reduction_applied=True,
                reduction_method=reduction_method,
                reduction_target=reduction_target,
                num_topics_before_reduction=num_topics_before_reduction,
                topic_hierarchy=json.dumps(topic_hierarchy) if topic_hierarchy else None,
                topic_count=topic_count,
            )
        else:
            db.update_topic_extraction(ext_id, topic_count=topic_count)

        topic_info_records = [
            {
                "topic_id": info.topic_id,
                "label": info.label,
                "count": info.count,
                "keywords": [{"word": w, "score": round(s, 4)} for w, s in info.keywords[:10]],
                "subtopics": info.subtopics,
            }
            for info in topic_infos
        ]
        db.insert_topic_info_batch(ext_id, topic_info_records)

        # Topic assignments, with subtopics when the topics were reduced
        assignment_records = []
        for item_id, topic_id in topic_assignments.items():
            record = {
                "item_id": item_id,
                "topic_id": topic_id,
                "topic_label": topic_labels.get(topic_id, "Unclustered"),
            }
            if subtopic_assignments is not None and subtopic_labels is not None:
                sub_id = subtopic_assignments.get(item_id, -1)
                record["subtopic_id"] = sub_id
                record["subtopic_label"] = subtopic_labels.get(sub_id, "Unclustered")
            assignment_records.append(record)
        db.insert_topic_assignments_batch(ext_id, assignment_records)

        for tree in merge_trees:
            db.replace_topic_merge_tree(
                ext_id,
                tree.linkage,
                leaf_ids=tree.leaf_ids,
                leaf_sizes=tree.leaf_sizes,
                children=tree.children,
                heights=tree.heights,
                noise_item_ids=list(tree.noise_merges),
                noise_merges=list(tree.noise_merges.values()),
            )

        db.set_collection_has_topics(collection_name)
        db.activate_topic_extraction(collection_name, ext_id)
    except Exception:
        try:
            db.delete_topic_extraction(ext_id)
        except Exception:
            logger.exception("Could not remove the partial topic extraction %s", ext_id)
        raise

    # The Explore page caches its colour options (topic fields included) in the
    # dataset metadata; they describe the previous topics, so drop them.
    try:
        db.update_collection_extra_metadata(collection_name, {"field_analysis": None})
    except Exception:
        logger.exception("Could not clear the cached field analysis of %s", collection_name)

    logger.info(
        "DuckDB: synced %d topics, %d assignments for %s",
        len(topic_info_records),
        len(assignment_records),
        collection_name,
    )
    return ext_id


def _refresh_topic_centroids(
    collection_name: str, extraction_id: str | None, preloaded: Preloaded | None = None
) -> None:
    """Store centroids for a freshly synced extraction; failures never fail the job."""
    if extraction_id is None:
        return
    try:
        rows = TopicCentroidService(duckdb=_get_duckdb()).refresh(
            collection_name, extraction_id, preloaded=preloaded
        )
        logger.info("Stored %d topic centroids for %s", rows, collection_name)
    except Exception as e:
        logger.error(
            "Topic centroids not computed for %s (topic search unavailable): %s",
            collection_name,
            e,
        )


def _merge_tree_service(duckdb, cancel_event: threading.Event | None):
    """Merge trees for one job: its vector loads stop at a cancel."""
    # Deferred: merge trees import scipy (lean schema import graph).
    from .topic_merge_tree_service import TopicMergeTreeService

    centroids = TopicCentroidService(
        duckdb=duckdb, vector_batches=cancellable_embedding_batches(cancel_event)
    )
    return TopicMergeTreeService(duckdb, centroid_service=centroids)


def _density_tree_for_items(clustering: "ClusteringOutcome", ids: list[str]):
    """HDBSCAN's density merge tree with its noise points named by item id."""
    if clustering.density is None:
        return None
    rows = clustering.density.noise_rows.tolist()
    return clustering.density.tree.with_noise(
        {ids[i]: row for i, row in enumerate(rows) if row >= 0}
    )


def _report_running(
    job_id: str, total_items: int, current_batch: int, total_batches: int, message: str
) -> None:
    emit_progress(
        job_id=job_id,
        status="running",
        items_processed=0,
        total_items=total_items,
        current_batch=current_batch,
        total_batches=total_batches,
        message=message,
    )


def _llm_progress(job_id: str, current_batch: int, total_batches: int):
    """Progress callback for LLM labelling: one event per labelled topic."""

    def report(done: int, total: int) -> None:
        emit_progress(
            job_id=job_id,
            status="running",
            items_processed=done,
            total_items=total,
            current_batch=current_batch,
            total_batches=total_batches,
            message=f"Generating LLM labels ({done}/{total})...",
        )

    return report


def extract_topics(
    config: TopicExtractionConfig, cancel_event: threading.Event | None = None
) -> TopicExtractionResult:
    """Extract topic clusters from an existing collection.

    Pipeline:
    1. Load projection coordinates from ChromaDB
    2. Run HDBSCAN clustering on projections
    3. Extract keywords via c-TF-IDF
    4. Optionally generate LLM labels
    5. Update item metadata with topic_id and topic_label
    6. Update collection metadata with topic summary

    A cancel stops the extraction before anything is stored (the clustering
    kernels stop mid-step, see cluster_documents); it reports CANCELLED_MESSAGE
    and leaves job state to the caller (the embed pipeline owns its job record).

    Args:
        config: Topic extraction configuration
        cancel_event: The job's cancel event

    Returns:
        TopicExtractionResult with topic information
    """
    start_time = time.time()
    job_id = JobIds.topic_extraction(config.collection_name)

    try:
        # Step 1: Load projection data from DuckDB
        emit_progress(
            job_id=job_id,
            status="running",
            items_processed=0,
            total_items=0,
            current_batch=0,
            total_batches=5,
            message="Loading projection data...",
        )
        logger.info(f"Loading projection data for {config.collection_name}")

        # DuckDB: load items + projections
        duckdb = _get_duckdb()
        projection_data = duckdb.get_projection_data(config.collection_name, config.projection_type)

        if not projection_data:
            return TopicExtractionResult(
                collection_name=config.collection_name,
                num_topics=0,
                num_noise_points=0,
                topics=[],
                duration_seconds=time.time() - start_time,
                error=f"No projection data found for {config.collection_name} ({config.projection_type})",
            )

        ids = projection_data["ids"]
        documents = [d or "" for d in projection_data["documents"] or [None] * len(ids)]
        coords = projection_data["coordinates"]

        total_items = len(ids)
        logger.info(f"Loaded {total_items} items from DuckDB")
        raise_if_cancelled(cancel_event)

        reduced_embeddings = np.array(coords, dtype=np.float64)

        # Validate we have actual projections (not all zeros)
        if np.allclose(reduced_embeddings, 0):
            return TopicExtractionResult(
                collection_name=config.collection_name,
                num_topics=0,
                num_noise_points=0,
                topics=[],
                duration_seconds=time.time() - start_time,
                error=f"No {config.projection_type} projections found. Compute projections first.",
            )

        # Step 2: Run HDBSCAN clustering
        emit_progress(
            job_id=job_id,
            status="running",
            items_processed=0,
            total_items=total_items,
            current_batch=1,
            total_batches=5,
            message=f"Running {config.clustering_method.upper()} clustering...",
        )
        logger.info(
            f"Clustering with method={config.clustering_method}, min_topic_size={config.min_topic_size}, n_clusters={config.n_clusters}"
        )

        # The raw-vector modes load the embeddings here (reused below for the
        # topic centroids); on a load failure clustering degrades to the
        # projection coords rather than hard-failing.
        raw = None
        if config.cluster_on in ("embedding", "cluster_umap"):
            logger.info("Clustering on %s space for %s", config.cluster_on, config.collection_name)
            raw = load_embeddings_for_ids(config.collection_name, ids, cancel_event=cancel_event)
            if raw is None:
                logger.warning(
                    "Could not load embeddings for %s; falling back to projection coords.",
                    config.collection_name,
                )

        # Local import: cluster_and_label pulls in hdbscan/sklearn/scipy (~94 MB RSS),
        # keep it out of module scope so server startup stays light (mirrors umap guard).
        from ..topic_extraction.cluster_and_label import GenerateTopics

        generator = GenerateTopics(
            documents=documents,
            min_topic_size=config.min_topic_size,
            language=config.language,
            clustering_method=config.clustering_method,
            n_clusters=config.n_clusters,
        )

        def clustering_progress(message: str) -> None:
            emit_progress(
                job_id=job_id,
                status="running",
                items_processed=0,
                total_items=total_items,
                current_batch=1,
                total_batches=5,
                message=message,
            )

        clustering = CancellableTask(
            cancel_event,
            isolate=cancellable_task.SETTINGS.should_isolate(total_items),
            on_item=clustering_progress,
        ).run(cluster_documents, config, reduced_embeddings, raw)
        documents_df = generator.documents_frame(clustering.labels)
        raise_if_cancelled(cancel_event)

        # Count topics and noise
        topic_counts = documents_df["Topic"].value_counts().to_dict()
        num_noise = topic_counts.get(-1, 0)
        num_topics = len([t for t in topic_counts.keys() if t != -1])
        logger.info(f"Found {num_topics} topics, {num_noise} noise points")

        # Step 3: Extract keywords with c-TF-IDF
        emit_progress(
            job_id=job_id,
            status="running",
            items_processed=0,
            total_items=total_items,
            current_batch=2,
            total_batches=5,
            message="Extracting keywords with c-TF-IDF...",
        )

        topics_data = generator.extract_topics(documents_df, n_words=config.n_keywords)

        # Deferred like GenerateTopics: the reduction pulls in the clustering stack.
        from .topic_reduction_service import BaseTopics, TopicReductionService

        preloaded = (ids, raw) if raw is not None else None
        density_tree = _density_tree_for_items(clustering, ids)
        base = BaseTopics.from_clustering(
            config.collection_name,
            ids,
            documents,
            clustering.labels,
            topics_data,
            config=_config_snapshot(config),
            preloaded=preloaded,
            density_tree=density_tree,
        )
        service = TopicReductionService(
            duckdb=duckdb, trees=_merge_tree_service(duckdb, cancel_event)
        )

        # Step 3.5: Topic Reduction (optional; needs two topics to merge)
        plan = None
        if config.reduce_topics and len(base.topic_ids) >= 2:
            raise_if_cancelled(cancel_event)
            emit_progress(
                job_id=job_id,
                status="running",
                items_processed=0,
                total_items=total_items,
                current_batch=2,
                total_batches=5,
                message="Reducing topics...",
            )
            request = ReductionRequest.create(
                strategy=config.reduction_strategy,
                method=config.reduction_method,
                n_topics=config.nr_topics,
                use_ctfidf=config.use_ctfidf_for_reduction,
                distance_threshold=config.reduction_distance_threshold,
                absorb_noise=config.reduction_absorb_noise,
                regenerate_labels=config.use_llm_labels,
                llm_provider=config.llm_provider,
                llm_model=config.llm_model,
            )
            logger.info("Running topic reduction: %s", request.snapshot())
            plan = service.plan(base, request, cancel_event=cancel_event)
            if plan.fallback:
                _report_running(job_id, total_items, 2, 5, plan.fallback)

        # Step 4: Labels (optional LLM labels for every final topic)
        if config.use_llm_labels:
            emit_progress(
                job_id=job_id,
                status="running",
                items_processed=0,
                total_items=total_items,
                current_batch=3,
                total_batches=5,
                message="Generating LLM labels...",
            )
        llm_progress = _llm_progress(job_id, current_batch=3, total_batches=5)
        if plan is not None:
            outcome = service.assemble(
                plan, keep_unmerged_labels=False, progress=llm_progress, cancel_event=cancel_event
            )
        else:
            outcome = service.unreduced(
                base,
                llm_provider=config.llm_provider if config.use_llm_labels else None,
                llm_model=config.llm_model,
                progress=llm_progress,
                cancel_event=cancel_event,
            )
        reduced = plan is not None and plan.changed

        # Step 5: Store topic assignments in DuckDB
        emit_progress(
            job_id=job_id,
            status="running",
            items_processed=0,
            total_items=total_items,
            current_batch=4,
            total_batches=5,
            message="Updating metadata...",
        )
        logger.info("Storing topic assignments in DuckDB")

        # Last cancel point: once synced, the extraction is the active one.
        raise_if_cancelled(cancel_event)

        # DuckDB: sole destination for topic data
        extraction_id = _sync_topics_to_duckdb(
            collection_name=config.collection_name,
            topic_infos=outcome.topic_infos,
            topic_assignments=outcome.topic_assignments,
            topic_labels=outcome.topic_labels,
            config_snapshot=service.config_snapshot(plan) if reduced else base.config,
            subtopic_assignments=outcome.subtopic_assignments,
            subtopic_labels=outcome.subtopic_labels,
            reduction_applied=reduced,
            num_topics_before_reduction=num_topics if reduced else None,
            reduction_method=config.reduction_method if reduced else None,
            reduction_target=config.nr_topics if reduced else None,
            topic_hierarchy=outcome.topic_hierarchy,
            merge_trees=[density_tree] if density_tree is not None else [],
        )

        emit_progress(
            job_id=job_id,
            status="running",
            items_processed=0,
            total_items=total_items,
            current_batch=4,
            total_batches=5,
            message="Computing topic centroids...",
        )
        _refresh_topic_centroids(config.collection_name, extraction_id, preloaded=preloaded)

        duration = time.time() - start_time

        # Emit completion
        emit_progress(
            job_id=job_id,
            status="completed",
            items_processed=total_items,
            total_items=total_items,
            current_batch=5,
            total_batches=5,
            message="Complete!",
        )

        logger.info(f"Topic extraction completed in {duration:.1f}s")

        return TopicExtractionResult(
            collection_name=config.collection_name,
            num_topics=outcome.num_topics,
            num_noise_points=outcome.num_noise,
            topics=outcome.topic_infos,
            duration_seconds=duration,
            num_topics_before_reduction=num_topics if reduced else None,
            reduction_applied=reduced,
            topic_mappings=plan.result.topic_mappings if reduced else None,
            absorbed_noise=plan.result.absorbed_noise if reduced else 0,
        )

    except JobCancelled:
        logger.info("Topic extraction cancelled for %s", config.collection_name)
        emit_cancelled(job_id)
        return _cancelled_topics_result(config.collection_name, start_time)

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Topic extraction failed: {e}")

        emit_progress(
            job_id=job_id,
            status="failed",
            items_processed=0,
            total_items=0,
            current_batch=0,
            total_batches=0,
            error=str(e),
            message=f"Failed: {str(e)}",
        )

        return TopicExtractionResult(
            collection_name=config.collection_name,
            num_topics=0,
            num_noise_points=0,
            topics=[],
            duration_seconds=duration,
            error=str(e),
        )


def reduce_existing_topics(
    collection_name: str,
    method: str = "auto",
    n_topics: int | None = None,
    use_ctfidf: bool = True,
    regenerate_labels: bool = False,
    llm_provider: str = "gemini",
    llm_model: str = "gemini-3-flash-preview",
    cancel_event: threading.Event | None = None,
    request: ReductionRequest | None = None,
) -> TopicExtractionResult:
    """Reduce the topics of a collection's active extraction (standalone post-processing).

    Always starts from the extraction's base topics (its original clusters,
    the subtopics of an already reduced extraction), so reductions never
    compound and a larger topic count undoes a smaller one. The result is a
    new active extraction: merged topics get keyword labels (LLM labels with
    ``regenerate_labels``), unmerged ones keep theirs, and the base topics
    become its subtopics.

    Args:
        collection_name: Name of collection with existing topics
        request: The reduction (strategy, cut, labels); without it, a legacy
            request is built from ``method`` / ``n_topics`` / ``use_ctfidf``
            and the labelling arguments
        cancel_event: Checked between steps and before each LLM call; a
            cancel stores nothing and reports CANCELLED_MESSAGE

    Returns:
        TopicExtractionResult with reduced topics
    """
    start_time = time.time()
    job_id = JobIds.topic_reduction(collection_name)

    try:
        _report_running(job_id, 0, 1, 4, "Loading existing topics...")
        logger.info(f"Reducing topics for collection: {collection_name}")
        if request is None:
            request = ReductionRequest.create(
                strategy=None,
                method=method,
                n_topics=n_topics,
                use_ctfidf=use_ctfidf,
                regenerate_labels=regenerate_labels,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )

        # Deferred: the reduction pulls in the clustering stack (lean schema imports).
        from .topic_reduction_service import BaseTopics, TopicReductionService

        duckdb = _get_duckdb()
        trees = _merge_tree_service(duckdb, cancel_event)
        service = TopicReductionService(duckdb=duckdb, trees=trees)
        base = BaseTopics.load(duckdb, collection_name, trees)
        total_items = len(base.item_ids)
        logger.info(f"Loaded {total_items} items from DuckDB")
        raise_if_cancelled(cancel_event)

        _report_running(job_id, total_items, 2, 4, "Reducing topics...")
        logger.info("Running reduction: %s", request.snapshot())
        plan = service.plan(base, request, cancel_event=cancel_event)
        if plan.fallback:
            _report_running(job_id, total_items, 2, 4, plan.fallback)
        logger.info(
            f"Reduced from {plan.num_topics_before} to {plan.result.num_topics_after} topics"
        )

        _report_running(job_id, total_items, 3, 4, "Labelling topics...")
        outcome = service.assemble(
            plan,
            keep_unmerged_labels=True,
            progress=_llm_progress(job_id, current_batch=3, total_batches=4),
            cancel_event=cancel_event,
        )

        _report_running(job_id, total_items, 4, 4, "Updating metadata...")
        # Last cancel point: once synced, the reduced extraction is the active one.
        raise_if_cancelled(cancel_event)

        extraction_id = _sync_topics_to_duckdb(
            collection_name=collection_name,
            topic_infos=outcome.topic_infos,
            topic_assignments=outcome.topic_assignments,
            topic_labels=outcome.topic_labels,
            config_snapshot=service.config_snapshot(plan),
            subtopic_assignments=outcome.subtopic_assignments,
            subtopic_labels=outcome.subtopic_labels,
            reduction_applied=plan.changed,
            num_topics_before_reduction=plan.num_topics_before,
            reduction_method=request.method if plan.changed else None,
            reduction_target=request.n_topics if plan.changed else None,
            topic_hierarchy=outcome.topic_hierarchy,
            merge_trees=[base.density_tree] if base.density_tree is not None else [],
        )
        _refresh_topic_centroids(collection_name, extraction_id)

        duration = time.time() - start_time
        emit_progress(
            job_id=job_id,
            status="completed",
            items_processed=total_items,
            total_items=total_items,
            current_batch=4,
            total_batches=4,
            message="Complete!",
        )
        logger.info(f"Topic reduction completed in {duration:.1f}s")

        return TopicExtractionResult(
            collection_name=collection_name,
            num_topics=outcome.num_topics,
            num_noise_points=outcome.num_noise,
            topics=outcome.topic_infos,
            duration_seconds=duration,
            num_topics_before_reduction=plan.num_topics_before,
            reduction_applied=plan.changed,
            topic_mappings=plan.result.topic_mappings,
            absorbed_noise=plan.result.absorbed_noise,
        )

    except JobCancelled:
        logger.info("Topic reduction cancelled for %s", collection_name)
        emit_cancelled(job_id)
        return _cancelled_topics_result(collection_name, start_time)

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Topic reduction failed: {e}")

        emit_progress(
            job_id=job_id,
            status="failed",
            items_processed=0,
            total_items=0,
            current_batch=0,
            total_batches=0,
            error=str(e),
            message=f"Failed: {str(e)}",
        )

        return TopicExtractionResult(
            collection_name=collection_name,
            num_topics=0,
            num_noise_points=0,
            topics=[],
            duration_seconds=duration,
            error=str(e),
        )


@dataclass
class LlmLabelingResult:
    """Result of standalone LLM label generation."""

    collection_name: str
    topics_labeled: int
    subtopics_labeled: int
    total_topics: int
    total_subtopics: int
    duration_seconds: float
    error: str | None = None


_KEYWORD_LABEL_PATTERN = re.compile(r"^[\w\-/]+( \| [\w\-/]+)+$")


def _is_keyword_label(label: str) -> bool:
    """Return True if label looks like a keyword-pattern label 'word | word | word'."""
    return bool(_KEYWORD_LABEL_PATTERN.match(label))


def generate_llm_labels_for_collection(
    collection_name: str,
    llm_provider: str = "gemini",
    llm_model: str = "gemini-3-flash-preview",
    label_scope: str = "both",
    resume: bool = False,
    cancel_event: threading.Event | None = None,
) -> LlmLabelingResult:
    """Generate LLM labels for existing topics in a collection.

    Incrementally saves labels to DuckDB after each LLM call.
    Supports resume by detecting already-labeled topics (non-keyword labels).

    Args:
        collection_name: Name of collection with existing topics
        llm_provider: "gemini" or "openai"
        llm_model: Model name for the provider
        label_scope: "both", "topics_only", or "subtopics_only"
        resume: If True, skip topics that already have non-keyword labels
        cancel_event: Checked before each LLM call; when set, stops with the
            labels saved so far and leaves the job resumable

    Returns:
        LlmLabelingResult with counts and timing
    """
    start_time = time.time()
    job_id = JobIds.llm_labeling(collection_name)
    job_state_service = None

    try:
        # Register job state
        from .job_state import get_job_state_service

        job_state_service = get_job_state_service()
        job_state_service.start_job(
            collection_name=job_id,
            job_type="llm_labeling",
            total_expected=0,
            total_batches=0,
            config={
                "collection_name": collection_name,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "label_scope": label_scope,
            },
        )

        # Step 1: Load collection and validate
        emit_progress(
            job_id=job_id,
            status="running",
            items_processed=0,
            total_items=0,
            current_batch=0,
            total_batches=1,
            message="Loading collection...",
        )
        logger.info(f"LLM labeling for collection: {collection_name}")

        # DuckDB: load items and topic data
        duckdb = _get_duckdb()
        active_topics = duckdb.get_active_topics(collection_name)
        if not active_topics:
            raise ValueError(
                f"Collection '{collection_name}' has no topics. Run extractTopics first."
            )

        extraction_id = active_topics["id"]
        topic_summary = active_topics.get("topics", [])
        has_hierarchy = bool(active_topics.get("topic_hierarchy"))

        # Load items from DuckDB
        items_rows = duckdb.get_items_columns(collection_name, ("id", "document"))
        all_ids = [r[0] for r in items_rows]
        all_documents = [r[1] or "" for r in items_rows]

        # Load topic + subtopic assignments from DuckDB
        assign_rows = duckdb.get_topic_assignments_raw(
            extraction_id,
            ["item_id", "topic_id", "topic_label", "subtopic_id", "subtopic_label"],
        )
        assign_map = {}
        for r in assign_rows:
            assign_map[r[0]] = {
                "topic_id": r[1],
                "topic_label": r[2],
                "subtopic_id": r[3],
                "subtopic_label": r[4],
            }

        # Build topics_data from summary
        topics_data: dict[int, list[tuple[str, float]]] = {}
        topic_labels_map: dict[int, str] = {}
        for entry in topic_summary:
            tid = entry["topic_id"]
            keywords = [(kw["word"], kw["score"]) for kw in entry.get("keywords", [])]
            topics_data[tid] = keywords
            topic_labels_map[tid] = entry.get("label", "")

        # Create labeler
        labeler = _create_labeler(llm_provider, llm_model)

        # Build documents grouped by topic for sample retrieval
        topic_docs: dict[int, list[str]] = {}
        subtopic_docs: dict[int, list[str]] = {}
        for item_id, doc in zip(all_ids, all_documents, strict=True):
            a = assign_map.get(item_id, {})
            tid = a.get("topic_id", -1)
            if tid not in topic_docs:
                topic_docs[tid] = []
            topic_docs[tid].append(doc)

            stid = a.get("subtopic_id")
            if stid is not None:
                if stid not in subtopic_docs:
                    subtopic_docs[stid] = []
                subtopic_docs[stid].append(doc)

        topics_labeled = 0
        subtopics_labeled = 0
        total_topics = len([t for t in topics_data.keys() if t != -1])

        # Gather subtopic info
        subtopic_labels_map: dict[int, str] = {}
        subtopic_keywords: dict[int, list[tuple[str, float]]] = {}
        if has_hierarchy and label_scope in ("both", "subtopics_only"):
            # Collect unique subtopics from assignments
            for a in assign_map.values():
                stid = a.get("subtopic_id")
                if stid is not None:
                    stid = int(stid)
                    if stid != -1 and stid not in subtopic_labels_map:
                        subtopic_labels_map[stid] = a.get("subtopic_label", "")

            # Extract subtopic keywords via c-TF-IDF
            if subtopic_docs and subtopic_labels_map:
                import pandas as pd
                from sklearn.feature_extraction.text import CountVectorizer

                from ..topic_extraction.cluster_and_label import ClassTfidfTransformer

                sub_doc_ids = []
                sub_doc_texts = []
                sub_doc_topics = []
                for idx, (item_id, doc) in enumerate(zip(all_ids, all_documents, strict=True)):
                    a = assign_map.get(item_id, {})
                    stid_raw = a.get("subtopic_id")
                    if stid_raw is not None:
                        sub_doc_ids.append(idx)
                        sub_doc_texts.append(doc)
                        sub_doc_topics.append(int(stid_raw))

                sub_df = pd.DataFrame(
                    {
                        "Document_ID": sub_doc_ids,
                        "Document": sub_doc_texts,
                        "Topic": sub_doc_topics,
                    }
                )

                docs_per_sub = sub_df.groupby(["Topic"], as_index=False).agg({"Document": " ".join})
                try:
                    count_vec = CountVectorizer(stop_words="english", ngram_range=(1, 1))
                    X = count_vec.fit_transform(docs_per_sub.Document.values)
                    words = count_vec.get_feature_names_out()
                    ctfidf = ClassTfidfTransformer()
                    ctfidf_matrix = ctfidf.fit_transform(X)

                    for row_idx, sub_tid in enumerate(docs_per_sub.Topic.values):
                        row = ctfidf_matrix[row_idx].toarray().flatten()
                        top_indices = row.argsort()[-10:][::-1]
                        subtopic_keywords[int(sub_tid)] = [
                            (words[i], float(row[i])) for i in top_indices if row[i] > 0
                        ]
                except ValueError:
                    logger.warning("Failed to extract subtopic keywords (empty vocabulary)")

        total_subtopics = len(subtopic_labels_map)
        total_work = (total_topics if label_scope in ("both", "topics_only") else 0) + (
            total_subtopics if label_scope in ("both", "subtopics_only") else 0
        )

        # Update job state with total count now that we know it
        job_state_service.update_total_expected(job_id, total_expected=total_work, total_batches=1)
        job_state_service.update_progress(job_id, 0, 0)

        progress_idx = 0

        # Step 3: Label topics
        if label_scope in ("both", "topics_only"):
            for topic_id in sorted(topics_data.keys()):
                if topic_id == -1:
                    continue

                current_label = topic_labels_map.get(topic_id, "")
                if resume and current_label and not _is_keyword_label(current_label):
                    logger.info(f"Skipping topic {topic_id} (already labeled: '{current_label}')")
                    progress_idx += 1
                    continue

                raise_if_cancelled(cancel_event)
                keywords = topics_data[topic_id]
                docs = topic_docs.get(topic_id, [])

                label = generate_llm_label_for_topic(
                    topic_id=topic_id,
                    keywords=keywords,
                    sample_documents=docs,
                    labeler=labeler,
                )

                if label is not None:
                    # Incremental save: update DuckDB topic label
                    duckdb.update_topic_label(extraction_id, topic_id, label)
                    topics_labeled += 1

                progress_idx += 1
                emit_progress(
                    job_id=job_id,
                    status="running",
                    items_processed=progress_idx,
                    total_items=total_work,
                    current_batch=0,
                    total_batches=1,
                    message=f"Labeling topic {progress_idx}/{total_work}...",
                )
                job_state_service.update_progress(job_id, progress_idx, 0)

        # Step 4: Label subtopics
        if label_scope in ("both", "subtopics_only") and has_hierarchy:
            for sub_id in sorted(subtopic_labels_map.keys()):
                current_label = subtopic_labels_map.get(sub_id, "")
                if resume and current_label and not _is_keyword_label(current_label):
                    logger.info(f"Skipping subtopic {sub_id} (already labeled: '{current_label}')")
                    progress_idx += 1
                    continue

                kws = subtopic_keywords.get(sub_id, [])
                docs = subtopic_docs.get(sub_id, [])

                if not kws and not docs:
                    progress_idx += 1
                    continue

                raise_if_cancelled(cancel_event)
                label = generate_llm_label_for_topic(
                    topic_id=sub_id,
                    keywords=kws,
                    sample_documents=docs,
                    labeler=labeler,
                )

                if label is not None:
                    # Incremental save: update DuckDB subtopic label
                    duckdb.update_subtopic_label(extraction_id, sub_id, label)
                    subtopics_labeled += 1

                progress_idx += 1
                emit_progress(
                    job_id=job_id,
                    status="running",
                    items_processed=progress_idx,
                    total_items=total_work,
                    current_batch=0,
                    total_batches=1,
                    message=f"Labeling subtopic {progress_idx}/{total_work}...",
                )
                job_state_service.update_progress(job_id, progress_idx, 0)

        duration = time.time() - start_time

        # Mark complete
        job_state_service.complete_job(job_id)
        emit_progress(
            job_id=job_id,
            status="completed",
            items_processed=total_work,
            total_items=total_work,
            current_batch=1,
            total_batches=1,
            message="Complete!",
        )

        logger.info(
            f"LLM labeling complete: {topics_labeled} topics, {subtopics_labeled} subtopics in {duration:.1f}s"
        )

        return LlmLabelingResult(
            collection_name=collection_name,
            topics_labeled=topics_labeled,
            subtopics_labeled=subtopics_labeled,
            total_topics=total_topics,
            total_subtopics=total_subtopics,
            duration_seconds=duration,
        )

    except JobCancelled:
        # Raised only from the labeling loops, after the counters are set.
        # Labels saved so far stay; the job stays resumable.
        logger.info(f"LLM labeling cancelled for {collection_name} at {progress_idx}/{total_work}")
        report_cancelled(job_id, items_processed=progress_idx, total_items=total_work)
        return LlmLabelingResult(
            collection_name=collection_name,
            topics_labeled=topics_labeled,
            subtopics_labeled=subtopics_labeled,
            total_topics=total_topics,
            total_subtopics=total_subtopics,
            duration_seconds=time.time() - start_time,
            error=CANCELLED_MESSAGE,
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"LLM labeling failed: {e}")

        if job_state_service:
            job_state_service.fail_job(job_id, str(e))

        emit_progress(
            job_id=job_id,
            status="failed",
            items_processed=0,
            total_items=0,
            current_batch=0,
            total_batches=0,
            error=str(e),
            message=f"Failed: {str(e)}",
        )

        return LlmLabelingResult(
            collection_name=collection_name,
            topics_labeled=0,
            subtopics_labeled=0,
            total_topics=0,
            total_subtopics=0,
            duration_seconds=duration,
            error=str(e),
        )
