"""
Topic extraction service for clustering embeddings and generating labels.

Orchestrates HDBSCAN clustering on projection coordinates and c-TF-IDF
keyword extraction. Optionally generates human-readable labels via LLM
(Gemini default, OpenAI supported).
"""

import json
import logging
import re
import time
from dataclasses import dataclass

import chromadb
import numpy as np
from chromadb.config import Settings

from ..topic_extraction.llm_labeling import (
    _create_labeler,
    generate_llm_label_for_topic,
    generate_llm_labels,
)
from ..utils.duckdb_sync import _get_db as _get_duckdb
from ..utils.embedding_loader import load_embeddings_for_ids
from ..utils.resource_paths import CHROMA_DB_PATH as DB_PATH
from .progress_emitter import emit_progress

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

    # Reduction config
    reduce_topics: bool = False
    reduction_method: str = "auto"  # "auto" or "fixed_n"
    nr_topics: int | None = None
    use_ctfidf_for_reduction: bool = True


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


def _sync_topics_to_duckdb(
    collection_name: str,
    topic_infos: list[TopicInfoResult],
    ids: list[str],
    topic_assignments: dict[str, int],
    topic_labels: dict[int, str],
    config=None,
    subtopic_assignments: dict[str, int] | None = None,
    subtopic_labels: dict[int, str] | None = None,
    reduction_applied: bool = False,
    num_topics_before_reduction: int | None = None,
    reduction_method: str | None = None,
    reduction_target: int | None = None,
    topic_hierarchy: dict | None = None,
) -> None:
    """Store topic extraction results in DuckDB.

    Creates a new topic_extraction record, inserts topic_info and
    topic_assignments (with optional subtopic fields for reduction).
    """
    db = _get_duckdb()
    if not db:
        return

    try:
        vc = db.get_vector_collection(collection_name)
        if not vc:
            logger.warning(
                "DuckDB: vector collection %r not found, skipping topic sync", collection_name
            )
            return

        dataset_name = vc["dataset_name"]

        # Build extraction config snapshot
        extraction_config = None
        if config:
            extraction_config = {
                "min_topic_size": getattr(config, "min_topic_size", None),
                "n_keywords": getattr(config, "n_keywords", None),
                "projection_type": getattr(config, "projection_type", None),
                "used_llm": getattr(config, "use_llm_labels", None),
                "clustering_method": getattr(config, "clustering_method", None),
                "n_clusters": getattr(config, "n_clusters", None),
                "cluster_on": getattr(config, "cluster_on", None),
                "cluster_n_components": getattr(config, "cluster_n_components", None),
                "cluster_min_dist": getattr(config, "cluster_min_dist", None),
                "cluster_n_neighbors": getattr(config, "cluster_n_neighbors", None),
            }

        ext_id = db.create_topic_extraction(
            collection_name,
            dataset_name,
            config=extraction_config,
        )

        # Store reduction metadata if applicable
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

        # Insert topic info
        topic_info_records = []
        for info in topic_infos:
            topic_info_records.append(
                {
                    "topic_id": info.topic_id,
                    "label": info.label,
                    "count": info.count,
                    "keywords": [{"word": w, "score": round(s, 4)} for w, s in info.keywords[:10]],
                    "subtopics": info.subtopics,
                }
            )
        db.insert_topic_info_batch(ext_id, topic_info_records)

        # Insert topic assignments (with subtopics if reduction was applied)
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

        # Update vector collection flag
        db.set_collection_has_topics(collection_name)

        logger.info(
            "DuckDB: synced %d topics, %d assignments for %s",
            len(topic_info_records),
            len(assignment_records),
            collection_name,
        )

    except Exception as e:
        logger.error("DuckDB topic sync failed: %s", e)


def extract_topics(config: TopicExtractionConfig) -> TopicExtractionResult:
    """Extract topic clusters from an existing collection.

    Pipeline:
    1. Load projection coordinates from ChromaDB
    2. Run HDBSCAN clustering on projections
    3. Extract keywords via c-TF-IDF
    4. Optionally generate LLM labels
    5. Update item metadata with topic_id and topic_label
    6. Update collection metadata with topic summary

    Args:
        config: Topic extraction configuration

    Returns:
        TopicExtractionResult with topic information
    """
    start_time = time.time()
    job_id = config.collection_name

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
        documents = projection_data["documents"] or [""] * len(ids)
        coords = projection_data["coordinates"]

        total_items = len(ids)
        logger.info(f"Loaded {total_items} items from DuckDB")

        # ChromaDB client still needed for topic_reducer semantic embeddings
        db_path = str(DB_PATH.resolve())
        client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )

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

        # Select the space to cluster on:
        #   "projection"   — stored UMAP/PCA viz coords (fast, but tuned for looks)
        #   "cluster_umap" — BERTopic-style fresh UMAP (5D, min_dist=0) for sharper density (default)
        #   "embedding"    — raw L2-normalised vectors (euclidean ≈ cosine for HDBSCAN)
        # Both raw-vector modes share the loader + None-guard; on a load failure we
        # degrade gracefully to projection coords rather than hard-fail (cluster_umap
        # is the default path).
        if config.cluster_on in ("embedding", "cluster_umap"):
            logger.info("Clustering on %s space for %s", config.cluster_on, config.collection_name)
            raw = load_embeddings_for_ids(config.collection_name, ids)
            if raw is None:
                logger.warning(
                    "Could not load embeddings for %s; falling back to projection coords.",
                    config.collection_name,
                )
                cluster_input = reduced_embeddings
            elif config.cluster_on == "cluster_umap":
                emit_progress(
                    job_id=job_id,
                    status="running",
                    items_processed=0,
                    total_items=total_items,
                    current_batch=1,
                    total_batches=5,
                    message="Reducing vectors for clustering (UMAP)...",
                )
                try:
                    cluster_input = _reduce_for_clustering(
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
                    cluster_input = reduced_embeddings
            else:  # "embedding": L2-normalise so HDBSCAN euclidean approximates cosine
                norms = np.linalg.norm(raw, axis=1, keepdims=True)
                cluster_input = raw / np.where(norms == 0, 1.0, norms)
        else:  # "projection"
            cluster_input = reduced_embeddings

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
        documents_df = generator.generate_clusters(cluster_input)

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

        # Step 3.5: Topic Reduction (optional)
        num_topics_before_reduction = None
        reduction_result = None
        pre_reduction_labels = {}  # topic_id -> label (before reduction)
        pre_reduction_assignments = {}  # item_id -> original topic_id
        if config.reduce_topics:
            emit_progress(
                job_id=job_id,
                status="running",
                items_processed=0,
                total_items=total_items,
                current_batch=2.5,
                total_batches=5,
                message="Reducing topics...",
            )
            logger.info(
                f"Running topic reduction: method={config.reduction_method}, target={config.nr_topics}"
            )

            # Store pre-reduction count
            num_topics_before_reduction = num_topics

            # Capture pre-reduction labels (top-3-keyword labels) and assignments
            for tid, kws in topics_data.items():
                if tid == -1:
                    pre_reduction_labels[tid] = "Unclustered"
                else:
                    top_words = [w for w, _ in kws[:3]]
                    pre_reduction_labels[tid] = " | ".join(top_words)

            for _, row in documents_df.iterrows():
                doc_idx = int(row["Document_ID"])
                item_id = ids[doc_idx]
                pre_reduction_assignments[item_id] = int(row["Topic"])

            # Initialize reducer
            from ..topic_extraction.topic_reducer import TopicReducer

            reducer = TopicReducer(
                documents_df=documents_df,
                topics_data=topics_data,
                ctfidf_matrix=generator.c_tf_idf_matrix,
                ctfidf_words=generator.words,
                language=config.language,
                collection_name=config.collection_name,
                chromadb_client=client,
            )

            # Run reduction
            if config.reduction_method == "fixed_n":
                if config.nr_topics is None:
                    raise ValueError("nr_topics required when reduction_method='fixed_n'")
                if config.nr_topics >= num_topics + 1:  # +1 for noise
                    logger.info(
                        f"Target ({config.nr_topics}) >= extracted ({num_topics}), skipping reduction"
                    )
                else:
                    reduction_result = reducer.reduce_to_n_topics(
                        n_topics=config.nr_topics, use_ctfidf=config.use_ctfidf_for_reduction
                    )
                    documents_df = reduction_result.documents_df
                    topics_data = reduction_result.topics_data
                    logger.info(
                        f"Reduced from {reduction_result.num_topics_before} to {reduction_result.num_topics_after} topics"
                    )
            elif config.reduction_method == "auto":
                reduction_result = reducer.auto_reduce_topics(
                    use_ctfidf=config.use_ctfidf_for_reduction
                )
                documents_df = reduction_result.documents_df
                topics_data = reduction_result.topics_data
                logger.info(
                    f"Auto-reduced from {reduction_result.num_topics_before} to {reduction_result.num_topics_after} topics"
                )
            else:
                raise ValueError(f"Invalid reduction_method: {config.reduction_method}")

            # Recalculate topic counts after reduction
            topic_counts = documents_df["Topic"].value_counts().to_dict()
            num_noise = topic_counts.get(-1, 0)
            num_topics = len([t for t in topic_counts.keys() if t != -1])
            logger.info(f"After reduction: {num_topics} topics, {num_noise} noise points")

        # Build labeled hierarchy: reduced_label -> [subtopic_labels]
        labeled_hierarchy = {}
        if reduction_result and reduction_result.topic_hierarchy:
            # We'll populate this after building topic_labels below
            pass

        # Build topic info list
        topic_labels: dict[int, str] = {}
        topic_infos: list[TopicInfoResult] = []

        for topic_id in sorted(topics_data.keys()):
            keywords = topics_data[topic_id]
            count = topic_counts.get(topic_id, 0)

            # Default label from top keywords
            if topic_id == -1:
                label = "Unclustered"
            else:
                top_words = [w for w, _ in keywords[:3]]
                label = " | ".join(top_words)

            topic_labels[topic_id] = label
            topic_infos.append(
                TopicInfoResult(topic_id=int(topic_id), keywords=keywords, label=label, count=count)
            )

        # Step 4: Optional LLM labeling
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

            def llm_progress(done, total):
                emit_progress(
                    job_id=job_id,
                    status="running",
                    items_processed=done,
                    total_items=total,
                    current_batch=3,
                    total_batches=5,
                    message=f"Generating LLM labels ({done}/{total})...",
                )

            llm_labels = generate_llm_labels(
                topics_data=topics_data,
                documents_df=documents_df,
                llm_provider=config.llm_provider,
                llm_model=config.llm_model,
                progress_callback=llm_progress,
            )

            # Update labels with LLM-generated ones
            for topic_info in topic_infos:
                if topic_info.topic_id in llm_labels and topic_info.topic_id != -1:
                    topic_info.label = llm_labels[topic_info.topic_id]
                    topic_labels[topic_info.topic_id] = llm_labels[topic_info.topic_id]

        # Build subtopic info after labels are finalized
        if reduction_result and reduction_result.topic_hierarchy:
            for new_id, old_ids in reduction_result.topic_hierarchy.items():
                new_label = topic_labels.get(new_id, f"Topic {new_id}")
                subtopic_label_list = [
                    pre_reduction_labels.get(old_id, f"Topic {old_id}") for old_id in old_ids
                ]
                labeled_hierarchy[new_label] = subtopic_label_list

                # Attach subtopics to the corresponding TopicInfoResult
                for info in topic_infos:
                    if info.topic_id == new_id:
                        info.subtopics = subtopic_label_list
                        break

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

        # Map document IDs to topic assignments
        topic_assignments = {}
        for _, row in documents_df.iterrows():
            doc_idx = int(row["Document_ID"])
            topic_id = int(row["Topic"])
            item_id = ids[doc_idx]
            topic_assignments[item_id] = topic_id

        # DuckDB: sole destination for topic data
        _sync_topics_to_duckdb(
            collection_name=config.collection_name,
            topic_infos=topic_infos,
            ids=ids,
            topic_assignments=topic_assignments,
            topic_labels=topic_labels,
            config=config,
            subtopic_assignments=pre_reduction_assignments if reduction_result else None,
            subtopic_labels=pre_reduction_labels if reduction_result else None,
            reduction_applied=bool(reduction_result),
            num_topics_before_reduction=num_topics_before_reduction,
            reduction_method=config.reduction_method if config.reduce_topics else None,
            reduction_target=config.nr_topics if config.reduce_topics else None,
            topic_hierarchy=labeled_hierarchy if labeled_hierarchy else None,
        )

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
            num_topics=num_topics,
            num_noise_points=num_noise,
            topics=topic_infos,
            duration_seconds=duration,
            num_topics_before_reduction=num_topics_before_reduction,
            reduction_applied=config.reduce_topics,
            topic_mappings=reduction_result.topic_mappings if reduction_result else None,
        )

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
    method: str,
    n_topics: int | None,
    use_ctfidf: bool,
    regenerate_labels: bool,
    llm_provider: str,
    llm_model: str,
    language: str = "english",
) -> TopicExtractionResult:
    """Reduce topics on an existing collection with extracted topics.

    Pipeline:
    1. Load collection metadata (validate has_topics=true)
    2. Load all items with topic_id, documents
    3. Reconstruct documents_df from items
    4. Reconstruct topics_data from metadata
    5. Reconstruct c-TF-IDF matrix (re-run CountVectorizer + c-TF-IDF)
    6. Initialize TopicReducer and run reduction
    7. Optional: Re-label with LLM
    8. Update metadata

    Args:
        collection_name: Name of collection with existing topics
        method: "fixed_n" or "auto"
        n_topics: Target topic count (required for fixed_n)
        use_ctfidf: Use c-TF-IDF (True) or semantic (False) embeddings
        regenerate_labels: Re-label with LLM after reduction
        llm_provider: "gemini" or "openai"
        llm_model: LLM model name
        language: Stop words language for keyword extraction

    Returns:
        TopicExtractionResult with reduced topics
    """
    start_time = time.time()
    job_id = f"{collection_name}_reduce"

    try:
        # Step 1: Load collection and validate
        emit_progress(
            job_id=job_id,
            status="running",
            items_processed=0,
            total_items=0,
            current_batch=1,
            total_batches=4,
            message="Loading existing topics...",
        )
        logger.info(f"Reducing topics for collection: {collection_name}")

        # DuckDB: load items and topic data
        duckdb = _get_duckdb()

        # ChromaDB client still needed for topic_reducer semantic embeddings
        db_path = str(DB_PATH.resolve())
        client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )

        # Validate has_topics via DuckDB
        active_topics = duckdb.get_active_topics(collection_name)
        if not active_topics:
            raise ValueError(
                f"Collection '{collection_name}' has no topics. Run extractTopics first."
            )

        extraction_id = active_topics["id"]

        # Step 2: Load items + topic assignments from DuckDB
        items_rows = duckdb.get_items_columns(collection_name, ("id", "document"))
        ids = [r[0] for r in items_rows]
        documents = [r[1] or "" for r in items_rows]

        # Load topic assignments
        assign_rows = duckdb.get_topic_assignments_raw(extraction_id, ["item_id", "topic_id"])
        assign_map = {r[0]: r[1] for r in assign_rows}

        total_items = len(ids)
        logger.info(f"Loaded {total_items} items from DuckDB")

        # Step 3: Reconstruct documents_df
        emit_progress(
            job_id=job_id,
            status="running",
            items_processed=0,
            total_items=total_items,
            current_batch=2,
            total_batches=4,
            message="Reconstructing topic data...",
        )

        doc_ids = []
        doc_texts = []
        doc_topics = []

        for idx, (item_id, doc) in enumerate(zip(ids, documents, strict=True)):
            doc_ids.append(idx)
            doc_texts.append(doc)
            topic_id = assign_map.get(item_id, -1)
            doc_topics.append(topic_id)

        import pandas as pd

        documents_df = pd.DataFrame(
            {"Document_ID": doc_ids, "Document": doc_texts, "Topic": doc_topics}
        )

        # Step 4: Reconstruct topics_data from DuckDB topic_info
        topic_summary = active_topics.get("topics", [])

        topics_data = {}
        for topic_info in topic_summary:
            topic_id = topic_info["topic_id"]
            keywords = [(kw["word"], kw["score"]) for kw in topic_info.get("keywords", [])]
            topics_data[topic_id] = keywords

        # Step 5: Reconstruct c-TF-IDF matrix
        from sklearn.feature_extraction.text import CountVectorizer

        from ..topic_extraction.cluster_and_label import ClassTfidfTransformer

        # Group documents by topic (mega-document step)
        docs_per_topic = documents_df.groupby(["Topic"], as_index=False).agg({"Document": " ".join})

        count_vectorizer = CountVectorizer(stop_words=language, ngram_range=(1, 1))
        X = count_vectorizer.fit_transform(docs_per_topic.Document.values)
        words = count_vectorizer.get_feature_names_out()

        ctfidf = ClassTfidfTransformer()
        ctfidf_matrix = ctfidf.fit_transform(X)

        # Step 6: Run reduction
        emit_progress(
            job_id=job_id,
            status="running",
            items_processed=0,
            total_items=total_items,
            current_batch=3,
            total_batches=4,
            message="Reducing topics...",
        )

        num_topics_before = len([t for t in topics_data.keys() if t != -1])
        logger.info(f"Running reduction: method={method}, use_ctfidf={use_ctfidf}")

        # Capture pre-reduction labels and assignments from DuckDB
        pre_reduction_labels: dict[int, str] = {}
        pre_reduction_assignments: dict[str, int] = {}

        # Build label map from topic_info
        for t in topic_summary:
            pre_reduction_labels[t["topic_id"]] = t.get("label", "Unclustered")

        # Assignments already loaded from DuckDB
        for item_id, topic_id in assign_map.items():
            pre_reduction_assignments[item_id] = topic_id

        from ..topic_extraction.topic_reducer import TopicReducer

        reducer = TopicReducer(
            documents_df=documents_df,
            topics_data=topics_data,
            ctfidf_matrix=ctfidf_matrix,
            ctfidf_words=words,
            language=language,
            collection_name=collection_name,
            chromadb_client=client,
        )

        # Run reduction based on method
        if method == "fixed_n":
            if n_topics is None:
                raise ValueError("n_topics required when method='fixed_n'")
            result = reducer.reduce_to_n_topics(n_topics=n_topics, use_ctfidf=use_ctfidf)
        elif method == "auto":
            result = reducer.auto_reduce_topics(use_ctfidf=use_ctfidf)
        else:
            raise ValueError(f"Invalid method: {method}")

        documents_df = result.documents_df
        topics_data = result.topics_data
        num_topics_after = result.num_topics_after

        logger.info(f"Reduced from {result.num_topics_before} to {result.num_topics_after} topics")

        # Recalculate topic counts
        topic_counts = documents_df["Topic"].value_counts().to_dict()
        num_noise = topic_counts.get(-1, 0)

        # Build topic info list
        topic_labels: dict[int, str] = {}
        topic_infos: list[TopicInfoResult] = []

        for topic_id in sorted(topics_data.keys()):
            keywords = topics_data[topic_id]
            count = topic_counts.get(topic_id, 0)

            # Default label from top keywords
            if topic_id == -1:
                label = "Unclustered"
            else:
                top_words = [w for w, _ in keywords[:3]]
                label = " | ".join(top_words)

            topic_labels[topic_id] = label
            topic_infos.append(
                TopicInfoResult(topic_id=int(topic_id), keywords=keywords, label=label, count=count)
            )

        # Step 7: Optional LLM re-labeling
        if regenerate_labels:
            logger.info("Re-generating LLM labels after reduction")

            def llm_progress(done, total):
                emit_progress(
                    job_id=job_id,
                    status="running",
                    items_processed=done,
                    total_items=total,
                    current_batch=3,
                    total_batches=4,
                    message=f"Generating LLM labels ({done}/{total})...",
                )

            llm_labels = generate_llm_labels(
                topics_data=topics_data,
                documents_df=documents_df,
                llm_provider=llm_provider,
                llm_model=llm_model,
                progress_callback=llm_progress,
            )

            for topic_info in topic_infos:
                if topic_info.topic_id in llm_labels and topic_info.topic_id != -1:
                    topic_info.label = llm_labels[topic_info.topic_id]
                    topic_labels[topic_info.topic_id] = llm_labels[topic_info.topic_id]

        # Build subtopic hierarchy from reduction result
        labeled_hierarchy = {}
        if result.topic_hierarchy:
            for new_id, old_ids in result.topic_hierarchy.items():
                new_label = topic_labels.get(new_id, f"Topic {new_id}")
                subtopic_label_list = [
                    pre_reduction_labels.get(old_id, f"Topic {old_id}") for old_id in old_ids
                ]
                labeled_hierarchy[new_label] = subtopic_label_list

                # Attach subtopics to the corresponding TopicInfoResult
                for info in topic_infos:
                    if info.topic_id == new_id:
                        info.subtopics = subtopic_label_list
                        break

        # Step 8: Update metadata
        emit_progress(
            job_id=job_id,
            status="running",
            items_processed=0,
            total_items=total_items,
            current_batch=4,
            total_batches=4,
            message="Updating metadata...",
        )

        # Build topic assignments
        topic_assignments = {}
        for _, row in documents_df.iterrows():
            doc_idx = int(row["Document_ID"])
            topic_id = int(row["Topic"])
            item_id = ids[doc_idx]
            topic_assignments[item_id] = topic_id

        # DuckDB: store reduced topics (with subtopic data from pre-reduction)
        _sync_topics_to_duckdb(
            collection_name=collection_name,
            topic_infos=topic_infos,
            ids=ids,
            topic_assignments=topic_assignments,
            topic_labels=topic_labels,
            subtopic_assignments=pre_reduction_assignments,
            subtopic_labels=pre_reduction_labels,
            reduction_applied=True,
            num_topics_before_reduction=num_topics_before,
            reduction_method=method,
            reduction_target=n_topics,
            topic_hierarchy=labeled_hierarchy if labeled_hierarchy else None,
        )

        duration = time.time() - start_time

        # Emit completion
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
            num_topics=num_topics_after,
            num_noise_points=num_noise,
            topics=topic_infos,
            duration_seconds=duration,
            num_topics_before_reduction=num_topics_before,
            reduction_applied=True,
            topic_mappings=result.topic_mappings,
        )

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


# ========== Standalone LLM Label Generation ==========


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
) -> LlmLabelingResult:
    """Generate LLM labels for existing topics in a collection.

    Incrementally saves labels to ChromaDB after each LLM call.
    Supports resume by detecting already-labeled topics (non-keyword labels).

    Args:
        collection_name: Name of collection with existing topics
        llm_provider: "gemini" or "openai"
        llm_model: Model name for the provider
        label_scope: "both", "topics_only", or "subtopics_only"
        resume: If True, skip topics that already have non-keyword labels

    Returns:
        LlmLabelingResult with counts and timing
    """
    start_time = time.time()
    job_id = f"{collection_name}_llm_labeling"
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
