"""
Topic extraction service for clustering embeddings and generating labels.

Orchestrates HDBSCAN clustering on projection coordinates and c-TF-IDF
keyword extraction. Optionally generates human-readable labels via LLM
(Gemini default, OpenAI supported).
"""

import time
import json
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import chromadb
from chromadb.config import Settings

from ..topic_extraction.cluster_and_label import GenerateTopics
from ..topic_extraction.llm_labeling import generate_llm_labels, generate_llm_label_for_topic, _create_labeler
from ..embedding_functions.config import DB_PATH
from .progress_emitter import emit_progress

logger = logging.getLogger('star_map.' + __name__)


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
    language: Optional[str] = "english"  # Stop words language for CountVectorizer

    # Reduction config
    reduce_topics: bool = False
    reduction_method: str = "auto"  # "auto" or "fixed_n"
    nr_topics: Optional[int] = None
    use_ctfidf_for_reduction: bool = True


@dataclass
class TopicInfoResult:
    """Information about a single extracted topic."""
    topic_id: int
    keywords: List[Tuple[str, float]]
    label: Optional[str]
    count: int
    subtopics: Optional[List[str]] = None


@dataclass
class TopicExtractionResult:
    """Result of topic extraction."""
    collection_name: str
    num_topics: int
    num_noise_points: int
    topics: List[TopicInfoResult]
    duration_seconds: float
    error: Optional[str] = None

    # Reduction tracking
    num_topics_before_reduction: Optional[int] = None
    reduction_applied: bool = False
    topic_mappings: Optional[Dict[int, int]] = None


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
        # Step 1: Load projection data
        emit_progress(
            job_id=job_id, status="running",
            items_processed=0, total_items=0,
            current_batch=0, total_batches=5,
            message="Loading projection data..."
        )
        logger.info(f"Loading projection data for {config.collection_name}")

        db_path = str(DB_PATH.resolve())
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(
            name=config.collection_name,
            embedding_function=None
        )

        # Get all items with metadata and documents
        results = collection.get(include=["metadatas", "documents"])
        ids = results["ids"]
        documents = results["documents"] or [""] * len(ids)
        raw_metadatas = results["metadatas"] or [{}] * len(ids)

        total_items = len(ids)
        logger.info(f"Loaded {total_items} items")

        # Extract projection coordinates
        projection_key = config.projection_type  # e.g., "umap_2d"
        coords = []
        for metadata in raw_metadatas:
            try:
                coord = json.loads(metadata.get(projection_key, "[0, 0]"))
                coords.append(coord)
            except (json.JSONDecodeError, TypeError):
                dims = 3 if "3d" in projection_key else 2
                coords.append([0.0] * dims)

        reduced_embeddings = np.array(coords, dtype=np.float64)

        # Validate we have actual projections (not all zeros)
        if np.allclose(reduced_embeddings, 0):
            return TopicExtractionResult(
                collection_name=config.collection_name,
                num_topics=0,
                num_noise_points=0,
                topics=[],
                duration_seconds=time.time() - start_time,
                error=f"No {config.projection_type} projections found. Compute projections first."
            )

        # Step 2: Run HDBSCAN clustering
        emit_progress(
            job_id=job_id, status="running",
            items_processed=0, total_items=total_items,
            current_batch=1, total_batches=5,
            message="Running HDBSCAN clustering..."
        )
        logger.info(f"Clustering with min_topic_size={config.min_topic_size}")

        generator = GenerateTopics(
            documents=documents,
            min_topic_size=config.min_topic_size,
            language=config.language
        )
        documents_df = generator.generate_clusters(reduced_embeddings)

        # Count topics and noise
        topic_counts = documents_df["Topic"].value_counts().to_dict()
        num_noise = topic_counts.get(-1, 0)
        num_topics = len([t for t in topic_counts.keys() if t != -1])
        logger.info(f"Found {num_topics} topics, {num_noise} noise points")

        # Step 3: Extract keywords with c-TF-IDF
        emit_progress(
            job_id=job_id, status="running",
            items_processed=0, total_items=total_items,
            current_batch=2, total_batches=5,
            message="Extracting keywords with c-TF-IDF..."
        )

        topics_data = generator.extract_topics(documents_df, n_words=config.n_keywords)

        # Step 3.5: Topic Reduction (optional)
        num_topics_before_reduction = None
        reduction_result = None
        pre_reduction_labels = {}  # topic_id -> label (before reduction)
        pre_reduction_assignments = {}  # item_id -> original topic_id
        if config.reduce_topics:
            emit_progress(
                job_id=job_id, status="running",
                items_processed=0, total_items=total_items,
                current_batch=2.5, total_batches=5,
                message="Reducing topics..."
            )
            logger.info(f"Running topic reduction: method={config.reduction_method}, target={config.nr_topics}")

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
                chromadb_client=client
            )

            # Run reduction
            if config.reduction_method == "fixed_n":
                if config.nr_topics is None:
                    raise ValueError("nr_topics required when reduction_method='fixed_n'")
                if config.nr_topics >= num_topics + 1:  # +1 for noise
                    logger.info(f"Target ({config.nr_topics}) >= extracted ({num_topics}), skipping reduction")
                else:
                    reduction_result = reducer.reduce_to_n_topics(
                        n_topics=config.nr_topics,
                        use_ctfidf=config.use_ctfidf_for_reduction
                    )
                    documents_df = reduction_result.documents_df
                    topics_data = reduction_result.topics_data
                    logger.info(f"Reduced from {reduction_result.num_topics_before} to {reduction_result.num_topics_after} topics")
            elif config.reduction_method == "auto":
                reduction_result = reducer.auto_reduce_topics(use_ctfidf=config.use_ctfidf_for_reduction)
                documents_df = reduction_result.documents_df
                topics_data = reduction_result.topics_data
                logger.info(f"Auto-reduced from {reduction_result.num_topics_before} to {reduction_result.num_topics_after} topics")
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
        topic_labels: Dict[int, str] = {}
        topic_infos: List[TopicInfoResult] = []

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
            topic_infos.append(TopicInfoResult(
                topic_id=int(topic_id),
                keywords=keywords,
                label=label,
                count=count
            ))

        # Step 4: Optional LLM labeling
        if config.use_llm_labels:
            emit_progress(
                job_id=job_id, status="running",
                items_processed=0, total_items=total_items,
                current_batch=3, total_batches=5,
                message="Generating LLM labels..."
            )

            llm_labels = generate_llm_labels(
                topics_data=topics_data,
                documents_df=documents_df,
                llm_provider=config.llm_provider,
                llm_model=config.llm_model
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
                    pre_reduction_labels.get(old_id, f"Topic {old_id}")
                    for old_id in old_ids
                ]
                labeled_hierarchy[new_label] = subtopic_label_list

                # Attach subtopics to the corresponding TopicInfoResult
                for info in topic_infos:
                    if info.topic_id == new_id:
                        info.subtopics = subtopic_label_list
                        break

        # Step 5: Update ChromaDB metadata
        emit_progress(
            job_id=job_id, status="running",
            items_processed=0, total_items=total_items,
            current_batch=4, total_batches=5,
            message="Updating metadata..."
        )
        logger.info("Updating item metadata with topic assignments")

        # Map document IDs to topic assignments
        topic_assignments = {}
        for _, row in documents_df.iterrows():
            doc_idx = int(row["Document_ID"])
            topic_id = int(row["Topic"])
            item_id = ids[doc_idx]
            topic_assignments[item_id] = topic_id

        _batch_update_topic_metadata(
            collection=collection,
            ids=ids,
            raw_metadatas=raw_metadatas,
            topic_assignments=topic_assignments,
            topic_labels=topic_labels,
            subtopic_assignments=pre_reduction_assignments if reduction_result else None,
            subtopic_labels=pre_reduction_labels if reduction_result else None
        )

        # Update collection-level metadata
        _update_collection_topic_metadata(
            collection=collection,
            topic_infos=topic_infos,
            config=config,
            num_topics_before_reduction=num_topics_before_reduction,
            topic_hierarchy=labeled_hierarchy if labeled_hierarchy else None
        )

        duration = time.time() - start_time

        # Emit completion
        emit_progress(
            job_id=job_id, status="completed",
            items_processed=total_items, total_items=total_items,
            current_batch=5, total_batches=5,
            message="Complete!"
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
            topic_mappings=reduction_result.topic_mappings if reduction_result else None
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Topic extraction failed: {e}")

        emit_progress(
            job_id=job_id, status="failed",
            items_processed=0, total_items=0,
            current_batch=0, total_batches=0,
            error=str(e),
            message=f"Failed: {str(e)}"
        )

        return TopicExtractionResult(
            collection_name=config.collection_name,
            num_topics=0,
            num_noise_points=0,
            topics=[],
            duration_seconds=duration,
            error=str(e)
        )


def _batch_update_topic_metadata(
    collection,
    ids: List[str],
    raw_metadatas: List[dict],
    topic_assignments: Dict[str, int],
    topic_labels: Dict[int, str],
    batch_size: int = 1000,
    subtopic_assignments: Optional[Dict[str, int]] = None,
    subtopic_labels: Optional[Dict[int, str]] = None
) -> None:
    """Update item metadata with topic_id and topic_label in batches.

    Args:
        collection: ChromaDB collection
        ids: All item IDs
        raw_metadatas: Current metadata for each item
        topic_assignments: Mapping of item ID -> topic_id
        topic_labels: Mapping of topic_id -> label
        batch_size: Items per batch
        subtopic_assignments: Mapping of item ID -> original topic_id (pre-reduction)
        subtopic_labels: Mapping of original topic_id -> label (pre-reduction)
    """
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_metadatas = []

        for j, item_id in enumerate(batch_ids):
            meta = raw_metadatas[i + j].copy()
            topic_id = topic_assignments.get(item_id, -1)
            meta["topic_id"] = str(topic_id)
            meta["topic_label"] = topic_labels.get(topic_id, "Unclustered")

            # Store pre-reduction topic as subtopic if reduction was applied
            if subtopic_assignments is not None and subtopic_labels is not None:
                subtopic_id = subtopic_assignments.get(item_id, -1)
                meta["subtopic_id"] = str(subtopic_id)
                meta["subtopic_label"] = subtopic_labels.get(subtopic_id, "Unclustered")

            batch_metadatas.append(meta)

        collection.update(ids=batch_ids, metadatas=batch_metadatas)

    logger.info(f"Updated {len(ids)} items with topic metadata")


def _update_collection_topic_metadata(
    collection,
    topic_infos: List[TopicInfoResult],
    config: TopicExtractionConfig,
    num_topics_before_reduction: Optional[int] = None,
    topic_hierarchy: Optional[Dict[str, List[str]]] = None
) -> None:
    """Update collection-level metadata with topic summary.

    Args:
        collection: ChromaDB collection
        topic_infos: List of topic information
        config: Topic extraction configuration
        num_topics_before_reduction: Number of topics before reduction (if applied)
        topic_hierarchy: Mapping of reduced topic label -> list of original subtopic labels
    """
    # Build topic summary for collection metadata
    topic_summary = []
    for info in topic_infos:
        entry = {
            "topic_id": info.topic_id,
            "label": info.label,
            "count": info.count,
            "keywords": [{"word": w, "score": round(s, 4)} for w, s in info.keywords[:5]]
        }
        if info.subtopics:
            entry["subtopics"] = info.subtopics
        topic_summary.append(entry)

    current_metadata = collection.metadata or {}
    current_metadata.update({
        "has_topics": True,
        "topics_extracted_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        "topic_count": len([t for t in topic_infos if t.topic_id != -1]),
        "topic_config": json.dumps({
            "min_topic_size": config.min_topic_size,
            "n_keywords": config.n_keywords,
            "projection_type": config.projection_type,
            "used_llm": config.use_llm_labels,
        }),
        "topic_summary": json.dumps(topic_summary),
    })
    # Invalidate cached field analysis — topic extraction adds/changes topic_id and
    # topic_label fields, so the frontend must re-analyze field metadata on next load.
    current_metadata.pop("field_analysis", None)

    # Add reduction info if applied
    if config.reduce_topics and num_topics_before_reduction is not None:
        current_metadata.update({
            "reduction_applied": True,
            "num_topics_before_reduction": num_topics_before_reduction,
            "reduction_method": config.reduction_method,
            "reduction_target": config.nr_topics,
        })

    # Store topic hierarchy if provided
    if topic_hierarchy:
        current_metadata["topic_hierarchy"] = json.dumps(topic_hierarchy)

    collection.modify(metadata=current_metadata)
    logger.info("Updated collection metadata with topic summary")




def reduce_existing_topics(
    collection_name: str,
    method: str,
    n_topics: Optional[int],
    use_ctfidf: bool,
    regenerate_labels: bool,
    llm_provider: str,
    llm_model: str,
    language: str = "english"
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
            job_id=job_id, status="running",
            items_processed=0, total_items=0,
            current_batch=1, total_batches=4,
            message="Loading existing topics..."
        )
        logger.info(f"Reducing topics for collection: {collection_name}")

        db_path = str(DB_PATH.resolve())
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(
            name=collection_name,
            embedding_function=None
        )

        # Validate has_topics
        metadata = collection.metadata or {}
        if not metadata.get("has_topics", False):
            raise ValueError(f"Collection '{collection_name}' has no topics. Run extractTopics first.")

        # Step 2: Load all items
        results = collection.get(include=["metadatas", "documents"])
        ids = results["ids"]
        documents = results["documents"] or [""] * len(ids)
        raw_metadatas = results["metadatas"] or [{}] * len(ids)

        total_items = len(ids)
        logger.info(f"Loaded {total_items} items")

        # Step 3: Reconstruct documents_df
        emit_progress(
            job_id=job_id, status="running",
            items_processed=0, total_items=total_items,
            current_batch=2, total_batches=4,
            message="Reconstructing topic data..."
        )

        doc_ids = []
        doc_texts = []
        doc_topics = []

        for idx, (doc, meta) in enumerate(zip(documents, raw_metadatas)):
            doc_ids.append(idx)
            doc_texts.append(doc)
            topic_id = int(meta.get("topic_id", -1))
            doc_topics.append(topic_id)

        import pandas as pd
        documents_df = pd.DataFrame({
            "Document_ID": doc_ids,
            "Document": doc_texts,
            "Topic": doc_topics
        })

        # Step 4: Reconstruct topics_data from metadata
        topic_summary_json = metadata.get("topic_summary", "[]")
        topic_summary = json.loads(topic_summary_json)

        topics_data = {}
        for topic_info in topic_summary:
            topic_id = topic_info["topic_id"]
            keywords = [(kw["word"], kw["score"]) for kw in topic_info.get("keywords", [])]
            topics_data[topic_id] = keywords

        # Step 5: Reconstruct c-TF-IDF matrix
        from ..topic_extraction.cluster_and_label import GenerateTopics
        from sklearn.feature_extraction.text import CountVectorizer
        from ..topic_extraction.cluster_and_label import ClassTfidfTransformer

        # Group documents by topic (mega-document step)
        docs_per_topic = documents_df.groupby(['Topic'], as_index=False).agg({
            'Document': ' '.join
        })

        count_vectorizer = CountVectorizer(stop_words=language, ngram_range=(1, 1))
        X = count_vectorizer.fit_transform(docs_per_topic.Document.values)
        words = count_vectorizer.get_feature_names_out()

        ctfidf = ClassTfidfTransformer()
        ctfidf_matrix = ctfidf.fit_transform(X)

        # Step 6: Run reduction
        emit_progress(
            job_id=job_id, status="running",
            items_processed=0, total_items=total_items,
            current_batch=3, total_batches=4,
            message="Reducing topics..."
        )

        num_topics_before = len([t for t in topics_data.keys() if t != -1])
        logger.info(f"Running reduction: method={method}, use_ctfidf={use_ctfidf}")

        # Capture pre-reduction labels and assignments from existing metadata
        pre_reduction_labels: Dict[int, str] = {}
        pre_reduction_assignments: Dict[str, int] = {}

        for idx, (item_id, meta) in enumerate(zip(ids, raw_metadatas)):
            orig_topic_id = int(meta.get("topic_id", -1))
            pre_reduction_assignments[item_id] = orig_topic_id
            if orig_topic_id not in pre_reduction_labels:
                pre_reduction_labels[orig_topic_id] = meta.get("topic_label", "Unclustered")

        from ..topic_extraction.topic_reducer import TopicReducer
        reducer = TopicReducer(
            documents_df=documents_df,
            topics_data=topics_data,
            ctfidf_matrix=ctfidf_matrix,
            ctfidf_words=words,
            language=language,
            collection_name=collection_name,
            chromadb_client=client
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
        topic_labels: Dict[int, str] = {}
        topic_infos: List[TopicInfoResult] = []

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
            topic_infos.append(TopicInfoResult(
                topic_id=int(topic_id),
                keywords=keywords,
                label=label,
                count=count
            ))

        # Step 7: Optional LLM re-labeling
        if regenerate_labels:
            logger.info("Re-generating LLM labels after reduction")
            llm_labels = generate_llm_labels(
                topics_data=topics_data,
                documents_df=documents_df,
                llm_provider=llm_provider,
                llm_model=llm_model
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
                    pre_reduction_labels.get(old_id, f"Topic {old_id}")
                    for old_id in old_ids
                ]
                labeled_hierarchy[new_label] = subtopic_label_list

                # Attach subtopics to the corresponding TopicInfoResult
                for info in topic_infos:
                    if info.topic_id == new_id:
                        info.subtopics = subtopic_label_list
                        break

        # Step 8: Update metadata
        emit_progress(
            job_id=job_id, status="running",
            items_processed=0, total_items=total_items,
            current_batch=4, total_batches=4,
            message="Updating metadata..."
        )

        # Update item metadata
        topic_assignments = {}
        for _, row in documents_df.iterrows():
            doc_idx = int(row["Document_ID"])
            topic_id = int(row["Topic"])
            item_id = ids[doc_idx]
            topic_assignments[item_id] = topic_id

        _batch_update_topic_metadata(
            collection=collection,
            ids=ids,
            raw_metadatas=raw_metadatas,
            topic_assignments=topic_assignments,
            topic_labels=topic_labels,
            subtopic_assignments=pre_reduction_assignments,
            subtopic_labels=pre_reduction_labels
        )

        # Update collection metadata (with reduction info)
        topic_config = TopicExtractionConfig(
            collection_name=collection_name,
            reduce_topics=True,
            reduction_method=method,
            nr_topics=n_topics,
            use_ctfidf_for_reduction=use_ctfidf
        )

        _update_collection_topic_metadata(
            collection=collection,
            topic_infos=topic_infos,
            config=topic_config,
            num_topics_before_reduction=num_topics_before,
            topic_hierarchy=labeled_hierarchy if labeled_hierarchy else None
        )

        duration = time.time() - start_time

        # Emit completion
        emit_progress(
            job_id=job_id, status="completed",
            items_processed=total_items, total_items=total_items,
            current_batch=4, total_batches=4,
            message="Complete!"
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
            topic_mappings=result.topic_mappings
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Topic reduction failed: {e}")

        emit_progress(
            job_id=job_id, status="failed",
            items_processed=0, total_items=0,
            current_batch=0, total_batches=0,
            error=str(e),
            message=f"Failed: {str(e)}"
        )

        return TopicExtractionResult(
            collection_name=collection_name,
            num_topics=0,
            num_noise_points=0,
            topics=[],
            duration_seconds=duration,
            error=str(e)
        )


# ========== Standalone LLM Label Generation ==========

import re

@dataclass
class LlmLabelingResult:
    """Result of standalone LLM label generation."""
    collection_name: str
    topics_labeled: int
    subtopics_labeled: int
    total_topics: int
    total_subtopics: int
    duration_seconds: float
    error: Optional[str] = None


_KEYWORD_LABEL_PATTERN = re.compile(r'^[\w\-/]+( \| [\w\-/]+)+$')


def _is_keyword_label(label: str) -> bool:
    """Return True if label looks like a keyword-pattern label 'word | word | word'."""
    return bool(_KEYWORD_LABEL_PATTERN.match(label))


def _update_label_for_group(
    collection,
    all_ids: List[str],
    all_metadatas: List[dict],
    group_id: int,
    new_label: str,
    id_field: str = "topic_id",
    label_field: str = "topic_label",
    batch_size: int = 1000,
) -> None:
    """Update label for all items belonging to a specific topic/subtopic group."""
    ids_to_update = []
    metas_to_update = []
    for item_id, meta in zip(all_ids, all_metadatas):
        if int(meta.get(id_field, -1)) == group_id:
            updated = meta.copy()
            updated[label_field] = new_label
            ids_to_update.append(item_id)
            metas_to_update.append(updated)

    for i in range(0, len(ids_to_update), batch_size):
        collection.update(
            ids=ids_to_update[i:i + batch_size],
            metadatas=metas_to_update[i:i + batch_size],
        )


def _update_topic_summary_label(collection, topic_id: int, new_label: str) -> None:
    """Update one topic's label in the collection-level topic_summary JSON."""
    metadata = collection.metadata or {}
    summary = json.loads(metadata.get("topic_summary", "[]"))
    for entry in summary:
        if entry["topic_id"] == topic_id:
            entry["label"] = new_label
            break
    metadata["topic_summary"] = json.dumps(summary)
    collection.modify(metadata=metadata)


def _update_subtopic_label_in_hierarchy(collection, old_label: str, new_label: str) -> None:
    """Update a subtopic label in the topic_hierarchy JSON and topic_summary subtopics."""
    metadata = collection.metadata or {}

    # Update topic_hierarchy
    hierarchy_json = metadata.get("topic_hierarchy")
    if hierarchy_json:
        hierarchy = json.loads(hierarchy_json)
        for parent_label, subtopic_list in hierarchy.items():
            hierarchy[parent_label] = [
                new_label if s == old_label else s for s in subtopic_list
            ]
        metadata["topic_hierarchy"] = json.dumps(hierarchy)

    # Update subtopics lists in topic_summary
    summary = json.loads(metadata.get("topic_summary", "[]"))
    for entry in summary:
        if "subtopics" in entry:
            entry["subtopics"] = [
                new_label if s == old_label else s for s in entry["subtopics"]
            ]
    metadata["topic_summary"] = json.dumps(summary)

    collection.modify(metadata=metadata)


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
            job_id=job_id, status="running",
            items_processed=0, total_items=0,
            current_batch=0, total_batches=1,
            message="Loading collection..."
        )
        logger.info(f"LLM labeling for collection: {collection_name}")

        db_path = str(DB_PATH.resolve())
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(
            name=collection_name,
            embedding_function=None
        )

        metadata = collection.metadata or {}
        if not metadata.get("has_topics", False):
            raise ValueError(f"Collection '{collection_name}' has no topics. Run extractTopics first.")

        # Step 2: Load items and topic summary
        results = collection.get(include=["metadatas", "documents"])
        all_ids = results["ids"]
        all_documents = results["documents"] or [""] * len(all_ids)
        all_metadatas = results["metadatas"] or [{}] * len(all_ids)

        topic_summary = json.loads(metadata.get("topic_summary", "[]"))
        has_hierarchy = bool(metadata.get("topic_hierarchy"))

        # Build topics_data from summary
        topics_data: Dict[int, List[Tuple[str, float]]] = {}
        topic_labels_map: Dict[int, str] = {}
        for entry in topic_summary:
            tid = entry["topic_id"]
            keywords = [(kw["word"], kw["score"]) for kw in entry.get("keywords", [])]
            topics_data[tid] = keywords
            topic_labels_map[tid] = entry.get("label", "")

        # Create labeler
        labeler = _create_labeler(llm_provider, llm_model)

        # Build documents grouped by topic for sample retrieval
        topic_docs: Dict[int, List[str]] = {}
        subtopic_docs: Dict[int, List[str]] = {}
        for doc, meta in zip(all_documents, all_metadatas):
            tid = int(meta.get("topic_id", -1))
            if tid not in topic_docs:
                topic_docs[tid] = []
            topic_docs[tid].append(doc)

            stid_raw = meta.get("subtopic_id")
            if stid_raw is not None:
                stid = int(stid_raw)
                if stid not in subtopic_docs:
                    subtopic_docs[stid] = []
                subtopic_docs[stid].append(doc)

        topics_labeled = 0
        subtopics_labeled = 0
        total_topics = len([t for t in topics_data.keys() if t != -1])

        # Gather subtopic info
        subtopic_labels_map: Dict[int, str] = {}
        subtopic_keywords: Dict[int, List[Tuple[str, float]]] = {}
        if has_hierarchy and label_scope in ("both", "subtopics_only"):
            # Collect unique subtopics from item metadata
            for meta in all_metadatas:
                stid_raw = meta.get("subtopic_id")
                if stid_raw is not None:
                    stid = int(stid_raw)
                    if stid != -1 and stid not in subtopic_labels_map:
                        subtopic_labels_map[stid] = meta.get("subtopic_label", "")

            # Extract subtopic keywords via c-TF-IDF
            if subtopic_docs and subtopic_labels_map:
                import pandas as pd
                from sklearn.feature_extraction.text import CountVectorizer
                from ..topic_extraction.cluster_and_label import ClassTfidfTransformer

                sub_doc_ids = []
                sub_doc_texts = []
                sub_doc_topics = []
                for idx, (doc, meta) in enumerate(zip(all_documents, all_metadatas)):
                    stid_raw = meta.get("subtopic_id")
                    if stid_raw is not None:
                        sub_doc_ids.append(idx)
                        sub_doc_texts.append(doc)
                        sub_doc_topics.append(int(stid_raw))

                sub_df = pd.DataFrame({
                    "Document_ID": sub_doc_ids,
                    "Document": sub_doc_texts,
                    "Topic": sub_doc_topics,
                })

                docs_per_sub = sub_df.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
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
        total_work = (total_topics if label_scope in ("both", "topics_only") else 0) + \
                     (total_subtopics if label_scope in ("both", "subtopics_only") else 0)

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
                    # Incremental save: update items
                    _update_label_for_group(
                        collection, all_ids, all_metadatas,
                        group_id=topic_id, new_label=label,
                        id_field="topic_id", label_field="topic_label",
                    )
                    # Update in-memory metadata to reflect changes for subsequent saves
                    for meta in all_metadatas:
                        if int(meta.get("topic_id", -1)) == topic_id:
                            meta["topic_label"] = label

                    # Incremental save: update topic_summary
                    _update_topic_summary_label(collection, topic_id, label)
                    topics_labeled += 1

                progress_idx += 1
                emit_progress(
                    job_id=job_id, status="running",
                    items_processed=progress_idx, total_items=total_work,
                    current_batch=0, total_batches=1,
                    message=f"Labeling topic {progress_idx}/{total_work}..."
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
                    old_label = current_label

                    # Incremental save: update items
                    _update_label_for_group(
                        collection, all_ids, all_metadatas,
                        group_id=sub_id, new_label=label,
                        id_field="subtopic_id", label_field="subtopic_label",
                    )
                    for meta in all_metadatas:
                        stid_raw = meta.get("subtopic_id")
                        if stid_raw is not None and int(stid_raw) == sub_id:
                            meta["subtopic_label"] = label

                    # Incremental save: update hierarchy and summary
                    if old_label:
                        _update_subtopic_label_in_hierarchy(collection, old_label, label)
                    subtopics_labeled += 1

                progress_idx += 1
                emit_progress(
                    job_id=job_id, status="running",
                    items_processed=progress_idx, total_items=total_work,
                    current_batch=0, total_batches=1,
                    message=f"Labeling subtopic {progress_idx}/{total_work}..."
                )
                job_state_service.update_progress(job_id, progress_idx, 0)

        duration = time.time() - start_time

        # Mark complete
        job_state_service.complete_job(job_id)
        emit_progress(
            job_id=job_id, status="completed",
            items_processed=total_work, total_items=total_work,
            current_batch=1, total_batches=1,
            message="Complete!"
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
            job_id=job_id, status="failed",
            items_processed=0, total_items=0,
            current_batch=0, total_batches=0,
            error=str(e),
            message=f"Failed: {str(e)}"
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
