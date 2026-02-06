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
from ..topic_extraction.llm_labeling import generate_llm_labels
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
                    result = reducer.reduce_to_n_topics(
                        n_topics=config.nr_topics,
                        use_ctfidf=config.use_ctfidf_for_reduction
                    )
                    documents_df = result.documents_df
                    topics_data = result.topics_data
                    logger.info(f"Reduced from {result.num_topics_before} to {result.num_topics_after} topics")
            elif config.reduction_method == "auto":
                result = reducer.auto_reduce_topics(use_ctfidf=config.use_ctfidf_for_reduction)
                documents_df = result.documents_df
                topics_data = result.topics_data
                logger.info(f"Auto-reduced from {result.num_topics_before} to {result.num_topics_after} topics")
            else:
                raise ValueError(f"Invalid reduction_method: {config.reduction_method}")

            # Recalculate topic counts after reduction
            topic_counts = documents_df["Topic"].value_counts().to_dict()
            num_noise = topic_counts.get(-1, 0)
            num_topics = len([t for t in topic_counts.keys() if t != -1])
            logger.info(f"After reduction: {num_topics} topics, {num_noise} noise points")

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
            topic_labels=topic_labels
        )

        # Update collection-level metadata
        _update_collection_topic_metadata(
            collection=collection,
            topic_infos=topic_infos,
            config=config,
            num_topics_before_reduction=num_topics_before_reduction
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
            reduction_applied=config.reduce_topics
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
    batch_size: int = 1000
) -> None:
    """Update item metadata with topic_id and topic_label in batches.

    Args:
        collection: ChromaDB collection
        ids: All item IDs
        raw_metadatas: Current metadata for each item
        topic_assignments: Mapping of item ID -> topic_id
        topic_labels: Mapping of topic_id -> label
        batch_size: Items per batch
    """
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_metadatas = []

        for j, item_id in enumerate(batch_ids):
            meta = raw_metadatas[i + j].copy()
            topic_id = topic_assignments.get(item_id, -1)
            meta["topic_id"] = str(topic_id)
            meta["topic_label"] = topic_labels.get(topic_id, "Unclustered")
            batch_metadatas.append(meta)

        collection.update(ids=batch_ids, metadatas=batch_metadatas)

    logger.info(f"Updated {len(ids)} items with topic metadata")


def _update_collection_topic_metadata(
    collection,
    topic_infos: List[TopicInfoResult],
    config: TopicExtractionConfig,
    num_topics_before_reduction: Optional[int] = None
) -> None:
    """Update collection-level metadata with topic summary.

    Args:
        collection: ChromaDB collection
        topic_infos: List of topic information
        config: Topic extraction configuration
        num_topics_before_reduction: Number of topics before reduction (if applied)
    """
    # Build topic summary for collection metadata
    topic_summary = []
    for info in topic_infos:
        topic_summary.append({
            "topic_id": info.topic_id,
            "label": info.label,
            "count": info.count,
            "keywords": [{"word": w, "score": round(s, 4)} for w, s in info.keywords[:5]]
        })

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
            topic_labels=topic_labels
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
            num_topics_before_reduction=num_topics_before
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
            reduction_applied=True
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
