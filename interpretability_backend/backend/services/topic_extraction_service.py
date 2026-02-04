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
            config=config
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
            duration_seconds=duration
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
    config: TopicExtractionConfig
) -> None:
    """Update collection-level metadata with topic summary.

    Args:
        collection: ChromaDB collection
        topic_infos: List of topic information
        config: Topic extraction configuration
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

    collection.modify(metadata=current_metadata)
    logger.info("Updated collection metadata with topic summary")


