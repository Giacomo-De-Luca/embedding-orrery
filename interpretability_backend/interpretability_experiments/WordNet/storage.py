"""
Qdrant Storage Module for SPLADE Sparse Embeddings

This module provides storage for sparse SPLADE embeddings using Qdrant,
which natively supports sparse vectors. It also stores dense embeddings
(MiniLM) for comparison.
"""

import os
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

try:
    from .splade_embedder import SparseEmbedding
except ImportError:
    from splade_embedder import SparseEmbedding


@dataclass
class WordEmbeddingRecord:
    """Record for a word embedding with metadata."""

    id: str
    word: str
    strategy: str  # "word_level" or "definition"
    context: str  # The sentence/definition used for embedding
    synset_id: str | None = None
    pos: str | None = None
    sparse_embedding: SparseEmbedding | None = None
    dense_embedding: list[float] | None = None
    top_tokens: list[str] | None = None  # Decoded top tokens for quick lookup


class QdrantStorage:
    """
    Qdrant-based storage for word embeddings.

    Supports both sparse (SPLADE) and dense (MiniLM) vectors,
    stored in the same collection for easy comparison.
    """

    DEFAULT_DB_PATH = "interpretability/resources/vector_db_qdrant"
    DENSE_DIM = 384  # MiniLM dimension

    # Shared client instances per path
    _clients: dict[str, QdrantClient] = {}

    def __init__(self, db_path: str | None = None, collection_name: str = "wordnet_splade"):
        """
        Initialize Qdrant storage.

        Args:
            db_path: Path to store Qdrant database
            collection_name: Name of the collection
        """
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self.collection_name = collection_name

        # Ensure directory exists
        os.makedirs(self.db_path, exist_ok=True)

        # Use shared client per path to avoid locking issues
        if self.db_path not in QdrantStorage._clients:
            QdrantStorage._clients[self.db_path] = QdrantClient(path=self.db_path)
            print(f"Qdrant storage initialized at: {self.db_path}")

        self.client = QdrantStorage._clients[self.db_path]

    def create_collection(self, recreate: bool = False, dense_dim: int = DENSE_DIM) -> None:
        """
        Create a collection with both dense and sparse vector support.

        Args:
            recreate: If True, delete existing collection first
            dense_dim: Dimension for dense vectors
        """
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": VectorParams(size=dense_dim, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams()},
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection already exists: {self.collection_name}")

    def add_record(self, record: WordEmbeddingRecord) -> None:
        """Add a single record to the collection."""
        self.add_records([record])

    def add_records(self, records: list[WordEmbeddingRecord]) -> None:
        """
        Add multiple records to the collection.

        Args:
            records: List of WordEmbeddingRecord to add
        """
        points = []
        for record in records:
            # Build vectors dict (includes both dense and sparse)
            vectors = {}
            if record.dense_embedding is not None:
                vectors["dense"] = record.dense_embedding

            if record.sparse_embedding is not None:
                vectors["sparse"] = SparseVector(
                    indices=record.sparse_embedding.indices, values=record.sparse_embedding.values
                )

            # Build payload
            payload = {
                "word": record.word,
                "strategy": record.strategy,
                "context": record.context,
            }
            if record.synset_id:
                payload["synset_id"] = record.synset_id
            if record.pos:
                payload["pos"] = record.pos
            if record.top_tokens:
                payload["top_tokens"] = record.top_tokens

            point = PointStruct(
                id=hash(record.id) & 0xFFFFFFFFFFFFFFFF,  # Convert to positive int
                vector=vectors if vectors else {},
                payload=payload,
            )
            points.append(point)

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def add_records_batch(self, records: list[WordEmbeddingRecord], batch_size: int = 100) -> None:
        """
        Add records in batches.

        Args:
            records: List of records to add
            batch_size: Number of records per batch
        """
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            self.add_records(batch)
            if (i + batch_size) % 1000 == 0:
                print(f"Added {min(i + batch_size, len(records))}/{len(records)} records")

    def get_all_records(
        self, limit: int = 10000, offset: int = 0, filter_strategy: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve all records from the collection.

        Args:
            limit: Maximum number of records to retrieve
            offset: Offset for pagination
            filter_strategy: Filter by strategy ("word_level" or "definition")

        Returns:
            List of records with payloads and vectors
        """
        scroll_filter = None
        if filter_strategy:
            scroll_filter = Filter(
                must=[FieldCondition(key="strategy", match=MatchValue(value=filter_strategy))]
            )

        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=True,
            scroll_filter=scroll_filter,
        )

        records = []
        for point in results:
            record = {
                "id": point.id,
                "payload": point.payload,
            }
            if point.vector:
                record["dense"] = point.vector.get("dense")
            records.append(record)

        return records

    def get_sparse_embeddings(
        self, filter_strategy: str | None = None, limit: int = 100000
    ) -> list[dict[str, Any]]:
        """
        Get all sparse embeddings from the collection.

        Args:
            filter_strategy: Filter by strategy
            limit: Maximum number to retrieve

        Returns:
            List of dicts with word, sparse indices/values, and metadata
        """
        scroll_filter = None
        if filter_strategy:
            scroll_filter = Filter(
                must=[FieldCondition(key="strategy", match=MatchValue(value=filter_strategy))]
            )

        results = []
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=min(1000, limit - len(results)),
                offset=offset,
                with_payload=True,
                with_vectors=["sparse"],
                scroll_filter=scroll_filter,
            )

            for point in points:
                sparse_vec = None
                if hasattr(point, "vector") and point.vector:
                    if isinstance(point.vector, dict) and "sparse" in point.vector:
                        sv = point.vector["sparse"]
                        sparse_vec = SparseEmbedding(
                            indices=list(sv.indices)
                            if hasattr(sv, "indices")
                            else sv.get("indices", []),
                            values=list(sv.values)
                            if hasattr(sv, "values")
                            else sv.get("values", []),
                        )

                results.append(
                    {
                        "word": point.payload.get("word"),
                        "strategy": point.payload.get("strategy"),
                        "context": point.payload.get("context"),
                        "synset_id": point.payload.get("synset_id"),
                        "pos": point.payload.get("pos"),
                        "top_tokens": point.payload.get("top_tokens"),
                        "sparse": sparse_vec,
                    }
                )

            if offset is None or len(results) >= limit:
                break

        return results

    def count(self) -> int:
        """Get the number of records in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def delete_collection(self) -> None:
        """Delete the collection."""
        self.client.delete_collection(self.collection_name)
        print(f"Deleted collection: {self.collection_name}")


def test_storage():
    """Test the storage module."""
    print("=" * 60)
    print("Testing Qdrant Storage")
    print("=" * 60)

    # Create storage with test collection
    storage = QdrantStorage(
        db_path="interpretability/resources/vector_db_qdrant_test",
        collection_name="test_collection",
    )
    storage.create_collection(recreate=True)

    # Create test records
    records = [
        WordEmbeddingRecord(
            id="cat_def_1",
            word="cat",
            strategy="definition",
            context="cat: a small feline mammal",
            synset_id="oewn-02086723-n",
            pos="n",
            sparse_embedding=SparseEmbedding(indices=[100, 200, 300], values=[1.5, 2.0, 0.8]),
            dense_embedding=[0.1] * 384,
            top_tokens=["cat", "feline", "mammal"],
        ),
        WordEmbeddingRecord(
            id="cat_word_1",
            word="cat",
            strategy="word_level",
            context="The cat sat on the mat.",
            synset_id="oewn-02086723-n",
            pos="n",
            sparse_embedding=SparseEmbedding(indices=[100, 150], values=[2.0, 1.2]),
            dense_embedding=[0.2] * 384,
            top_tokens=["cat", "cats"],
        ),
    ]

    # Add records
    storage.add_records(records)
    print(f"Added {len(records)} records")
    print(f"Total count: {storage.count()}")

    # Retrieve records
    print("\nRetrieving all records:")
    all_records = storage.get_all_records()
    for r in all_records:
        print(f"  - {r['payload']['word']} ({r['payload']['strategy']})")

    # Filter by strategy
    print("\nFiltering by word_level strategy:")
    word_level = storage.get_sparse_embeddings(filter_strategy="word_level")
    for r in word_level:
        print(f"  - {r['word']}: {r['top_tokens']}")

    # Cleanup
    storage.delete_collection()
    print("\nTest complete!")


if __name__ == "__main__":
    test_storage()
