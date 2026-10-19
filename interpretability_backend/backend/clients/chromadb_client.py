"""ChromaDB client — vector-only storage and similarity search.

After the DuckDB migration, ChromaDB is used exclusively for:
  - Storing dense embedding vectors (IDs + vectors, no documents/metadata)
  - Semantic similarity search (query by text or pre-computed embedding)
  - Reading raw embeddings (for projection computation, topic reduction)

All document storage, metadata, projections, and topic data are in DuckDB.
"""

import logging
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings
from chromadb.errors import InternalError

from ..embedding_functions.config import EmbeddingModelConfig, EmbeddingProvider
from ..embedding_functions.create_embedding_function import create_embedding_function, get_device
from ..utils.chroma_segments import ChromaSegmentJanitor
from ..utils.resource_paths import CHROMA_DB_PATH

logger = logging.getLogger("orrery")

# Ids per existence lookup when pruning a search scope (SQLite variable limit).
_ID_LOOKUP_CHUNK = 10_000

# Gemini task type mapping: document task type -> query task type
GEMINI_QUERY_TASK_MAP = {
    "RETRIEVAL_DOCUMENT": "RETRIEVAL_QUERY",
}


def _map_gemini_task_type_for_query(task_type: str | None) -> str | None:
    if task_type is None:
        return None
    return GEMINI_QUERY_TASK_MAP.get(task_type, task_type)


class ChromaDBClient:
    """Vector-only wrapper for ChromaDB operations."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = CHROMA_DB_PATH
        else:
            db_path = Path(db_path)

        self.db_path = db_path.resolve()
        self.client = chromadb.PersistentClient(
            path=str(self.db_path), settings=Settings(anonymized_telemetry=False)
        )

    def get_collection(
        self,
        name: str,
        load_embedding_function: bool = False,
        for_query: bool = False,
        query_prompt: str | None = None,
    ):
        """Get a collection by name.

        Args:
            name: Collection name
            load_embedding_function: If True, loads the embedding function for query operations.
                                     If False (default), returns collection without EF for read-only ops.
            for_query: If True, configures EF for query embedding (QWEN adds instruction prefix)
            query_prompt: Override prompt for query embedding

        Returns:
            ChromaDB collection (with or without embedding function)
        """
        try:
            if not load_embedding_function:
                return self.client.get_collection(name=name)

            # Load embedding function for semantic search with text
            collection = self.client.get_collection(name=name)
            try:
                ef = self._embedding_function_for(
                    collection.metadata or {}, for_query=for_query, query_prompt=query_prompt
                )
            except Exception as e:
                logger.warning("Could not load embedding function for '%s': %s", name, e)
                return collection
            if ef is None:
                return collection
            return self.client.get_collection(name=name, embedding_function=ef)
        except Exception as e:
            raise ValueError(f"Collection '{name}' not found: {e}") from e

    def delete_collection(self, name: str) -> list[Path]:
        """Delete a collection together with its HNSW index directory.

        chromadb's own ``delete_collection`` leaves the directory on disk (see
        ``utils/chroma_segments.py``). Raises chromadb's ``NotFoundError`` when
        the collection does not exist; returns the directories removed.
        """
        return ChromaSegmentJanitor.delete_collection(self.client, name)

    @staticmethod
    def _embedding_function_for(
        metadata: dict[str, Any], for_query: bool, query_prompt: str | None
    ):
        """Build the embedding function recorded in a collection's metadata.

        Returns None when the metadata names no provider/model. Raises if the
        recorded provider is unknown or the function cannot be created.
        """
        provider_str = metadata.get("embedding_provider")
        model_name = metadata.get("embedding_model")
        if not (provider_str and model_name):
            return None

        provider = EmbeddingProvider(provider_str)
        task_type = metadata.get("embedding_task_type")
        if for_query and provider == EmbeddingProvider.GEMINI:
            task_type = _map_gemini_task_type_for_query(task_type)

        config = EmbeddingModelConfig(
            provider=provider,
            model_name=model_name,
            task=metadata.get("embedding_task"),
            task_type=task_type,
            prompt=query_prompt or metadata.get("embedding_prompt"),
        )
        ef, _ = create_embedding_function(
            config,
            get_device(),
            known_dimension=metadata.get("embedding_dim"),
            is_query=for_query,
        )
        return ef

    def embed_query(
        self, collection_name: str, text: str, query_prompt: str | None = None
    ) -> np.ndarray:
        """Embed a text query with the collection's own (query-mode) model.

        Unlike ``get_collection(load_embedding_function=True)`` this never
        falls back to ChromaDB's default embedder: a wrong-model query vector
        would silently produce meaningless similarities.

        Raises:
            ValueError: the collection is missing or records no embedding model.
        """
        collection = self.get_collection(collection_name)
        ef = self._embedding_function_for(
            collection.metadata or {}, for_query=True, query_prompt=query_prompt
        )
        if ef is None:
            raise ValueError(
                f"Collection '{collection_name}' records no embedding model; "
                "cannot embed a text query"
            )
        return np.asarray(ef([text])[0], dtype=np.float64)

    def semantic_search(
        self,
        collection_name: str,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        distance_metric: str = "cosine",
        query_prompt: str | None = None,
        ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Perform semantic similarity search.

        ``ids`` restricts the nearest-neighbour search to that subset (an
        empty list yields no hits); ``n_results`` is capped at its size, and
        ids the collection holds no vector for are ignored.

        Returns IDs, distances, and similarities. Documents and metadata
        should be enriched from DuckDB by the caller.
        """
        if query_texts is None and query_embeddings is None:
            raise ValueError("Either query_texts or query_embeddings must be provided")
        if query_texts is not None and query_embeddings is not None:
            raise ValueError("Provide either query_texts or query_embeddings, not both")
        if ids is not None and not ids:
            return self._no_hits()

        needs_ef = query_texts is not None
        collection = self.get_collection(
            collection_name,
            load_embedding_function=needs_ef,
            for_query=needs_ef,
            query_prompt=query_prompt,
        )

        def run_query(scope: list[str] | None) -> dict[str, Any]:
            return collection.query(
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                n_results=n_results if scope is None else min(n_results, len(scope)),
                where=where,
                ids=scope,
                include=["distances", "embeddings"],
            )

        try:
            results = run_query(ids)
        except InternalError:
            # chromadb rejects a scope containing ids it holds no vector for
            # (dataset items skipped at embed time, partial collections).
            # Retry once with only the ids that exist.
            if ids is None:
                raise
            present = self._existing_ids(collection, ids)
            if len(present) == len(ids):
                raise
            if not present:
                return self._no_hits()
            results = run_query(present)

        # Convert distances to similarities
        distances = results.get("distances", [[]])[0]
        if distance_metric == "cosine":
            similarities = [1 - d for d in distances]
        elif distance_metric == "l2":
            similarities = [1 / (1 + d) for d in distances]
        elif distance_metric == "ip":
            similarities = [-d for d in distances]
        else:
            similarities = [1 - d for d in distances]

        results["similarities"] = [similarities]
        return results

    @staticmethod
    def _no_hits() -> dict[str, Any]:
        return {"ids": [[]], "distances": [[]], "similarities": [[]], "embeddings": None}

    @staticmethod
    def _existing_ids(collection, ids: list[str]) -> list[str]:
        """The subset of ``ids`` present in ``collection``, in input order.

        Looked up in chunks: a single ``get`` over ~100k ids exceeds SQLite's
        bound-variable limit.
        """
        present: set[str] = set()
        for start in range(0, len(ids), _ID_LOOKUP_CHUNK):
            batch = collection.get(ids=ids[start : start + _ID_LOOKUP_CHUNK], include=[])
            present.update(batch["ids"])
        return [item_id for item_id in ids if item_id in present]
