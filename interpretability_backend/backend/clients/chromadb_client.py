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
from chromadb.config import Settings

from ..embedding_functions.config import EmbeddingModelConfig, EmbeddingProvider
from ..embedding_functions.create_embedding_function import create_embedding_function, get_device

logger = logging.getLogger("star_map")

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
            interpretability_root = Path(__file__).parent.parent.parent
            db_path = interpretability_root / "resources" / "vector_db"
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
            metadata = collection.metadata or {}
            provider_str = metadata.get("embedding_provider")
            model_name = metadata.get("embedding_model")

            if provider_str and model_name:
                try:
                    provider = EmbeddingProvider(provider_str)
                    task = metadata.get("embedding_task")
                    task_type = metadata.get("embedding_task_type")
                    prompt = query_prompt or metadata.get("embedding_prompt")

                    if for_query and provider == EmbeddingProvider.GEMINI:
                        task_type = _map_gemini_task_type_for_query(task_type)

                    config = EmbeddingModelConfig(
                        provider=provider,
                        model_name=model_name,
                        task=task,
                        task_type=task_type,
                        prompt=prompt,
                    )

                    embedding_dim = metadata.get("embedding_dim")
                    device = get_device()
                    ef, _ = create_embedding_function(
                        config, device, known_dimension=embedding_dim, is_query=for_query
                    )

                    return self.client.get_collection(name=name, embedding_function=ef)

                except Exception as e:
                    logger.warning("Could not load embedding function for '%s': %s", name, e)
                    return collection

            return collection
        except Exception as e:
            raise ValueError(f"Collection '{name}' not found: {e}")

    def semantic_search(
        self,
        collection_name: str,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        distance_metric: str = "cosine",
        query_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Perform semantic similarity search.

        Returns IDs, distances, and similarities. Documents and metadata
        should be enriched from DuckDB by the caller.
        """
        needs_ef = query_texts is not None
        collection = self.get_collection(
            collection_name,
            load_embedding_function=needs_ef,
            for_query=needs_ef,
            query_prompt=query_prompt,
        )

        if query_texts is None and query_embeddings is None:
            raise ValueError("Either query_texts or query_embeddings must be provided")
        if query_texts is not None and query_embeddings is not None:
            raise ValueError("Provide either query_texts or query_embeddings, not both")

        results = collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            include=["distances", "embeddings"],
        )

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
