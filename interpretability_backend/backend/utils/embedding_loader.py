"""Embedding loading utilities.

Reads dense embedding vectors from ChromaDB, either streamed in batches
(``iter_embedding_batches``, e.g. for topic centroids) or aligned to a
caller-supplied list of item IDs (``load_embeddings_for_ids``, for
embedding-space topic clustering, probing and the topic-quality evaluator).
"""

import logging
from collections.abc import Iterator

import chromadb
import numpy as np
from chromadb.config import Settings

from ..utils.resource_paths import CHROMA_DB_PATH

logger = logging.getLogger("orrery." + __name__)

DEFAULT_LOAD_BATCH_SIZE = 5000


def iter_embedding_batches(
    collection_name: str, batch_size: int = DEFAULT_LOAD_BATCH_SIZE
) -> Iterator[tuple[list[str], np.ndarray]]:
    """Yield ``(ids, vectors)`` batches covering every vector in a collection.

    Raises:
        ValueError: if the collection cannot be opened.
    """
    try:
        client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH.resolve()),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(name=collection_name, embedding_function=None)
    except Exception as e:
        raise ValueError(f"Could not open ChromaDB collection {collection_name!r}: {e}") from e

    count = collection.count()
    for offset in range(0, count, batch_size):
        limit = min(batch_size, count - offset)
        batch = collection.get(include=["embeddings"], limit=limit, offset=offset)
        yield list(batch["ids"]), np.asarray(batch["embeddings"], dtype=np.float64)


def load_embeddings_for_ids(
    collection_name: str, ids: list[str], load_batch_size: int = DEFAULT_LOAD_BATCH_SIZE
) -> np.ndarray | None:
    """Load embeddings from ChromaDB ordered to match ``ids``.

    Reads all ``(id, embedding)`` pairs from the collection in batches, builds an
    ``id -> vector`` lookup, then returns a ``(len(ids), dim)`` array in the exact
    order of the requested ``ids``.

    Args:
        collection_name: Name of the ChromaDB collection.
        ids: Item IDs whose embeddings to return, in the desired output order.
        load_batch_size: Number of records to read per ChromaDB ``get`` call.

    Returns:
        A float64 numpy array aligned to ``ids``, or ``None`` if the collection is
        missing, empty, or any requested id has no embedding.
    """
    id_to_vector: dict[str, np.ndarray] = {}
    try:
        for batch_ids, vectors in iter_embedding_batches(collection_name, load_batch_size):
            for item_id, vector in zip(batch_ids, vectors, strict=True):
                id_to_vector[item_id] = vector
    except ValueError as e:
        logger.error("%s", e)
        return None

    if not id_to_vector:
        logger.warning("Collection %r has no vectors", collection_name)
        return None

    missing = [i for i in ids if i not in id_to_vector]
    if missing:
        logger.error(
            "%d/%d requested ids have no embedding in %r (e.g. %s)",
            len(missing),
            len(ids),
            collection_name,
            missing[:3],
        )
        return None

    return np.array([id_to_vector[i] for i in ids], dtype=np.float64)
