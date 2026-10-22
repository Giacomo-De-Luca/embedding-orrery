"""Embedding loading utilities.

Reads dense embedding vectors from ChromaDB, either streamed in batches
(``iter_embedding_batches``, e.g. for topic centroids) or aligned to a
caller-supplied list of item IDs (``load_embeddings_for_ids``, for
embedding-space topic clustering, probing and the topic-quality evaluator).
"""

import logging
import threading
from collections.abc import Iterator
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

from ..services.cancel_registry import raise_if_cancelled
from ..utils.resource_paths import CHROMA_DB_PATH

logger = logging.getLogger("orrery." + __name__)

DEFAULT_LOAD_BATCH_SIZE = 5000


def iter_embedding_batches(
    collection_name: str,
    batch_size: int = DEFAULT_LOAD_BATCH_SIZE,
    chroma_path: Path | None = None,
) -> Iterator[tuple[list[str], np.ndarray]]:
    """Yield ``(ids, vectors)`` batches covering every vector in a collection.

    ``chroma_path``: another vector store than the live one (offline runners
    on a copy).

    Raises:
        ValueError: if the collection cannot be opened.
    """
    try:
        client = chromadb.PersistentClient(
            path=str((chroma_path or CHROMA_DB_PATH).resolve()),
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


def cancellable_embedding_batches(cancel_event: threading.Event | None):
    """``iter_embedding_batches`` that raises JobCancelled between batches once
    ``cancel_event`` is set (a drop-in vector-batch source for long loads)."""

    def batches(
        collection_name: str, batch_size: int = DEFAULT_LOAD_BATCH_SIZE
    ) -> Iterator[tuple[list[str], np.ndarray]]:
        for batch in iter_embedding_batches(collection_name, batch_size):
            raise_if_cancelled(cancel_event)
            yield batch

    return batches


def load_all_embeddings(
    collection_name: str,
    cancel_event: threading.Event | None = None,
    dtype=np.float32,
) -> tuple[list[str], np.ndarray]:
    """Every ``(id, vector)`` of a collection as ``(ids, matrix)``; a cancel
    raises JobCancelled between batches. float32 by default: PCA/UMAP inputs
    need no more, and it halves memory on large collections."""
    ids: list[str] = []
    blocks: list[np.ndarray] = []
    for batch_ids, vectors in cancellable_embedding_batches(cancel_event)(collection_name):
        ids.extend(batch_ids)
        blocks.append(np.asarray(vectors, dtype=dtype))
    raise_if_cancelled(cancel_event)
    return ids, np.concatenate(blocks) if blocks else np.empty((0, 0), dtype=dtype)


def load_embeddings_for_ids(
    collection_name: str,
    ids: list[str],
    load_batch_size: int = DEFAULT_LOAD_BATCH_SIZE,
    cancel_event: threading.Event | None = None,
) -> np.ndarray | None:
    """Load embeddings from ChromaDB ordered to match ``ids``.

    Reads all ``(id, embedding)`` pairs from the collection in batches, builds an
    ``id -> vector`` lookup, then returns a ``(len(ids), dim)`` array in the exact
    order of the requested ``ids``.

    Args:
        collection_name: Name of the ChromaDB collection.
        ids: Item IDs whose embeddings to return, in the desired output order.
        load_batch_size: Number of records to read per ChromaDB ``get`` call.
        cancel_event: The calling job's cancel event; checked per batch
            (raises JobCancelled).

    Returns:
        A float64 numpy array aligned to ``ids``, or ``None`` if the collection is
        missing, empty, or any requested id has no embedding.
    """
    id_to_vector: dict[str, np.ndarray] = {}
    batches = cancellable_embedding_batches(cancel_event)
    try:
        for batch_ids, vectors in batches(collection_name, load_batch_size):
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
