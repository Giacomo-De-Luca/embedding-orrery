"""Shared ChromaDB client instance for API layer."""

from ..clients.chromadb_client import ChromaDBClient

# Shared ChromaDB client instance (lazy singleton)
_chromadb_client: ChromaDBClient | None = None


def get_chromadb_client() -> ChromaDBClient:
    """Get shared ChromaDB client instance.

    Returns:
        Singleton ChromaDBClient instance, created on first call.
    """
    global _chromadb_client
    if _chromadb_client is None:
        _chromadb_client = ChromaDBClient()
    return _chromadb_client
