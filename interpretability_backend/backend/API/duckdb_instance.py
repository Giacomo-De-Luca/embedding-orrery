"""Shared DuckDB client instance for API layer."""

from ..clients.duckdb_client import DuckDBClient

# Shared DuckDB client instance (lazy singleton)
_duckdb_client: DuckDBClient | None = None


def get_duckdb_client() -> DuckDBClient:
    """Get shared DuckDB client instance.

    Returns:
        Singleton DuckDBClient instance, created on first call.
    """
    global _duckdb_client
    if _duckdb_client is None:
        _duckdb_client = DuckDBClient()
    return _duckdb_client
