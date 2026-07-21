"""Unit tests for first-run seed bootstrap."""

import pytest

from backend.utils import seed_bootstrap


@pytest.fixture
def seed_env(tmp_path, monkeypatch):
    """Wire the bootstrap module's paths into a temp layout.

    Returns the live + seed paths so tests can populate/inspect them.
    """
    live_duckdb = tmp_path / "resources" / "main.duckdb"
    live_vector_db = tmp_path / "resources" / "vector_db"
    seed_dir = tmp_path / "resources" / "seed"
    seed_duckdb = seed_dir / "main.duckdb"
    seed_vector_db = seed_dir / "vector_db"

    monkeypatch.setattr(seed_bootstrap, "DUCKDB_PATH", live_duckdb)
    monkeypatch.setattr(seed_bootstrap, "CHROMA_DB_PATH", live_vector_db)
    monkeypatch.setattr(seed_bootstrap, "SEED_DIR", seed_dir)
    monkeypatch.setattr(seed_bootstrap, "SEED_DUCKDB_PATH", seed_duckdb)
    monkeypatch.setattr(seed_bootstrap, "SEED_VECTOR_DB", seed_vector_db)

    return {
        "live_duckdb": live_duckdb,
        "live_vector_db": live_vector_db,
        "seed_duckdb": seed_duckdb,
        "seed_vector_db": seed_vector_db,
    }


def _write_seed(env, duckdb_bytes=b"SEED_DB", vector_file=("chroma.sqlite3", b"SEED_VEC")):
    env["seed_duckdb"].parent.mkdir(parents=True, exist_ok=True)
    env["seed_duckdb"].write_bytes(duckdb_bytes)
    env["seed_vector_db"].mkdir(parents=True, exist_ok=True)
    (env["seed_vector_db"] / vector_file[0]).write_bytes(vector_file[1])


def test_copies_seed_when_live_db_absent(seed_env):
    _write_seed(seed_env)

    result = seed_bootstrap.ensure_seed_loaded()

    assert result is True
    assert seed_env["live_duckdb"].read_bytes() == b"SEED_DB"
    assert (seed_env["live_vector_db"] / "chroma.sqlite3").read_bytes() == b"SEED_VEC"


def test_noop_when_live_db_exists(seed_env):
    """Must never clobber an existing (real) database."""
    _write_seed(seed_env)
    seed_env["live_duckdb"].parent.mkdir(parents=True, exist_ok=True)
    seed_env["live_duckdb"].write_bytes(b"REAL_USER_DATA")

    result = seed_bootstrap.ensure_seed_loaded()

    assert result is False
    # Existing DB is untouched and no vector_db was created.
    assert seed_env["live_duckdb"].read_bytes() == b"REAL_USER_DATA"
    assert not seed_env["live_vector_db"].exists()


def test_noop_when_no_seed_present(seed_env):
    """Fresh dev with no committed seed: stay empty, don't error."""
    result = seed_bootstrap.ensure_seed_loaded()

    assert result is False
    assert not seed_env["live_duckdb"].exists()


def test_handles_missing_seed_vector_db(seed_env):
    """Seed DuckDB present but no vector_db dir: copy DB only, no crash."""
    seed_env["seed_duckdb"].parent.mkdir(parents=True, exist_ok=True)
    seed_env["seed_duckdb"].write_bytes(b"SEED_DB")

    result = seed_bootstrap.ensure_seed_loaded()

    assert result is True
    assert seed_env["live_duckdb"].read_bytes() == b"SEED_DB"
    assert not seed_env["live_vector_db"].exists()
