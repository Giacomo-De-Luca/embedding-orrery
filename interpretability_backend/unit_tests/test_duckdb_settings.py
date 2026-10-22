"""DuckDB connection settings (``config/duckdb.json``): the memory cap.

Without a cap DuckDB's buffer pool may grow to 80% of RAM and keep cached
blocks for the life of the backend. A percentage is resolved against the
machine's memory, since DuckDB itself only accepts absolute sizes.
"""

import pytest

from backend.clients.duckdb_client import DuckDBClient
from backend.utils.duckdb_settings import DuckDBSettings, MachineMemory

GIB = 1024**3


class TestMachineMemory:
    def test_container_limit_caps_physical_memory(self, tmp_path):
        v2 = tmp_path / "memory.max"
        v2.write_text(f"{4 * GIB}\n")
        assert MachineMemory.available_bytes(64 * GIB, cgroup_files=[v2]) == 4 * GIB

    def test_unlimited_markers_are_ignored(self, tmp_path):
        v2 = tmp_path / "memory.max"
        v2.write_text("max\n")
        v1 = tmp_path / "memory.limit_in_bytes"
        v1.write_text("9223372036854771712\n")  # cgroup v1 "no limit"
        assert MachineMemory.available_bytes(16 * GIB, cgroup_files=[v2, v1]) == 16 * GIB

    def test_missing_cgroup_files_mean_physical_memory(self, tmp_path):
        assert MachineMemory.available_bytes(8 * GIB, cgroup_files=[tmp_path / "x"]) == 8 * GIB


class TestSettings:
    def test_defaults_cap_memory_at_a_quarter_of_ram(self):
        config = DuckDBSettings().connect_config(memory_bytes=16 * GIB)
        assert config == {"memory_limit": "4096MiB"}

    def test_tiny_percentages_resolve_to_a_floor(self):
        config = DuckDBSettings(memory_limit="0.001%").connect_config(memory_bytes=GIB)
        assert config == {"memory_limit": "64MiB"}

    @pytest.mark.parametrize("bad", ["0GB", "0MiB", "0.0 GiB"])
    def test_zero_sizes_are_rejected(self, bad):
        with pytest.raises(ValueError, match="memory_limit"):
            DuckDBSettings(memory_limit=bad)

    def test_absolute_sizes_pass_through(self):
        settings = DuckDBSettings(memory_limit="2GB", threads=4)
        assert settings.connect_config() == {"memory_limit": "2GB", "threads": 4}

    def test_no_limit_means_duckdb_default(self):
        assert DuckDBSettings(memory_limit=None).connect_config() == {}

    @pytest.mark.parametrize("bad", ["lots", "0%", "101%", "-1GB", "12", 5])
    def test_invalid_memory_limits_are_rejected(self, bad):
        with pytest.raises(ValueError, match="memory_limit"):
            DuckDBSettings(memory_limit=bad)

    @pytest.mark.parametrize("bad", [0, -2, "4", True])
    def test_invalid_thread_counts_are_rejected(self, bad):
        with pytest.raises(ValueError, match="threads"):
            DuckDBSettings(threads=bad)

    def test_config_file_rejects_unknown_keys(self, tmp_path):
        path = tmp_path / "duckdb.json"
        path.write_text('{"memory_limt": "2GB"}')
        with pytest.raises(ValueError, match="Unknown"):
            DuckDBSettings.from_file(path)

    def test_shipped_config_is_valid(self):
        from backend.utils.duckdb_settings import SETTINGS

        assert "memory_limit" in SETTINGS.connect_config()


def test_client_connection_applies_the_cap(tmp_path):
    client = DuckDBClient(
        db_path=str(tmp_path / "test.duckdb"), settings=DuckDBSettings(memory_limit="1GiB")
    )
    try:
        limit = client._conn.execute("SELECT current_setting('memory_limit')").fetchone()[0]
    finally:
        client._conn.close()
    assert limit == "1.0 GiB"
