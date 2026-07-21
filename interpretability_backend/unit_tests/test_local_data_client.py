"""
Tests for local data client file handling.

These tests verify file type detection and data loading
without requiring actual file I/O (unit tests with mocking).
"""

import pandas as pd

# Import the functions we're testing
from backend.clients.local_data_client import (
    LocalFileInfo,
    LocalFilePreview,
    _detect_file_type,
)


class TestFileTypeDetection:
    """Test file extension detection logic."""

    def test_detect_parquet(self):
        """Should detect .parquet files."""
        assert _detect_file_type("data.parquet") == "parquet"
        assert _detect_file_type("/path/to/file.parquet") == "parquet"
        assert _detect_file_type("my.data.parquet") == "parquet"

    def test_detect_json(self):
        """Should detect .json files."""
        assert _detect_file_type("data.json") == "json"
        assert _detect_file_type("/path/to/file.json") == "json"

    def test_detect_jsonl(self):
        """Should detect .jsonl and .ndjson files."""
        assert _detect_file_type("data.jsonl") == "jsonl"
        assert _detect_file_type("data.ndjson") == "jsonl"
        assert _detect_file_type("/path/to/file.jsonl") == "jsonl"

    def test_detect_csv(self):
        """Should detect .csv files."""
        assert _detect_file_type("data.csv") == "csv"
        assert _detect_file_type("/path/to/file.csv") == "csv"

    def test_detect_tsv(self):
        """Should detect .tsv files."""
        assert _detect_file_type("data.tsv") == "tsv"
        assert _detect_file_type("/path/to/file.tsv") == "tsv"

    def test_detect_unknown(self):
        """Should return 'unknown' for unsupported extensions."""
        assert _detect_file_type("data.txt") == "unknown"
        assert _detect_file_type("data.xlsx") == "unknown"
        assert _detect_file_type("data.xml") == "unknown"
        assert _detect_file_type("noextension") == "unknown"

    def test_case_insensitive(self):
        """File extension detection should be case-insensitive."""
        assert _detect_file_type("DATA.PARQUET") == "parquet"
        assert _detect_file_type("file.JSON") == "json"
        assert _detect_file_type("file.Csv") == "csv"

    def test_empty_path(self):
        """Empty path should return unknown."""
        assert _detect_file_type("") == "unknown"

    def test_path_with_dots(self):
        """Should handle paths with multiple dots correctly."""
        assert _detect_file_type("my.data.file.csv") == "csv"
        assert _detect_file_type("version.1.0.parquet") == "parquet"


class TestLocalFileInfoDataclass:
    """Test LocalFileInfo dataclass behavior."""

    def test_create_success_info(self):
        """Should create info object for successful file read."""
        info = LocalFileInfo(
            file_path="/path/to/data.csv",
            file_type="csv",
            columns=["id", "name", "value"],
            num_rows=1000,
            file_size_bytes=50000,
        )
        assert info.file_path == "/path/to/data.csv"
        assert info.file_type == "csv"
        assert info.columns == ["id", "name", "value"]
        assert info.num_rows == 1000
        assert info.file_size_bytes == 50000
        assert info.error is None

    def test_create_error_info(self):
        """Should create info object with error message."""
        info = LocalFileInfo(
            file_path="/path/to/missing.csv",
            file_type="csv",
            columns=[],
            num_rows=0,
            file_size_bytes=0,
            error="File not found: /path/to/missing.csv",
        )
        assert info.error is not None
        assert "File not found" in info.error


class TestLocalFilePreviewDataclass:
    """Test LocalFilePreview dataclass behavior."""

    def test_create_preview(self):
        """Should create preview with sample rows."""
        preview = LocalFilePreview(
            file_path="/path/to/data.csv",
            columns=["id", "name"],
            rows=[
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ],
            total_rows=100,
        )
        assert len(preview.rows) == 2
        assert preview.columns == ["id", "name"]
        assert preview.total_rows == 100
        assert preview.error is None

    def test_create_preview_with_error(self):
        """Should create preview object with error."""
        preview = LocalFilePreview(
            file_path="/path/to/bad.csv",
            columns=[],
            rows=[],
            total_rows=0,
            error="Failed to parse CSV",
        )
        assert preview.error == "Failed to parse CSV"
        assert preview.rows == []


class TestDataFrameRowConversion:
    """Test the row conversion logic used in previews."""

    def test_convert_basic_types(self):
        """Should preserve basic Python types."""
        df = pd.DataFrame({
            "str_col": ["hello", "world"],
            "int_col": [1, 2],
            "float_col": [1.5, 2.5],
            "bool_col": [True, False],
        })

        rows = []
        for _, row in df.iterrows():
            row_dict = {}
            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    row_dict[col] = None
                elif isinstance(value, (str, int, float, bool)):
                    row_dict[col] = value
                else:
                    row_dict[col] = str(value)
            rows.append(row_dict)

        assert rows[0]["str_col"] == "hello"
        assert rows[0]["int_col"] == 1
        assert rows[0]["float_col"] == 1.5
        assert rows[0]["bool_col"] is True

    def test_handle_null_values(self):
        """Should convert NaN/None to None."""
        df = pd.DataFrame({
            "col": [1.0, None, 3.0],
        })

        rows = []
        for _, row in df.iterrows():
            row_dict = {}
            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    row_dict[col] = None
                else:
                    row_dict[col] = value
            rows.append(row_dict)

        assert rows[0]["col"] == 1.0
        assert rows[1]["col"] is None
        assert rows[2]["col"] == 3.0

    def test_truncate_long_lists(self):
        """Should truncate lists longer than 10 items."""
        long_list = list(range(20))
        df = pd.DataFrame({
            "list_col": [long_list],
        })

        for _, row in df.iterrows():
            value = row["list_col"]
            if isinstance(value, list):
                truncated = value[:10] if len(value) > 10 else value

        assert len(truncated) == 10 # type: ignore
        assert truncated == list(range(10)) # type: ignore


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_special_characters_in_path(self):
        """Should handle paths with special characters."""
        # Just testing the detection part
        assert _detect_file_type("/path/with spaces/file.csv") == "csv"
        assert _detect_file_type("/path/with-dashes/file.parquet") == "parquet"
        assert _detect_file_type("/path/with_underscores/file.json") == "json"

    def test_hidden_files(self):
        """Should handle hidden files (starting with dot)."""
        assert _detect_file_type(".hidden.csv") == "csv"
        assert _detect_file_type("/path/.config.json") == "json"

    def test_unicode_paths(self):
        """Should handle unicode in file paths."""
        assert _detect_file_type("/путь/файл.csv") == "csv"
        assert _detect_file_type("/路径/文件.json") == "json"
