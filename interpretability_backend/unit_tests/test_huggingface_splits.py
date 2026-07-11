"""
Tests for multi-split assembly in the HuggingFace embedding pipeline.

`_load_rows_for_splits` loads each requested split via an injected loader,
tags every row with its split under the reserved key, and concatenates the
result so a single embedding pass can store one collection containing all
splits. Pure / no network — the loader is faked.
"""

from backend.embedding_functions.embed_huggingface import (
    _SPLIT_KEY,
    _load_rows_for_splits,
)


def _fake_loader(per_split):
    """Build a load_fn(split) -> (rows, total) from a {split: rows} dict."""

    def load_fn(split):
        rows = [dict(r) for r in per_split[split]]
        return rows, len(rows)

    return load_fn


class TestLoadRowsForSplits:
    def test_tags_each_row_with_its_split(self):
        per_split = {
            "train": [{"text": "a"}, {"text": "b"}],
            "test": [{"text": "c"}],
        }
        rows, total = _load_rows_for_splits(["train", "test"], _fake_loader(per_split))

        assert total == 3
        assert [r[_SPLIT_KEY] for r in rows] == ["train", "train", "test"]
        assert [r["text"] for r in rows] == ["a", "b", "c"]

    def test_single_split_behaves_like_before(self):
        per_split = {"train": [{"text": "a"}]}
        rows, total = _load_rows_for_splits(["train"], _fake_loader(per_split))

        assert total == 1
        assert rows[0][_SPLIT_KEY] == "train"

    def test_preserves_split_order(self):
        per_split = {"validation": [{"id": 1}], "train": [{"id": 2}]}
        rows, _ = _load_rows_for_splits(
            ["validation", "train"], _fake_loader(per_split)
        )
        assert [r[_SPLIT_KEY] for r in rows] == ["validation", "train"]
