"""
Tests for IDDeduplicator (collision-only suffixing).

The deduplicator preserves the original id on first occurrence and only appends
`_1`, `_2`, ... to subsequent collisions, bumping the counter until a genuinely
free id is found (so a generated suffix can never silently overwrite a real id
that already contains that suffix).
"""

from backend.utils.id_utils import IDDeduplicator


class TestIDDeduplicator:
    def test_unique_ids_unchanged(self):
        """Non-colliding ids are returned verbatim (clean ids preserved)."""
        dedup = IDDeduplicator()
        assert dedup.get_unique_id("a") == "a"
        assert dedup.get_unique_id("b") == "b"
        assert dedup.get_unique_id("c") == "c"

    def test_collisions_get_suffix(self):
        """Repeated base ids get _1, _2 on the 2nd+ occurrence."""
        dedup = IDDeduplicator()
        assert dedup.get_unique_id("cat") == "cat"
        assert dedup.get_unique_id("cat") == "cat_1"
        assert dedup.get_unique_id("cat") == "cat_2"

    def test_generated_suffix_never_overwrites_real_id(self):
        """If a real id equals a would-be generated suffix, no duplicates emitted."""
        dedup = IDDeduplicator()
        # Source order: "5", "5_1" (real), "5" (dup) -> must all be distinct.
        assert dedup.get_unique_id("5") == "5"
        assert dedup.get_unique_id("5_1") == "5_1"  # real id, kept clean
        # The duplicate "5" cannot reuse "5_1"; bumps to "5_2".
        assert dedup.get_unique_id("5") == "5_2"

    def test_all_ids_remain_unique(self):
        """Exhaustive: every emitted id across a messy stream is unique."""
        dedup = IDDeduplicator()
        stream = ["5", "5", "5_1", "5", "x", "x", "5_1"]
        out = [dedup.get_unique_id(s) for s in stream]
        assert len(out) == len(set(out)), f"duplicate emitted: {out}"
