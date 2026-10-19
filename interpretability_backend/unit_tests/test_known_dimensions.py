"""Known embedding dimensions per model (backend/utils/known_dimensions.py)."""

import json

from backend.utils.known_dimensions import KNOWN_DIMENSIONS_FILE, KnownDimensions


def test_shipped_file_knows_embeddinggemma():
    known = KnownDimensions(KNOWN_DIMENSIONS_FILE)
    assert known.get("google/embeddinggemma-2") == 768
    assert known.get("all-MiniLM-L6-v2") == 384


def test_missing_or_invalid_file_means_nothing_known(tmp_path):
    assert KnownDimensions(tmp_path / "absent.json").load() == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert KnownDimensions(broken).get("m") is None


def test_lookup_prefers_the_exact_name_then_ignores_case(tmp_path):
    path = tmp_path / "dims.json"
    path.write_text(json.dumps({"Org/Model": 512, "org/model": 256}))
    known = KnownDimensions(path)
    assert known.get("Org/Model") == 512
    assert known.get("org/model") == 256
    assert known.get("ORG/MODEL") in (256, 512)
    assert known.get("other") is None


def test_record_keeps_other_entries_sorted(tmp_path):
    path = tmp_path / "dims.json"
    path.write_text(json.dumps({"b-model": 3}))

    KnownDimensions(path).record("a-model", 7)

    assert path.read_text() == json.dumps({"a-model": 7, "b-model": 3}, indent=1, sort_keys=True)


def test_record_creates_the_file(tmp_path):
    path = tmp_path / "new.json"
    KnownDimensions(path).record("m", 5)
    assert KnownDimensions(path).get("m") == 5
