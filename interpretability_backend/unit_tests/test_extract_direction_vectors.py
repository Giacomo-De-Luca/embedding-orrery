"""Tests for the extract_direction_vectors preprocessing script.

The script slices the multi-candidate refusal tensor (5, 34, 2560) down to
the single 1-D vector chosen via (source_position, source_layer), writes
matching metadata JSON, and is idempotent on already-processed files.
"""

import json
from pathlib import Path

import pytest
import torch

from scripts.extract_direction_vectors import (
    REFUSAL_METADATA,
    process_poetry,
    process_refusal,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def directions_dir(tmp_path: Path) -> Path:
    """Empty directions/ folder under a tmp_path."""
    d = tmp_path / "directions"
    d.mkdir()
    return d


@pytest.fixture
def candidate_refusal_pt(directions_dir: Path) -> Path:
    """A fake (5, 34, 2560) refusal candidate tensor, with a distinct value
    at the [-3, 14] slice so we can verify the slice survives intact."""
    t = torch.zeros((5, 34, 2560), dtype=torch.float64)
    # position -3 == index 2 (5 positions); layer 14
    t[2, 14] = torch.arange(2560, dtype=torch.float64) + 100.0
    path = directions_dir / "refusal_direction.pt"
    torch.save(t, path)
    return path


@pytest.fixture
def already_1d_refusal_pt(directions_dir: Path) -> Path:
    """A pre-sliced 1-D float32 refusal tensor (idempotency target)."""
    t = torch.arange(2560, dtype=torch.float32) + 7.0
    path = directions_dir / "refusal_direction.pt"
    torch.save(t, path)
    return path


@pytest.fixture
def poetry_pt(directions_dir: Path) -> Path:
    """A 1-D poetry tensor (already correct shape)."""
    t = torch.arange(2560, dtype=torch.float32)
    path = directions_dir / "poetry_direction.pt"
    torch.save(t, path)
    return path


@pytest.fixture
def poetry_metadata_json(directions_dir: Path) -> Path:
    """Existing poetry metadata, without model_id (will be added by script)."""
    path = directions_dir / "poetry_direction_metadata.json"
    path.write_text(
        json.dumps(
            {
                "name": "poetry_prose",
                "intermediate": "post_attn",
                "position": -2,
                "layer": 11,
                "coefficient": 1.0,
            }
        )
    )
    return path


# ---------------------------------------------------------------------------
# Refusal slicing
# ---------------------------------------------------------------------------


def test_refusal_slices_candidate_tensor(directions_dir: Path, candidate_refusal_pt: Path):
    """A (5, 34, 2560) tensor is sliced to the configured (position, layer)."""
    process_refusal(directions_dir)

    out = torch.load(candidate_refusal_pt, map_location="cpu", weights_only=False)
    assert out.shape == (2560,)
    assert out.dtype == torch.float32
    # The sentinel values we wrote at [2, 14] should survive
    expected = torch.arange(2560, dtype=torch.float32) + 100.0
    assert torch.allclose(out, expected)


def test_refusal_writes_metadata(directions_dir: Path, candidate_refusal_pt: Path):
    """Metadata JSON is written with the schema mirroring poetry's."""
    process_refusal(directions_dir)
    meta_path = directions_dir / "refusal_direction_metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta == REFUSAL_METADATA


def test_refusal_is_idempotent(directions_dir: Path, already_1d_refusal_pt: Path):
    """Re-running on an already-1-D tensor leaves it untouched."""
    before = torch.load(already_1d_refusal_pt, map_location="cpu", weights_only=False).clone()
    process_refusal(directions_dir)
    after = torch.load(already_1d_refusal_pt, map_location="cpu", weights_only=False)
    assert torch.equal(before, after)


def test_refusal_raises_on_unexpected_shape(directions_dir: Path):
    """A tensor with neither expected shape is an explicit error."""
    bad = torch.zeros((7, 17, 999), dtype=torch.float32)
    path = directions_dir / "refusal_direction.pt"
    torch.save(bad, path)
    with pytest.raises(ValueError, match="unexpected refusal shape"):
        process_refusal(directions_dir)


# ---------------------------------------------------------------------------
# Poetry metadata
# ---------------------------------------------------------------------------


def test_poetry_metadata_gets_model_id(
    directions_dir: Path, poetry_pt: Path, poetry_metadata_json: Path
):
    """The script adds model_id to poetry metadata if missing."""
    process_poetry(directions_dir)
    meta = json.loads(poetry_metadata_json.read_text())
    assert meta["model_id"] == "gemma-3-4b-it"
    # Other fields preserved
    assert meta["layer"] == 11
    assert meta["intermediate"] == "post_attn"


def test_poetry_metadata_idempotent(
    directions_dir: Path, poetry_pt: Path, poetry_metadata_json: Path
):
    """Running twice does not flip or duplicate fields."""
    process_poetry(directions_dir)
    first = poetry_metadata_json.read_text()
    process_poetry(directions_dir)
    second = poetry_metadata_json.read_text()
    assert first == second


def test_poetry_does_not_modify_tensor(directions_dir: Path, poetry_pt: Path, poetry_metadata_json: Path):
    """Poetry tensor is already 1-D — script must not touch it."""
    before = torch.load(poetry_pt, map_location="cpu", weights_only=False).clone()
    process_poetry(directions_dir)
    after = torch.load(poetry_pt, map_location="cpu", weights_only=False)
    assert torch.equal(before, after)
