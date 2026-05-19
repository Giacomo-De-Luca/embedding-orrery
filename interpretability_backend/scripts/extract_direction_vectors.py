"""Pre-process activation-additive steering directions into runtime-ready 1-D vectors.

The refusal-direction pipeline saves a candidate tensor of shape
``(n_positions, n_layers, d_model)``. At runtime we only need one 1-D
slice — the one chosen by ``(SOURCE_POSITION, SOURCE_LAYER)``. This
script bakes the slice in (overwriting the file in place) and writes
matching metadata JSON in the same schema as ``poetry_direction_metadata.json``.

Poetry is already 1-D — this script only normalises its metadata
(adds ``model_id`` if missing) and leaves the tensor untouched.

Run once after extracting a new candidate tensor:

    uv run python -m interpretability_backend.scripts.extract_direction_vectors
"""

import json
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DIRECTIONS_DIR = Path(__file__).resolve().parents[1] / "resources" / "directions"

# Refusal: extracted at pre_attn / position=-3 / layer=14 for Gemma-3-4b-it.
# The candidate tensor shape is (5, 34, 2560); we want the (-3, 14) slice.
REFUSAL_SOURCE_POSITION = -3
REFUSAL_SOURCE_LAYER = 14
REFUSAL_METADATA = {
    "name": "refusal",
    "intermediate": "pre_attn",
    "source_position": REFUSAL_SOURCE_POSITION,
    "layer": REFUSAL_SOURCE_LAYER,
    "coefficient": -1.0,
    "model_id": "gemma-3-4b-it",
}

POETRY_MODEL_ID = "gemma-3-4b-it"


# ---------------------------------------------------------------------------
# Refusal
# ---------------------------------------------------------------------------


def process_refusal(directions_dir: Path) -> None:
    """Slice the refusal candidate tensor to a 1-D vector and write metadata.

    Idempotent: if the tensor is already 1-D, only metadata is refreshed.
    """
    tensor_path = directions_dir / "refusal_direction.pt"
    meta_path = directions_dir / "refusal_direction_metadata.json"

    if not tensor_path.exists():
        raise FileNotFoundError(f"missing {tensor_path}")

    t = torch.load(tensor_path, map_location="cpu", weights_only=False)
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"refusal_direction.pt is not a tensor: {type(t).__name__}")

    if t.ndim == 1:
        print(f"refusal: already 1-D ({tuple(t.shape)}), skipping slice")
    elif t.ndim == 3 and t.shape[-1] == 2560:
        sliced = t[REFUSAL_SOURCE_POSITION, REFUSAL_SOURCE_LAYER].to(torch.float32).contiguous()
        torch.save(sliced, tensor_path)
        print(
            f"refusal: sliced {tuple(t.shape)} → {tuple(sliced.shape)} "
            f"at (pos={REFUSAL_SOURCE_POSITION}, layer={REFUSAL_SOURCE_LAYER})"
        )
    else:
        raise ValueError(f"unexpected refusal shape: {tuple(t.shape)}")

    meta_path.write_text(json.dumps(REFUSAL_METADATA, indent=2) + "\n")
    print(f"refusal: wrote metadata to {meta_path.name}")


# ---------------------------------------------------------------------------
# Poetry
# ---------------------------------------------------------------------------


def process_poetry(directions_dir: Path) -> None:
    """Normalise poetry metadata (add model_id if missing). Tensor untouched."""
    meta_path = directions_dir / "poetry_direction_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing {meta_path}")

    meta = json.loads(meta_path.read_text())
    if meta.get("model_id") == POETRY_MODEL_ID:
        print("poetry: metadata already has correct model_id, skipping")
        return

    meta["model_id"] = POETRY_MODEL_ID
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"poetry: added model_id to {meta_path.name}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    process_refusal(DIRECTIONS_DIR)
    process_poetry(DIRECTIONS_DIR)


if __name__ == "__main__":
    main()
