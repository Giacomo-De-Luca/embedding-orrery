"""Load harmful/harmless splits and processed eval sets for the pipeline.

Mirrors `references/refusal_direction/dataset/load_dataset.py` but parameterised
on the `splits_dir` / `eval_dir` paths from `RefusalConfig`. Datasets were
relocated to `resources/refusal_direction/` (gitignored) — see `resources.md`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HARMTYPES = ("harmful", "harmless")
SPLITS = ("train", "val", "test")


def load_split(harmtype: str, split: str, splits_dir: Path) -> list[dict]:
    """Load one of the six split files (harm × split)."""
    if harmtype not in HARMTYPES:
        raise ValueError(f"harmtype must be one of {HARMTYPES}, got {harmtype}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split}")
    path = Path(splits_dir) / f"{harmtype}_{split}.json"
    with path.open("r") as f:
        return json.load(f)


def load_eval_dataset(name: str, eval_dir: Path) -> list[dict]:
    """Load a processed evaluation dataset (e.g. ``"jailbreakbench"``)."""
    path = Path(eval_dir) / f"{name}.json"
    with path.open("r") as f:
        return json.load(f)


def sample(items: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic random sample of `n` items (no replacement)."""
    rng = random.Random(seed)
    if n >= len(items):
        return list(items)
    return rng.sample(items, n)


def instructions_only(items: list[dict]) -> list[str]:
    return [item["instruction"] for item in items]
