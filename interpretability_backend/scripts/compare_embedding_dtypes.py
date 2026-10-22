"""Compare one model's embeddings in float32 and bfloat16 on this machine.

The backend now keeps bf16 checkpoints (EmbeddingGemma 2) in bf16 on Apple
GPUs instead of upcasting them to float32, which halves their memory. This
script measures what that costs in embedding agreement: it embeds the same
texts with both dtypes, one model load at a time, and prints the cosine
between each text's two vectors, how many top-k neighbours both agree on,
and the Metal memory each load held.

Config: ``compare_embedding_dtypes_config.toml`` next to this file (override
with ``ORRERY_DTYPE_COMPARISON_CONFIG``). It loads the model twice, so run it
while the backend is idle:
    uv run python -m interpretability_backend.scripts.compare_embedding_dtypes
"""

import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from interpretability_backend.backend.utils.model_load_options import (
    SentenceTransformerLoadOptions,
)
from interpretability_backend.backend.utils.topic_centroids import unit_rows
from interpretability_backend.evaluation.utils.runner_common import (
    load_config,
    resolve_config_path,
)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("compare_embedding_dtypes_config.toml")
DTYPES = ("float32", "bfloat16")


class FixedDtypePolicy:
    """A dtype policy that always answers ``dtype`` (to force each load's dtype)."""

    def __init__(self, dtype: str):
        self.dtype = dtype

    def for_device(self, device: str, model_name: str | None = None) -> str:
        return self.dtype


@dataclass(frozen=True)
class EmbeddingAgreement:
    """How closely a candidate embedding matrix reproduces a reference one."""

    mean_cosine: float
    min_cosine: float
    # Mean share of each text's top-k neighbours (cosine) that both matrices agree on.
    neighbour_overlap: float

    @classmethod
    def between(cls, reference: np.ndarray, candidate: np.ndarray, k: int = 10) -> Self:
        a = unit_rows(reference)
        b = unit_rows(candidate)
        cosines = (a * b).sum(axis=1)
        k = min(k, len(a) - 1)
        overlaps = [
            len(set(na) & set(nb)) / k
            for na, nb in zip(cls._neighbours(a, k), cls._neighbours(b, k), strict=True)
        ]
        return cls(float(cosines.mean()), float(cosines.min()), float(np.mean(overlaps)))

    @staticmethod
    def _neighbours(unit: np.ndarray, k: int) -> np.ndarray:
        similarity = unit @ unit.T
        np.fill_diagonal(similarity, -np.inf)
        return np.argsort(-similarity, axis=1)[:, :k]


def _texts(config: dict) -> list[str]:
    texts_file = config.get("texts_file")
    if texts_file:
        lines = Path(texts_file).expanduser().read_text().splitlines()
        return [line for line in lines if line.strip()]
    return list(config["texts"])


def _embed(config: dict, dtype: str, texts: list[str]) -> tuple[np.ndarray, int | None]:
    """Embed ``texts`` with a fresh load in ``dtype``; returns vectors and Metal bytes held."""
    model_name, device = config["model"], config.get("device", "mps")
    options = SentenceTransformerLoadOptions.for_modalities(
        model_name, device, ["text"], dtype_policy=FixedDtypePolicy(dtype)
    )
    model = SentenceTransformer(model_name, device=device, **options.constructor_kwargs())
    encode_kwargs = {"batch_size": config.get("batch_size", 16), "convert_to_numpy": True}
    if config.get("prompt"):
        encode_kwargs["prompt_name"] = config["prompt"]
    if config.get("max_length"):
        encode_kwargs["processing_kwargs"] = {"text": {"max_length": config["max_length"]}}
    vectors = np.asarray(model.encode(texts, **encode_kwargs), dtype=np.float32)
    held = torch.mps.driver_allocated_memory() if device == "mps" else None
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return vectors, held


def main() -> int:
    config = load_config(resolve_config_path("ORRERY_DTYPE_COMPARISON_CONFIG", DEFAULT_CONFIG_PATH))
    texts = _texts(config)
    if len(texts) < 2:
        print("ERROR: need at least two texts to compare", file=sys.stderr)
        return 1

    vectors = {}
    for dtype in DTYPES:
        vectors[dtype], held = _embed(config, dtype, texts)
        memory = f"{held / 1024**3:.2f} GiB Metal" if held is not None else "n/a"
        print(f"{dtype:>9}: {len(texts)} texts embedded, {memory} held after encode")

    agreement = EmbeddingAgreement.between(
        vectors["float32"], vectors["bfloat16"], k=config.get("neighbours", 10)
    )
    print(f"\nmean cosine fp32 vs bf16:   {agreement.mean_cosine:.5f}")
    print(f"lowest cosine:              {agreement.min_cosine:.5f}")
    print(f"top-k neighbour agreement:  {agreement.neighbour_overlap:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
