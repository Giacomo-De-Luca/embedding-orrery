"""Extraction-stage configs: encoder + gemma.

Each extraction has a user-chosen `name` that uniquely identifies it
within an experiment AND drives the cache filename in the per-experiment
`resources/extracted_activations/<experiment_name>/` cache.

If two configs share a name but differ in any other field, the cache
layer's sidecar validation raises `CacheMismatchError` — fail loud.

All current extractors emit ONE pooled vector per sample (controlled by
`token_position` for Gemma, by `pooling` for encoders). Downstream
consumers cannot re-pool. See `TokenPosition` docstring for the future
extension to token-level extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from interpret.probing.utils.enums import TokenPosition


@dataclass
class EncoderExtractionConfig:
    """Per-layer hidden-state extraction from an HF encoder model."""

    name: str  # required: drives the cache filename and probe folder name
    type: Literal["encoder"] = "encoder"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    pooling: Literal["cls", "mean", "last"] = "cls"
    batch_size: int = 64
    # `None` = auto-select first/middle/last transformer layers.
    layers: list[int] | None = None
    device: str | None = None  # auto-detect when None

    def __post_init__(self) -> None:
        if self.type != "encoder":
            raise ValueError(
                f"EncoderExtractionConfig.type must be 'encoder', got {self.type!r}",
            )
        if not self.name:
            raise ValueError("EncoderExtractionConfig.name is required.")

    def cache_filename(self) -> str:
        return self.name


@dataclass
class GemmaExtractionConfig:
    """Activation extraction from Gemma3 via the PyTorch wrapper.

    Emits a single `[hidden]` vector per sample, pooled per the
    `token_position` field. Downstream SAE encoding + probes consume that
    pre-pooled vector. To compare pooling strategies, define multiple
    Gemma extractions with distinct `name` + `token_position`.
    """

    name: str  # required
    type: Literal["gemma"] = "gemma"
    checkpoint_path: str | None = None  # auto-resolve from HF cache when None
    layers: list[int] = field(default_factory=lambda: list(range(34)))
    intermediates: list[str] = field(default_factory=lambda: ["post_mlp"])
    token_position: TokenPosition = TokenPosition.LAST
    cache_phase: Literal["prefill", "last"] = "prefill"

    # Either prompt_column (text-only) or image_column (multimodal) must be set.
    prompt_column: str | None = None
    prompt_template: str | None = None
    image_column: str | None = None
    image_dir: str | None = None

    hidden_size: int = 2560

    def __post_init__(self) -> None:
        if self.type != "gemma":
            raise ValueError(
                f"GemmaExtractionConfig.type must be 'gemma', got {self.type!r}",
            )
        if not self.name:
            raise ValueError("GemmaExtractionConfig.name is required.")
        if isinstance(self.token_position, str):
            self.token_position = TokenPosition(self.token_position)
        if self.token_position is TokenPosition.WORD_LAST:
            if self.prompt_template is None or "{word}" not in self.prompt_template:
                raise ValueError(
                    "WORD_LAST requires prompt_template containing '{word}'.",
                )
        if self.prompt_column is None and self.image_column is None:
            raise ValueError(
                "GemmaExtractionConfig: set either prompt_column or image_column.",
            )

    def cache_filename(self) -> str:
        return self.name
