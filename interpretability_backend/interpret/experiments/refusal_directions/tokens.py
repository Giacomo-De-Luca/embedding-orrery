"""Backward-compatible token/template helpers — thin shims over ``DirectionModel``.

The chat template, end-of-instruction window, and refusal-token resolution are
now backend-specific and live on the ``DirectionModel`` adapter
(``interpret.experiments.directions_common.model_adapter``). These free
functions delegate to it so existing call sites (and notebooks) that import
``format_chat`` / ``compute_eoi_token_ids`` / ``verify_refusal_tokens`` keep
working — they accept the adapter as the first argument in place of the old
raw wrapper.
"""

from __future__ import annotations

from interpret.experiments.directions_common import DirectionModel
from interpret.experiments.directions_common.model_adapter import GemmaDirectionModel

# Preserved for callers/notebooks that referenced the constant directly.
EOI_TEMPLATE_SUFFIX = GemmaDirectionModel.EOI_TEMPLATE_SUFFIX


def compute_eoi_token_ids(model: DirectionModel) -> list[int]:
    """Token ids of the end-of-instruction window (mean-capture slice size)."""
    return model.eoi_token_ids()


def format_chat(model: DirectionModel, instruction: str) -> str:
    """Apply the model's chat template to a single user instruction."""
    return model.format_chat(instruction)


def verify_refusal_tokens(
    model: DirectionModel, ids: tuple[int, ...] = (235285,)
) -> tuple[int, ...]:
    """Resolve/verify the refusal-cue token ids for this model.

    Gemma verifies the configured ids map to ``"I"`` (returns them unchanged,
    warning on drift); Qwen recomputes ids for its own tokenizer.
    """
    return model.refusal_token_ids(ids)
