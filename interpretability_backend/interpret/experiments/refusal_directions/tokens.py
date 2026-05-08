"""Gemma-3 SentencePiece helpers for the refusal-direction pipeline.

The paper extracts mean activations at every "post-instruction" position —
the tokens that close the user turn and open the model turn. For Gemma the
chat template ends with ``<end_of_turn>\\n<start_of_turn>model\\n``; tokenising
that suffix gives the position window. The reference implementation does the
same: see [gemma_model.py:108](references/refusal_direction/pipeline/model_utils/gemma_model.py#L108).
"""

from __future__ import annotations

import warnings

EOI_TEMPLATE_SUFFIX = "<end_of_turn>\n<start_of_turn>model\n"


def compute_eoi_token_ids(wrapper) -> list[int]:
    """Tokenise the end-of-instruction suffix and return its token IDs.

    The list length is the number of post-instruction positions to slice
    activations at when computing the mean-of-difference direction.
    """
    return wrapper.tokenize(EOI_TEMPLATE_SUFFIX, bos=False)


def format_chat(wrapper, instruction: str) -> str:
    """Apply the Gemma-3 chat template via the wrapper's static helper.

    Note: ``GemmaPytorchInference.format_prompt`` includes a trailing ``\\n``
    after ``<start_of_turn>model``; ``GemmaPytorchInference.generate`` does
    not. This pipeline always feeds the model via ``generate_from_template``
    so that the template matches the reference paper (and so that
    ``EOI_TEMPLATE_SUFFIX`` is exactly the suffix the model sees).
    """
    return wrapper.format_prompt(instruction)


def verify_refusal_tokens(wrapper, ids: tuple[int, ...] = (235285,)) -> tuple[int, ...]:
    """Sanity-check that the configured refusal token IDs map to ``"I"``.

    Returns the IDs unchanged. Emits a warning (rather than raising) if the
    SentencePiece vocab differs from the Gemma 1/2 default — Gemma 3 may
    have re-numbered, in which case the caller should override
    ``RefusalConfig.refusal_token_ids`` after inspecting the warning.
    """
    expected = wrapper.tokenize("I", bos=False)
    if tuple(expected) != tuple(ids):
        warnings.warn(
            f"Refusal token mismatch: configured {ids}, "
            f"tokeniser produced {expected} for 'I'. "
            "Update RefusalConfig.refusal_token_ids if this is wrong.",
            stacklevel=2,
        )
    return ids
