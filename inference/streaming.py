"""Shared streaming-generation types for inference wrappers.

Deliberately stdlib-only so any wrapper can import the streaming event type
without pulling in a model-specific backend (e.g. the forked ``gemma_pytorch``
package, which is heavy and Gemma-specific). Both ``GemmaPytorchInference`` and
``Qwen3Inference`` yield ``TokenStreamEvent`` from their ``generate_stream`` /
``generate_chat_stream`` methods, so a consumer (e.g. the backend
``InterpretService.generate_stream`` subscription) can treat either wrapper
identically.
"""

from typing import NamedTuple


class TokenStreamEvent(NamedTuple):
    """A single token yielded during streaming generation."""

    token_index: int  # 0-based position in generated output
    token_id: int  # raw token ID from the tokenizer
    text_delta: str  # clean text chunk for display
    is_done: bool  # True on the last token
