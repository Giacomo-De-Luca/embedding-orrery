"""Count the documents an embedding run truncates at its input-length cap.

Local embedders truncate inputs longer than the cap (the model's limit, or a
lower one chosen in the embed form). The run never stops for it: each
embedder keeps a ``TruncationTracker`` and the batch job turns its count into
a warning on the progress stream and in the job result.

Torch-free: lengths come from the model's own tokenizer, called without
truncation or padding. Counting is best-effort: a model whose tokenizer cannot
be called on plain text (or has none) stops counting rather than failing the
job, and models that wrap inputs in a chat template are under-counted by the
template's tokens.
"""

import logging
from collections.abc import Iterable, Sequence

logger = logging.getLogger("orrery." + __name__)


class TokenLengths:
    """Token counts of texts, special tokens included, nothing truncated."""

    @staticmethod
    def of(tokenizer, texts: Sequence[str]) -> list[int]:
        encoded = tokenizer(
            list(texts),
            add_special_tokens=True,
            truncation=False,
            padding=False,
            verbose=False,
        )
        return [len(ids) for ids in encoded["input_ids"]]


class TruncationTracker:
    """Running count of inputs longer than ``max_length`` (inactive without a cap)."""

    def __init__(self, max_length: int | None):
        self.max_length = max_length
        self.truncated = 0
        self.seen = 0
        self.longest = 0
        self._disabled = False

    @property
    def active(self) -> bool:
        return self.max_length is not None and not self._disabled

    def count(self, tokenizer, texts: Sequence[str]) -> int:
        """Tokenize ``texts`` and record them; returns how many exceed the cap.

        A tokenizer that is missing or fails stops the counting (logged once):
        the warning is informative, it must never fail the embedding.
        """
        if not self.active:
            return 0
        try:
            lengths = TokenLengths.of(tokenizer, texts)
        except Exception:
            logger.warning("Truncation counting disabled: the tokenizer failed", exc_info=True)
            self._disabled = True
            return 0
        return self.observe(lengths)

    def observe(self, lengths: Iterable[int]) -> int:
        """Record one batch of token counts; returns how many of them exceed the cap."""
        if not self.active:
            return 0
        lengths = list(lengths)
        over = sum(1 for n in lengths if n > self.max_length)
        self.truncated += over
        self.seen += len(lengths)
        self.longest = max([self.longest, *lengths])
        return over

    def warning(self) -> str | None:
        """Summary for the user, or None while nothing was truncated."""
        if not self.truncated:
            return None
        return (
            f"{self.truncated:,} of {self.seen:,} documents exceeded the "
            f"{self.max_length:,}-token limit and were truncated "
            f"(longest: {self.longest:,} tokens)"
        )
