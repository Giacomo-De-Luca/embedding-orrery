"""Shared WordNet sample iteration for the autointerpreter collectors.

Both the SAE collector (:class:`ActivationCollector`) and the embedding
collector (:class:`EmbeddingCollector`) walk WordNet the same way: for every
``(word, synset)`` pair with a non-empty definition, format the prompt
template and yield a ``{word, synset_id, pos, definition, prompt}`` dict.

This is the source-agnostic core of "collect" — it has no dependency on
SAEs, hooks, or embedding models — so it lives in one place and both
collectors delegate to it. ``skip_keys`` lets a collector skip rows already
on disk (resume).
"""

from __future__ import annotations

from collections.abc import Iterator

from interpret.utils.wordnet_parser import WordNetParser


class WordNetSampleIterator:
    """Yield prompt samples for every eligible ``(word, synset)`` in WordNet.

    Args:
        parser: a :class:`WordNetParser` (injectable for tests).
        pos_filter: keep only these parts of speech (``n``/``v``/``a``/``r``);
            ``None`` keeps all.
        prompt_template: ``str.format`` template with ``{word}``/``{definition}``.
        limit: stop after this many yielded samples; ``None`` = no limit.
    """

    def __init__(
        self,
        parser: WordNetParser,
        *,
        pos_filter: list[str] | None = None,
        prompt_template: str = "{word}: {definition}.",
        limit: int | None = None,
    ) -> None:
        self.parser = parser
        self.pos_filter = set(pos_filter) if pos_filter else None
        self.prompt_template = prompt_template
        self.limit = limit

    def iter_samples(
        self, skip_keys: set[tuple[str, str]] | frozenset = frozenset(),
    ) -> Iterator[dict]:
        """Yield ``{word, synset_id, pos, definition, prompt}`` dicts.

        ``skip_keys`` is a set of ``(word, synset_id)`` pairs to skip (already
        collected). The ``limit`` counts *yielded* samples, not skipped ones,
        matching the historical single-SAE behaviour.
        """
        yielded = 0
        for word in self.parser.get_all_words():
            for synset in self.parser.get_synsets_for_word(word):
                if self.pos_filter and synset.part_of_speech not in self.pos_filter:
                    continue
                definition = (synset.definition or "").strip()
                if not definition:
                    continue
                if (word, synset.id) in skip_keys:
                    continue
                prompt = self.prompt_template.format(word=word, definition=definition)
                yield {
                    "word": word,
                    "synset_id": synset.id,
                    "pos": synset.part_of_speech,
                    "definition": definition,
                    "prompt": prompt,
                }
                yielded += 1
                if self.limit is not None and yielded >= self.limit:
                    return
