"""Prompt resolution for embedding models with task prompts (EmbeddingGemma, ...).

Document side (``DocumentPromptResolver``): a collection's prompt is either a
prompt *name* the loaded model defines (sent as ``prompt_name``) or literal
prompt text (sent as ``prompt``). A known prompt name the model lacks is an
error, not literal text: it would silently prefix every input with e.g. "STS".
The vocabulary of known names comes from the catalog.

Query side (``QueryPromptResolver``): an explicit query prompt wins. Otherwise
the collection's document prompt maps to its query counterpart
(``Retrieval-document`` -> ``Retrieval-query``) when the model defines it, else
the stored prompt itself is reused (symmetric prompts such as ``Clustering``,
or fine-tuned checkpoints that only kept their document prompt). Gemini task
types map the same way. Both maps live in ``config/embedding_models.json``.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from .embedding_model_catalog import EmbeddingModelCatalog


@dataclass(frozen=True)
class ResolvedPrompt:
    """A prompt ready for ``SentenceTransformer.encode``: a name, literal text, or nothing."""

    name: str | None = None
    text: str | None = None

    def encode_kwargs(self) -> dict[str, str]:
        if self.name is not None:
            return {"prompt_name": self.name}
        if self.text is not None:
            return {"prompt": self.text}
        return {}


class DocumentPromptResolver:
    """Decide whether a prompt string is a prompt name of the loaded model or literal text."""

    def __init__(self, catalog: EmbeddingModelCatalog | None = None):
        self.catalog = catalog or EmbeddingModelCatalog.shipped()

    def resolve(self, prompt: str | None, model_prompts: Mapping[str, str]) -> ResolvedPrompt:
        """Raises ``ValueError`` for a known prompt name ``model_prompts`` does not define."""
        if not prompt:
            return ResolvedPrompt()
        if prompt in model_prompts:
            return ResolvedPrompt(name=prompt)
        if prompt in self.catalog.prompt_vocabulary:
            available = ", ".join(sorted(model_prompts))
            detail = f"available: {available}" if available else "this model defines no prompts"
            raise ValueError(
                f"{prompt!r} is a prompt name, but the loaded model does not define it ({detail})"
            )
        return ResolvedPrompt(text=prompt)


class QueryPromptResolver:
    """Choose the prompt (or Gemini task type) a text query is embedded with."""

    def __init__(self, catalog: EmbeddingModelCatalog | None = None):
        self.catalog = catalog or EmbeddingModelCatalog.shipped()

    def candidates(self, explicit: str | None, document_prompt: str | None) -> tuple[str, ...]:
        """Prompts to try, in order: the explicit one, else the mapped and the stored prompt."""
        if explicit:
            return (explicit,)
        if not document_prompt:
            return ()
        mapped = self.catalog.query_prompt_for(document_prompt)
        return tuple(dict.fromkeys(p for p in (mapped, document_prompt) if p))

    def for_model(
        self, explicit: str | None, document_prompt: str | None, model_prompts: Mapping[str, str]
    ) -> str | None:
        """The query prompt for a loaded model with ``model_prompts``.

        An explicit prompt is returned as is (the document-side resolver
        reports a name the model lacks); otherwise the first candidate that is
        either defined by the model or literal text.
        """
        if explicit:
            return explicit
        for candidate in self.candidates(None, document_prompt):
            if candidate in model_prompts or candidate not in self.catalog.prompt_vocabulary:
                return candidate
        return None

    def task_type(self, document_task_type: str | None) -> str | None:
        """The Gemini query task type for a collection's document task type."""
        return self.catalog.query_task_type_for(document_task_type)
