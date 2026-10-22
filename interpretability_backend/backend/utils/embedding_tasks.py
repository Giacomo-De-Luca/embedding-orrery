"""Task prefixes of the models that take Google's prompt format.

EmbeddingGemma (sentence-transformers) and Gemini Embedding 2 read their task
from a text prefix instead of a prompt name or an API field (Gemini Embedding 2
has no ``task_type``). Search-type tasks are asymmetric: queries get
``task: search result | query: ``, documents ``title: {title} | text: `` (with
``title: none`` when there is none); symmetric tasks use one prefix for both.
The tasks live in the catalog (``task_prefixes`` in
``config/embedding_models.json``); the models are marked there with
``"prompt_format": "task_prefix"``.

How a collection carries them (no schema change):

- ``EmbeddingModelConfig.prompt`` is the document prefix template. Its
  ``{column}`` placeholders may name any column of the row (the title is
  usually a metadata column), and ``DocumentPrefix`` renders it into the model
  input only: the stored document stays the plain text.
- ``EmbeddingModelConfig.task`` is the query prefix (query time only, the same
  meaning Qwen's instruction has).
- ``EmbeddingModelConfig.prompt_format`` = ``TASK_PREFIX_FORMAT`` marks such a
  config, and the collection's ``embedding_prompt_format`` metadata marks the
  collection. Configs and collections without the marker (prompt names such
  as ``Retrieval-document``, or built before task prefixes) keep the
  sentence-transformers prompt path.

``EmbeddingTexts`` is what the text embedders (local file, HuggingFace,
re-embed) use per row: the stored document, and the model input behind the
prefix. Media inputs never get a prefix.

Torch-free: safe to import from the GraphQL schema.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from .embedding_model_catalog import TASK_PREFIX_FORMAT, EmbeddingModelCatalog
from .text_processing import ColumnTemplate, format_text_for_embedding

# Value of an empty placeholder, as in the documented "title: none".
MISSING_VALUE = "none"


class DocumentPrefix:
    """The document prefix of one embedding job, rendered per row."""

    def __init__(self, template: str):
        self.template = ColumnTemplate(template)

    @property
    def columns(self) -> tuple[str, ...]:
        """The columns the prefix reads."""
        return self.template.fields

    def validate(self, columns: Iterable[str]) -> None:
        """Raise ``ValueError`` when a placeholder names none of ``columns``."""
        self.template.require_columns(columns, "Task prefix")

    def model_input(self, row: Mapping[str, Any], text: str) -> str:
        """What the model embeds for this row: the rendered prefix, then ``text``."""
        return self.template.render(row, missing=MISSING_VALUE) + text


class TaskPrefixPolicy:
    """The catalog-backed rules for task prefixes (see module docstring)."""

    def __init__(self, catalog: EmbeddingModelCatalog | None = None):
        self.catalog = catalog or EmbeddingModelCatalog.shipped()

    def uses_task_prefixes(self, provider: str, model_name: str) -> bool:
        """Whether the model reads its task from a prefix (and takes no ``task_type``)."""
        return self.catalog.uses_task_prefixes(provider, model_name)

    def prompt_format_for(self, provider: str, model_name: str, prompt: str | None) -> str | None:
        """The marker of a new embedding config.

        ``TASK_PREFIX_FORMAT`` for a task-prefix model unless ``prompt`` is a
        prompt name: such configs (resumed jobs, older forms) keep resolving
        the name through sentence-transformers.
        """
        if not self.uses_task_prefixes(provider, model_name):
            return None
        if prompt and prompt in self.catalog.prompt_vocabulary:
            return None
        return TASK_PREFIX_FORMAT

    @staticmethod
    def document_prefix(prompt_format: str | None, prompt: str | None) -> DocumentPrefix | None:
        """The per-row document prefix of a marked config with a prompt, else None."""
        if prompt_format != TASK_PREFIX_FORMAT or not prompt:
            return None
        return DocumentPrefix(prompt)

    def query_prefix(self, stored: str | None, explicit: str | None) -> str | None:
        """The prefix a text query of a task-prefix collection gets.

        An explicit query prompt wins: literal text as is, a prompt name
        through the task it aliases (``Retrieval-query`` -> the search prefix).
        Otherwise the collection's stored query prefix; empty means none.

        Raises:
            ValueError: ``explicit`` is a prompt name no task aliases
                (e.g. the document prompt ``Retrieval-document``).
        """
        if explicit:
            task = self.catalog.task_prefixes.task_for_prompt_name(explicit)
            if task is not None:
                return task.query_prefix
            if explicit in self.catalog.prompt_vocabulary:
                raise ValueError(
                    f"{explicit!r} is a prompt name with no task in the task-prefix table "
                    f"(document prompts such as 'Retrieval-document' are not query "
                    f"prompts); use a query prompt name or the prefix text, e.g. "
                    f"{self.catalog.task_prefixes.default.query_prefix!r}"
                )
            return explicit
        return stored or None


class EmbeddingTexts:
    """What one text embed job stores and embeds for each row.

    The document (stored in DuckDB and shown everywhere) is the job's text:
    its template over the embedded columns, or those columns joined. The
    model input is that document behind the rendered task prefix, when the
    model config has one.
    """

    def __init__(
        self,
        columns: Iterable[str],
        template: str | None,
        prompt_format: str | None,
        prompt: str | None,
    ):
        """Raises ``ValueError`` for a malformed text template or one naming a
        column outside ``columns`` (the embedded columns)."""
        self.columns = list(columns)
        self.template = ColumnTemplate.parse(template) if template else None
        if self.template is not None:
            self.template.require_columns(self.columns, "Text template")
        self.prefix = TaskPrefixPolicy.document_prefix(prompt_format, prompt)

    def validate(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Raise ``ValueError`` when a prefix placeholder names no column of any row.

        Runs before the model loads and before an overwrite deletes the old
        collection; rows may omit absent values, so the keys of all rows count.
        """
        if self.prefix is not None:
            self.prefix.validate(set().union(*(row.keys() for row in rows)))

    def document(self, row: Mapping[str, Any]) -> str:
        """The stored document text of ``row``."""
        if self.template is not None:
            return self.template.render(row)
        return format_text_for_embedding(row, self.columns)

    def model_input(self, row: Mapping[str, Any], document: str) -> str:
        """What the model embeds for ``row`` (its ``document`` behind any task prefix)."""
        return document if self.prefix is None else self.prefix.model_input(row, document)
