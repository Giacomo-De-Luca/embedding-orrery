"""Catalog of known embedding models per provider (``config/embedding_models.json``).

One config file is the single source for:

- the known models of each provider (the factory's and the embed forms'
  per-model facts below);
- the prompt-name vocabulary, which tells a prompt *name* the loaded model lacks
  (an error) from a literal prompt string;
- the task-prefix table (``task_prefixes``) of the models marked
  ``"prompt_format": "task_prefix"`` (EmbeddingGemma, Gemini Embedding 2): each
  task's query prefix (``task: search result | query: ``), document prefix
  (``title: none | text: ``; symmetric tasks use the query prefix for both) and
  the prompt names that alias it. ``utils/embedding_tasks.py`` applies it; the
  GraphQL ``embeddingTaskPrefixes`` query serves it to the embed forms;
- the document -> query maps used when a query carries no explicit prompt
  (sentence-transformers prompt names, and Gemini task types);
- the default model of the multimodal data type;
- the encoder towers a multimodal model can skip loading (``encoders`` maps a
  media modality to the model-config attribute to disable, e.g. ``"vision_config"``);
- the input-length limit in tokens (``max_length``), for models whose own files
  don't state it reliably (EmbeddingGemma 2's tokenizer declares no limit). It
  wins over the model files in ``utils/model_max_length.py``.

Providers are keyed by their ``EmbeddingProviderEnum`` value (``"sentence_transformers"``,
``"openai"``, ...), the same strings DuckDB stores in ``vector_collections``.
Torch-free: safe to import from the GraphQL schema.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache, cached_property
from types import MappingProxyType
from typing import ClassVar, Self

from .json_settings import JsonFileSettings
from .provider_list import EmbeddingProviderEnum
from .resource_paths import PACKAGE_ROOT

CATALOG_FILE = PACKAGE_ROOT / "config" / "embedding_models.json"

MODALITIES = ("text", "image", "audio", "video")
MEDIA_MODALITIES = tuple(m for m in MODALITIES if m != "text")
# Data types whose default model comes from the catalog.
DATA_TYPES_WITH_DEFAULTS = ("multimodal",)
PROVIDERS = frozenset(p.value for p in EmbeddingProviderEnum)
# Models whose task instruction is a text prefix from ``task_prefixes``.
TASK_PREFIX_FORMAT = "task_prefix"
PROMPT_FORMATS = (TASK_PREFIX_FORMAT,)


def _require_names(value: object, label: str) -> None:
    """Raise unless ``value`` is a tuple of distinct non-empty strings."""
    if not isinstance(value, tuple) or not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"{label} must be a list of non-empty strings, got {value!r}")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} lists a name twice: {list(value)}")


def _require_text(value: object, label: str) -> None:
    """Raise unless ``value`` is a non-empty string."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")


@dataclass(frozen=True)
class EmbeddingTask(JsonFileSettings):
    """One task of the task-prefix table.

    ``document_prefix`` is None for symmetric tasks (classification,
    clustering...), which embed queries and documents with ``query_prefix``.
    ``prompt_names`` are the sentence-transformers prompt names whose prefix is
    this task's query prefix, so an explicit query prompt name resolves to it.
    """

    SETTINGS_LABEL: ClassVar[str] = "embedding task"
    TUPLE_FIELDS: ClassVar[tuple[str, ...]] = ("prompt_names",)

    id: str = ""
    label: str = ""
    query_prefix: str = ""
    document_prefix: str | None = None
    prompt_names: tuple[str, ...] = ()

    def __post_init__(self):
        _require_text(self.id, "task id")
        _require_text(self.label, f"task {self.id}: label")
        _require_text(self.query_prefix, f"task {self.id}: query_prefix")
        if self.document_prefix is None:
            object.__setattr__(self, "document_prefix", self.query_prefix)
        else:
            _require_text(self.document_prefix, f"task {self.id}: document_prefix")
        _require_names(self.prompt_names, f"task {self.id}: prompt_names")

    @property
    def symmetric(self) -> bool:
        """Queries and documents share one prefix."""
        return self.document_prefix == self.query_prefix


@dataclass(frozen=True)
class TaskPrefixTable(JsonFileSettings):
    """The ordered tasks of ``task_prefixes`` and the default one."""

    SETTINGS_LABEL: ClassVar[str] = "task prefix table"
    TUPLE_FIELDS: ClassVar[tuple[str, ...]] = ("tasks",)

    default_task: str = ""
    tasks: tuple[EmbeddingTask, ...] = ()

    def __post_init__(self):
        if not isinstance(self.tasks, tuple):
            raise ValueError(f"tasks must be a list of tasks, got {self.tasks!r}")
        parsed = tuple(
            task if isinstance(task, EmbeddingTask) else EmbeddingTask.from_mapping(task)
            for task in self.tasks
        )
        object.__setattr__(self, "tasks", parsed)
        _require_names(tuple(task.id for task in parsed), "task ids")
        _require_names(
            tuple(name for task in parsed for name in task.prompt_names), "task prompt_names"
        )
        if parsed and self.get(self.default_task) is None:
            raise ValueError(f"default_task {self.default_task!r} is not one of the task ids")
        if not parsed and self.default_task:
            raise ValueError("default_task needs tasks")

    def get(self, task_id: str | None) -> EmbeddingTask | None:
        """The task with this id, if listed."""
        return next((task for task in self.tasks if task.id == task_id), None)

    @property
    def default(self) -> EmbeddingTask:
        """The default task; raises on an empty table."""
        task = self.get(self.default_task)
        if task is None:
            raise ValueError("The task prefix table lists no tasks")
        return task

    def task_for_prompt_name(self, name: str | None) -> EmbeddingTask | None:
        """The task a prompt name aliases (``Retrieval-query`` -> search), if any."""
        return next((task for task in self.tasks if name in task.prompt_names), None)


@dataclass(frozen=True)
class CatalogModel(JsonFileSettings):
    """One known model of one provider."""

    SETTINGS_LABEL: ClassVar[str] = "catalog model"
    TUPLE_FIELDS: ClassVar[tuple[str, ...]] = ("modalities", "prompts", "default_for")

    model: str = ""
    provider: str = ""
    description: str | None = None
    modalities: tuple[str, ...] = ("text",)
    prompts: tuple[str, ...] = ()
    default_for: tuple[str, ...] = ()
    encoders: Mapping[str, str] = field(default_factory=dict, hash=False)
    max_length: int | None = None
    # TASK_PREFIX_FORMAT: the task instruction is a text prefix from task_prefixes.
    prompt_format: str | None = None

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError(f"model must be a non-empty string, got {self.model!r}")
        if self.description is not None and (
            not isinstance(self.description, str) or not self.description.strip()
        ):
            raise ValueError(f"{self.model}: description must be a non-empty string")
        _require_names(self.modalities, f"{self.model}: modalities")
        if "text" not in self.modalities:
            raise ValueError(f"{self.model}: modalities must include 'text'")
        unknown = set(self.modalities) - set(MODALITIES)
        if unknown:
            raise ValueError(f"{self.model}: unknown modalities {sorted(unknown)}")
        _require_names(self.prompts, f"{self.model}: prompts")
        _require_names(self.default_for, f"{self.model}: default_for")
        if not set(self.default_for) <= set(DATA_TYPES_WITH_DEFAULTS):
            raise ValueError(
                f"{self.model}: default_for must be a subset of {list(DATA_TYPES_WITH_DEFAULTS)}"
            )
        if "multimodal" in self.default_for and not self.media_modalities:
            raise ValueError(f"{self.model}: a multimodal default needs a media modality")
        self._validate_encoders()
        object.__setattr__(self, "encoders", MappingProxyType(dict(self.encoders)))
        if self.max_length is not None:
            self.require_int("max_length", minimum=1)
        if self.prompt_format is not None and self.prompt_format not in PROMPT_FORMATS:
            raise ValueError(
                f"{self.model}: prompt_format must be one of {list(PROMPT_FORMATS)}, "
                f"got {self.prompt_format!r}"
            )

    def _validate_encoders(self) -> None:
        if not isinstance(self.encoders, Mapping):
            raise ValueError(f"{self.model}: encoders must map modalities to config attributes")
        for modality, attribute in self.encoders.items():
            if modality not in self.media_modalities:
                raise ValueError(
                    f"{self.model}: encoders may only name the model's media modalities "
                    f"{list(self.media_modalities)}, got {modality!r}"
                )
            if not isinstance(attribute, str) or not attribute:
                raise ValueError(f"{self.model}: encoders[{modality!r}] must be a config attribute")

    @property
    def media_modalities(self) -> tuple[str, ...]:
        """The modalities other than text this model embeds."""
        return tuple(m for m in self.modalities if m != "text")


@dataclass(frozen=True)
class EmbeddingModelCatalog(JsonFileSettings):
    """All known models, keyed by provider value (see module docstring)."""

    SETTINGS_LABEL: ClassVar[str] = "embedding model catalog"
    TUPLE_FIELDS: ClassVar[tuple[str, ...]] = ("extra_prompt_names",)

    providers: Mapping[str, tuple[CatalogModel, ...]] = field(default_factory=dict, hash=False)
    # Document prompt name -> query prompt name (sentence-transformers prompts).
    query_prompt_map: Mapping[str, str] = field(default_factory=dict, hash=False)
    # Document task type -> query task type (Gemini task types).
    query_task_type_map: Mapping[str, str] = field(default_factory=dict, hash=False)
    # Prompt names some models use but no catalog entry lists (e.g. "s2p").
    extra_prompt_names: tuple[str, ...] = ()
    # Tasks of the models marked prompt_format "task_prefix".
    task_prefixes: TaskPrefixTable = field(default_factory=TaskPrefixTable, hash=False)

    def __post_init__(self):
        if not isinstance(self.providers, Mapping):
            raise ValueError("providers must map provider names to lists of models")
        parsed = {
            provider: self._parse_models(provider, entries)
            for provider, entries in self.providers.items()
        }
        object.__setattr__(self, "providers", MappingProxyType(parsed))
        if not isinstance(self.task_prefixes, TaskPrefixTable):
            object.__setattr__(
                self,
                "task_prefixes",
                TaskPrefixTable.from_mapping(self.task_prefixes, source="task_prefixes"),
            )
        if self.task_prefix_models() and not self.task_prefixes.tasks:
            raise ValueError("Models with prompt_format 'task_prefix' need tasks in task_prefixes")
        for name in ("query_prompt_map", "query_task_type_map"):
            mapping = getattr(self, name)
            if not isinstance(mapping, Mapping) or not all(
                isinstance(k, str) and k and isinstance(v, str) and v for k, v in mapping.items()
            ):
                raise ValueError(f"{name} must map names to non-empty names")
            object.__setattr__(self, name, MappingProxyType(dict(mapping)))
        _require_names(self.extra_prompt_names, "extra_prompt_names")
        self._validate_defaults()

    @staticmethod
    def _parse_models(provider: str, entries: object) -> tuple[CatalogModel, ...]:
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {provider!r}; expected one of {sorted(PROVIDERS)}")
        if not isinstance(entries, (list, tuple)):
            raise ValueError(f"providers[{provider!r}] must be a list of models")
        models = []
        for entry in entries:
            if isinstance(entry, CatalogModel):  # already parsed (dataclasses.replace)
                models.append(entry)
                continue
            if isinstance(entry, dict):
                if "provider" in entry:
                    raise ValueError(f"providers[{provider!r}]: entries must not set 'provider'")
                entry = {**entry, "provider": provider}
            models.append(CatalogModel.from_mapping(entry, source=f"providers[{provider!r}]"))
        seen: set[str] = set()
        for model in models:
            key = model.model.casefold()
            if key in seen:
                raise ValueError(f"providers[{provider!r}] lists {model.model!r} twice")
            seen.add(key)
        return tuple(models)

    def _validate_defaults(self) -> None:
        for data_type in DATA_TYPES_WITH_DEFAULTS:
            defaults = [m.model for m in self.models() if data_type in m.default_for]
            if len(defaults) > 1:
                raise ValueError(f"more than one default model for {data_type!r}: {defaults}")

    @classmethod
    @cache
    def shipped(cls) -> Self:
        """The catalog in ``config/embedding_models.json``, loaded once per process."""
        return cls.from_file(CATALOG_FILE)

    def models(self, provider: str | None = None) -> tuple[CatalogModel, ...]:
        """Every model, or those of one provider (unknown providers have none)."""
        if provider is not None:
            return self.providers.get(provider, ())
        return tuple(model for models in self.providers.values() for model in models)

    def find(self, provider: str, model_name: str) -> CatalogModel | None:
        """The entry for ``model_name`` (case-insensitive) under ``provider``, if listed."""
        wanted = model_name.casefold()
        return next((m for m in self.models(provider) if m.model.casefold() == wanted), None)

    def default_model_for(self, data_type: str) -> CatalogModel:
        """The model marked ``default_for`` this data type; raises if none is."""
        model = next((m for m in self.models() if data_type in m.default_for), None)
        if model is None:
            raise ValueError(
                f"The embedding model catalog names no default model for {data_type!r}"
            )
        return model

    def uses_task_prefixes(self, provider: str, model_name: str) -> bool:
        """Whether the model takes its task instruction as a ``task_prefixes`` prefix."""
        model = self.find(provider, model_name)
        return model is not None and model.prompt_format == TASK_PREFIX_FORMAT

    def task_prefix_models(self) -> tuple[CatalogModel, ...]:
        """Every model marked ``prompt_format: task_prefix``."""
        return tuple(m for m in self.models() if m.prompt_format == TASK_PREFIX_FORMAT)

    @cached_property
    def prompt_vocabulary(self) -> frozenset[str]:
        """Every prompt name any catalog model defines or a task aliases, plus
        ``extra_prompt_names``."""
        names = {name for model in self.models() for name in model.prompts}
        aliases = {name for task in self.task_prefixes.tasks for name in task.prompt_names}
        return frozenset(names | aliases | set(self.extra_prompt_names))

    def query_prompt_for(self, document_prompt: str | None) -> str | None:
        """The query-side counterpart of a document prompt name, if it has one."""
        if document_prompt is None:
            return None
        return self.query_prompt_map.get(document_prompt)

    def query_task_type_for(self, task_type: str | None) -> str | None:
        """The query-side task type of a document task type (others pass through)."""
        if task_type is None:
            return None
        return self.query_task_type_map.get(task_type, task_type)
