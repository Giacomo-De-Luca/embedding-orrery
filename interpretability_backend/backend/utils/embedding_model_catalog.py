"""Catalog of known embedding models per provider (``config/embedding_models.json``).

One config file is the single source for:

- the model-name suggestions of the embed forms (GraphQL ``embeddingModelCatalog``);
- the prompt-name vocabulary, which tells a prompt *name* the loaded model lacks
  (an error) from a literal prompt string;
- the document -> query maps used when a query carries no explicit prompt
  (sentence-transformers prompt names, and Gemini task types);
- the default model of the multimodal data type;
- the encoder towers a multimodal model can skip loading (``encoders`` maps a
  media modality to the model-config attribute to disable, e.g. ``"vision_config"``).

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


def _require_names(value: object, label: str) -> None:
    """Raise unless ``value`` is a tuple of distinct non-empty strings."""
    if not isinstance(value, tuple) or not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"{label} must be a list of non-empty strings, got {value!r}")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} lists a name twice: {list(value)}")


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

    def __post_init__(self):
        if not isinstance(self.providers, Mapping):
            raise ValueError("providers must map provider names to lists of models")
        parsed = {
            provider: self._parse_models(provider, entries)
            for provider, entries in self.providers.items()
        }
        object.__setattr__(self, "providers", MappingProxyType(parsed))
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

    @cached_property
    def prompt_vocabulary(self) -> frozenset[str]:
        """Every prompt name any catalog model defines, plus ``extra_prompt_names``."""
        names = {name for model in self.models() for name in model.prompts}
        return frozenset(names | set(self.extra_prompt_names))

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
