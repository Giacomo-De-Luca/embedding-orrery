"""Converters between GraphQL types and internal service/config types."""

import dataclasses
from enum import Enum
from typing import Any

from strawberry.utils.str_converters import to_camel_case

from ..embed_dataset import (
    DataType,
    EmbeddingConfig,
    EmbeddingModelConfig,
    EmbeddingProvider,
    LocalFileEmbeddingConfig,
    PortionConfig,
    PortionStrategy,
)
from ..embedding_functions.config import ReEmbedConfig
from ..services.probing_types import (
    MLP_ACTIVATIONS,
    PROBE_KINDS,
    ProbeConfig,
    ProbeTask,
    derive_probe_task,
    score_field_names,
)
from ..services.topic_extraction_service import TopicExtractionConfig
from ..services.topic_reduction_types import ReductionRequest
from ..utils.embedding_model_catalog import EmbeddingModelCatalog
from ..utils.embedding_tasks import TaskPrefixPolicy
from ..utils.known_dimensions import KnownDimensions
from ..utils.media_settings import MediaEmbeddingSettings, MediaOptions
from .types import (
    DataTypeEnum,
    EmbeddingModelCatalogEntry,
    EmbeddingModelInput,
    EmbeddingProviderEnum,
    EmbeddingTaskPrefix,
    EmbeddingTaskPrefixModel,
    EmbeddingTaskPrefixTable,
    MediaEmbeddingSettingsInfo,
    MediaOptionsInfo,
    PortionStrategyEnum,
    ProbeInfo,
    TopicInfo,
    TopicKeyword,
    TopicMergeTreeInfo,
    TopicTreeLeaf,
    TopicTreeMerge,
)

# ========== Enum Maps ==========

PORTION_STRATEGY_MAP = {
    PortionStrategyEnum.FIRST_N: PortionStrategy.FIRST_N,
    PortionStrategyEnum.RANDOM_SAMPLE: PortionStrategy.RANDOM_SAMPLE,
    PortionStrategyEnum.ROW_RANGE: PortionStrategy.ROW_RANGE,
    PortionStrategyEnum.ALL: PortionStrategy.ALL,
}

# Auto-generate provider mapping - no manual maintenance needed
# When a new provider is added to provider_list.py, it automatically appears here
EMBEDDING_PROVIDER_MAP = {
    getattr(EmbeddingProviderEnum, member.name): member for member in EmbeddingProvider
}

DATA_TYPE_MAP = {
    DataTypeEnum.TEXT: DataType.TEXT,
    DataTypeEnum.MULTIMODAL: DataType.MULTIMODAL,
    DataTypeEnum.VECTOR: DataType.VECTOR,
}


class MediaJobInputs:
    """The media fields of an embed input (``DataTypeEnum.MULTIMODAL`` jobs).

    ``resolved`` fills the catalog's multimodal default model (EmbeddingGemma 2)
    when none is given, so the stored job input names the model it ran with
    and a resume can never pick up a different default. It rejects jobs a
    model cannot run: no media columns, a provider other than
    sentence-transformers, or a catalog model that embeds text only (models
    outside the catalog, such as local checkpoints, are checked when they
    load). ``options`` merges ``MediaOptionsInput`` over the defaults of
    ``config/media_embedding.json``.
    """

    def __init__(
        self,
        catalog: EmbeddingModelCatalog | None = None,
        settings: MediaEmbeddingSettings | None = None,
    ):
        self.catalog = catalog or EmbeddingModelCatalog.shipped()
        self.settings = settings or MediaEmbeddingSettings.shipped()

    @staticmethod
    def is_multimodal(input) -> bool:
        return getattr(input, "data_type", None) == DataTypeEnum.MULTIMODAL

    def resolved(self, input):
        """``input`` with its model filled in; unchanged for other data types."""
        if not self.is_multimodal(input):
            return input
        if not input.media_columns:
            raise ValueError("Select at least one media column (images, audio or video)")
        default = self.catalog.default_model_for(DataType.MULTIMODAL.value).model
        model = input.embedding_model
        if model is None:
            model = EmbeddingModelInput(
                provider=EmbeddingProviderEnum.SENTENCE_TRANSFORMERS, model_name=default
            )
        elif not model.model_name.strip():
            model = dataclasses.replace(model, model_name=default)
        if model.provider != EmbeddingProviderEnum.SENTENCE_TRANSFORMERS:
            raise ValueError(
                f"Images, audio and video need a sentence-transformers model such as {default}"
            )
        entry = self.catalog.find(model.provider.value, model.model_name.strip())
        if entry is not None and not entry.media_modalities:
            raise ValueError(f"{entry.model} embeds text only; use a media model such as {default}")
        return dataclasses.replace(input, embedding_model=model)

    def options(self, input) -> MediaOptions | None:
        """The job's effective media options (None for other data types)."""
        if not self.is_multimodal(input):
            return None
        overrides = input.media_options
        values = dataclasses.asdict(overrides) if overrides is not None else None
        return self.settings.with_overrides(values)

    def columns(self, input) -> list[str] | None:
        return list(input.media_columns) if self.is_multimodal(input) else None


# ========== Input -> Config Builders ==========


def graphql_input_payload(value: Any) -> Any:
    """The GraphQL variables form (camelCase keys, enum names) of a Strawberry
    input, so a stored job can be replayed as mutation variables (resume)."""
    definition = getattr(value, "__strawberry_definition__", None)
    if definition is not None:
        return {
            (field.graphql_name or to_camel_case(field.python_name)): graphql_input_payload(
                getattr(value, field.python_name)
            )
            for field in definition.fields
        }
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, list):
        return [graphql_input_payload(item) for item in value]
    return value


def build_sae_link(input) -> dict | None:
    """SaeLinkInput → the collection extra-metadata keys the SAE services read."""
    if input is None:
        return None
    return {"sae_model_id": input.sae_model_id, "sae_id": input.sae_id}


def build_embedding_model_config(
    input, batch_size: int | None = None
) -> EmbeddingModelConfig | None:
    """Convert GraphQL EmbeddingModelInput to internal EmbeddingModelConfig.

    ``batch_size`` is the job's batch size: local models embed that many texts
    per forward pass (it is also the rows-per-write chunk of the job).
    Task-prefix models (``utils/embedding_tasks.py``) get the
    ``prompt_format`` marker and no ``task_type``, which they don't support.
    """
    if input is None:
        return None
    provider = EMBEDDING_PROVIDER_MAP[input.provider]
    model_name = input.model_name.strip()
    policy = TaskPrefixPolicy()
    return EmbeddingModelConfig(
        provider=provider,
        model_name=model_name,
        ollama_url=input.ollama_url,
        task=input.task,
        task_type=(
            None if policy.uses_task_prefixes(provider.value, model_name) else input.task_type
        ),
        prompt=input.prompt,
        max_length=input.max_length,
        batch_size=batch_size,
        prompt_format=policy.prompt_format_for(provider.value, model_name, input.prompt),
    )


def build_portion_config(input) -> PortionConfig | None:
    """Convert GraphQL PortionInput to internal PortionConfig."""
    if input is None:
        return None
    return PortionConfig(
        strategy=PORTION_STRATEGY_MAP[input.strategy],
        n=input.n,
        start=input.start,
        end=input.end,
        seed=input.seed,
    )


def build_hf_embedding_config(input) -> EmbeddingConfig:
    """Convert EmbedDatasetInput to internal EmbeddingConfig."""
    if input.data_type == DataTypeEnum.VECTOR:
        raise ValueError("Pre-computed vectors can only be embedded from local files")
    media = MediaJobInputs()
    input = media.resolved(input)
    return EmbeddingConfig(
        dataset_id=input.dataset_id,
        collection_name=input.collection_name,
        config=input.config,
        split=input.split,
        splits=input.splits,
        data_type=DATA_TYPE_MAP[input.data_type],
        media_columns=media.columns(input),
        media_options=media.options(input),
        columns=input.columns,
        text_template=input.text_template,
        id_column=input.id_column,
        portion=build_portion_config(input.portion),
        metadata_columns=input.metadata_columns,
        embedding_model=build_embedding_model_config(
            input.embedding_model, batch_size=input.batch_size or 100
        ),
        batch_size=input.batch_size or 100,
        resume=input.resume,
        job_input=graphql_input_payload(input),
    )


def build_local_file_embedding_config(input) -> LocalFileEmbeddingConfig:
    """Convert EmbedLocalFileInput to internal LocalFileEmbeddingConfig."""
    media = MediaJobInputs()
    input = media.resolved(input)
    return LocalFileEmbeddingConfig(
        file_path=input.file_path,
        collection_name=input.collection_name,
        data_type=DATA_TYPE_MAP[input.data_type],
        media_columns=media.columns(input),
        media_options=media.options(input),
        columns=input.columns,
        text_template=input.text_template,
        vector_column=input.vector_column,
        id_column=input.id_column,
        metadata_columns=input.metadata_columns,
        n_rows=input.n_rows,
        sample_n=input.sample_n,
        sample_seed=input.sample_seed,
        embedding_model=build_embedding_model_config(
            input.embedding_model, batch_size=input.batch_size or 100
        ),
        batch_size=input.batch_size or 100,
        resume=input.resume,
        job_input=graphql_input_payload(input),
    )


def build_reembed_config(input) -> ReEmbedConfig:
    """Convert ReEmbedDatasetInput to internal ReEmbedConfig."""
    return ReEmbedConfig(
        source_dataset_name=input.source_dataset_name,
        collection_name=input.collection_name,
        embedding_model=build_embedding_model_config(
            input.embedding_model, batch_size=input.batch_size or 100
        ),
        columns=input.columns,
        text_template=input.text_template,
        batch_size=input.batch_size or 100,
        resume=input.resume,
        job_input=graphql_input_payload(input),
    )


def build_topic_extraction_config(
    collection_name: str,
    tc=None,
) -> TopicExtractionConfig:
    """Convert GraphQL TopicConfigInput to internal TopicExtractionConfig.

    Handles tc=None by using defaults.
    """
    reduce_topics = False
    reduction_method = "auto"
    nr_topics = None
    use_ctfidf_for_reduction = True
    reduction_strategy = None
    reduction_distance_threshold = None
    reduction_absorb_noise = False

    if tc and getattr(tc, "reduction", None) and tc.reduction.enabled:
        reduction = tc.reduction
        reduce_topics = True
        reduction_method = reduction.method
        nr_topics = reduction.n_topics
        use_ctfidf_for_reduction = reduction.use_ctfidf
        reduction_strategy = getattr(reduction, "strategy", None)
        reduction_distance_threshold = getattr(reduction, "distance_threshold", None)
        reduction_absorb_noise = getattr(reduction, "absorb_noise", False)
        # Fail at the API boundary, not after clustering.
        ReductionRequest.create(
            strategy=reduction_strategy,
            method=reduction_method,
            n_topics=nr_topics,
            use_ctfidf=use_ctfidf_for_reduction,
            distance_threshold=reduction_distance_threshold,
            absorb_noise=reduction_absorb_noise,
        )

    return TopicExtractionConfig(
        collection_name=collection_name,
        min_topic_size=tc.min_topic_size if tc else 10,
        n_keywords=tc.n_keywords if tc else 10,
        use_llm_labels=tc.use_llm_labels if tc else False,
        llm_provider=tc.llm_provider if tc else "gemini",
        llm_model=tc.llm_model if tc else "gemini-3-flash-preview",
        projection_type=tc.projection_type if tc else "umap_2d",
        clustering_method=tc.clustering_method if tc else "hdbscan",
        n_clusters=tc.n_clusters if tc else None,
        cluster_on=tc.cluster_on if tc else "cluster_umap",
        cluster_n_components=tc.cluster_n_components if tc else 5,
        cluster_min_dist=tc.cluster_min_dist if tc else 0.0,
        cluster_n_neighbors=tc.cluster_n_neighbors if tc else 15,
        reduce_topics=reduce_topics,
        reduction_method=reduction_method,
        nr_topics=nr_topics,
        use_ctfidf_for_reduction=use_ctfidf_for_reduction,
        reduction_strategy=reduction_strategy,
        reduction_distance_threshold=reduction_distance_threshold,
        reduction_absorb_noise=reduction_absorb_noise,
    )


def reduction_request_from_input(input) -> ReductionRequest:
    """Validated ReductionRequest from a GraphQL ReduceTopicsInput."""
    return ReductionRequest.create(
        strategy=input.strategy,
        method=input.method,
        n_topics=input.n_topics,
        use_ctfidf=input.use_ctfidf,
        distance_threshold=input.distance_threshold,
        absorb_noise=input.absorb_noise,
        regenerate_labels=input.regenerate_labels,
        llm_provider=input.llm_provider,
        llm_model=input.llm_model,
    )


def topic_merge_tree_info(view, include_noise: bool = False) -> TopicMergeTreeInfo:
    """GraphQL TopicMergeTree from a ``TopicTreeView`` (topic_merge_tree_service)."""
    tree = view.tree
    k = tree.n_leaves
    sizes = tree.node_sizes()
    noise_ids = noise_rows = None
    if include_noise and tree.linkage == "density":
        noise_ids = list(tree.noise_merges)
        noise_rows = [int(tree.noise_merges[item]) for item in noise_ids]
    return TopicMergeTreeInfo(
        extraction_id=view.extraction_id,
        linkage=tree.linkage,
        base_level=view.base_level,
        leaves=[
            TopicTreeLeaf(topic_id=topic_id, label=view.labels[topic_id], size=size)
            for topic_id, size in zip(tree.leaf_ids.tolist(), tree.leaf_sizes.tolist(), strict=True)
        ],
        merges=[
            TopicTreeMerge(left=left, right=right, height=height, size=int(sizes[k + row]))
            for row, ((left, right), height) in enumerate(
                zip(tree.children.tolist(), tree.heights.tolist(), strict=True)
            )
        ],
        noise_count=view.noise_count,
        available_linkages=list(view.available_linkages),
        noise_item_ids=noise_ids,
        noise_merges=noise_rows,
    )


def build_probe_config(input) -> ProbeConfig:
    """Convert GraphQL TrainProbeInput to internal ProbeConfig.

    Optional hyperparams fall back to ProbeConfig defaults; the kind is
    validated here so bad input fails before any data is loaded.
    """
    if input.kind not in PROBE_KINDS:
        raise ValueError(f"Unknown probe kind {input.kind!r}. Expected one of {PROBE_KINDS}.")
    defaults = ProbeConfig(collection_name="", target_field="")

    def _or(value, fallback):
        return value if value is not None else fallback

    # class_weight is a nullable enum-like string; only "balanced" is valid.
    class_weight = input.class_weight or None
    if class_weight not in (None, "balanced"):
        raise ValueError(f"class_weight must be null or 'balanced', got {class_weight!r}.")

    # Range-check the numeric knobs here so bad input fails as a clean
    # converter error instead of an opaque sklearn traceback mid-job.
    if input.train_split is not None and not (0.0 < input.train_split < 1.0):
        raise ValueError(f"train_split must be in (0, 1), got {input.train_split}.")
    if input.c is not None and input.c <= 0:
        raise ValueError(f"c must be > 0, got {input.c}.")
    if input.alpha is not None and input.alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {input.alpha}.")
    if input.kernel is not None and input.kernel not in ("rbf", "linear", "poly", "sigmoid"):
        raise ValueError(f"kernel must be one of rbf/linear/poly/sigmoid, got {input.kernel!r}.")
    if input.max_train_samples is not None and input.max_train_samples < 1:
        raise ValueError(f"max_train_samples must be >= 1, got {input.max_train_samples}.")
    if input.patience is not None and input.patience < 1:
        raise ValueError(f"patience must be >= 1, got {input.patience}.")
    if input.dev_split is not None and not (0.0 <= input.dev_split < 1.0):
        # 0 is legal: it disables early stopping (fixed epochs, final weights).
        raise ValueError(f"dev_split must be in [0, 1), got {input.dev_split}.")
    if input.activation is not None and input.activation not in MLP_ACTIVATIONS:
        raise ValueError(f"activation must be one of {MLP_ACTIVATIONS}, got {input.activation!r}.")
    if input.epochs is not None and input.epochs < 1:
        # epochs=0 would "succeed" by persisting a randomly initialized MLP.
        raise ValueError(f"epochs must be >= 1, got {input.epochs}.")
    if input.hidden_dims and any(h < 1 for h in input.hidden_dims):
        raise ValueError(f"hidden_dims entries must be >= 1, got {input.hidden_dims}.")
    if input.min_class_fraction is not None and not (0.0 <= input.min_class_fraction < 0.5):
        raise ValueError(f"min_class_fraction must be in [0, 0.5), got {input.min_class_fraction}.")

    return ProbeConfig(
        collection_name=input.collection_name,
        target_field=input.target_field,
        kind=input.kind,
        alpha=_or(input.alpha, defaults.alpha),
        c=_or(input.c, defaults.c),
        kernel=_or(input.kernel, defaults.kernel),
        class_weight=class_weight,
        epochs=_or(input.epochs, defaults.epochs),
        hidden_dims=list(input.hidden_dims) if input.hidden_dims else None,
        patience=_or(input.patience, defaults.patience),
        dev_split=_or(input.dev_split, defaults.dev_split),
        activation=_or(input.activation, defaults.activation),
        seed=_or(input.seed, defaults.seed),
        train_split=_or(input.train_split, defaults.train_split),
        max_train_samples=_or(input.max_train_samples, defaults.max_train_samples),
        min_class_fraction=_or(input.min_class_fraction, defaults.min_class_fraction),
    )


def probe_info_from_record(record: dict) -> ProbeInfo:
    """Map a stored probe row (``DuckDBClient.list_probes`` shape) to ProbeInfo."""
    config = record.get("config") or {}
    score_field, residual_field = score_field_names(record["target_field"], record["kind"])
    task = derive_probe_task(record["kind"], config)
    return ProbeInfo(
        target_field=record["target_field"],
        kind=record["kind"],
        score_field=score_field,
        residual_field=residual_field,
        metrics=record["metrics"],
        n_train=record["n_train"] or 0,
        n_val=record["n_val"] or 0,
        created_at=record["created_at"],
        target_mapping=config.get("target_mapping"),
        task=task,
        classes=config.get("classes"),
        excluded_classes=config.get("excluded_classes"),
        # Every snapshot records it; it only applies to classifiers.
        min_class_fraction=(
            config.get("min_class_fraction") if task != ProbeTask.REGRESSION else None
        ),
    )


# ========== Result -> GraphQL Converters ==========


def convert_topic_infos(topics) -> list:
    """Convert list of service TopicInfoResult to GraphQL TopicInfo list."""
    return [
        TopicInfo(
            topic_id=topic.topic_id,
            keywords=[TopicKeyword(word=w, score=s) for w, s in topic.keywords],
            label=topic.label,
            count=topic.count,
            subtopics=topic.subtopics,
        )
        for topic in topics
    ]


def task_prefix_table_from_catalog(catalog: EmbeddingModelCatalog) -> EmbeddingTaskPrefixTable:
    """The catalog's task-prefix table and its models, for the embed forms."""
    table = catalog.task_prefixes
    return EmbeddingTaskPrefixTable(
        default_task=table.default_task,
        tasks=[
            EmbeddingTaskPrefix(
                id=task.id,
                label=task.label,
                query_prefix=task.query_prefix,
                document_prefix=task.document_prefix,
                symmetric=task.symmetric,
            )
            for task in table.tasks
        ],
        models=[
            EmbeddingTaskPrefixModel(
                provider=EmbeddingProviderEnum(model.provider), model_name=model.model
            )
            for model in catalog.task_prefix_models()
        ],
    )


def embedding_model_catalog_entries(
    catalog: EmbeddingModelCatalog,
    dimensions: KnownDimensions,
    in_use: list[tuple[str, str, int]],
) -> list[EmbeddingModelCatalogEntry]:
    """The embed forms' model suggestions: the catalog models, then models only
    existing collections use (``in_use``: provider, model, collection count;
    providers outside ``EmbeddingProviderEnum``, e.g. SAE label vectors, skipped).
    """
    counts = {(provider, model.casefold()): count for provider, model, count in in_use}
    entries = []
    for model in catalog.models():
        key = (model.provider, model.model.casefold())
        entries.append(
            EmbeddingModelCatalogEntry(
                provider=EmbeddingProviderEnum(model.provider),
                model_name=model.model,
                description=model.description,
                modalities=list(model.modalities),
                prompts=list(model.prompts),
                dimension=dimensions.get(model.model),
                default_for=list(model.default_for),
                suggested=True,
                collection_count=counts.pop(key, 0),
            )
        )
    providers = {provider.value for provider in EmbeddingProviderEnum}
    for provider, model_name, count in in_use:
        if provider not in providers or (provider, model_name.casefold()) not in counts:
            continue
        counts.pop((provider, model_name.casefold()))
        entries.append(
            EmbeddingModelCatalogEntry(
                provider=EmbeddingProviderEnum(provider),
                model_name=model_name,
                dimension=dimensions.get(model_name),
                suggested=False,
                collection_count=count,
            )
        )
    return entries


def media_settings_info(
    settings: MediaEmbeddingSettings, default_model: str
) -> MediaEmbeddingSettingsInfo:
    """Defaults, limits and token constants of the media options, for the embed forms."""
    return MediaEmbeddingSettingsInfo(
        defaults=MediaOptionsInfo(**settings.defaults.to_dict()),
        soft_token_budgets=list(settings.soft_token_budgets),
        video_fps_limit=settings.video_fps_limit,
        video_frames_limit=settings.video_frames_limit,
        audio_seconds_limit=settings.audio_seconds_limit,
        context_tokens=settings.context_tokens,
        reserved_tokens=settings.reserved_tokens,
        audio_tokens_per_second=settings.audio_tokens_per_second,
        default_model=default_model,
    )
