"""Converters between GraphQL types and internal service/config types."""

from ..embed_dataset import (
    DataType,
    EmbeddingConfig,
    EmbeddingModelConfig,
    EmbeddingProvider,
    LocalFileEmbeddingConfig,
    PortionConfig,
    PortionStrategy,
)
from ..services.probing_types import MLP_ACTIVATIONS, PROBE_KINDS, ProbeConfig
from ..services.topic_extraction_service import TopicExtractionConfig
from .types import (
    DataTypeEnum,
    EmbeddingProviderEnum,
    PortionStrategyEnum,
    TopicInfo,
    TopicKeyword,
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
    DataTypeEnum.IMAGE: DataType.IMAGE,
    DataTypeEnum.VECTOR: DataType.VECTOR,
}


# ========== Input -> Config Builders ==========


def build_embedding_model_config(input) -> EmbeddingModelConfig | None:
    """Convert GraphQL EmbeddingModelInput to internal EmbeddingModelConfig."""
    if input is None:
        return None
    return EmbeddingModelConfig(
        provider=EMBEDDING_PROVIDER_MAP[input.provider],
        model_name=input.model_name,
        ollama_url=input.ollama_url,
        task=input.task,
        task_type=input.task_type,
        prompt=input.prompt,
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
    return EmbeddingConfig(
        dataset_id=input.dataset_id,
        collection_name=input.collection_name,
        config=input.config,
        split=input.split,
        splits=input.splits,
        columns=input.columns,
        text_template=input.text_template,
        id_column=input.id_column,
        portion=build_portion_config(input.portion),
        metadata_columns=input.metadata_columns,
        embedding_model=build_embedding_model_config(input.embedding_model),
        batch_size=input.batch_size or 100,
        resume=input.resume,
    )


def build_local_file_embedding_config(input) -> LocalFileEmbeddingConfig:
    """Convert EmbedLocalFileInput to internal LocalFileEmbeddingConfig."""
    return LocalFileEmbeddingConfig(
        file_path=input.file_path,
        collection_name=input.collection_name,
        data_type=DATA_TYPE_MAP[input.data_type],
        columns=input.columns,
        text_template=input.text_template,
        image_column=input.image_column,
        vector_column=input.vector_column,
        id_column=input.id_column,
        metadata_columns=input.metadata_columns,
        n_rows=input.n_rows,
        sample_n=input.sample_n,
        sample_seed=input.sample_seed,
        embedding_model=build_embedding_model_config(input.embedding_model),
        batch_size=input.batch_size or 100,
        resume=input.resume,
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

    if tc and getattr(tc, "reduction", None) and tc.reduction.enabled:
        reduce_topics = True
        reduction_method = tc.reduction.method
        nr_topics = tc.reduction.n_topics
        use_ctfidf_for_reduction = tc.reduction.use_ctfidf

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
