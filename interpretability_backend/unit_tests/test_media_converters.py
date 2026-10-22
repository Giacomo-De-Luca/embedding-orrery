"""GraphQL media inputs -> embedding configs (API/converters.py)."""

import pytest
from graphql import coerce_input_value

from backend.API import schema
from backend.API.converters import (
    build_hf_embedding_config,
    build_local_file_embedding_config,
    graphql_input_payload,
)
from backend.API.types import (
    DataTypeEnum,
    EmbedDatasetInput,
    EmbeddingModelInput,
    EmbedLocalFileInput,
    MediaOptionsInput,
)
from backend.embedding_functions.config import DataType, EmbeddingProvider
from backend.utils.media_settings import MediaOptions
from backend.utils.provider_list import EmbeddingProviderEnum

EG2 = "google/embeddinggemma-2"
ST = EmbeddingProviderEnum.SENTENCE_TRANSFORMERS


def _local(**overrides) -> EmbedLocalFileInput:
    values = dict(
        file_path="/data/pets.parquet",
        collection_name="pets",
        data_type=DataTypeEnum.MULTIMODAL,
        media_columns=["img"],
        columns=["caption"],
    )
    values.update(overrides)
    return EmbedLocalFileInput(**values)


class TestLocalMultimodal:
    def test_defaults_to_the_catalog_model_and_options(self):
        config = build_local_file_embedding_config(_local())

        assert config.data_type is DataType.MULTIMODAL
        assert config.media_columns == ["img"]
        assert config.embedding_model.provider is EmbeddingProvider.SENTENCE_TRANSFORMERS
        assert config.embedding_model.model_name == EG2
        assert config.media_options == MediaOptions()

    def test_the_stored_input_carries_the_resolved_model(self):
        config = build_local_file_embedding_config(_local())
        assert config.job_input["embeddingModel"]["modelName"] == EG2
        assert config.job_input["dataType"] == "MULTIMODAL"

    def test_a_blank_model_name_means_the_default(self):
        model = EmbeddingModelInput(provider=ST, model_name="  ")
        config = build_local_file_embedding_config(_local(embedding_model=model))
        assert config.embedding_model.model_name == EG2

    def test_overrides_merge_into_the_defaults(self):
        options = MediaOptionsInput(video_max_frames=8, image_max_soft_tokens=70)
        config = build_local_file_embedding_config(_local(media_options=options))
        assert config.media_options.video_max_frames == 8
        assert config.media_options.image_max_soft_tokens == 70
        assert config.media_options.video_fps == 1.0

    def test_invalid_overrides_are_rejected(self):
        with pytest.raises(ValueError, match="image_max_soft_tokens"):
            build_local_file_embedding_config(
                _local(media_options=MediaOptionsInput(image_max_soft_tokens=300))
            )

    def test_media_needs_sentence_transformers(self):
        model = EmbeddingModelInput(provider=EmbeddingProviderEnum.OPENAI, model_name="m")
        with pytest.raises(ValueError, match="sentence-transformers"):
            build_local_file_embedding_config(_local(embedding_model=model))

    def test_text_only_catalog_models_are_rejected(self):
        model = EmbeddingModelInput(provider=ST, model_name="all-MiniLM-L6-v2")
        with pytest.raises(ValueError, match="text only"):
            build_local_file_embedding_config(_local(embedding_model=model))

    def test_models_outside_the_catalog_are_checked_at_load(self):
        model = EmbeddingModelInput(provider=ST, model_name="/models/my-clip")
        config = build_local_file_embedding_config(_local(embedding_model=model))
        assert config.embedding_model.model_name == "/models/my-clip"

    def test_media_columns_are_required(self):
        with pytest.raises(ValueError, match="media column"):
            build_local_file_embedding_config(_local(media_columns=[]))

    def test_text_jobs_ignore_media_fields(self):
        config = build_local_file_embedding_config(
            _local(data_type=DataTypeEnum.TEXT, media_columns=None)
        )
        assert config.data_type is DataType.TEXT and config.media_options is None


class TestHuggingFaceMultimodal:
    def test_builds_a_media_config(self):
        config = build_hf_embedding_config(
            EmbedDatasetInput(
                dataset_id="org/pets",
                collection_name="pets",
                data_type=DataTypeEnum.MULTIMODAL,
                media_columns=["image"],
            )
        )
        assert config.data_type is DataType.MULTIMODAL and config.media_columns == ["image"]
        assert config.embedding_model.model_name == EG2

    def test_vectors_are_local_files_only(self):
        with pytest.raises(ValueError, match="local"):
            build_hf_embedding_config(
                EmbedDatasetInput(
                    dataset_id="org/pets", collection_name="pets", data_type=DataTypeEnum.VECTOR
                )
            )


def test_payload_is_valid_mutation_input():
    payload = graphql_input_payload(_local(media_options=MediaOptionsInput(video_fps=2.0)))
    coerce_input_value(payload, schema._schema.get_type("EmbedLocalFileInput"))
    assert payload["mediaOptions"]["videoFps"] == 2.0
