"""Stored job inputs replay as mutation variables (resume).

Embedding jobs keep the original GraphQL input in their job-state config
(``config["input"]``) so the frontend can resume a job by re-sending it with
``resume: true`` — columns, topic settings and the SAE link included.
"""

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
    PortionInput,
    PortionStrategyEnum,
    SaeLinkInput,
    TopicConfigInput,
    TopicReductionInput,
)
from backend.utils.provider_list import EmbeddingProviderEnum


def _hf_input() -> EmbedDatasetInput:
    return EmbedDatasetInput(
        dataset_id="org/data",
        collection_name="col",
        columns=["title", "text"],
        text_template="{title}: {text}",
        portion=PortionInput(strategy=PortionStrategyEnum.FIRST_N, n=100),
        embedding_model=EmbeddingModelInput(
            provider=EmbeddingProviderEnum.SENTENCE_TRANSFORMERS, model_name="all-MiniLM-L6-v2"
        ),
        extract_topics=True,
        topic_config=TopicConfigInput(
            min_topic_size=5, reduction=TopicReductionInput(enabled=True, n_topics=8)
        ),
        sae_link=SaeLinkInput(sae_model_id="gemma-3-4b-it", sae_id="9-gemmascope-2-res-16k"),
    )


def _graphql_type(name: str):
    return schema._schema.get_type(name)


class TestGraphqlInputPayload:
    def test_camel_case_keys_and_enum_names(self):
        payload = graphql_input_payload(_hf_input())

        assert payload["datasetId"] == "org/data"
        assert payload["portion"] == {
            "strategy": "FIRST_N",
            "n": 100,
            "start": None,
            "end": None,
            "seed": 42,
        }
        assert payload["embeddingModel"]["provider"] == "SENTENCE_TRANSFORMERS"
        assert payload["topicConfig"]["reduction"]["nTopics"] == 8
        assert payload["saeLink"] == {
            "saeModelId": "gemma-3-4b-it",
            "saeId": "9-gemmascope-2-res-16k",
        }

    def test_payload_is_valid_mutation_input(self):
        coerce_input_value(graphql_input_payload(_hf_input()), _graphql_type("EmbedDatasetInput"))
        local = EmbedLocalFileInput(
            file_path="/tmp/v.parquet",
            collection_name="vecs",
            data_type=DataTypeEnum.VECTOR,
            vector_column="vector",
        )
        coerce_input_value(graphql_input_payload(local), _graphql_type("EmbedLocalFileInput"))


class TestJobConfigKeepsInput:
    def test_hf_job_config_stores_the_input(self):
        config = build_hf_embedding_config(_hf_input())

        job_config = config.to_job_config()

        assert job_config["input"] == graphql_input_payload(_hf_input())
        assert job_config["dataset_id"] == "org/data"  # summary fields still present

    def test_local_job_config_stores_the_input(self):
        local = EmbedLocalFileInput(file_path="/tmp/a.csv", collection_name="c", columns=["text"])

        job_config = build_local_file_embedding_config(local).to_job_config()

        assert job_config["input"]["filePath"] == "/tmp/a.csv"
        assert job_config["data_type"] == "text"
