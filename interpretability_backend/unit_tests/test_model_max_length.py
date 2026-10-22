"""Model input-length limits: the catalog field, the resolution order and request checks.

The form shows a model's real limit and embedding truncates at it, so the
lookup must not trust the "no limit" sentinels some model files carry
(EmbeddingGemma 2's tokenizer declares 1e30). No test touches the network:
model files come from a fake reader, or from a temporary folder.
"""

import json

import pytest
from huggingface_hub.errors import EntryNotFoundError

from backend.utils.embedding_model_catalog import CatalogModel, EmbeddingModelCatalog
from backend.utils.model_files import ModelConfigFiles
from backend.utils.model_max_length import ModelMaxLength, ModelMaxLengthResolver

EG2 = "google/embeddinggemma-2"
PLAIN = "plain-model"
QWEN = "Qwen/Qwen3-Embedding-0.6B"


class FakeFiles:
    """Stands in for ModelConfigFiles: ``{(model, filename): parsed JSON}``."""

    def __init__(self, files: dict | None = None):
        self.files = files or {}
        self.reads: list[tuple[str, str]] = []

    def read(self, model_name: str, filename: str) -> dict | None:
        self.reads.append((model_name, filename))
        return self.files.get((model_name, filename))


def _catalog(**eg2_fields) -> EmbeddingModelCatalog:
    return EmbeddingModelCatalog.from_mapping(
        {
            "providers": {
                "sentence_transformers": [{"model": EG2, **eg2_fields}, {"model": PLAIN}],
                "qwen": [{"model": QWEN}],
            }
        }
    )


class TestCatalogField:
    def test_positive_limit_is_kept(self):
        assert CatalogModel.from_mapping({"model": "m", "max_length": 512}).max_length == 512

    def test_defaults_to_unknown(self):
        assert CatalogModel.from_mapping({"model": "m"}).max_length is None

    @pytest.mark.parametrize("bad", [0, -3, "512", True, 1.5])
    def test_invalid_limits_are_rejected(self, bad):
        with pytest.raises(ValueError, match="max_length"):
            CatalogModel.from_mapping({"model": "m", "max_length": bad})

    def test_shipped_catalog_states_the_embeddinggemma_limits(self):
        catalog = EmbeddingModelCatalog.shipped()
        assert catalog.find("sentence_transformers", EG2).max_length == 8192
        assert (
            catalog.find("sentence_transformers", "google/embeddinggemma-300m").max_length == 2048
        )


class TestResolve:
    def test_catalog_wins_over_misleading_model_files(self):
        files = FakeFiles(
            {
                (EG2, "tokenizer_config.json"): {"model_max_length": int(1e30)},
                (EG2, "config.json"): {"text_config": {"max_position_embeddings": 262144}},
            }
        )
        resolver = ModelMaxLengthResolver(catalog=_catalog(max_length=8192), files=files)

        assert resolver.resolve("sentence_transformers", EG2) == ModelMaxLength(8192, "catalog")
        assert files.reads == []

    def test_sentence_transformers_setting_comes_next(self):
        files = FakeFiles(
            {
                (PLAIN, "sentence_bert_config.json"): {"max_seq_length": 256},
                (PLAIN, "tokenizer_config.json"): {"model_max_length": 512},
            }
        )
        resolver = ModelMaxLengthResolver(catalog=_catalog(), files=files)

        assert resolver.resolve("sentence_transformers", PLAIN) == ModelMaxLength(
            256, "sentence_transformers_config"
        )

    def test_tokenizer_limit_when_sentence_transformers_sets_none(self):
        files = FakeFiles(
            {
                (PLAIN, "sentence_bert_config.json"): {"do_lower_case": False},
                (PLAIN, "tokenizer_config.json"): {"model_max_length": 512},
            }
        )
        resolver = ModelMaxLengthResolver(catalog=_catalog(), files=files)

        assert resolver.resolve("sentence_transformers", PLAIN) == ModelMaxLength(512, "tokenizer")

    def test_tokenizer_limit_is_capped_by_the_position_table(self):
        files = FakeFiles(
            {
                (QWEN, "tokenizer_config.json"): {"model_max_length": 131072},
                (QWEN, "config.json"): {"max_position_embeddings": 32768},
            }
        )
        resolver = ModelMaxLengthResolver(catalog=_catalog(), files=files)

        assert resolver.resolve("qwen", QWEN) == ModelMaxLength(32768, "model_config")

    def test_tokenizer_limit_below_the_position_table_stands(self):
        files = FakeFiles(
            {
                (PLAIN, "tokenizer_config.json"): {"model_max_length": 512},
                (PLAIN, "config.json"): {"max_position_embeddings": 514},
            }
        )
        resolver = ModelMaxLengthResolver(catalog=_catalog(), files=files)

        assert resolver.resolve("sentence_transformers", PLAIN) == ModelMaxLength(512, "tokenizer")

    def test_tokenizer_sentinel_falls_through_to_nested_position_embeddings(self):
        files = FakeFiles(
            {
                (QWEN, "tokenizer_config.json"): {"model_max_length": int(1e30)},
                (QWEN, "config.json"): {"text_config": {"max_position_embeddings": 32768}},
            }
        )
        resolver = ModelMaxLengthResolver(catalog=_catalog(), files=files)

        assert resolver.resolve("qwen", QWEN) == ModelMaxLength(32768, "model_config")

    def test_top_level_position_embeddings(self):
        files = FakeFiles({(PLAIN, "config.json"): {"max_position_embeddings": 514}})
        resolver = ModelMaxLengthResolver(catalog=_catalog(), files=files)

        assert resolver.resolve("sentence_transformers", PLAIN) == ModelMaxLength(
            514, "model_config"
        )

    def test_unknown_when_nothing_states_a_limit(self):
        resolver = ModelMaxLengthResolver(catalog=_catalog(), files=FakeFiles())

        assert resolver.resolve("sentence_transformers", PLAIN) == ModelMaxLength(None, None)

    def test_api_providers_have_no_local_limit(self):
        files = FakeFiles()
        resolver = ModelMaxLengthResolver(catalog=_catalog(), files=files)

        assert resolver.resolve("gemini", "gemini-embedding-001") == ModelMaxLength(None, None)
        assert files.reads == []


class TestEffective:
    @pytest.fixture
    def resolver(self) -> ModelMaxLengthResolver:
        return ModelMaxLengthResolver(catalog=_catalog(max_length=8192), files=FakeFiles())

    def test_no_request_means_the_model_limit(self, resolver):
        assert resolver.effective("sentence_transformers", EG2, None) == 8192

    @pytest.mark.parametrize("requested", [1, 2048, 8192])
    def test_requests_up_to_the_limit_are_used(self, resolver, requested):
        assert resolver.effective("sentence_transformers", EG2, requested) == requested

    def test_request_above_the_limit_is_rejected(self, resolver):
        with pytest.raises(ValueError, match="8,192"):
            resolver.effective("sentence_transformers", EG2, 9000)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_requests_are_rejected(self, resolver, bad):
        with pytest.raises(ValueError, match="max_length"):
            resolver.effective("sentence_transformers", EG2, bad)

    def test_unknown_limit_accepts_any_positive_request(self, resolver):
        assert resolver.effective("sentence_transformers", PLAIN, 4096) == 4096
        assert resolver.effective("sentence_transformers", PLAIN, None) is None

    def test_api_providers_ignore_the_request(self, resolver):
        assert resolver.effective("gemini", "gemini-embedding-001", 100) is None


def _never(*args, **kwargs):
    raise AssertionError("must not be called")


class TestModelConfigFiles:
    def test_reads_from_a_local_model_folder(self, tmp_path):
        (tmp_path / "sentence_bert_config.json").write_text(json.dumps({"max_seq_length": 128}))
        files = ModelConfigFiles(cache_lookup=_never, downloader=_never)

        assert files.read(str(tmp_path), "sentence_bert_config.json") == {"max_seq_length": 128}
        assert files.read(str(tmp_path), "missing.json") is None

    def test_uses_the_hub_cache_before_downloading(self, tmp_path):
        cached = tmp_path / "config.json"
        cached.write_text(json.dumps({"dtype": "bfloat16"}))
        files = ModelConfigFiles(cache_lookup=lambda repo, name: str(cached), downloader=_never)

        assert files.read("org/model", "config.json") == {"dtype": "bfloat16"}

    def test_downloads_when_not_cached(self, tmp_path):
        downloaded = tmp_path / "config.json"
        downloaded.write_text(json.dumps({"max_position_embeddings": 512}))
        calls = []

        def download(repo_id, filename):
            calls.append((repo_id, filename))
            return str(downloaded)

        files = ModelConfigFiles(cache_lookup=lambda repo, name: None, downloader=download)

        assert files.read("org/model", "config.json") == {"max_position_embeddings": 512}
        assert calls == [("org/model", "config.json")]

    def test_a_file_known_to_be_absent_is_not_downloaded(self):
        absent = object()  # huggingface_hub's "cached as missing" marker
        files = ModelConfigFiles(cache_lookup=lambda repo, name: absent, downloader=_never)

        assert files.read("org/model", "sentence_bert_config.json") is None

    def test_bare_names_resolve_to_the_sentence_transformers_org(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("{}")
        repos = []

        def lookup(repo_id, filename):
            repos.append(repo_id)
            return str(path)

        ModelConfigFiles(cache_lookup=lookup, downloader=_never).read(
            "all-MiniLM-L6-v2", "config.json"
        )
        assert repos == ["sentence-transformers/all-MiniLM-L6-v2"]

    def test_transient_download_errors_are_retried_later(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text('{"dtype": "bfloat16"}')
        attempts = []

        def flaky(repo_id, filename):
            attempts.append(filename)
            if len(attempts) == 1:
                raise OSError("offline")
            return str(path)

        files = ModelConfigFiles(cache_lookup=lambda repo, name: None, downloader=flaky)

        assert files.read("org/model", "config.json") is None
        assert files.read("org/model", "config.json") == {"dtype": "bfloat16"}
        assert len(attempts) == 2

    def test_a_missing_file_is_remembered(self):
        attempts = []

        def missing(repo_id, filename):
            attempts.append(filename)
            raise EntryNotFoundError("no such file")

        files = ModelConfigFiles(cache_lookup=lambda repo, name: None, downloader=missing)
        files.read("org/model", "sentence_bert_config.json")
        files.read("org/model", "sentence_bert_config.json")

        assert attempts == ["sentence_bert_config.json"]

    def test_cache_only_reader_never_downloads_or_grows(self):
        files = ModelConfigFiles.cache_only()

        assert files.read("orrery-test/no-such-model-xyz", "config.json") is None
        assert files._read == {}

    def test_results_are_read_once(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("{}")
        calls = []

        def lookup(repo_id, filename):
            calls.append(filename)
            return str(path)

        files = ModelConfigFiles(cache_lookup=lookup, downloader=_never)
        files.read("org/model", "config.json")
        files.read("org/model", "config.json")

        assert calls == ["config.json"]


class TestGraphqlQuery:
    QUERY = (
        "query($provider: EmbeddingProviderEnum!, $model: String!) {"
        " embeddingModelMaxLength(provider: $provider, modelName: $model) { maxLength source } }"
    )

    def _run(self, provider: str, model: str) -> dict:
        import asyncio

        from backend.API import schema

        result = asyncio.run(
            schema.execute(self.QUERY, variable_values={"provider": provider, "model": model})
        )
        assert result.errors is None, result.errors
        return result.data["embeddingModelMaxLength"]

    def test_catalog_limit_for_a_known_model(self):
        assert self._run("SENTENCE_TRANSFORMERS", EG2) == {"maxLength": 8192, "source": "catalog"}

    def test_api_models_have_no_limit(self):
        assert self._run("GEMINI", "gemini-embedding-001") == {"maxLength": None, "source": None}

    def test_read_only_mode_uses_the_cache_only_resolver(self, monkeypatch):
        from backend.API import queries

        asked: list[str] = []

        class CacheOnly:
            def resolve(self, provider, model_name):
                asked.append(model_name)
                return ModelMaxLength(None, None)

        monkeypatch.setenv("ORRERY_READ_ONLY", "1")
        monkeypatch.setattr(
            queries.ModelMaxLengthResolver, "cache_only", classmethod(lambda cls: CacheOnly())
        )

        assert self._run("SENTENCE_TRANSFORMERS", "any/model") == {
            "maxLength": None,
            "source": None,
        }
        assert asked == ["any/model"]
