"""SentenceTransformer load options: dtype policy and encoder towers per modality."""

import pytest

from backend.utils.embedding_model_catalog import EmbeddingModelCatalog
from backend.utils.model_load_options import EmbeddingDtypePolicy, SentenceTransformerLoadOptions

EG2 = "google/embeddinggemma-2"


@pytest.fixture(scope="module")
def catalog() -> EmbeddingModelCatalog:
    return EmbeddingModelCatalog.from_mapping(
        {
            "providers": {
                "sentence_transformers": [
                    {"model": "all-MiniLM-L6-v2"},
                    {
                        "model": EG2,
                        "modalities": ["text", "image", "audio", "video"],
                        "encoders": {
                            "image": "vision_config",
                            "video": "vision_config",
                            "audio": "audio_config",
                        },
                    },
                ]
            }
        }
    )


def _options(catalog, modalities, model=EG2, dtype="float32"):
    policy = EmbeddingDtypePolicy(cuda_bf16_supported=lambda: True)
    return SentenceTransformerLoadOptions.for_modalities(
        model,
        "mps" if dtype == "float32" else "cuda",
        modalities,
        catalog=catalog,
        dtype_policy=policy,
    )


class TestDtypePolicy:
    def test_bf16_only_on_bf16_capable_cuda(self):
        assert (
            EmbeddingDtypePolicy(cuda_bf16_supported=lambda: True).for_device("cuda") == "bfloat16"
        )
        assert (
            EmbeddingDtypePolicy(cuda_bf16_supported=lambda: True).for_device("cuda:1")
            == "bfloat16"
        )
        assert (
            EmbeddingDtypePolicy(cuda_bf16_supported=lambda: False).for_device("cuda") == "float32"
        )

    @pytest.mark.parametrize("device", ["mps", "cpu"])
    def test_float32_elsewhere_without_probing_cuda(self, device):
        def probe():
            raise AssertionError("must not probe CUDA for " + device)

        assert EmbeddingDtypePolicy(cuda_bf16_supported=probe).for_device(device) == "float32"

    def test_never_float16(self):
        for supported in (True, False):
            policy = EmbeddingDtypePolicy(cuda_bf16_supported=lambda s=supported: s)
            assert all(policy.for_device(d) != "float16" for d in ("cuda", "mps", "cpu"))


class TestEncoderSelection:
    @pytest.mark.parametrize(
        "modalities, disabled",
        [
            ({"text"}, ("audio_config", "vision_config")),
            ({"text", "image"}, ("audio_config",)),
            ({"text", "video"}, ("audio_config",)),
            ({"text", "audio"}, ("vision_config",)),
            ({"text", "image", "audio"}, ()),
            ({"text", "video", "audio"}, ()),
        ],
    )
    def test_only_the_needed_towers_load(self, catalog, modalities, disabled):
        options = _options(catalog, modalities)
        assert options.disabled_encoders == disabled
        assert options.config_kwargs() == dict.fromkeys(disabled)

    def test_models_without_encoder_map_load_whole(self, catalog):
        options = _options(catalog, {"text"}, model="all-MiniLM-L6-v2")
        assert options.disabled_encoders == () and options.config_kwargs() == {}

    def test_unknown_models_load_whole(self, catalog):
        assert _options(catalog, {"text"}, model="/local/checkpoint").config_kwargs() == {}

    def test_catalog_lookup_ignores_case(self, catalog):
        assert _options(catalog, {"text"}, model=EG2.upper()).disabled_encoders != ()


class TestLoadKwargs:
    def test_model_kwargs_carry_the_dtype(self, catalog):
        assert _options(catalog, {"text"}).model_kwargs() == {"dtype": "float32"}
        assert _options(catalog, {"text"}, dtype="bfloat16").model_kwargs() == {"dtype": "bfloat16"}

    def test_constructor_kwargs_skip_empty_config_kwargs(self, catalog):
        whole = _options(catalog, {"text"}, model="all-MiniLM-L6-v2")
        assert whole.constructor_kwargs() == {"model_kwargs": {"dtype": "float32"}}
        text_only = _options(catalog, {"text"})
        assert text_only.constructor_kwargs()["config_kwargs"] == {
            "audio_config": None,
            "vision_config": None,
        }


class TestCoverage:
    def test_a_fuller_load_covers_a_text_only_request(self, catalog):
        full = _options(catalog, {"text", "image", "audio"})
        text_only = _options(catalog, {"text"})
        assert full.covers(text_only)
        assert not text_only.covers(full)

    def test_vision_load_covers_video_but_not_audio(self, catalog):
        vision = _options(catalog, {"text", "image"})
        assert vision.covers(_options(catalog, {"text", "video"}))
        assert not vision.covers(_options(catalog, {"text", "audio"}))

    def test_different_dtype_or_model_never_covers(self, catalog):
        fp32 = _options(catalog, {"text", "image", "audio"})
        assert not fp32.covers(_options(catalog, {"text"}, dtype="bfloat16"))
        assert not fp32.covers(_options(catalog, {"text"}, model="all-MiniLM-L6-v2"))

    def test_cache_key_distinguishes_loads(self, catalog):
        keys = {
            _options(catalog, {"text"}).cache_key,
            _options(catalog, {"text", "image"}).cache_key,
            _options(catalog, {"text"}, dtype="bfloat16").cache_key,
        }
        assert len(keys) == 3
