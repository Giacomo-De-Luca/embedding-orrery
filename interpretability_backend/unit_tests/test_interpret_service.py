"""Tests for InterpretService — SAE inference integration.

All tests use mocked inference wrappers and SAE objects so no GPU or
model weights are required.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from backend.services.interpret_service import (
    ActiveFeatureResult,
    FeatureActivation,
    InterpretService,
    LayerActivationsResult,
    ModelStatusResult,
    PromptActivationsResult,
    SteeredGenerationResult,
    SteeringSpec,
    TokenFeaturesResult,
)
from interpret.inference.gemma_pytorch import GemmaPytorchInference
from interpret.inference.qwen3_transformers import Qwen3Inference
from interpret.sae.sae_config import GemmaScopeSAEConfig, HookType, QwenScopeSAEConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    """Fresh InterpretService instance (no model loaded)."""
    return InterpretService()


def _make_mock_wrapper():
    """Build a mock GemmaPytorchInference with the attributes the service needs."""
    wrapper = MagicMock()
    wrapper.device = torch.device("cpu")
    wrapper.model.model.layers = MagicMock(spec_set=["__len__", "__getitem__"])
    wrapper.model.model.layers.__len__ = MagicMock(return_value=34)
    wrapper.generate.return_value = "mock response"
    wrapper.generate_from_template.return_value = "mock response"
    wrapper.tokenizer = MagicMock()
    wrapper.tokenizer.sp_model = MagicMock()
    return wrapper


def _make_mock_db():
    """Mock DuckDB client for the label-lookup path (no real DB access)."""
    db = MagicMock()
    db.get_sae_feature_labels_batch.return_value = {}
    return db


def _make_mock_layer_result(layer: int, width: str, feature_index: int = 42):
    """Mock toolkit LayerResult: one token at position 0 with one feature."""
    feature = MagicMock()
    feature.index = feature_index
    feature.activation = 3.5
    feature.label = "test label"
    feature.density = 0.01
    token = MagicMock()
    token.token = "hello"
    token.position = 0
    token.features = [feature]
    lr = MagicMock()
    lr.layer = layer
    lr.width = width
    lr.tokens = [token]
    lr.feature_acts = torch.zeros(1, 16)
    return lr


def _make_mock_prompt_result(specs: list[tuple[int, str]]):
    """Mock toolkit PromptResult keyed by (layer, width) as the explorer returns."""
    prompt_result = MagicMock()
    prompt_result.prompt = "hello"
    prompt_result.token_strings = ["hello"]
    prompt_result.layers = {
        (layer, width): _make_mock_layer_result(layer, width) for layer, width in specs
    }
    prompt_result.generated_text = None
    return prompt_result


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_status_initial(self, service):
        status = service.get_status()
        assert isinstance(status, ModelStatusResult)
        assert status.loaded is False
        assert status.model_name is None
        assert status.device is None

    @patch("backend.services.interpret_service.GemmaPytorchInference")
    def test_load_model(self, mock_cls, service):
        mock_cls.return_value = _make_mock_wrapper()
        status = service.load_model("google/gemma-3-4b-it")
        assert status.loaded is True
        assert status.model_name == "google/gemma-3-4b-it"
        assert status.device is not None

    @patch("backend.services.interpret_service.GemmaPytorchInference")
    def test_load_model_already_loaded(self, mock_cls, service):
        mock_cls.return_value = _make_mock_wrapper()
        service.load_model("google/gemma-3-4b-it")
        with pytest.raises(RuntimeError, match="already loaded"):
            service.load_model("google/gemma-3-4b-it")

    @patch("backend.services.interpret_service.GemmaPytorchInference")
    def test_unload_model(self, mock_cls, service):
        mock_cls.return_value = _make_mock_wrapper()
        service.load_model("google/gemma-3-4b-it")
        status = service.unload_model()
        assert status.loaded is False
        assert status.model_name is None
        assert service._wrapper is None

    def test_unload_when_not_loaded(self, service):
        # Should be a no-op, not an error.
        status = service.unload_model()
        assert status.loaded is False


# ---------------------------------------------------------------------------
# UC1: Prompt activations
# ---------------------------------------------------------------------------


class TestPromptActivations:
    def test_not_loaded_raises(self, service):
        with pytest.raises(RuntimeError, match="not loaded"):
            service.run_prompt_activations("hello", None, "16k", 10)

    @patch("backend.API.duckdb_instance.get_duckdb_client")
    @patch("backend.services.interpret_service.GemmaPytorchInference")
    @patch("backend.services.interpret_service.PromptExplorer")
    def test_returns_correct_structure(self, mock_explorer_cls, mock_inf_cls, mock_get_db, service):
        mock_inf_cls.return_value = _make_mock_wrapper()
        mock_get_db.return_value = _make_mock_db()
        service.load_model("google/gemma-3-4b-it")

        mock_prompt_result = _make_mock_prompt_result([(9, "16k")])
        mock_explorer_cls.return_value.run_prompt.return_value = mock_prompt_result

        result = service.run_prompt_activations("hello", [9], "16k", 10)

        # Legacy layers+width call resolves to a single (layer, width) spec
        config = mock_explorer_cls.call_args[0][0]
        assert config.effective_saes() == [(9, "16k")]

        assert isinstance(result, PromptActivationsResult)
        assert result.prompt == "hello"
        assert len(result.token_strings) == 1
        assert len(result.layers) == 1
        layer = result.layers[0]
        assert isinstance(layer, LayerActivationsResult)
        assert layer.layer == 9
        assert len(layer.tokens) == 1
        tok = layer.tokens[0]
        assert isinstance(tok, TokenFeaturesResult)
        assert tok.token == "hello"
        assert len(tok.features) == 1
        feat = tok.features[0]
        assert isinstance(feat, ActiveFeatureResult)
        assert feat.index == 42
        assert feat.activation == 3.5
        assert feat.label == "test label"

    @patch("backend.API.duckdb_instance.get_duckdb_client")
    @patch("backend.services.interpret_service.GemmaPytorchInference")
    @patch("backend.services.interpret_service.PromptExplorer")
    def test_multi_spec_single_forward_pass(
        self, mock_explorer_cls, mock_inf_cls, mock_get_db, service
    ):
        """Multiple SAEs — including two widths at the same layer — run in a
        single explorer call, with per-spec width-aware label lookups."""
        mock_inf_cls.return_value = _make_mock_wrapper()
        db = _make_mock_db()
        mock_get_db.return_value = db
        service.load_model("google/gemma-3-4b-it")

        specs = [(9, "16k"), (9, "65k"), (22, "16k")]
        explorer = MagicMock()
        explorer.run_prompt.return_value = _make_mock_prompt_result(specs)
        mock_explorer_cls.return_value = explorer

        # db_sae_id is deliberately set: with multiple specs the service must
        # ignore it and derive per-spec width-aware sae_ids instead.
        result = service.run_prompt_activations(
            "hello",
            None,
            "16k",
            10,
            db_model_id="gemma-3-4b-it",
            db_sae_id="9-gemmascope-2-res-16k",
            saes=specs,
        )

        # One explorer construction with the spec list, one forward pass
        assert mock_explorer_cls.call_count == 1
        config = mock_explorer_cls.call_args[0][0]
        assert config.saes == specs
        assert explorer.run_prompt.call_count == 1

        # Three layer entries, per-spec (layer, width)
        assert [(lr.layer, lr.width) for lr in result.layers] == specs

        # Per-spec DB label lookup with width-aware derived sae_ids
        lookup_calls = db.get_sae_feature_labels_batch.call_args_list
        assert {call.args[1] for call in lookup_calls} == {
            "9-gemmascope-2-res-16k",
            "9-gemmascope-2-res-65k",
            "22-gemmascope-2-res-16k",
        }
        assert all(call.args[0] == "gemma-3-4b-it" for call in lookup_calls)

    @patch("backend.API.duckdb_instance.get_duckdb_client")
    @patch("backend.services.interpret_service.GemmaPytorchInference")
    @patch("backend.services.interpret_service.PromptExplorer")
    def test_explorer_cached_for_identical_specs(
        self, mock_explorer_cls, mock_inf_cls, mock_get_db, service
    ):
        mock_inf_cls.return_value = _make_mock_wrapper()
        mock_get_db.return_value = _make_mock_db()
        service.load_model("google/gemma-3-4b-it")

        specs = [(9, "16k"), (9, "65k")]
        prompt_result = _make_mock_prompt_result(specs)

        def factory(config):
            inst = MagicMock()
            inst.config = config  # real config so the cache compare runs
            inst.run_prompt.return_value = prompt_result
            return inst

        mock_explorer_cls.side_effect = factory

        service.run_prompt_activations("hello", None, "16k", 10, saes=specs)
        service.run_prompt_activations("hello", None, "16k", 10, saes=specs)
        assert mock_explorer_cls.call_count == 1  # reused for identical specs

        service.run_prompt_activations("hello", None, "16k", 10, saes=[(9, "16k")])
        assert mock_explorer_cls.call_count == 2  # spec change → rebuild


# ---------------------------------------------------------------------------
# UC2: Steered generation
# ---------------------------------------------------------------------------


class TestSteeredGeneration:
    SINGLE_SPEC = [
        SteeringSpec(
            feature_index=42,
            layer=9,
            hook_type="resid_post",
            width="16k",
            strength=800.0,
        )
    ]

    MULTI_SPEC = [
        SteeringSpec(
            feature_index=42, layer=9, hook_type="resid_post", width="16k", strength=800.0
        ),
        SteeringSpec(
            feature_index=100, layer=17, hook_type="resid_post", width="16k", strength=400.0
        ),
    ]

    def test_not_loaded_raises(self, service):
        with pytest.raises(RuntimeError, match="not loaded"):
            service.generate_steered("hello", self.SINGLE_SPEC, output_len=64, temperature=None)

    @patch("backend.services.interpret_service.GemmaPytorchInference")
    @patch("backend.services.interpret_service.HookManager")
    def test_returns_baseline_and_steered(self, mock_hm_cls, mock_inf_cls, service):
        wrapper = _make_mock_wrapper()
        wrapper.generate.side_effect = ["baseline text", "steered text"]
        mock_inf_cls.return_value = wrapper
        service.load_model("google/gemma-3-4b-it")

        mock_manager = MagicMock()
        mock_manager.session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_manager.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_hm_cls.return_value = mock_manager

        result = service.generate_steered(
            "hello", self.SINGLE_SPEC, output_len=64, temperature=None
        )

        assert isinstance(result, SteeredGenerationResult)
        assert result.baseline_text == "baseline text"
        assert result.steered_text == "steered text"
        assert len(result.steering) == 1
        assert result.steering[0].feature_index == 42

    @patch("backend.services.interpret_service.GemmaPytorchInference")
    @patch("backend.services.interpret_service.HookManager")
    def test_multi_feature_steering(self, mock_hm_cls, mock_inf_cls, service):
        wrapper = _make_mock_wrapper()
        wrapper.generate.side_effect = ["baseline text", "multi-steered text"]
        mock_inf_cls.return_value = wrapper
        service.load_model("google/gemma-3-4b-it")

        mock_manager = MagicMock()
        mock_manager.session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_manager.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_hm_cls.return_value = mock_manager

        result = service.generate_steered("hello", self.MULTI_SPEC, output_len=64, temperature=None)

        assert isinstance(result, SteeredGenerationResult)
        assert result.steered_text == "multi-steered text"
        assert len(result.steering) == 2
        assert result.steering[0].feature_index == 42
        assert result.steering[1].feature_index == 100

    def test_empty_steering_raises(self, service):
        """At least one steering spec is required."""
        with pytest.raises((RuntimeError, ValueError)):
            service.generate_steered("hello", [], output_len=64, temperature=None)


# ---------------------------------------------------------------------------
# UC3: Prompt highlight (max-pooled activations)
# ---------------------------------------------------------------------------


class TestPromptHighlight:
    def test_not_loaded_raises(self, service):
        with pytest.raises(RuntimeError, match="not loaded"):
            service.run_prompt_highlight("hello", layer=9, width="16k", hook_type="resid_post")

    @patch("backend.services.interpret_service.GemmaPytorchInference")
    @patch("backend.services.interpret_service.HookManager")
    def test_returns_feature_activations(self, mock_hm_cls, mock_inf_cls, service):
        wrapper = _make_mock_wrapper()
        mock_inf_cls.return_value = wrapper
        service.load_model("google/gemma-3-4b-it")

        # Simulate prefill feature_acts: shape (1, 5, 16384) with a few nonzero features
        feature_acts = torch.zeros(1, 5, 100)  # use small d_sae for test
        feature_acts[0, 0, 10] = 3.0
        feature_acts[0, 2, 10] = 5.0  # max-pool should pick 5.0 for feature 10
        feature_acts[0, 1, 42] = 2.5
        feature_acts[0, 3, 99] = 1.0

        mock_record = MagicMock()
        mock_record.feature_acts = feature_acts

        mock_store = MagicMock()
        mock_store.prefill.return_value = mock_record

        mock_manager = MagicMock()
        mock_manager.session.return_value.__enter__ = MagicMock(return_value=mock_store)
        mock_manager.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_hm_cls.return_value = mock_manager

        result = service.run_prompt_highlight("hello", layer=9, width="16k", hook_type="resid_post")

        assert isinstance(result, list)
        assert all(isinstance(f, FeatureActivation) for f in result)

        # Check max-pooled values
        result_dict = {f.feature_index: f.activation for f in result}
        assert result_dict[10] == pytest.approx(5.0)
        assert result_dict[42] == pytest.approx(2.5)
        assert result_dict[99] == pytest.approx(1.0)
        assert len(result) == 3  # only 3 nonzero features


# ---------------------------------------------------------------------------
# Streaming generation: seed handling
# ---------------------------------------------------------------------------


def _mock_stream_event(text: str = "hi", *, done: bool = True):
    """A single generate_chat_stream event (matches TokenStreamEvent's shape)."""
    return MagicMock(token_index=0, token_id=1, text_delta=text, is_done=done)


class TestGenerateStreamSeed:
    """The seed makes sampling reproducible; a steered and a baseline call
    sharing one seed each get an identical fresh RNG state (see
    ``generate_stream`` docstring)."""

    @patch("backend.services.interpret_service.GemmaPytorchInference")
    def test_seed_seeds_torch_before_sampling(self, mock_cls, service):
        wrapper = _make_mock_wrapper()
        wrapper.generate_chat_stream.return_value = iter([_mock_stream_event()])
        mock_cls.return_value = wrapper
        service.load_model("google/gemma-3-4b-it")

        with patch("backend.services.interpret_service.torch.manual_seed") as mock_seed:
            service.generate_stream([("user", "hello")], "sid-seeded", seed=123)

        mock_seed.assert_called_once_with(123)
        wrapper.generate_chat_stream.assert_called_once()

    @patch("backend.services.interpret_service.GemmaPytorchInference")
    def test_no_seed_leaves_rng_untouched(self, mock_cls, service):
        wrapper = _make_mock_wrapper()
        wrapper.generate_chat_stream.return_value = iter([_mock_stream_event()])
        mock_cls.return_value = wrapper
        service.load_model("google/gemma-3-4b-it")

        with patch("backend.services.interpret_service.torch.manual_seed") as mock_seed:
            service.generate_stream([("user", "hello")], "sid-unseeded")

        mock_seed.assert_not_called()


# ---------------------------------------------------------------------------
# Checkpoint parsing (family dispatch)
# ---------------------------------------------------------------------------


class TestCheckpointParsing:
    """_parse_checkpoint returns (family, model_size, variant);
    _normalize_checkpoint canonicalises Gemma names and passes Qwen through."""

    # -- Gemma regression --

    def test_parse_gemma_it(self):
        assert InterpretService._parse_checkpoint("google/gemma-3-4b-it") == (
            "gemma",
            "4b",
            "it",
        )

    def test_parse_gemma_base_defaults_pt(self):
        assert InterpretService._parse_checkpoint("google/gemma-3-1b") == ("gemma", "1b", "pt")

    def test_parse_gemma_no_org(self):
        assert InterpretService._parse_checkpoint("gemma-3-12b-it") == ("gemma", "12b", "it")

    def test_normalize_gemma_appends_variant(self):
        assert (
            InterpretService._normalize_checkpoint("google/gemma-3-1b", "gemma", "1b", "pt")
            == "google/gemma-3-1b-pt"
        )

    def test_normalize_gemma_already_canonical(self):
        assert (
            InterpretService._normalize_checkpoint("google/gemma-3-4b-it", "gemma", "4b", "it")
            == "google/gemma-3-4b-it"
        )

    # -- Qwen --

    def test_parse_qwen_instruct(self):
        assert InterpretService._parse_checkpoint("Qwen/Qwen3-1.7B") == ("qwen", "1.7B", "it")

    def test_parse_qwen_base(self):
        assert InterpretService._parse_checkpoint("Qwen/Qwen3-1.7B-Base") == (
            "qwen",
            "1.7B",
            "pt",
        )

    def test_parse_qwen35(self):
        assert InterpretService._parse_checkpoint("Qwen/Qwen3.5-27B") == ("qwen", "27B", "it")

    def test_parse_qwen_unknown_size_raises(self):
        with pytest.raises(ValueError, match="0.6B"):
            InterpretService._parse_checkpoint("Qwen/Qwen3-0.6B")

    def test_normalize_qwen_passthrough(self):
        # The gemma rule would mangle this into "Qwen/gemma-3-1.7B-pt".
        assert (
            InterpretService._normalize_checkpoint("Qwen/Qwen3-1.7B-Base", "qwen", "1.7B", "pt")
            == "Qwen/Qwen3-1.7B-Base"
        )


# ---------------------------------------------------------------------------
# Qwen family: lifecycle + SAE config dispatch
# ---------------------------------------------------------------------------


QWEN_CHECKPOINT = "Qwen/Qwen3-1.7B"


class TestQwenLifecycle:
    @patch("backend.services.interpret_service.Qwen3Inference")
    def test_load_qwen_model(self, mock_cls, service):
        mock_cls.return_value = _make_mock_wrapper()
        status = service.load_model(QWEN_CHECKPOINT)
        assert status.loaded is True
        assert status.model_name == QWEN_CHECKPOINT  # not gemma-normalised
        assert status.variant == "it"
        assert status.model_size == "1.7B"
        assert mock_cls.call_args.args[0] == QWEN_CHECKPOINT

    @patch("backend.services.interpret_service.Qwen3Inference")
    def test_unload_resets_family(self, mock_cls, service):
        mock_cls.return_value = _make_mock_wrapper()
        service.load_model(QWEN_CHECKPOINT)
        assert service._family == "qwen"
        service.unload_model()
        assert service._family == "gemma"

    @patch("backend.services.interpret_service.Qwen3Inference")
    def test_neuronpedia_model_id(self, mock_cls, service):
        mock_cls.return_value = _make_mock_wrapper()
        service.load_model(QWEN_CHECKPOINT)
        # Must match QwenScopeSAEConfig.neuronpedia_model_id exactly — it keys
        # the bootstrap ingest, preset bundle, and checkpoint registry.
        assert service._neuronpedia_model_id == "qwen3-1.7B-base"

    @patch("backend.API.duckdb_instance.get_duckdb_client")
    @patch("backend.services.interpret_service.PromptExplorer")
    @patch("backend.services.interpret_service.Qwen3Inference")
    def test_prompt_activations_supported(self, mock_cls, mock_explorer_cls, mock_get_db, service):
        """Phase 1.5: per-token prompt activations now work for Qwen."""
        mock_cls.return_value = _make_mock_wrapper()
        mock_get_db.return_value = _make_mock_db()
        service.load_model(QWEN_CHECKPOINT)

        mock_explorer_cls.return_value.run_prompt.return_value = _make_mock_prompt_result(
            [(14, "32k")]
        )

        result = service.run_prompt_activations("hello", [14], "32k", 0, saes=[(14, "32k")])

        assert isinstance(result, PromptActivationsResult)
        assert [(lr.layer, lr.width) for lr in result.layers] == [(14, "32k")]

        # The explorer is handed a family-aware factory that builds Qwen configs.
        config = mock_explorer_cls.call_args[0][0]
        cfg = config.sae_config_factory(14, "32k")
        assert isinstance(cfg, QwenScopeSAEConfig)
        assert cfg.k == 50 and cfg.prefill_only is True

    @patch("backend.API.duckdb_instance.get_duckdb_client")
    @patch("backend.services.interpret_service.PromptExplorer")
    @patch("backend.services.interpret_service.Qwen3Inference")
    def test_prompt_activations_derives_qwen_sae_id(
        self, mock_cls, mock_explorer_cls, mock_get_db, service
    ):
        """Label lookup uses the width-aware qwen-scope id + qwen model id."""
        mock_cls.return_value = _make_mock_wrapper()
        db = _make_mock_db()
        mock_get_db.return_value = db
        service.load_model(QWEN_CHECKPOINT)

        mock_explorer_cls.return_value.run_prompt.return_value = _make_mock_prompt_result(
            [(14, "32k")]
        )

        service.run_prompt_activations("hello", [14], "32k", 0, saes=[(14, "32k")])

        lookup = db.get_sae_feature_labels_batch.call_args
        assert lookup.args[0] == "qwen3-1.7B-base"
        assert lookup.args[1] == "14-qwenscope-1-res-32k"


class TestMakeSaeConfig:
    @patch("backend.services.interpret_service.Qwen3Inference")
    def test_qwen_config(self, mock_cls, service):
        mock_cls.return_value = _make_mock_wrapper()
        service.load_model(QWEN_CHECKPOINT)
        cfg = service._make_sae_config(14, "32k", HookType.RESID_POST, "cpu")
        assert isinstance(cfg, QwenScopeSAEConfig)
        assert cfg.layer_index == 14
        assert cfg.width == "32k"
        assert cfg.k == 50
        assert cfg.model_size == "1.7B"
        assert cfg.prefill_only is False
        assert cfg.read_only is True

    @patch("backend.services.interpret_service.Qwen3Inference")
    def test_qwen_prefill_only_passthrough(self, mock_cls, service):
        mock_cls.return_value = _make_mock_wrapper()
        service.load_model(QWEN_CHECKPOINT)
        cfg = service._make_sae_config(14, "32k", HookType.RESID_POST, "cpu", prefill_only=True)
        assert cfg.prefill_only is True

    @patch("backend.services.interpret_service.Qwen3Inference")
    def test_qwen_rejects_non_residual_hook(self, mock_cls, service):
        mock_cls.return_value = _make_mock_wrapper()
        service.load_model(QWEN_CHECKPOINT)
        with pytest.raises(ValueError, match="RESID_POST"):
            service._make_sae_config(14, "32k", HookType.MLP_OUT, "cpu")

    @patch("backend.services.interpret_service.GemmaPytorchInference")
    def test_gemma_config_unchanged(self, mock_cls, service):
        mock_cls.return_value = _make_mock_wrapper()
        service.load_model("google/gemma-3-4b-it")
        cfg = service._make_sae_config(9, "16k", HookType.RESID_POST, "cpu")
        assert isinstance(cfg, GemmaScopeSAEConfig)
        assert cfg.model_size == "4b"
        assert cfg.variant == "it"
        assert cfg.read_only is True


# ---------------------------------------------------------------------------
# _find_prompt_token_range: chat-template trimming (Gemma + Qwen)
# ---------------------------------------------------------------------------


class TestFindPromptTokenRange:
    """Trimming the chat template down to the user-prompt tokens.

    Previously untested; Qwen support adds a family branch, so pin both.
    """

    @staticmethod
    def _service(family: str, variant: str, prepends_bos: bool):
        svc = InterpretService()
        wrapper = MagicMock()
        wrapper.prepends_bos = prepends_bos
        # Gemma path counts the tokenized prefix "<start_of_turn>user\n".
        wrapper.tokenize.return_value = [0, 1, 2, 3]  # len 4 → content starts at 4
        svc._wrapper = wrapper
        svc._family = family
        svc._variant = variant
        return svc

    def test_gemma_it(self):
        svc = self._service("gemma", "it", prepends_bos=True)
        toks = [
            "<bos>",
            "<start_of_turn>",
            "user",
            "\n",  # 4 prefix tokens (mocked len)
            "hello",
            "world",
            "<end_of_turn>",
            "\n",
            "<start_of_turn>",
            "model",
        ]
        assert svc._find_prompt_token_range(toks, "hello world") == (4, 6)

    def test_gemma_pt_skips_bos(self):
        svc = self._service("gemma", "pt", prepends_bos=True)
        toks = ["<bos>", "hello", "world"]
        assert svc._find_prompt_token_range(toks, "hello world") == (1, 3)

    def test_qwen_it(self):
        svc = self._service("qwen", "it", prepends_bos=False)
        toks = [
            "<|im_start|>",
            "user",
            "Ċ",  # standalone newline piece → skipped
            "hello",
            "world",
            "<|im_end|>",
            "Ċ",
            "<|im_start|>",
            "assistant",
        ]
        assert svc._find_prompt_token_range(toks, "hello world") == (3, 5)

    def test_qwen_it_skips_injected_system_turn(self):
        svc = self._service("qwen", "it", prepends_bos=False)
        toks = [
            "<|im_start|>",
            "system",
            "Ċ",
            "You",
            "<|im_end|>",
            "Ċ",
            "<|im_start|>",
            "user",
            "Ċ",
            "hi",
            "<|im_end|>",
        ]
        assert svc._find_prompt_token_range(toks, "hi") == (9, 10)

    def test_qwen_pt_no_bos(self):
        svc = self._service("qwen", "pt", prepends_bos=False)
        toks = ["hello", "world"]
        assert svc._find_prompt_token_range(toks, "hello world") == (0, 2)


# ---------------------------------------------------------------------------
# Wrapper token_strings(): per-token pieces aligned to the prefill sequence
# ---------------------------------------------------------------------------


class TestTokenStrings:
    """Both wrappers expose a symmetric token_strings(); __init__ (model load)
    is bypassed with object.__new__ so no weights are needed."""

    def test_gemma_uses_bos_and_sentencepiece_pieces(self):
        w = object.__new__(GemmaPytorchInference)
        w.model = MagicMock()
        w.model.tokenizer.encode.return_value = [2, 5, 9]
        w.model.tokenizer.sp_model.IdToPiece.side_effect = lambda t: f"p{t}"

        assert w.token_strings("hi") == ["p2", "p5", "p9"]
        # Must prepend BOS so length matches the Gemma prefill sequence.
        w.model.tokenizer.encode.assert_called_once_with("hi", bos=True)

    def test_qwen_uses_special_tokens_and_bpe_pieces(self):
        w = object.__new__(Qwen3Inference)
        w._tokenizer = MagicMock()
        w._tokenizer.encode.return_value = [10, 11]
        w._tokenizer.convert_ids_to_tokens.return_value = ["Ġa", "b"]

        assert w.token_strings("hi") == ["Ġa", "b"]
        # add_special_tokens=True keeps ChatML markers as discrete pieces and
        # matches generate_from_template's tokenization (Qwen has no BOS).
        w._tokenizer.encode.assert_called_once_with("hi", add_special_tokens=True)
        w._tokenizer.convert_ids_to_tokens.assert_called_once_with([10, 11])


# ---------------------------------------------------------------------------
# Qwen family: generation paths
# ---------------------------------------------------------------------------


class TestQwenGeneration:
    QWEN_SPEC = [
        SteeringSpec(
            feature_index=42,
            layer=14,
            hook_type="resid_post",
            width="32k",
            strength=20.0,
        )
    ]

    @patch("backend.services.interpret_service.Qwen3Inference")
    def test_stream_translates_model_role(self, mock_cls, service):
        wrapper = _make_mock_wrapper()
        wrapper.generate_chat_stream.return_value = iter([_mock_stream_event()])
        mock_cls.return_value = wrapper
        service.load_model(QWEN_CHECKPOINT)

        turns = [("user", "hi"), ("model", "yo"), ("user", "again")]
        service.generate_stream(turns, "sid-qwen")

        call = wrapper.generate_chat_stream.call_args
        assert call.args[0] == [("user", "hi"), ("assistant", "yo"), ("user", "again")]

    @patch("backend.services.interpret_service.Qwen3Inference")
    def test_stream_thinking_default_off(self, mock_cls, service):
        wrapper = _make_mock_wrapper()
        wrapper.generate_chat_stream.return_value = iter([_mock_stream_event()])
        mock_cls.return_value = wrapper
        service.load_model(QWEN_CHECKPOINT)

        service.generate_stream([("user", "hi")], "sid-qwen")
        assert wrapper.generate_chat_stream.call_args.kwargs.get("enable_thinking") is False

    @patch("backend.services.interpret_service.Qwen3Inference")
    def test_stream_thinking_toggle_on(self, mock_cls, service):
        wrapper = _make_mock_wrapper()
        wrapper.generate_chat_stream.return_value = iter([_mock_stream_event()])
        mock_cls.return_value = wrapper
        service.load_model(QWEN_CHECKPOINT)

        service.generate_stream([("user", "hi")], "sid-qwen", enable_thinking=True)
        assert wrapper.generate_chat_stream.call_args.kwargs.get("enable_thinking") is True

    @patch("backend.services.interpret_service.GemmaPytorchInference")
    def test_gemma_stream_untouched(self, mock_cls, service):
        """Gemma keeps its role convention and never receives the qwen kwarg."""
        wrapper = _make_mock_wrapper()
        wrapper.generate_chat_stream.return_value = iter([_mock_stream_event()])
        mock_cls.return_value = wrapper
        service.load_model("google/gemma-3-4b-it")

        turns = [("user", "hi"), ("model", "yo")]
        service.generate_stream(turns, "sid-gemma")

        call = wrapper.generate_chat_stream.call_args
        assert call.args[0] == turns
        assert "enable_thinking" not in call.kwargs

    @patch("backend.services.interpret_service.Qwen3Inference")
    @patch("backend.services.interpret_service.HookManager")
    def test_generate_steered_thinking_off(self, mock_hm_cls, mock_inf_cls, service):
        wrapper = _make_mock_wrapper()
        wrapper.generate.side_effect = ["baseline text", "steered text"]
        mock_inf_cls.return_value = wrapper
        service.load_model(QWEN_CHECKPOINT)

        mock_manager = MagicMock()
        mock_manager.session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_manager.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_hm_cls.return_value = mock_manager

        result = service.generate_steered("hello", self.QWEN_SPEC, output_len=64, temperature=None)

        assert result.baseline_text == "baseline text"
        assert result.steered_text == "steered text"
        # Both generate calls suppress <think> blocks (chat default is off).
        for call in wrapper.generate.call_args_list:
            assert call.kwargs.get("enable_thinking") is False
        # The steering session was built with a Qwen-scope config.
        sae_arg = mock_manager.add_sae.call_args.args[0]
        assert isinstance(sae_arg, QwenScopeSAEConfig)
        assert sae_arg.k == 50

    @patch("backend.services.interpret_service.Qwen3Inference")
    @patch("backend.services.interpret_service.HookManager")
    def test_prompt_highlight_uses_qwen_config(self, mock_hm_cls, mock_inf_cls, service):
        wrapper = _make_mock_wrapper()
        mock_inf_cls.return_value = wrapper
        service.load_model(QWEN_CHECKPOINT)

        feature_acts = torch.zeros(1, 3, 50)
        feature_acts[0, 1, 7] = 2.0
        mock_record = MagicMock()
        mock_record.feature_acts = feature_acts
        mock_store = MagicMock()
        mock_store.prefill.return_value = mock_record

        mock_manager = MagicMock()
        mock_manager.session.return_value.__enter__ = MagicMock(return_value=mock_store)
        mock_manager.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_hm_cls.return_value = mock_manager

        result = service.run_prompt_highlight(
            "hello", layer=14, width="32k", hook_type="resid_post"
        )

        sae_arg = mock_manager.add_sae.call_args.args[0]
        assert isinstance(sae_arg, QwenScopeSAEConfig)
        assert sae_arg.prefill_only is True
        assert wrapper.generate.call_args.kwargs.get("enable_thinking") is False
        assert len(result) == 1
        assert result[0].feature_index == 7
