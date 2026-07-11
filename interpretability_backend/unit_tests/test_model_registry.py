"""Unit tests for the torch-free model-id ↔ checkpoint registry.

Guards the derivation used by the residual-norm profiler to key its JSON by
the stored/frontend model id (not the checkpoint basename), so the steering
hint matches the id the frontend store passes in.
"""

import pytest

from backend.services.model_registry import (
    MODEL_ID_TO_CHECKPOINT,
    checkpoint_for_model_id,
    model_id_for_checkpoint,
)


class TestCheckpointForModelId:
    def test_registered_qwen_id(self):
        assert checkpoint_for_model_id("qwen3-1.7B-base") == "Qwen/Qwen3-1.7B"

    def test_gemma_rule_fallback(self):
        assert checkpoint_for_model_id("gemma-3-4b-it") == "google/gemma-3-4b-it"

    def test_org_prefixed_passthrough(self):
        assert checkpoint_for_model_id("org/model") == "org/model"

    def test_unregistered_qwen_id_fails_fast(self):
        # The gemma rule would mint a nonexistent "google/qwen…" path → HF 404.
        with pytest.raises(ValueError, match="MODEL_ID_TO_CHECKPOINT"):
            checkpoint_for_model_id("qwen3.5-2B-base")


class TestModelIdForCheckpoint:
    def test_registered_qwen_checkpoint(self):
        # Frontend/DuckDB id is "qwen3-1.7B-base", NOT the basename "Qwen3-1.7B".
        assert model_id_for_checkpoint("Qwen/Qwen3-1.7B") == "qwen3-1.7B-base"

    def test_gemma_basename_fallback(self):
        assert model_id_for_checkpoint("google/gemma-3-4b-it") == "gemma-3-4b-it"
        assert model_id_for_checkpoint("google/gemma-3-1b-pt") == "gemma-3-1b-pt"

    def test_round_trips_registered_ids(self):
        for model_id, checkpoint in MODEL_ID_TO_CHECKPOINT.items():
            assert model_id_for_checkpoint(checkpoint) == model_id
            assert checkpoint_for_model_id(model_id) == checkpoint
