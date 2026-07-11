"""Tests for direction-vector steering support in InterpretService.

The runtime supports two kinds of steering specs:
  - SAE feature: identified by (feature_index, layer, hook_type, width)
  - Raw direction vector: identified by ``direction_name`` (e.g. "refusal").

These tests cover the direction-name path: registry lookup, file caching,
HookManager wiring (SteeringOp built with `vector=`, not `feature_index=`),
and error on unknown names. Uses mocked HookManager / wrappers — no GPU
or model weights required.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from backend.services.interpret_service import (
    DIRECTION_REGISTRY,
    DirectionPreset,
    InterpretService,
    SteeringSpec,
)
from interpret.sae.sae_config import HookType
from interpret.sae.steering import SteeringMode

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_directions(tmp_path: Path, monkeypatch):
    """Point InterpretService at a tmp directions/ folder with fake .pt files."""
    directions_dir = tmp_path / "directions"
    directions_dir.mkdir()

    # Refusal: 1-D tensor, distinct sentinel value so we can verify it survives
    refusal_vec = torch.arange(2560, dtype=torch.float32) + 100.0
    torch.save(refusal_vec, directions_dir / "refusal_direction.pt")

    # Poetry: 1-D tensor
    poetry_vec = torch.arange(2560, dtype=torch.float32)
    torch.save(poetry_vec, directions_dir / "poetry_direction.pt")

    monkeypatch.setattr(
        "backend.services.interpret_service.DIRECTIONS_DIR", directions_dir
    )
    return directions_dir


@pytest.fixture
def service_with_mock_model(isolated_directions):
    """Service with a mocked wrapper that pretends a model is loaded."""
    svc = InterpretService()
    wrapper = MagicMock()
    wrapper.device = torch.device("cpu")
    wrapper.model.model.layers = MagicMock(spec_set=["__len__", "__getitem__"])
    wrapper.model.model.layers.__len__ = MagicMock(return_value=34)
    wrapper.generate.return_value = "mock response"
    wrapper.generate_from_template.return_value = "mock response"
    svc._wrapper = wrapper
    svc._model_name = "google/gemma-3-4b-it"
    svc._model_size = "4b"
    svc._variant = "it"
    return svc


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_has_expected_presets():
    """The two presets we care about for gemma-3-4b-it are registered."""
    assert "refusal" in DIRECTION_REGISTRY
    assert "poetry" in DIRECTION_REGISTRY
    refusal = DIRECTION_REGISTRY["refusal"]
    poetry = DIRECTION_REGISTRY["poetry"]
    assert isinstance(refusal, DirectionPreset)
    assert isinstance(poetry, DirectionPreset)
    assert refusal.model_id == "gemma-3-4b-it"
    assert poetry.model_id == "gemma-3-4b-it"


# ---------------------------------------------------------------------------
# Direction loading + caching
# ---------------------------------------------------------------------------


def test_load_direction_returns_1d_tensor(service_with_mock_model):
    """Loader returns the 1-D vector as float32 on the wrapper's device."""
    vec = service_with_mock_model._load_direction("refusal")
    assert vec.ndim == 1
    assert vec.shape == (2560,)
    assert vec.dtype == torch.float32
    # Sentinel value survived
    assert torch.allclose(vec[:5], torch.tensor([100.0, 101.0, 102.0, 103.0, 104.0]))


def test_load_direction_is_cached(service_with_mock_model, isolated_directions):
    """Second load doesn't re-read the file."""
    svc = service_with_mock_model
    svc._load_direction("refusal")
    assert "refusal" in svc._direction_cache

    # Delete the file: second call should still succeed (cached)
    (isolated_directions / "refusal_direction.pt").unlink()
    vec = svc._load_direction("refusal")
    assert vec.shape == (2560,)


def test_load_direction_unknown_name(service_with_mock_model):
    """Asking for an unregistered direction is a clean error."""
    with pytest.raises(KeyError, match="unknown direction"):
        service_with_mock_model._load_direction("nonexistent")


def test_load_direction_requires_model(isolated_directions):
    """Without a loaded model we can't resolve the target device."""
    svc = InterpretService()
    with pytest.raises(RuntimeError, match="not loaded"):
        svc._load_direction("refusal")


def test_load_direction_rejects_wrong_model(service_with_mock_model):
    """Applying a 4b-bound direction to a different model is a clean error."""
    svc = service_with_mock_model
    svc._model_size = "12b"  # _neuronpedia_model_id derives "gemma-3-12b-it"
    with pytest.raises(ValueError, match="bound to model"):
        svc._load_direction("refusal")


# ---------------------------------------------------------------------------
# Wiring into generate_steered: direction path skips add_sae and uses vector=
# ---------------------------------------------------------------------------


def test_generate_steered_direction_path_builds_vector_op(service_with_mock_model):
    """A direction spec → SteeringOp(vector=…), no add_sae call."""
    svc = service_with_mock_model

    mock_manager = MagicMock()
    mock_manager.session.return_value.__enter__ = MagicMock(return_value=None)
    mock_manager.session.return_value.__exit__ = MagicMock(return_value=False)

    with patch(
        "backend.services.interpret_service.HookManager", return_value=mock_manager
    ):
        specs = [
            SteeringSpec(
                feature_index=0,  # unused for direction path
                layer=0,
                hook_type="resid_post",
                width="",
                strength=1.5,
                direction_name="refusal",
            )
        ]
        svc.generate_steered("hello", specs, output_len=8, temperature=None)

    # No SAE registered for direction-only spec
    assert mock_manager.add_sae.call_count == 0
    # One steering op added
    assert mock_manager.add_steering.call_count == 1
    op = mock_manager.add_steering.call_args.args[0]
    # Direction-based op: uses raw vector, application layer comes from registry
    assert op.feature_index is None
    assert op.vector is not None
    assert op.vector.shape == (2560,)
    assert op.layer_index == DIRECTION_REGISTRY["refusal"].layer
    assert op.hook_type == HookType.RESID_POST
    assert op.mode == SteeringMode.ADDITIVE
    assert op.strength == 1.5


def test_generate_steered_mixed_specs(service_with_mock_model):
    """One SAE feature spec + one direction spec → both correctly wired."""
    svc = service_with_mock_model

    mock_manager = MagicMock()
    mock_manager.session.return_value.__enter__ = MagicMock(return_value=None)
    mock_manager.session.return_value.__exit__ = MagicMock(return_value=False)

    with patch(
        "backend.services.interpret_service.HookManager", return_value=mock_manager
    ):
        specs = [
            SteeringSpec(
                feature_index=197,
                layer=9,
                hook_type="resid_post",
                width="16k",
                strength=800.0,
            ),
            SteeringSpec(
                feature_index=0,
                layer=0,
                hook_type="resid_post",
                width="",
                strength=-1.0,
                direction_name="refusal",
            ),
        ]
        svc.generate_steered("hello", specs, output_len=8, temperature=None)

    # SAE registered once for the feature spec; direction spec adds no SAE.
    assert mock_manager.add_sae.call_count == 1
    assert mock_manager.add_steering.call_count == 2

    feature_op = mock_manager.add_steering.call_args_list[0].args[0]
    direction_op = mock_manager.add_steering.call_args_list[1].args[0]

    assert feature_op.feature_index == 197
    assert feature_op.vector is None
    assert direction_op.feature_index is None
    assert direction_op.vector is not None
