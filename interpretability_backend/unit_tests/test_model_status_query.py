"""``modelStatus`` answers without creating the interpretability service.

Creating the service imports torch, transformers and the interpret toolkit
for the life of the process; the frontend polls this query, so it must only
read an existing service. The import itself is guarded in a subprocess by
``test_torch_free_import.py``.
"""

from types import SimpleNamespace

import pytest

from backend.API import interpret_instance, schema

STATUS_QUERY = "{ modelStatus { loaded modelName device variant modelSize } }"


@pytest.fixture
def no_service_creation(monkeypatch):
    def create():
        raise AssertionError("modelStatus must not create the interpretability service")

    monkeypatch.setattr(interpret_instance, "get_interpret_service", create)


def test_status_without_a_service_reports_not_loaded(monkeypatch, no_service_creation):
    monkeypatch.setattr(interpret_instance, "_interpret_service", None)

    result = schema.execute_sync(STATUS_QUERY)

    assert result.errors is None
    assert result.data["modelStatus"] == {
        "loaded": False,
        "modelName": None,
        "device": None,
        "variant": None,
        "modelSize": None,
    }
    assert interpret_instance._interpret_service is None


def test_status_reads_an_existing_service(monkeypatch, no_service_creation):
    status = SimpleNamespace(
        loaded=True, model_name="google/gemma-3-4b-it", device="mps", variant="it", model_size="4b"
    )
    monkeypatch.setattr(
        interpret_instance, "_interpret_service", SimpleNamespace(get_status=lambda: status)
    )

    result = schema.execute_sync(STATUS_QUERY)

    assert result.errors is None
    assert result.data["modelStatus"]["loaded"] is True
    assert result.data["modelStatus"]["modelName"] == "google/gemma-3-4b-it"
