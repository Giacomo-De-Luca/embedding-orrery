"""GraphQL round trip of ``runPromptActivations``: the hook site of each SAE
spec reaches the interpret service, and each result layer reports its site.

The interpret service is faked (no torch, no model), like the other
resolver tests.
"""

import asyncio
from types import SimpleNamespace

from backend.API import mutations, schema

QUERY = """
    mutation Run($input: RunPromptActivationsInput!) {
        runPromptActivations(input: $input) {
            layers { layer width hookType tokens { token } }
            error
        }
    }
"""


class _FakeInterpretService:
    """Records the SAE specs it is called with; returns one empty layer per spec."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self.saes = None

    def run_prompt_activations(
        self,
        prompt,
        layers,
        width,
        top_k,
        model_id,
        sae_id,
        skip_chat_template,
        filter_mode,
        saes,
    ):
        self.saes = saes
        return SimpleNamespace(
            prompt=prompt,
            token_strings=[],
            layers=[
                SimpleNamespace(layer=layer, width=w, hook_type=hook_type, tokens=[])
                for layer, w, hook_type in saes
            ],
        )


def _run(monkeypatch, saes: list[dict]) -> tuple[_FakeInterpretService, dict]:
    service = _FakeInterpretService()
    monkeypatch.setattr(mutations, "get_interpret_service", lambda: service)
    result = asyncio.run(
        schema.execute(QUERY, variable_values={"input": {"prompt": "hi", "saes": saes}})
    )
    assert result.errors is None, result.errors
    return service, result.data["runPromptActivations"]


def test_hook_site_reaches_the_service_and_comes_back(monkeypatch):
    service, data = _run(
        monkeypatch,
        [
            {"layer": 9, "width": "16k", "hookType": "RESID_POST"},
            {"layer": 9, "width": "16k", "hookType": "MLP_OUT"},
        ],
    )

    assert service.saes == [(9, "16k", "resid_post"), (9, "16k", "mlp_out")]
    assert data["error"] is None
    assert [(lr["layer"], lr["width"], lr["hookType"]) for lr in data["layers"]] == [
        (9, "16k", "RESID_POST"),
        (9, "16k", "MLP_OUT"),
    ]


def test_hook_site_defaults_to_the_residual_stream(monkeypatch):
    service, data = _run(monkeypatch, [{"layer": 22, "width": "16k"}])

    assert service.saes == [(22, "16k", "resid_post")]
    assert data["layers"][0]["hookType"] == "RESID_POST"
