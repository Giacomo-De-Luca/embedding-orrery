"""Guard the orjson response encoder (backend/utils/json_encoding.py).

GraphQL responses are encoded with orjson for speed and to skip the
intermediate ``str`` copy. Anything the stdlib encoded before must still
encode (orjson refusals fall back to ``json.dumps``), and the deliberate
differences — UTF-8 instead of escapes, NaN as ``null`` — must hold.
"""

import json
import math

import numpy as np
import pytest
import strawberry
from fastapi import FastAPI
from fastapi.testclient import TestClient
from strawberry.fastapi import GraphQLRouter

from backend.API.graphql_router import OrreryGraphQLRouter
from backend.utils.json_encoding import encode_json_bytes


def _roundtrip(data: object) -> object:
    return json.loads(encode_json_bytes(data))


class TestEncodeJsonBytes:
    def test_returns_bytes_with_stdlib_semantics(self):
        data = {
            "ids": ["a", "b"],
            "nested": {"x": [1, 2.5, None, True, False], "empty": {}},
            "unicode": "café 🪐 ガ",
        }
        encoded = encode_json_bytes(data)
        assert isinstance(encoded, bytes)
        assert json.loads(encoded) == json.loads(json.dumps(data))

    def test_non_ascii_is_written_as_utf8(self):
        assert encode_json_bytes({"s": "café"}) == '{"s":"café"}'.encode()

    def test_non_string_keys_follow_stdlib_conversion(self):
        data = {1: "a", None: "b", 2.5: "c", False: "d"}
        assert _roundtrip(data) == json.loads(json.dumps(data))

    def test_numpy_values_are_serialized(self):
        data = {
            "f64": np.float64(1.5),
            "f32": np.float32(0.25),
            "i64": np.int64(7),
            "arr": np.arange(3, dtype=np.float32),
        }
        assert _roundtrip(data) == {"f64": 1.5, "f32": 0.25, "i64": 7, "arr": [0.0, 1.0, 2.0]}

    def test_non_finite_floats_become_null(self):
        # json.dumps would emit bare NaN/Infinity, which JSON.parse rejects.
        assert _roundtrip({"nan": math.nan, "inf": math.inf}) == {"nan": None, "inf": None}

    def test_integers_beyond_64_bits_fall_back_to_stdlib(self):
        assert _roundtrip({"big": 2**70}) == {"big": 2**70}

    def test_lone_surrogates_fall_back_to_stdlib(self):
        assert _roundtrip({"s": "\ud800"}) == {"s": "\ud800"}

    def test_unsupported_types_still_raise_type_error(self):
        with pytest.raises(TypeError):
            encode_json_bytes({"obj": object()})


@strawberry.type
class _Query:
    @strawberry.field
    def greeting(self) -> str:
        return "café"

    @strawberry.field
    def broken(self) -> str:
        raise ValueError("boom")


def _client(router_cls: type[GraphQLRouter] = OrreryGraphQLRouter) -> TestClient:
    app = FastAPI()
    app.include_router(router_cls(strawberry.Schema(query=_Query)), prefix="/graphql")
    return TestClient(app)


class TestRouter:
    def test_responses_are_compact_orjson(self):
        response = _client().post("/graphql", json={"query": "{ greeting }"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        assert response.content == '{"data":{"greeting":"café"}}'.encode()

    @pytest.mark.parametrize(
        "request_kwargs",
        [
            {"json": {"query": "{ greeting }"}},
            {"json": {"query": "{ broken }"}},
            {"json": {"query": "{ doesNotExist }"}},
            {"content": b"{not json", "headers": {"content-type": "application/json"}},
        ],
        ids=["ok", "resolver-error", "validation-error", "malformed-body"],
    )
    def test_status_and_body_match_the_stock_router(self, request_kwargs):
        ours = _client().post("/graphql", **request_kwargs)
        stock = _client(GraphQLRouter).post("/graphql", **request_kwargs)
        assert ours.status_code == stock.status_code
        assert _body(ours) == _body(stock)


def _body(response) -> object:
    """Parsed JSON for JSON responses, raw text otherwise (e.g. a 400 message)."""
    if response.headers.get("content-type", "").startswith("application/json"):
        return response.json()
    return response.text
