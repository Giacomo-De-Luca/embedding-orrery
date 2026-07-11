"""Guard the read-only demo mode (ORRERY_READ_ONLY=1).

Public demo deployments (e.g. the HuggingFace Space) expose the GraphQL
endpoint to anyone, so hiding write UI is not enough: the schema itself must
reject mutations. ``ReadOnlyExtension`` short-circuits any mutation operation
before execution, and ``main.py`` skips mounting the ``/upload`` router.

The flag is read per-operation so these tests can flip it with monkeypatch;
the upload-router test runs in a subprocess because router mounting happens
at import time (same pattern as ``test_torch_free_import.py``).
"""

import asyncio
import os
import subprocess
import sys

import strawberry

from backend.API.read_only import (
    READ_ONLY_ENV,
    READ_ONLY_MESSAGE,
    ReadOnlyExtension,
    is_read_only,
)


@strawberry.type
class _Query:
    @strawberry.field
    def ping(self) -> str:
        return "pong"


@strawberry.type
class _Mutation:
    @strawberry.mutation
    def poke(self) -> str:
        return "poked"


_toy_schema = strawberry.Schema(query=_Query, mutation=_Mutation, extensions=[ReadOnlyExtension])


def test_is_read_only_reflects_env(monkeypatch):
    monkeypatch.delenv(READ_ONLY_ENV, raising=False)
    assert not is_read_only()
    # common truthy spellings all enable the gate (fail-closed for hand-set envs)
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv(READ_ONLY_ENV, value)
        assert is_read_only(), value
    for value in ("0", "false", "", "off"):
        monkeypatch.setenv(READ_ONLY_ENV, value)
        assert not is_read_only(), value


def test_mutation_blocked_when_read_only(monkeypatch):
    monkeypatch.setenv(READ_ONLY_ENV, "1")
    result = _toy_schema.execute_sync("mutation { poke }")
    assert result.data is None
    assert result.errors
    assert any(READ_ONLY_MESSAGE in str(e) for e in result.errors)


def test_query_allowed_when_read_only(monkeypatch):
    monkeypatch.setenv(READ_ONLY_ENV, "1")
    result = _toy_schema.execute_sync("query { ping }")
    assert result.errors is None
    assert result.data == {"ping": "pong"}


def test_mutation_executes_when_flag_unset(monkeypatch):
    monkeypatch.delenv(READ_ONLY_ENV, raising=False)
    result = _toy_schema.execute_sync("mutation { poke }")
    assert result.errors is None
    assert result.data == {"poke": "poked"}


def test_app_schema_blocks_mutations_when_read_only(monkeypatch):
    """The extension must actually be installed on the real schema."""
    monkeypatch.setenv(READ_ONLY_ENV, "1")
    from backend.API import schema

    # Short-circuited before execution, so no client/DB is ever touched.
    result = schema.execute_sync('mutation { deleteCollection(collectionName: "x") }')
    assert result.data is None
    assert result.errors
    assert any(READ_ONLY_MESSAGE in str(e) for e in result.errors)


def test_app_schema_allows_queries_when_read_only(monkeypatch):
    monkeypatch.setenv(READ_ONLY_ENV, "1")
    from backend.API import schema

    result = schema.execute_sync("query { __typename }")
    assert result.errors is None
    assert result.data == {"__typename": "Query"}


def test_generate_stream_blocked_when_read_only(monkeypatch):
    """The streaming subscription refuses before touching the interpret
    service (whose lazy torch import alone costs hundreds of MB of RSS)."""
    monkeypatch.setenv(READ_ONLY_ENV, "1")
    from backend.API.subscriptions import Subscription
    from backend.API.types import ChatTurnInput, GenerateStreamInput

    async def first_chunk():
        gen = Subscription().generate_stream(
            GenerateStreamInput(turns=[ChatTurnInput(role="user", content="hi")])
        )
        return await anext(gen)

    chunk = asyncio.run(first_chunk())
    assert chunk.done
    assert chunk.error == READ_ONLY_MESSAGE


_UPLOAD_ROUTE_SNIPPET = """
import sys

from interpretability_backend.backend.main import app

has_upload = any(getattr(r, "path", None) == "/upload" for r in app.routes)
sys.exit(0 if has_upload == {expected} else 1)
"""


def _upload_route_present(read_only_env: dict[str, str], expected: bool) -> bool:
    env = {**os.environ, **read_only_env}
    result = subprocess.run(
        [sys.executable, "-c", _UPLOAD_ROUTE_SNIPPET.format(expected=expected)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    return result.returncode == 0


def test_upload_router_absent_when_read_only():
    assert _upload_route_present({"ORRERY_READ_ONLY": "1"}, expected=False)


def test_upload_router_present_by_default():
    assert _upload_route_present({"ORRERY_READ_ONLY": ""}, expected=True)
