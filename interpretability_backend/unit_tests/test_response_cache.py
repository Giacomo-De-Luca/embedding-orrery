"""Guard the read-only GraphQL response cache (backend/utils/response_cache.py).

On a read-only deployment (the HF Space) collection data never changes, so the
first response to an allow-listed query is stored gzipped and replayed for
identical requests without re-running DuckDB, GraphQL completion or JSON
encoding. These tests pin what may be cached (allow-listed query root fields,
no errors, read-only only), how hits are served (stored gzip, or plain JSON
streamed to clients without gzip), and that nothing else changes behaviour.
"""

import asyncio
import gzip
import random

import httpx
import pytest
import strawberry
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from backend.API import queries as query_module, schema as real_schema
from backend.API.graphql_router import OrreryGraphQLRouter
from backend.API.read_only import READ_ONLY_ENV, ReadOnlyExtension
from backend.API.response_cache_extension import (
    ResponseCacheMarkerExtension,
    response_cache_active,
)
from backend.main import app as main_app, graphql_app as main_graphql_app
from backend.utils.http_compression import CompressionSettings, JsonGzipMiddleware
from backend.utils.resource_paths import PACKAGE_ROOT
from backend.utils.response_cache import (
    CACHE_STATUS_HEADER,
    CACHEABLE_HEADER,
    CachedResponse,
    GraphQLResponseCacheMiddleware,
    KeyedLocks,
    ResponseCache,
    ResponseCacheSettings,
    iter_gunzip,
    load_response_cache_settings,
    request_cache_key,
)

PAYLOAD_SIZE = 5_000  # resolver output large enough to be worth compressing
CALLS: dict[str, int] = {}


def _count(name: str) -> None:
    CALLS[name] = CALLS.get(name, 0) + 1


@strawberry.type
class _Query:
    @strawberry.field
    def collection(self, name: str) -> str:
        _count("collection")
        if name == "boom":
            raise ValueError("boom")
        return f"{name}:" + "x" * PAYLOAD_SIZE

    @strawberry.field
    async def slow_collection(self, name: str) -> str:
        _count("slow_collection")
        await asyncio.sleep(0.05)
        return f"{name}:" + "y" * PAYLOAD_SIZE

    @strawberry.field
    def other(self) -> str:
        _count("other")
        return "z" * PAYLOAD_SIZE


@strawberry.type
class _Mutation:
    @strawberry.mutation
    def touch(self) -> bool:
        _count("touch")
        return True


TEST_SETTINGS = ResponseCacheSettings(root_fields=("collection", "slowCollection"))
QUERY = "query GetCollection($name: String!) { collection(name: $name) }"


def _make_app(
    settings: ResponseCacheSettings = TEST_SETTINGS,
    schema: strawberry.Schema | None = None,
) -> FastAPI:
    """App wired like backend/main.py: cache innermost, gzip outermost."""
    schema = schema or strawberry.Schema(
        query=_Query,
        mutation=_Mutation,
        extensions=[ReadOnlyExtension, ResponseCacheMarkerExtension(settings=settings)],
    )
    app = FastAPI()
    app.add_middleware(
        GraphQLResponseCacheMiddleware,
        settings=settings,
        is_active=lambda: response_cache_active(settings),
    )
    app.add_middleware(JsonGzipMiddleware, settings=CompressionSettings())
    app.include_router(OrreryGraphQLRouter(schema), prefix="/graphql")
    return app


def _post(
    client: TestClient,
    name: str = "a",
    *,
    query: str = QUERY,
    encoding: str = "gzip",
) -> httpx.Response:
    return client.post(
        "/graphql",
        json={"query": query, "variables": {"name": name}},
        headers={"accept-encoding": encoding},
    )


@pytest.fixture(autouse=True)
def _reset_calls():
    CALLS.clear()


@pytest.fixture
def read_only(monkeypatch):
    monkeypatch.setenv(READ_ONLY_ENV, "1")


@pytest.fixture
def client(read_only):
    return TestClient(_make_app())


class TestCachedQueries:
    def test_identical_query_is_built_once(self, client):
        first = _post(client)
        second = _post(client)

        assert CALLS["collection"] == 1
        expected = {"data": {"collection": "a:" + "x" * PAYLOAD_SIZE}}
        assert first.json() == second.json() == expected
        assert first.headers[CACHE_STATUS_HEADER] == "miss"
        assert second.headers[CACHE_STATUS_HEADER] == "hit"
        assert CACHEABLE_HEADER not in first.headers
        assert CACHEABLE_HEADER not in second.headers

    def test_gzip_clients_receive_the_stored_gzip_body(self, client):
        first = _post(client)
        second = _post(client)

        for response in (first, second):
            assert response.headers["content-encoding"] == "gzip"
            assert "accept-encoding" in response.headers["vary"].lower()

    def test_identity_clients_receive_plain_json(self, read_only):
        # A tiny stream chunk forces the hit to arrive in many body messages.
        settings = ResponseCacheSettings(root_fields=("collection",), stream_chunk_bytes=64)
        client = TestClient(_make_app(settings))
        _post(client, encoding="gzip")  # stored by a gzip client
        miss_plain = _post(client, name="b", encoding="identity")
        hit_plain = _post(client, encoding="identity")

        assert CALLS["collection"] == 2
        assert miss_plain.headers[CACHE_STATUS_HEADER] == "miss"
        assert hit_plain.headers[CACHE_STATUS_HEADER] == "hit"
        for response in (miss_plain, hit_plain):
            assert "content-encoding" not in response.headers
            assert int(response.headers["content-length"]) == len(response.content)
        assert hit_plain.json() == {"data": {"collection": "a:" + "x" * PAYLOAD_SIZE}}

    def test_variables_are_part_of_the_key(self, client):
        _post(client, "a")
        _post(client, "b")
        _post(client, "a")
        assert CALLS["collection"] == 2

    def test_concurrent_identical_misses_build_once(self, read_only):
        app = _make_app()
        query = "query Slow($name: String!) { slowCollection(name: $name) }"

        async def fire() -> list[httpx.Response]:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                return await asyncio.gather(
                    *(
                        http.post("/graphql", json={"query": query, "variables": {"name": "a"}})
                        for _ in range(5)
                    )
                )

        responses = asyncio.run(fire())
        assert CALLS["slow_collection"] == 1
        assert len({r.content for r in responses}) == 1
        statuses = sorted(r.headers[CACHE_STATUS_HEADER] for r in responses)
        assert statuses == ["hit"] * 4 + ["miss"]


class TestNotCached:
    def _assert_built_twice(self, client: TestClient, field: str, **kwargs) -> None:
        responses = [_post(client, **kwargs) for _ in range(2)]
        assert CALLS.get(field, 0) == 2
        for response in responses:
            assert CACHE_STATUS_HEADER not in response.headers
            assert CACHEABLE_HEADER not in response.headers

    def test_resolver_errors(self, client):
        self._assert_built_twice(client, "collection", name="boom")

    def test_root_fields_outside_the_allow_list(self, client):
        self._assert_built_twice(client, "other", query="{ other }")

    def test_aliases_cannot_smuggle_other_fields(self, client):
        self._assert_built_twice(client, "other", query="{ collection: other }")

    def test_mixed_root_fields(self, client):
        self._assert_built_twice(client, "other", query='{ collection(name: "a") other }')

    def test_mutations(self, client):
        responses = [
            client.post("/graphql", json={"query": "mutation { touch }"}) for _ in range(2)
        ]
        assert CALLS.get("touch", 0) == 0  # blocked by the read-only gate
        for response in responses:
            assert response.json()["errors"]
            assert CACHE_STATUS_HEADER not in response.headers

    def test_get_queries_but_the_marker_is_stripped(self, client):
        for _ in range(2):
            response = client.get("/graphql", params={"query": '{ collection(name: "a") }'})
            assert response.status_code == 200
            assert CACHEABLE_HEADER not in response.headers
        assert CALLS["collection"] == 2

    def test_request_bodies_over_the_limit(self, read_only):
        settings = ResponseCacheSettings(root_fields=("collection",), max_request_bytes=64)
        client = TestClient(_make_app(settings))
        long_name = "n" * 200
        responses = [_post(client, long_name) for _ in range(2)]
        assert CALLS["collection"] == 2
        assert responses[1].json() == {"data": {"collection": f"{long_name}:" + "x" * PAYLOAD_SIZE}}

    def test_malformed_bodies(self, client):
        response = client.post(
            "/graphql", content=b"{not json", headers={"content-type": "application/json"}
        )
        assert 400 <= response.status_code < 500
        assert CACHE_STATUS_HEADER not in response.headers

    def test_entries_over_the_budget_are_served_but_not_stored(self, read_only):
        # Any gzip body exceeds 10 bytes (its header alone is 18).
        settings = ResponseCacheSettings(
            root_fields=("collection",), max_bytes=10, max_entry_bytes=10
        )
        client = TestClient(_make_app(settings))
        responses = [_post(client) for _ in range(2)]
        assert CALLS["collection"] == 2
        assert [r.headers[CACHE_STATUS_HEADER] for r in responses] == ["miss", "miss"]
        assert responses[1].json()["data"]["collection"].startswith("a:")

    def test_writable_instances(self, monkeypatch):
        monkeypatch.delenv(READ_ONLY_ENV, raising=False)
        self._assert_built_twice(TestClient(_make_app()), "collection")

    def test_disabled_in_settings(self, read_only):
        settings = ResponseCacheSettings(enabled=False, root_fields=("collection",))
        self._assert_built_twice(TestClient(_make_app(settings)), "collection")


def _seed_projection_collection(db) -> None:
    db.create_dataset("ds")
    db.insert_items_batch(
        "ds",
        ["a", "b"],
        ["document a", "document b"],
        [{"group": "one"}, {"group": "two"}],
    )
    db.register_vector_collection("ds", "chromadb", "ds_dense", "dense")
    db.insert_projections_batch(
        "ds_dense", ["a", "b"], "umap_3d", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )


def test_real_collection_query_is_served_from_cache(db, monkeypatch, read_only):
    """End to end through the real schema, resolver and shipped settings."""
    _seed_projection_collection(db)
    monkeypatch.setattr(query_module, "get_duckdb_client", lambda: db)
    loads: list[str] = []
    original = db.get_projection_data

    def counting_get_projection_data(name, projection_type):
        loads.append(name)
        return original(name, projection_type)

    monkeypatch.setattr(db, "get_projection_data", counting_get_projection_data)
    client = TestClient(_make_app(load_response_cache_settings(), schema=real_schema))
    body = {
        "query": (
            "query GetCollectionData($name: String!, $projectionTypes: [String!]) {"
            " collection(name: $name, projectionTypes: $projectionTypes)"
            " { ids documents itemMetadata umap3d itemSignature } }"
        ),
        "variables": {"name": "ds_dense", "projectionTypes": ["umap_3d"]},
    }

    first = client.post("/graphql", json=body)
    second = client.post("/graphql", json=body)

    assert loads == ["ds_dense"]
    assert second.headers[CACHE_STATUS_HEADER] == "hit"
    assert first.json() == second.json()
    assert first.json()["data"]["collection"]["ids"] == ["a", "b"]


def test_main_app_wiring():
    classes = [m.cls for m in main_app.user_middleware]  # outermost first
    gzip_at = classes.index(JsonGzipMiddleware)
    cors_at = classes.index(CORSMiddleware)
    cache_at = classes.index(GraphQLResponseCacheMiddleware)
    # CORS must wrap the cache so hits still carry CORS headers; gzip outermost.
    assert gzip_at < cors_at < cache_at
    assert isinstance(main_graphql_app, OrreryGraphQLRouter)
    assert any(isinstance(ext, ResponseCacheMarkerExtension) for ext in real_schema.extensions)


def _entry(size: int) -> CachedResponse:
    return CachedResponse(headers=(), gzip_body=b"x" * size, plain_size=size * 4)


class TestResponseCache:
    def test_stores_and_returns_entries(self):
        cache = ResponseCache(max_bytes=1_000, max_entry_bytes=500)
        entry = _entry(40)
        assert cache.put("k", entry)
        assert cache.get("k") is entry
        assert cache.get("missing") is None
        assert len(cache) == 1
        assert cache.total_bytes == 40

    def test_evicts_least_recently_used_beyond_the_byte_budget(self):
        cache = ResponseCache(max_bytes=100, max_entry_bytes=100)
        cache.put("a", _entry(40))
        cache.put("b", _entry(40))
        cache.get("a")  # refresh: b is now least recently used
        cache.put("c", _entry(40))
        assert cache.get("b") is None
        assert cache.get("a") is not None and cache.get("c") is not None
        assert cache.total_bytes == 80

    def test_rejects_entries_over_the_entry_limit(self):
        cache = ResponseCache(max_bytes=1_000, max_entry_bytes=50)
        assert not cache.put("big", _entry(60))
        assert cache.get("big") is None
        assert cache.total_bytes == 0

    def test_replacing_an_entry_updates_the_byte_count(self):
        cache = ResponseCache(max_bytes=1_000, max_entry_bytes=500)
        cache.put("k", _entry(40))
        cache.put("k", _entry(10))
        assert cache.total_bytes == 10

    def test_clear(self):
        cache = ResponseCache(max_bytes=1_000, max_entry_bytes=500)
        cache.put("k", _entry(40))
        cache.clear()
        assert len(cache) == 0
        assert cache.total_bytes == 0


class TestRequestCacheKey:
    def test_key_order_does_not_matter(self):
        one = b'{"query": "q", "variables": {"a": 1, "b": [1, 2]}, "operationName": "Op"}'
        two = b'{"operationName": "Op", "variables": {"b": [1, 2], "a": 1}, "query": "q"}'
        assert request_cache_key(one) == request_cache_key(two)

    def test_different_variables_give_different_keys(self):
        one = b'{"query": "q", "variables": {"name": "a"}}'
        two = b'{"query": "q", "variables": {"name": "b"}}'
        assert request_cache_key(one) != request_cache_key(two)

    @pytest.mark.parametrize(
        "body",
        [b"{not json", b"[]", b'[{"query": "q"}]', b'{"variables": {}}', b'{"query": 1}'],
    )
    def test_requests_without_a_single_query_have_no_key(self, body):
        assert request_cache_key(body) is None


COMPRESSIBLE = b"orrery " * 20_000 + bytes(range(256)) * 40
# Incompressible input spans several of iter_gunzip's 64 KiB input slices.
INCOMPRESSIBLE = random.Random(0).randbytes(300_000)


class TestIterGunzip:
    @pytest.mark.parametrize("chunk_size", [1, 7, 1024, 10_000_000])
    @pytest.mark.parametrize("data", [COMPRESSIBLE, INCOMPRESSIBLE], ids=["text", "random"])
    def test_round_trips_in_bounded_chunks(self, data, chunk_size):
        chunks = list(iter_gunzip(gzip.compress(data), chunk_size))
        assert b"".join(chunks) == data
        assert all(0 < len(chunk) <= chunk_size for chunk in chunks)

    def test_truncated_bodies_raise(self):
        with pytest.raises(ValueError):
            list(iter_gunzip(gzip.compress(INCOMPRESSIBLE)[:-50], 1024))


class TestKeyedLocks:
    def test_serializes_holders_of_one_key_and_cleans_up(self):
        locks = KeyedLocks()
        events: list[str] = []

        async def hold(name: str) -> None:
            async with locks.hold("key"):
                events.append(f"{name}-in")
                await asyncio.sleep(0.01)
                events.append(f"{name}-out")

        async def run() -> None:
            await asyncio.gather(hold("first"), hold("second"))

        asyncio.run(run())
        assert events == ["first-in", "first-out", "second-in", "second-out"]
        assert len(locks) == 0


async def _chunked_receive_app_test(max_request_bytes: int) -> bytes:
    """Run the middleware over a raw ASGI app that echoes the body it read."""
    received: list[bytes] = []

    async def downstream(scope, receive, send):
        while True:
            message = await receive()
            received.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = GraphQLResponseCacheMiddleware(
        downstream,
        settings=ResponseCacheSettings(max_request_bytes=max_request_bytes),
        is_active=lambda: True,
    )
    parts = [b'{"query": "', b"q" * 50, b'"}']
    messages = [
        {"type": "http.request", "body": part, "more_body": i < len(parts) - 1}
        for i, part in enumerate(parts)
    ]

    async def receive():
        return messages.pop(0)

    async def send(message):
        pass

    scope = {"type": "http", "method": "POST", "path": "/graphql", "headers": []}
    await middleware(scope, receive, send)
    return b"".join(received)


@pytest.mark.parametrize("max_request_bytes", [16, 65_536], ids=["over-limit", "buffered"])
def test_request_body_reaches_downstream_intact(max_request_bytes):
    body = asyncio.run(_chunked_receive_app_test(max_request_bytes))
    assert body == b'{"query": "' + b"q" * 50 + b'"}'


class TestSettings:
    def test_missing_file_uses_defaults(self, tmp_path):
        assert ResponseCacheSettings.from_file(tmp_path / "absent.json") == ResponseCacheSettings()

    def test_file_values_override_defaults(self, tmp_path):
        path = tmp_path / "response_cache.json"
        path.write_text('{"root_fields": ["collection", "collectionTopics"], "gzip_level": 9}')
        settings = ResponseCacheSettings.from_file(path)
        assert settings.root_fields == ("collection", "collectionTopics")
        assert settings.gzip_level == 9

    @pytest.mark.parametrize(
        "raw",
        [
            '{"enabled": "yes"}',
            '{"path": "graphql"}',
            '{"root_fields": []}',
            '{"root_fields": "collection"}',
            '{"max_bytes": 0}',
            '{"max_entry_bytes": 0}',
            '{"max_bytes": 100, "max_entry_bytes": 200}',
            '{"max_request_bytes": 0}',
            '{"gzip_level": 0}',
            '{"gzip_level": 10}',
            '{"stream_chunk_bytes": 0}',
            '{"unknown_key": 1}',
            "[]",
        ],
    )
    def test_invalid_values_are_rejected(self, tmp_path, raw):
        path = tmp_path / "response_cache.json"
        path.write_text(raw)
        with pytest.raises(ValueError):
            ResponseCacheSettings.from_file(path)

    def test_shipped_config_file_is_valid(self):
        settings = ResponseCacheSettings.from_file(PACKAGE_ROOT / "config" / "response_cache.json")
        assert settings.enabled
        assert "collection" in settings.root_fields
