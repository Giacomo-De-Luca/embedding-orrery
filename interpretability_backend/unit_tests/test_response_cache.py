"""Guard the read-only GraphQL response cache (backend/utils/response_cache.py).

On a read-only deployment (the HF Space) collection data never changes, so the
first response to an allow-listed query is stored gzipped and replayed for
identical requests without re-running DuckDB, GraphQL completion or JSON
encoding. These tests pin what may be cached (allow-listed query root fields,
no errors, body size within bounds, read-only only), how hits are served
(stored gzip, or plain JSON streamed to clients without gzip), and that
everything else behaves exactly as without the cache, including concurrency.
"""

import asyncio
import dataclasses
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
    InFlight,
    ResponseCache,
    ResponseCacheSettings,
    iter_gunzip,
    load_response_cache_settings,
    parse_request_body,
    request_cache_key,
)

PAYLOAD_SIZE = 5_000  # resolver output large enough to be worth compressing
ORIGIN = "http://localhost:3000"
CALLS: dict[str, int] = {}
ACTIVE = {"now": 0, "max": 0}  # concurrent executions of the slow resolvers


def _count(name: str) -> None:
    CALLS[name] = CALLS.get(name, 0) + 1


async def _busy(seconds: float = 0.05) -> None:
    ACTIVE["now"] += 1
    ACTIVE["max"] = max(ACTIVE["max"], ACTIVE["now"])
    try:
        await asyncio.sleep(seconds)
    finally:
        ACTIVE["now"] -= 1


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
        await _busy()
        if name == "boom":
            raise ValueError("boom")
        return f"{name}:" + "y" * PAYLOAD_SIZE

    @strawberry.field
    async def slow_other(self) -> str:
        _count("slow_other")
        await _busy()
        return "w" * PAYLOAD_SIZE

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


# min_body_bytes=1: the test payloads are far below the shipped 256 KiB floor.
TEST_SETTINGS = ResponseCacheSettings(
    root_fields=("collection", "slowCollection"), min_body_bytes=1
)
QUERY = "query GetCollection($name: String!) { collection(name: $name) }"
SLOW_QUERY = "query Slow($name: String!) { slowCollection(name: $name) }"


def _make_app(
    settings: ResponseCacheSettings = TEST_SETTINGS,
    schema: strawberry.Schema | None = None,
) -> FastAPI:
    """App wired like backend/main.py: cache innermost, then CORS, gzip outermost."""
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
    app.add_middleware(CORSMiddleware, allow_origins=[ORIGIN])
    app.add_middleware(JsonGzipMiddleware, settings=CompressionSettings())
    app.include_router(OrreryGraphQLRouter(schema), prefix="/graphql")
    return app


def _post(
    client: TestClient,
    name: str = "a",
    *,
    query: str = QUERY,
    encoding: str = "gzip",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return client.post(
        "/graphql",
        json={"query": query, "variables": {"name": name}},
        headers={"accept-encoding": encoding, **(headers or {})},
    )


def _fire(app: FastAPI, query: str, variables: dict, count: int = 5) -> list[httpx.Response]:
    """Send ``count`` identical requests concurrently."""

    async def run() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            body = {"query": query, "variables": variables}
            return await asyncio.gather(*(http.post("/graphql", json=body) for _ in range(count)))

    return asyncio.run(run())


@pytest.fixture(autouse=True)
def _reset_counters():
    CALLS.clear()
    ACTIVE.update(now=0, max=0)


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
        for response in (_post(client), _post(client)):
            assert response.headers["content-encoding"] == "gzip"
            assert "accept-encoding" in response.headers["vary"].lower()

    def test_identity_clients_receive_plain_json(self, read_only):
        # A tiny stream chunk forces the hit to arrive in many body messages.
        settings = dataclasses.replace(TEST_SETTINGS, stream_chunk_bytes=64)
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

    def test_hits_carry_cors_headers(self, client):
        _post(client, headers={"origin": ORIGIN})
        hit = _post(client, headers={"origin": ORIGIN})
        assert hit.headers[CACHE_STATUS_HEADER] == "hit"
        assert hit.headers["access-control-allow-origin"] == ORIGIN

    def test_variables_are_part_of_the_key(self, client):
        _post(client, "a")
        _post(client, "b")
        _post(client, "a")
        assert CALLS["collection"] == 2

    def test_concurrent_identical_misses_build_once(self, read_only):
        responses = _fire(_make_app(), SLOW_QUERY, {"name": "a"})
        assert CALLS["slow_collection"] == 1
        assert len({r.content for r in responses}) == 1
        statuses = sorted(r.headers[CACHE_STATUS_HEADER] for r in responses)
        assert statuses == ["hit"] * 4 + ["miss"]


class TestConcurrencyIsPreserved:
    def test_requests_outside_the_allow_list_run_in_parallel(self, read_only):
        responses = _fire(_make_app(), "{ slowOther }", {})
        assert CALLS["slow_other"] == 5
        assert ACTIVE["max"] == 5
        assert all(CACHE_STATUS_HEADER not in r.headers for r in responses)

    def test_waiters_build_their_own_response_when_the_first_is_not_stored(self, read_only):
        responses = _fire(_make_app(), SLOW_QUERY, {"name": "boom"})
        assert CALLS["slow_collection"] == 5
        assert ACTIVE["max"] >= 2  # released together once the first answered
        for response in responses:
            assert response.json()["errors"]
            assert CACHE_STATUS_HEADER not in response.headers


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

    @pytest.mark.parametrize(
        "bounds",
        [{"min_body_bytes": 10_000}, {"max_body_bytes": 1_000}],
        ids=["below-min", "above-max"],
    )
    def test_bodies_outside_the_size_bounds(self, read_only, bounds):
        client = TestClient(_make_app(dataclasses.replace(TEST_SETTINGS, **bounds)))
        self._assert_built_twice(client, "collection")
        assert _post(client).json()["data"]["collection"].startswith("a:")

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
        settings = dataclasses.replace(TEST_SETTINGS, max_request_bytes=64)
        client = TestClient(_make_app(settings))
        long_name = "n" * 200
        responses = [_post(client, long_name) for _ in range(2)]
        assert CALLS["collection"] == 2
        expected = {"data": {"collection": f"{long_name}:" + "x" * PAYLOAD_SIZE}}
        assert responses[1].json() == expected

    def test_malformed_bodies(self, client):
        response = client.post(
            "/graphql", content=b"{not json", headers={"content-type": "application/json"}
        )
        assert 400 <= response.status_code < 500
        assert CACHE_STATUS_HEADER not in response.headers

    def test_non_json_requests_are_answered_as_without_the_cache(self, read_only, monkeypatch):
        body = b'{"query": "{ collection(name: \\"a\\") }"}'
        text_plain = {"content-type": "text/plain"}
        client = TestClient(_make_app())
        client.post("/graphql", content=body, headers={"content-type": "application/json"})

        response = client.post("/graphql", content=body, headers=text_plain)
        monkeypatch.delenv(READ_ONLY_ENV)
        uncached = TestClient(_make_app()).post("/graphql", content=body, headers=text_plain)
        assert response.status_code == uncached.status_code
        assert CACHE_STATUS_HEADER not in response.headers

    def test_multipart_requests_are_answered_as_without_the_cache(self, read_only, monkeypatch):
        multipart = {"accept": "multipart/mixed;subscriptionSpec=1.0"}
        client = TestClient(_make_app(), raise_server_exceptions=False)
        _post(client)  # primes the cache for this exact body
        response = _post(client, headers=multipart)

        monkeypatch.delenv(READ_ONLY_ENV)
        uncached = _post(TestClient(_make_app(), raise_server_exceptions=False), headers=multipart)
        assert response.status_code == uncached.status_code
        assert CACHE_STATUS_HEADER not in response.headers

    def test_writable_instances(self, monkeypatch):
        monkeypatch.delenv(READ_ONLY_ENV, raising=False)
        self._assert_built_twice(TestClient(_make_app()), "collection")

    def test_disabled_in_settings(self, read_only):
        settings = dataclasses.replace(TEST_SETTINGS, enabled=False)
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
    """End to end through the real schema, resolver and shipped allow-list."""
    _seed_projection_collection(db)
    monkeypatch.setattr(query_module, "get_duckdb_client", lambda: db)
    loads: list[str] = []
    original = db.get_projection_data

    def counting_get_projection_data(name, projection_type):
        loads.append(name)
        return original(name, projection_type)

    monkeypatch.setattr(db, "get_projection_data", counting_get_projection_data)
    # The two-item collection is far below the shipped 256 KiB floor.
    settings = dataclasses.replace(load_response_cache_settings(), min_body_bytes=1)
    client = TestClient(_make_app(settings, schema=real_schema))
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
        cache = ResponseCache(max_bytes=10_000)
        entry = _entry(40)
        assert cache.put("k", entry)
        assert cache.get("k") is entry
        assert cache.get("missing") is None
        assert len(cache) == 1
        assert cache.total_bytes == entry.footprint

    def test_footprint_counts_a_per_entry_overhead(self):
        assert _entry(40).footprint > 40

    def test_evicts_least_recently_used_beyond_the_byte_budget(self):
        footprint = _entry(40).footprint
        cache = ResponseCache(max_bytes=2 * footprint + footprint // 2)
        cache.put("a", _entry(40))
        cache.put("b", _entry(40))
        cache.get("a")  # refresh: b is now least recently used
        cache.put("c", _entry(40))
        assert cache.get("b") is None
        assert cache.get("a") is not None and cache.get("c") is not None
        assert cache.total_bytes == 2 * footprint

    def test_rejects_entries_larger_than_the_whole_budget(self):
        cache = ResponseCache(max_bytes=_entry(60).footprint - 1)
        assert not cache.put("big", _entry(60))
        assert cache.get("big") is None
        assert cache.total_bytes == 0

    def test_replacing_an_entry_updates_the_byte_count(self):
        cache = ResponseCache(max_bytes=10_000)
        cache.put("k", _entry(40))
        cache.put("k", _entry(10))
        assert cache.total_bytes == _entry(10).footprint


class TestRequestParsing:
    def test_key_order_does_not_matter(self):
        one = parse_request_body(
            b'{"query": "q", "variables": {"a": 1, "b": [1, 2]}, "operationName": "Op"}'
        )
        two = parse_request_body(
            b'{"operationName": "Op", "variables": {"b": [1, 2], "a": 1}, "query": "q"}'
        )
        assert request_cache_key(one) == request_cache_key(two)

    def test_different_variables_give_different_keys(self):
        one = parse_request_body(b'{"query": "q", "variables": {"name": "a"}}')
        two = parse_request_body(b'{"query": "q", "variables": {"name": "b"}}')
        assert request_cache_key(one) != request_cache_key(two)

    @pytest.mark.parametrize(
        "body",
        [
            b"{not json",
            b"[]",
            b'[{"query": "q"}]',
            b'{"variables": {}}',
            b'{"query": 1}',
            b'{"query": "q", "operationName": 1}',
        ],
    )
    def test_bodies_without_a_single_query_are_rejected(self, body):
        assert parse_request_body(body) is None


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


class TestInFlight:
    def test_the_leaders_decision_reaches_every_waiter(self):
        flights = InFlight()
        entry = _entry(10)

        async def run() -> list:
            with flights.lead("k") as decide:
                waiters = [asyncio.ensure_future(flights.wait(flights.get("k"))) for _ in range(3)]
                await asyncio.sleep(0)
                decide(entry)
                return await asyncio.gather(*waiters)

        assert asyncio.run(run()) == [entry, entry, entry]
        assert len(flights) == 0

    def test_leaving_without_a_decision_releases_waiters_with_none(self):
        flights = InFlight()

        async def run():
            with flights.lead("k"):
                waiter = asyncio.ensure_future(flights.wait(flights.get("k")))
                await asyncio.sleep(0)
            return await waiter

        assert asyncio.run(run()) is None
        assert len(flights) == 0

    def test_a_cancelled_waiter_does_not_cancel_the_others(self):
        flights = InFlight()

        async def run():
            with flights.lead("k") as decide:
                future = flights.get("k")
                cancelled = asyncio.ensure_future(flights.wait(future))
                patient = asyncio.ensure_future(flights.wait(future))
                await asyncio.sleep(0)
                cancelled.cancel()
                await asyncio.sleep(0)
                decide(None)
                return await patient

        assert asyncio.run(run()) is None


ALLOWED_BODY = b'{"query": "{ collection(name: \\"a\\") }"}'


async def _drive(app, body: bytes = ALLOWED_BODY, chunk_sizes: tuple[int, ...] = ()):
    """Call ``app`` like a server would; return the messages it sent."""
    parts = [body]
    if chunk_sizes:
        parts, rest = [], body
        for size in chunk_sizes:
            parts.append(rest[:size])
            rest = rest[size:]
        parts.append(rest)
    messages = [
        {"type": "http.request", "body": part, "more_body": i < len(parts) - 1}
        for i, part in enumerate(parts)
    ]

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/graphql",
        "headers": [(b"content-type", b"application/json")],
    }
    await app(scope, receive, send)
    return sent


def _middleware(downstream, **overrides) -> GraphQLResponseCacheMiddleware:
    settings = dataclasses.replace(TEST_SETTINGS, **overrides)
    return GraphQLResponseCacheMiddleware(downstream, settings=settings, is_active=lambda: True)


@pytest.mark.parametrize("max_request_bytes", [16, 65_536], ids=["over-limit", "buffered"])
def test_request_body_reaches_downstream_intact(max_request_bytes):
    received: list[bytes] = []

    async def downstream(scope, receive, send):
        while True:
            message = await receive()
            received.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = _middleware(downstream, max_request_bytes=max_request_bytes)
    asyncio.run(_drive(middleware, chunk_sizes=(11, 9)))
    assert b"".join(received) == ALLOWED_BODY


def _marked_start(content_length: int | None = None) -> dict:
    headers = [(b"content-type", b"application/json"), (CACHEABLE_HEADER.encode(), b"1")]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    return {"type": "http.response.start", "status": 200, "headers": headers}


def test_downstream_failure_after_the_start_is_held_releases_the_key():
    calls = {"n": 0}
    body = b'{"data": {"collection": "a"}}'

    async def downstream(scope, receive, send):
        calls["n"] += 1
        await send(_marked_start(len(body)))
        if calls["n"] == 1:
            raise RuntimeError("downstream failed")
        await send({"type": "http.response.body", "body": body})

    middleware = _middleware(downstream)
    with pytest.raises(RuntimeError):
        asyncio.run(_drive(middleware))
    assert len(middleware.in_flight) == 0

    sent = asyncio.run(_drive(middleware))
    assert dict(sent[0]["headers"])[CACHE_STATUS_HEADER.encode()] == b"miss"
    assert len(middleware.cache) == 1


def test_non_body_message_releases_the_held_response_unchanged():
    trailers = {"type": "http.response.trailers", "headers": [], "more_trailers": False}

    async def downstream(scope, receive, send):
        await send(_marked_start())
        await send({"type": "http.response.body", "body": b'{"data":', "more_body": True})
        await send(trailers)

    middleware = _middleware(downstream)
    sent = asyncio.run(_drive(middleware))
    assert [m["type"] for m in sent] == [
        "http.response.start",
        "http.response.body",
        "http.response.trailers",
    ]
    assert CACHEABLE_HEADER.encode() not in dict(sent[0]["headers"])
    assert sent[1] == {"type": "http.response.body", "body": b'{"data":', "more_body": True}
    assert len(middleware.cache) == 0


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
            '{"min_body_bytes": 0}',
            '{"max_body_bytes": 0}',
            '{"min_body_bytes": 200, "max_body_bytes": 100}',
            '{"max_request_bytes": 0}',
            '{"gzip_level": 0}',
            '{"gzip_level": 10}',
            '{"gzip_level": true}',
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
