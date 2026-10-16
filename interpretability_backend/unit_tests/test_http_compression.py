"""Guard the JSON gzip middleware (backend/utils/http_compression.py).

The middleware compresses GraphQL JSON responses for every deployment
(local dev hits uvicorn directly, so nginx's gzip never applies there). It
must leave non-JSON payloads (e.g. Arrow), streaming bodies and
already-encoded responses untouched, and push large bodies to a worker thread
so the event loop keeps serving WebSocket progress while compressing.
"""

import gzip
import json

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.testclient import TestClient

from backend.main import app as main_app
from backend.utils import http_compression
from backend.utils.http_compression import CompressionSettings, JsonGzipMiddleware
from backend.utils.resource_paths import PACKAGE_ROOT

LARGE_PAYLOAD = {"values": list(range(5_000))}  # ~30 KB of JSON
SMALL_PAYLOAD = {"ok": True}


def _make_app(settings: CompressionSettings) -> FastAPI:
    app = FastAPI()
    app.add_middleware(JsonGzipMiddleware, settings=settings)

    @app.get("/large")
    def large():
        return JSONResponse(LARGE_PAYLOAD)

    @app.get("/small")
    def small():
        return JSONResponse(SMALL_PAYLOAD)

    @app.get("/graphql-json")
    def graphql_json():
        return Response(
            json.dumps(LARGE_PAYLOAD),
            media_type="application/graphql-response+json; charset=utf-8",
        )

    @app.get("/arrow")
    def arrow():
        return Response(b"\x00" * 50_000, media_type="application/vnd.apache.arrow.stream")

    @app.get("/stream")
    def stream():
        def chunks():
            yield b'{"values": ['
            yield b",".join(str(i).encode() for i in range(5_000))
            yield b"]}"

        return StreamingResponse(chunks(), media_type="application/json")

    @app.get("/no-content")
    def no_content():
        return Response(status_code=204, media_type="application/json")

    @app.api_route("/head-large", methods=["HEAD", "GET"])
    def head_large():
        return JSONResponse(LARGE_PAYLOAD)

    @app.get("/pre-encoded")
    def pre_encoded():
        body = gzip.compress(json.dumps(LARGE_PAYLOAD).encode())
        return Response(body, media_type="application/json", headers={"Content-Encoding": "gzip"})

    return app


@pytest.fixture
def client():
    return TestClient(_make_app(CompressionSettings()))


def _get(client: TestClient, path: str, encoding: str = "gzip"):
    return client.get(path, headers={"Accept-Encoding": encoding})


class TestCompressesJson:
    def test_large_json_is_gzipped_and_round_trips(self, client):
        response = _get(client, "/large")
        assert response.headers["content-encoding"] == "gzip"
        assert "accept-encoding" in response.headers["vary"].lower()
        assert response.json() == LARGE_PAYLOAD  # httpx decodes transparently

    def test_content_length_matches_compressed_body(self, client):
        with client.stream("GET", "/large", headers={"Accept-Encoding": "gzip"}) as response:
            raw = b"".join(response.iter_raw())
        assert int(response.headers["content-length"]) == len(raw)
        assert json.loads(gzip.decompress(raw)) == LARGE_PAYLOAD

    def test_graphql_response_json_media_type_is_gzipped(self, client):
        response = _get(client, "/graphql-json")
        assert response.headers["content-encoding"] == "gzip"
        assert response.json() == LARGE_PAYLOAD


class TestPassesThrough:
    def test_small_body_is_not_compressed(self, client):
        response = _get(client, "/small")
        assert "content-encoding" not in response.headers
        assert response.json() == SMALL_PAYLOAD

    def test_client_without_gzip_gets_identity(self, client):
        response = _get(client, "/large", encoding="identity")
        assert "content-encoding" not in response.headers
        assert response.json() == LARGE_PAYLOAD

    def test_arrow_payload_is_not_compressed(self, client):
        response = _get(client, "/arrow")
        assert "content-encoding" not in response.headers
        assert len(response.content) == 50_000

    def test_streaming_body_is_not_compressed(self, client):
        response = _get(client, "/stream")
        assert "content-encoding" not in response.headers
        assert response.json() == LARGE_PAYLOAD

    def test_pre_encoded_body_is_not_double_compressed(self, client):
        response = _get(client, "/pre-encoded")
        assert response.headers["content-encoding"] == "gzip"
        assert response.json() == LARGE_PAYLOAD

    def test_explicitly_refused_gzip_is_not_used(self, client):
        response = _get(client, "/large", encoding="gzip;q=0, identity")
        assert "content-encoding" not in response.headers
        assert response.json() == LARGE_PAYLOAD

    def test_wildcard_encoding_accepts_gzip(self, client):
        assert _get(client, "/large", encoding="*").headers["content-encoding"] == "gzip"

    def test_no_content_response_passes_through(self, client):
        response = _get(client, "/no-content")
        assert response.status_code == 204
        assert "content-encoding" not in response.headers

    def test_head_request_has_no_body(self, client):
        response = client.head("/head-large", headers={"Accept-Encoding": "gzip"})
        assert response.status_code == 200
        assert response.content == b""

    def test_configured_media_types_are_case_insensitive(self):
        settings = CompressionSettings(content_types=("Application/JSON",))
        client = TestClient(_make_app(settings))
        assert _get(client, "/large").headers["content-encoding"] == "gzip"

    def test_disabled_settings_pass_everything_through(self):
        client = TestClient(_make_app(CompressionSettings(enabled=False)))
        response = _get(client, "/large")
        assert "content-encoding" not in response.headers


class _ThreadSpy:
    """Stands in for anyio.to_thread: records calls, runs inline."""

    def __init__(self):
        self.calls = 0

    async def run_sync(self, func, *args):
        self.calls += 1
        return func(*args)


class TestThreadOffload:
    def test_bodies_above_threshold_compress_in_a_worker_thread(self, monkeypatch):
        spy = _ThreadSpy()
        monkeypatch.setattr(http_compression, "to_thread", spy)
        client = TestClient(_make_app(CompressionSettings(thread_threshold_bytes=2_048)))
        response = _get(client, "/large")
        assert response.headers["content-encoding"] == "gzip"
        assert spy.calls == 1

    def test_bodies_below_threshold_compress_inline(self, monkeypatch):
        spy = _ThreadSpy()
        monkeypatch.setattr(http_compression, "to_thread", spy)
        client = TestClient(_make_app(CompressionSettings(thread_threshold_bytes=10_000_000)))
        response = _get(client, "/large")
        assert response.headers["content-encoding"] == "gzip"
        assert spy.calls == 0


class TestSettings:
    def test_missing_file_uses_defaults(self, tmp_path):
        settings = CompressionSettings.from_file(tmp_path / "absent.json")
        assert settings == CompressionSettings()

    def test_file_values_override_defaults(self, tmp_path):
        path = tmp_path / "http_compression.json"
        path.write_text(json.dumps({"level": 6, "content_types": ["application/json"]}))
        settings = CompressionSettings.from_file(path)
        assert settings.level == 6
        assert settings.content_types == ("application/json",)

    @pytest.mark.parametrize(
        "raw",
        [
            {"level": 0},
            {"level": 10},
            {"minimum_size": -1},
            {"thread_threshold_bytes": -1},
            {"content_types": []},
            {"unknown_key": 1},
            {"content_types": "application/json"},
            {"level": "4"},
        ],
    )
    def test_invalid_values_are_rejected(self, tmp_path, raw):
        path = tmp_path / "http_compression.json"
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            CompressionSettings.from_file(path)

    def test_shipped_config_file_is_valid(self):
        settings = CompressionSettings.from_file(PACKAGE_ROOT / "config" / "http_compression.json")
        assert settings.enabled


def test_main_app_installs_the_middleware():
    assert any(m.cls is JsonGzipMiddleware for m in main_app.user_middleware)
