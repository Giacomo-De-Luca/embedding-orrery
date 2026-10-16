"""Gzip compression for JSON responses, configured by ``config/http_compression.json``.

Local dev talks to uvicorn directly (no nginx in front), so without this the
GraphQL payloads — collection loads, probe scores — travel uncompressed.
Starlette's stock ``GZipMiddleware`` is not used because it compresses every
content type (including binary Arrow payloads) on the event loop at
level 9; a large body would then stall WebSocket progress streams for
seconds. This middleware:

- compresses only allow-listed JSON media types,
- buffers single-message bodies only (streaming responses pass through),
- never re-encodes a response that already carries ``Content-Encoding``,
- moves compression of bodies above ``thread_threshold_bytes`` to a worker
  thread so the event loop keeps running.

Behind the HF Space nginx, ``proxy_set_header Accept-Encoding ""`` keeps
compression in nginx's worker instead (see deploy/hf-space/nginx.conf).
"""

import functools
import gzip
from dataclasses import dataclass
from typing import ClassVar

from anyio import to_thread
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .json_settings import JsonFileSettings


@dataclass(frozen=True)
class CompressionSettings(JsonFileSettings):
    """Tunables for ``JsonGzipMiddleware`` (all optional in the config file)."""

    SETTINGS_LABEL: ClassVar[str] = "compression settings"
    TUPLE_FIELDS: ClassVar[tuple[str, ...]] = ("content_types",)

    enabled: bool = True
    level: int = 4
    minimum_size: int = 1024
    thread_threshold_bytes: int = 1_048_576
    content_types: tuple[str, ...] = ("application/json", "application/graphql-response+json")

    def __post_init__(self):
        if not isinstance(self.level, int) or isinstance(self.level, bool):
            raise ValueError(f"Compression level must be an integer, got {self.level!r}")
        if not isinstance(self.content_types, tuple):
            raise ValueError("content_types must be a list of media types")
        if not 1 <= self.level <= 9:
            raise ValueError(f"Compression level must be in 1..9, got {self.level}")
        if self.minimum_size < 0:
            raise ValueError("minimum_size must be >= 0")
        if self.thread_threshold_bytes < 0:
            raise ValueError("thread_threshold_bytes must be >= 0")
        if not self.content_types or not all(isinstance(t, str) and t for t in self.content_types):
            raise ValueError("content_types must be a non-empty list of media types")
        # Media types compare case-insensitively.
        object.__setattr__(self, "content_types", tuple(t.lower() for t in self.content_types))


class JsonGzipMiddleware:
    """ASGI middleware gzipping eligible JSON responses (see module docstring)."""

    def __init__(self, app: ASGIApp, settings: CompressionSettings | None = None):
        self.app = app
        self.settings = settings or CompressionSettings()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or not self.settings.enabled
            or not accepts_gzip(Headers(scope=scope).get("accept-encoding", ""))
        ):
            await self.app(scope, receive, send)
            return
        await self.app(scope, receive, _GzipResponder(self.settings, send))


def accepts_gzip(accept_encoding: str) -> bool:
    """Whether an Accept-Encoding header allows gzip (``gzip;q=0`` refuses it)."""
    for part in accept_encoding.split(","):
        coding, *params = (piece.strip() for piece in part.split(";"))
        if coding.lower() not in ("gzip", "*"):
            continue
        quality = 1.0
        for param in params:
            name, _, value = param.partition("=")
            if name.strip().lower() == "q":
                try:
                    quality = float(value)
                except ValueError:
                    quality = 0.0
        return quality > 0
    return False


class _GzipResponder:
    """Per-request ``send`` wrapper that decides and applies compression."""

    def __init__(self, settings: CompressionSettings, send: Send):
        self.settings = settings
        self.send = send
        self.start_message: Message | None = None
        self.passthrough = False

    async def __call__(self, message: Message) -> None:
        if message["type"] == "http.response.start":
            self.start_message = message
            self.passthrough = not self._eligible(Headers(raw=message["headers"]))
            if self.passthrough:
                await self.send(message)
            return
        if self.passthrough or self.start_message is None:
            await self.send(message)
            return
        if message["type"] != "http.response.body":
            # Not a body this can buffer (e.g. a pathsend): release the held start first.
            start, self.start_message = self.start_message, None
            self.passthrough = True
            await self.send(start)
            await self.send(message)
            return

        # First body message of an eligible response: decide now.
        start, self.start_message = self.start_message, None
        self.passthrough = True  # every later message is forwarded verbatim
        body = message.get("body", b"")
        if message.get("more_body", False) or len(body) < self.settings.minimum_size:
            await self.send(start)
            await self.send(message)
            return

        compressed = await self._compress(body)
        headers = MutableHeaders(raw=start["headers"])
        headers["Content-Encoding"] = "gzip"
        headers["Content-Length"] = str(len(compressed))
        headers.add_vary_header("Accept-Encoding")
        await self.send(start)
        await self.send({"type": "http.response.body", "body": compressed, "more_body": False})

    def _eligible(self, headers: Headers) -> bool:
        if "content-encoding" in headers:
            return False
        media_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
        return media_type in self.settings.content_types

    async def _compress(self, body: bytes) -> bytes:
        compress = functools.partial(gzip.compress, compresslevel=self.settings.level, mtime=0)
        if len(body) > self.settings.thread_threshold_bytes:
            return await to_thread.run_sync(compress, body)
        return compress(body)
