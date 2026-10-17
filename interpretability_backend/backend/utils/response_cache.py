"""In-memory cache of GraphQL responses for read-only deployments.

On the read-only HF Space the data never changes, yet every Explore page load
rebuilt the same full-collection response: DuckDB fetch, per-item JSON
parsing, GraphQL completion and encoding (WordNet: about 8 s of one core and
0.5 GB transient memory on the Space for 81 MB of JSON). This middleware
stores the first response to an allow-listed query, gzipped, and replays it
for identical requests.

Who decides what:
- ``RootFieldAllowList`` (``graphql_operations.py``) pre-filters requests on
  their query text: only queries whose root fields are all allow-listed take
  the cache path. Everything else passes straight through.
- ``ResponseCacheMarkerExtension`` (``backend/API/response_cache_extension.py``)
  marks a result cacheable after execution, with the ``CACHEABLE_HEADER``
  response header, when it also produced no errors. The middleware strips the
  marker from every response.
- The middleware stores a marked 200 response whose JSON body size lies in
  ``[min_body_bytes, max_body_bytes]`` (decided from ``Content-Length``
  before anything is buffered), and serves hits.

Details:
- Keys: SHA-256 of the request's query, variables, operation name and
  extensions, encoded with sorted keys. Only ``application/json`` POSTs up to
  ``max_request_bytes`` that do not ask for a multipart reply are keyed, so a
  hit never answers a request the router would have rejected.
- Storage: gzipped bodies in an LRU whose budget counts a fixed per-entry
  overhead on top of the gzip bytes.
- Serving: clients accepting gzip get the stored bytes as they are; others
  get plain JSON decompressed in ``stream_chunk_bytes`` slices.
- Single flight (``InFlight``): identical requests arriving while the first
  is being built wait for its decision. Once it is stored they replay it; if
  it will not be stored they are released as soon as that is known and build
  their own responses concurrently.

The cache runs only where data is immutable: ``enabled`` in
``config/response_cache.json`` and the read-only flag set (see
``response_cache_active``). Writable instances never cache, since mutations
and background jobs would leave entries stale.

Order in ``backend/main.py``: inside CORS, so hits still carry CORS headers,
and inside ``JsonGzipMiddleware``, which passes already-encoded responses
through untouched. See ``documentation/GRAPHQL_RESPONSE_CACHE.md``.
"""

import asyncio
import functools
import hashlib
import logging
import zlib
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar

import orjson
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .graphql_operations import RootFieldAllowList
from .http_compression import accepts_gzip, compress_gzip
from .json_settings import JsonFileSettings
from .resource_paths import PACKAGE_ROOT

logger = logging.getLogger("orrery." + __name__)

CACHEABLE_HEADER = "x-orrery-cacheable"
CACHE_STATUS_HEADER = "x-orrery-cache"
RESPONSE_CACHE_CONFIG = PACKAGE_ROOT / "config" / "response_cache.json"

# Recomputed for every reply instead of stored with the entry.
_PER_REPLY_HEADERS = frozenset(
    {"content-length", "content-encoding", "vary", CACHEABLE_HEADER, CACHE_STATUS_HEADER}
)
_KEY_FIELDS = ("query", "variables", "operationName", "extensions")
_GUNZIP_INPUT_SLICE = 64 * 1024
# Key string, headers, dataclass and LRU node of one entry (measured ~0.5 KB).
_ENTRY_OVERHEAD_BYTES = 1024

Decision = Callable[["CachedResponse | None"], None]


@dataclass(frozen=True)
class ResponseCacheSettings(JsonFileSettings):
    """Tunables for ``GraphQLResponseCacheMiddleware`` (all optional in the file).

    ``root_fields`` are GraphQL field names as written in queries (camelCase),
    matched against field names, never response aliases. Body bounds apply
    to the uncompressed JSON; ``max_bytes`` budgets the stored gzip bytes.
    """

    SETTINGS_LABEL: ClassVar[str] = "response cache settings"
    TUPLE_FIELDS: ClassVar[tuple[str, ...]] = ("root_fields",)

    enabled: bool = True
    path: str = "/graphql"
    root_fields: tuple[str, ...] = ("collection",)
    max_bytes: int = 256 * 1024 * 1024
    min_body_bytes: int = 256 * 1024
    max_body_bytes: int = 256 * 1024 * 1024
    max_request_bytes: int = 64 * 1024
    gzip_level: int = 6
    stream_chunk_bytes: int = 1024 * 1024

    def __post_init__(self):
        self.require_bool("enabled")
        if not isinstance(self.path, str) or not self.path.startswith("/"):
            raise ValueError(f"path must start with '/', got {self.path!r}")
        if (
            not isinstance(self.root_fields, tuple)
            or not self.root_fields
            or not all(isinstance(name, str) and name for name in self.root_fields)
        ):
            raise ValueError("root_fields must be a non-empty list of GraphQL field names")
        for name in (
            "max_bytes",
            "min_body_bytes",
            "max_body_bytes",
            "max_request_bytes",
            "stream_chunk_bytes",
        ):
            self.require_int(name, minimum=1)
        self.require_int("gzip_level", minimum=1, maximum=9)
        if self.min_body_bytes > self.max_body_bytes:
            raise ValueError("min_body_bytes cannot exceed max_body_bytes")

    def stores_body_of(self, size: int) -> bool:
        """Whether a JSON body of ``size`` bytes is worth storing."""
        return self.min_body_bytes <= size <= self.max_body_bytes


@functools.cache
def load_response_cache_settings() -> ResponseCacheSettings:
    """Shipped settings, read once per process (middleware and extension share them)."""
    return ResponseCacheSettings.from_file(RESPONSE_CACHE_CONFIG)


@dataclass(frozen=True)
class CachedResponse:
    """A stored 200 response: its content headers and gzipped body."""

    headers: tuple[tuple[bytes, bytes], ...]
    gzip_body: bytes
    plain_size: int

    @property
    def footprint(self) -> int:
        """Bytes charged against the cache budget."""
        return len(self.gzip_body) + _ENTRY_OVERHEAD_BYTES


class ResponseCache:
    """Byte-budgeted LRU of cached responses.

    Only touched from the event loop, so it needs no locking.
    """

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._entries: OrderedDict[str, CachedResponse] = OrderedDict()
        self._bytes = 0

    def get(self, key: str) -> CachedResponse | None:
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
        return entry

    def put(self, key: str, entry: CachedResponse) -> bool:
        """Store ``entry``, evicting least recently used ones; False if it cannot fit."""
        if entry.footprint > self.max_bytes:
            return False
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._bytes -= previous.footprint
        self._entries[key] = entry
        self._bytes += entry.footprint
        while self._bytes > self.max_bytes:
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= evicted.footprint
        return True

    @property
    def total_bytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._entries)


class InFlight:
    """Responses being built, keyed by request (single flight).

    The first request for a key leads: it builds the response and decides as
    early as it can whether the result is stored. Identical requests arriving
    meanwhile wait for that decision, then replay the stored entry or, on
    ``None``, build their own response.
    """

    def __init__(self):
        self._futures: dict[str, asyncio.Future] = {}

    def get(self, key: str) -> asyncio.Future | None:
        return self._futures.get(key)

    @contextmanager
    def lead(self, key: str) -> Iterator[Decision]:
        """Register this request as the builder of ``key``; yields its ``decide``."""
        future = asyncio.get_running_loop().create_future()
        self._futures[key] = future

        def decide(entry: CachedResponse | None) -> None:
            if not future.done():
                future.set_result(entry)

        try:
            yield decide
        finally:
            decide(None)  # failures and early exits release the waiters
            if self._futures.get(key) is future:
                del self._futures[key]

    @staticmethod
    async def wait(future: asyncio.Future) -> CachedResponse | None:
        # Shielded: a cancelled waiter must not cancel the future others share.
        return await asyncio.shield(future)

    def __len__(self) -> int:
        return len(self._futures)


def parse_request_body(body: bytes) -> dict[str, Any] | None:
    """The JSON payload of a single GraphQL request, or None."""
    try:
        payload = orjson.loads(body)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("query"), str):
        return None
    operation_name = payload.get("operationName")
    if operation_name is not None and not isinstance(operation_name, str):
        return None
    return payload


def request_cache_key(payload: dict[str, Any]) -> str:
    """Canonical SHA-256 key of a parsed request (JSON key order cannot split it)."""
    canonical = orjson.dumps(
        {name: payload.get(name) for name in _KEY_FIELDS}, option=orjson.OPT_SORT_KEYS
    )
    return hashlib.sha256(canonical).hexdigest()


def iter_gunzip(data: bytes, chunk_size: int) -> Iterator[bytes]:
    """Decompress a gzip body in slices of at most ``chunk_size`` bytes.

    Compressed input is fed ``_GUNZIP_INPUT_SLICE`` bytes at a time: zlib's
    ``unconsumed_tail`` is a fresh copy of the remaining input, so feeding the
    whole body would copy it again for every output slice.
    """
    decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)
    view = memoryview(data)
    for start in range(0, len(view), _GUNZIP_INPUT_SLICE):
        pending = view[start : start + _GUNZIP_INPUT_SLICE]
        while pending:
            chunk = decompressor.decompress(pending, chunk_size)
            pending = decompressor.unconsumed_tail
            if chunk:
                yield chunk
    while not decompressor.eof:  # output still held inside zlib
        chunk = decompressor.decompress(b"", chunk_size)
        if not chunk:
            raise ValueError("Truncated gzip body")
        yield chunk


class GraphQLResponseCacheMiddleware:
    """ASGI middleware replaying cached GraphQL responses (see module docstring)."""

    def __init__(
        self,
        app: ASGIApp,
        settings: ResponseCacheSettings | None = None,
        is_active: Callable[[], bool] | None = None,
    ):
        self.app = app
        self.settings = settings or ResponseCacheSettings()
        self.is_active = is_active or (lambda: self.settings.enabled)
        self.allow_list = RootFieldAllowList(self.settings.root_fields)
        self.cache = ResponseCache(self.settings.max_bytes)
        self.in_flight = InFlight()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["path"].rstrip("/") != self.settings.path.rstrip("/")
            or not self.is_active()
        ):
            await self.app(scope, receive, send)
            return
        request_headers = Headers(scope=scope)
        if scope["method"] != "POST" or not _expects_single_json_reply(request_headers):
            await self.app(scope, receive, _strip_marker(send))
            return

        messages, body = await _read_request_body(receive, self.settings.max_request_bytes)
        receive = _replaying(messages, receive)
        payload = parse_request_body(body) if body is not None else None
        if payload is None or not self.allow_list.allows_query_text(
            payload["query"], payload.get("operationName")
        ):
            await self.app(scope, receive, _strip_marker(send))
            return

        wants_gzip = accepts_gzip(request_headers.get("accept-encoding", ""))
        key = request_cache_key(payload)
        entry = self.cache.get(key)
        if entry is None:
            pending = self.in_flight.get(key)
            if pending is None:
                with self.in_flight.lead(key) as decide:
                    responder = _CachingResponder(self, key, wants_gzip, send, decide)
                    await self.app(scope, receive, responder)
                return
            entry = await self.in_flight.wait(pending)
            if entry is None:  # the first response was not stored: build this one
                await self.app(scope, receive, _strip_marker(send))
                return
        await self.reply(entry, wants_gzip, send, "hit")

    async def reply(
        self,
        entry: CachedResponse,
        wants_gzip: bool,
        send: Send,
        cache_status: str | None,
        plain: bytes | None = None,
    ) -> None:
        """Send ``entry``: stored gzip bytes, or plain JSON (given or streamed)."""
        headers = MutableHeaders(raw=list(entry.headers))
        headers.add_vary_header("Accept-Encoding")
        if cache_status is not None:
            headers[CACHE_STATUS_HEADER] = cache_status
        if wants_gzip:
            headers["content-encoding"] = "gzip"
            headers["content-length"] = str(len(entry.gzip_body))
            await send({"type": "http.response.start", "status": 200, "headers": headers.raw})
            await send({"type": "http.response.body", "body": entry.gzip_body})
            return

        headers["content-length"] = str(entry.plain_size)
        await send({"type": "http.response.start", "status": 200, "headers": headers.raw})
        if plain is not None:
            await send({"type": "http.response.body", "body": plain})
            return
        for chunk in iter_gunzip(entry.gzip_body, self.settings.stream_chunk_bytes):
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b""})


class _CachingResponder:
    """``send`` wrapper for the leading request of a key.

    Holds back a response the extension marked cacheable, stores it, decides
    for the waiters and replies. Anything it will not store is forwarded
    unchanged, and the waiters are released as soon as that is known.
    """

    def __init__(
        self,
        middleware: GraphQLResponseCacheMiddleware,
        key: str,
        wants_gzip: bool,
        send: Send,
        decide: Decision,
    ):
        self.middleware = middleware
        self.key = key
        self.wants_gzip = wants_gzip
        self.send = send
        self.decide = decide
        self.start: Message | None = None
        self.chunks: list[bytes] = []
        self.holding = False

    async def __call__(self, message: Message) -> None:
        if message["type"] == "http.response.start":
            self.holding = self._may_store(message)
            message = _without_marker(message)
            if self.holding:
                self.start = message
            else:
                self.decide(None)
                await self.send(message)
            return
        if not self.holding:
            await self.send(message)
            return
        if message["type"] != "http.response.body":
            await self._release_held()
            await self.send(message)
            return
        self.chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            await self._store_and_reply()

    def _may_store(self, start: Message) -> bool:
        headers = Headers(raw=start["headers"])
        if (
            headers.get(CACHEABLE_HEADER) != "1"
            or start["status"] != 200
            or "content-encoding" in headers
        ):
            return False
        length = headers.get("content-length", "")
        return not length.isdigit() or self.middleware.settings.stores_body_of(int(length))

    async def _release_held(self) -> None:
        """Give up storing: forward what was held back, unchanged."""
        self.holding = False
        self.decide(None)
        await self.send(self.start)
        if self.chunks:
            body = b"".join(self.chunks)
            self.chunks = []
            await self.send({"type": "http.response.body", "body": body, "more_body": True})

    async def _store_and_reply(self) -> None:
        plain = b"".join(self.chunks)
        self.chunks = []
        settings = self.middleware.settings
        if not settings.stores_body_of(len(plain)):  # no Content-Length to decide on earlier
            self.decide(None)
            await self.send(self.start)
            await self.send({"type": "http.response.body", "body": plain})
            return
        entry = CachedResponse(
            headers=_content_headers(self.start["headers"]),
            gzip_body=await compress_gzip(plain, settings.gzip_level),
            plain_size=len(plain),
        )
        cache = self.middleware.cache
        stored = cache.put(self.key, entry)
        self.decide(entry if stored else None)
        if stored:
            logger.info(
                "Cached GraphQL response %s…: %d bytes JSON, %d gzipped (cache %d entries, %d bytes)",
                self.key[:12],
                len(plain),
                len(entry.gzip_body),
                len(cache),
                cache.total_bytes,
            )
        status = "miss" if stored else None
        await self.middleware.reply(entry, self.wants_gzip, self.send, status, plain=plain)


def _expects_single_json_reply(headers: Headers) -> bool:
    """A JSON request body and no multipart reply: what a hit can answer."""
    media_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
    return (
        media_type == "application/json"
        and "multipart/mixed" not in headers.get("accept", "").lower()
    )


async def _read_request_body(receive: Receive, limit: int) -> tuple[list[Message], bytes | None]:
    """Read the request up to ``limit`` bytes; the body is None when larger or cut off."""
    messages: list[Message] = []
    size = 0
    while True:
        message = await receive()
        messages.append(message)
        if message["type"] != "http.request":
            return messages, None
        size += len(message.get("body", b""))
        if size > limit:
            return messages, None
        if not message.get("more_body", False):
            return messages, b"".join(m.get("body", b"") for m in messages)


def _replaying(messages: list[Message], receive: Receive) -> Receive:
    """A ``receive`` that first returns the already-read messages."""
    pending = deque(messages)

    async def replay() -> Message:
        if pending:
            return pending.popleft()
        return await receive()

    return replay


def _without_marker(message: Message) -> Message:
    headers = MutableHeaders(raw=list(message["headers"]))
    del headers[CACHEABLE_HEADER]
    return {**message, "headers": headers.raw}


def _strip_marker(send: Send) -> Send:
    """A ``send`` that removes the cacheable marker from the response headers."""

    async def wrapped(message: Message) -> None:
        if message["type"] == "http.response.start":
            message = _without_marker(message)
        await send(message)

    return wrapped


def _content_headers(raw: list[tuple[bytes, bytes]]) -> tuple[tuple[bytes, bytes], ...]:
    return tuple(
        (name, value)
        for name, value in raw
        if name.decode("latin-1").lower() not in _PER_REPLY_HEADERS
    )
