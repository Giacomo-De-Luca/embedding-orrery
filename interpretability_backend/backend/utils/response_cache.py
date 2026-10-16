"""In-memory cache of GraphQL responses for read-only deployments.

On the read-only HF Space the data never changes, yet every Explore page load
rebuilt the same full-collection response: DuckDB fetch, per-item JSON
parsing, GraphQL completion and encoding (WordNet: about 8 s of one core and
0.5 GB transient memory on the Space for 81 MB of JSON). This middleware
stores the first response to an allow-listed query, gzipped, and replays it
for identical requests.

Who decides what. ``ResponseCacheMarkerExtension``
(``backend/API/response_cache_extension.py``) knows GraphQL semantics and marks
a response cacheable with the ``CACHEABLE_HEADER`` header: a query whose root
fields are all allow-listed, executed without errors, while the cache is
active. This middleware works on HTTP bytes: it keys requests on their
canonical JSON body, stores marked 200 responses, strips the marker from
every response, and serves hits.

- Keys: SHA-256 of the request's query, variables, operation name and
  extensions, encoded with sorted keys (JSON key order cannot split entries).
- Storage: gzipped bodies in a byte-budgeted LRU (``ResponseCache``).
- Serving: clients accepting gzip get the stored bytes as they are; others
  get plain JSON decompressed in ``stream_chunk_bytes`` slices, so a hit never
  materializes the whole uncompressed body.
- Concurrency: identical misses are single-flighted (``KeyedLocks``): the first
  request builds while the others wait, then hit.
- Only POST bodies up to ``max_request_bytes`` are keyed; everything else
  passes through (with the marker stripped).

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
import gzip
import hashlib
import logging
import zlib
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import ClassVar

import orjson
from anyio import to_thread
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .http_compression import accepts_gzip
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
_POSITIVE_INT_FIELDS = ("max_bytes", "max_entry_bytes", "max_request_bytes", "stream_chunk_bytes")


@dataclass(frozen=True)
class ResponseCacheSettings(JsonFileSettings):
    """Tunables for ``GraphQLResponseCacheMiddleware`` (all optional in the file).

    ``root_fields`` are GraphQL field names as written in queries (camelCase),
    matched against field names, never response aliases.
    """

    SETTINGS_LABEL: ClassVar[str] = "response cache settings"
    TUPLE_FIELDS: ClassVar[tuple[str, ...]] = ("root_fields",)

    enabled: bool = True
    path: str = "/graphql"
    root_fields: tuple[str, ...] = ("collection",)
    max_bytes: int = 256 * 1024 * 1024
    max_entry_bytes: int = 64 * 1024 * 1024
    max_request_bytes: int = 64 * 1024
    gzip_level: int = 6
    stream_chunk_bytes: int = 1024 * 1024

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError(f"enabled must be true or false, got {self.enabled!r}")
        if not isinstance(self.path, str) or not self.path.startswith("/"):
            raise ValueError(f"path must start with '/', got {self.path!r}")
        if (
            not isinstance(self.root_fields, tuple)
            or not self.root_fields
            or not all(isinstance(name, str) and name for name in self.root_fields)
        ):
            raise ValueError("root_fields must be a non-empty list of GraphQL field names")
        for name in _POSITIVE_INT_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.max_entry_bytes > self.max_bytes:
            raise ValueError("max_entry_bytes cannot exceed max_bytes")
        level = self.gzip_level
        if not isinstance(level, int) or isinstance(level, bool) or not 1 <= level <= 9:
            raise ValueError(f"gzip_level must be an integer in 1..9, got {level!r}")


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
    def size(self) -> int:
        return len(self.gzip_body)


class ResponseCache:
    """Byte-budgeted LRU of cached responses.

    Only touched from the event loop, so it needs no locking.
    """

    def __init__(self, max_bytes: int, max_entry_bytes: int):
        self.max_bytes = max_bytes
        self.max_entry_bytes = max_entry_bytes
        self._entries: OrderedDict[str, CachedResponse] = OrderedDict()
        self._bytes = 0

    def get(self, key: str) -> CachedResponse | None:
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
        return entry

    def put(self, key: str, entry: CachedResponse) -> bool:
        """Store ``entry``, evicting least recently used ones; False if too large."""
        if entry.size > self.max_entry_bytes:
            return False
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._bytes -= previous.size
        self._entries[key] = entry
        self._bytes += entry.size
        while self._bytes > self.max_bytes:
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= evicted.size
        return True

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0

    @property
    def total_bytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._entries)


@dataclass
class _KeyLock:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class KeyedLocks:
    """Per-key asyncio locks, dropped once nobody holds or waits on them."""

    def __init__(self):
        self._locks: dict[str, _KeyLock] = {}

    @asynccontextmanager
    async def hold(self, key: str) -> AsyncIterator[None]:
        entry = self._locks.setdefault(key, _KeyLock())
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if entry.users == 0:
                del self._locks[key]

    def __len__(self) -> int:
        return len(self._locks)


def request_cache_key(body: bytes) -> str | None:
    """Canonical SHA-256 key of a single GraphQL request body, or None."""
    try:
        payload = orjson.loads(body)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("query"), str):
        return None
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
        self.cache = ResponseCache(self.settings.max_bytes, self.settings.max_entry_bytes)
        self.locks = KeyedLocks()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["path"].rstrip("/") != self.settings.path.rstrip("/")
            or not self.is_active()
        ):
            await self.app(scope, receive, send)
            return
        if scope["method"] != "POST":
            await self.app(scope, receive, _strip_marker(send))
            return

        messages, body = await _read_request_body(receive, self.settings.max_request_bytes)
        receive = _replaying(messages, receive)
        key = request_cache_key(body) if body is not None else None
        if key is None:
            await self.app(scope, receive, _strip_marker(send))
            return

        wants_gzip = accepts_gzip(Headers(scope=scope).get("accept-encoding", ""))
        entry = self.cache.get(key)
        if entry is None:
            async with self.locks.hold(key):
                entry = self.cache.get(key)  # filled while waiting on the lock?
                if entry is None:
                    responder = _CachingResponder(self, key, wants_gzip, send)
                    await self.app(scope, receive, responder)
                    return
        await self.reply(entry, wants_gzip, send, "hit")

    async def reply(
        self,
        entry: CachedResponse,
        wants_gzip: bool,
        send: Send,
        cache_status: str,
        plain: bytes | None = None,
    ) -> None:
        """Send ``entry``: stored gzip bytes, or plain JSON (given or streamed)."""
        headers = MutableHeaders(raw=list(entry.headers))
        headers.add_vary_header("Accept-Encoding")
        headers[CACHE_STATUS_HEADER] = cache_status
        if wants_gzip:
            headers["content-encoding"] = "gzip"
            headers["content-length"] = str(entry.size)
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
    """``send`` wrapper for a miss: stores a marked 200 response, then replies."""

    def __init__(
        self,
        middleware: GraphQLResponseCacheMiddleware,
        key: str,
        wants_gzip: bool,
        send: Send,
    ):
        self.middleware = middleware
        self.key = key
        self.wants_gzip = wants_gzip
        self.send = send
        self.start: Message | None = None
        self.chunks: list[bytes] = []
        self.buffering = False

    async def __call__(self, message: Message) -> None:
        if message["type"] == "http.response.start":
            headers = Headers(raw=message["headers"])
            self.buffering = (
                headers.get(CACHEABLE_HEADER) == "1"
                and message["status"] == 200
                and "content-encoding" not in headers
            )
            message = _without_marker(message)
            if self.buffering:
                self.start = message
            else:
                await self.send(message)
            return
        if not self.buffering:
            await self.send(message)
            return
        if message["type"] != "http.response.body":
            await self._release_held()
            await self.send(message)
            return
        self.chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            await self._store_and_reply()

    async def _release_held(self) -> None:
        """Give up caching: forward what was held back, unchanged."""
        self.buffering = False
        await self.send(self.start)
        if self.chunks:
            body = b"".join(self.chunks)
            self.chunks = []
            await self.send({"type": "http.response.body", "body": body, "more_body": True})

    async def _store_and_reply(self) -> None:
        plain = b"".join(self.chunks)
        self.chunks = []
        compress = functools.partial(
            gzip.compress, compresslevel=self.middleware.settings.gzip_level, mtime=0
        )
        gzip_body = await to_thread.run_sync(compress, plain)
        entry = CachedResponse(
            headers=_content_headers(self.start["headers"]),
            gzip_body=gzip_body,
            plain_size=len(plain),
        )
        cache = self.middleware.cache
        if cache.put(self.key, entry):
            logger.info(
                "Cached GraphQL response %s…: %d bytes JSON, %d gzipped (cache %d entries, %d bytes)",
                self.key[:12],
                len(plain),
                entry.size,
                len(cache),
                cache.total_bytes,
            )
        await self.middleware.reply(entry, self.wants_gzip, self.send, "miss", plain=plain)


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
