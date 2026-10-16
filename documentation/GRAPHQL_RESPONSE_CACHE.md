# GraphQL response cache and orjson encoding

The Explore page loads a whole collection in one `collection` GraphQL query.
On the read-only HF Space the data never changes, yet every visitor's load
rebuilt the same response from scratch. This change does two things:

1. **orjson encoding** for every GraphQL response (all deployments).
2. **A response cache** that stores the first response to an allow-listed
   query, gzipped, and replays it for identical requests (read-only
   deployments only).

## Why

Measured on the live Space (cpu-basic, 2026-10-10), one full load each:

| Collection | JSON | Gzipped | Server time | Container memory peak |
|---|---|---|---|---|
| EMNLP abstracts (13,980 items, demo default) | 39 MB | 12.8 MB | about 1.5 s | +105 MB |
| WordNet senses (212,478 items) | 81 MB | 22.4 MB | about 8 s, one core, event loop blocked | +572 MB |

Every request repeated the same work: DuckDB fetch, `json.loads` of each
item's metadata, topic merge, GraphQL completion, `json.dumps` into a `str`,
`.encode()` into `bytes`, then gzip in nginx.

## What was built

| Piece | File | Role |
|---|---|---|
| `OrreryGraphQLRouter` | `backend/API/graphql_router.py` | `GraphQLRouter` whose `create_response` encodes with orjson. `encode_json` is left alone because Strawberry's multipart framing concatenates its output as `str`. |
| `encode_json_bytes` | `backend/utils/json_encoding.py` | orjson (`OPT_NON_STR_KEYS`, `OPT_SERIALIZE_NUMPY`) with a `json.dumps` fallback for what orjson refuses (integers beyond 64 bits, lone surrogates). |
| `ResponseCacheMarkerExtension` | `backend/API/response_cache_extension.py` | Schema extension that marks a result cacheable with the `x-orrery-cacheable` response header. |
| `GraphQLResponseCacheMiddleware`, `ResponseCache`, `KeyedLocks`, `ResponseCacheSettings` | `backend/utils/response_cache.py` | ASGI middleware: keys requests, stores marked responses gzipped, serves hits, strips the marker. |
| `JsonFileSettings` | `backend/utils/json_settings.py` | Shared `from_file` loader for settings dataclasses; `CompressionSettings` now uses it too. |
| Settings | `config/response_cache.json` | See "Settings" below. |

orjson was already installed as a chromadb dependency. It is now declared
directly in `pyproject.toml`.

### orjson output vs `json.dumps`

Same data, different bytes:
- compact separators;
- raw UTF-8 instead of `\uXXXX` escapes;
- NaN/Infinity become `null` instead of bare `NaN`, which is not JSON and
  makes the browser's `JSON.parse` fail.

On both demo collections the cached and orjson responses parse identical to
the stock router's output (checked by the benchmark below).

## Request flow

Middleware order, outermost first: `JsonGzipMiddleware` → CORS → response
cache → GraphQL router.

- **Miss** (cacheable): the router builds the response and the extension
  sets the marker. The cache gzips the body in a worker thread (level 6),
  stores it, and replies with `X-Orrery-Cache: miss`. Clients that accept
  gzip get the gzip bytes, which the outer gzip middleware passes through
  untouched; other clients get the plain body.
- **Hit**: no resolver, DuckDB or encoding work. Clients that accept gzip get
  the stored bytes as they are. Others get plain JSON decompressed in 1 MiB
  slices, so a hit never holds the whole uncompressed body.
- **Concurrent identical misses**: single-flighted. The first request builds
  while the others wait on a per-key lock, then hit.
- **Not cacheable**: passes through unchanged, with the marker stripped.

### What is cached

The cache is active only when `enabled` is true in the config **and**
`ORRERY_READ_ONLY` is truthy; both are checked per request. Writable
instances (local dev) never cache, because mutations and background jobs
would leave entries stale.

A response is stored when all of these hold:
- the operation is a query;
- every root selection is a field listed in `root_fields`, matched by field
  name, not alias (default: `collection`);
- execution produced no errors;
- the status is 200;
- the request was a POST JSON body of at most `max_request_bytes`;
- the gzipped body fits `max_entry_bytes`.

The key is the SHA-256 of the request's `query`, `variables`,
`operationName` and `extensions`, encoded with sorted keys. Entries live for
the process lifetime in a byte-budgeted LRU.

## Settings (`config/response_cache.json`)

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Master switch (still read-only only) |
| `path` | `/graphql` | GraphQL endpoint path |
| `root_fields` | `["collection"]` | Query root fields whose results may be cached |
| `max_bytes` | 256 MiB | Total gzipped bytes kept (LRU eviction beyond) |
| `max_entry_bytes` | 64 MiB | Larger responses are served but not stored |
| `max_request_bytes` | 64 KiB | Larger request bodies are never keyed |
| `gzip_level` | 6 | Stored compression level, paid once per entry |
| `stream_chunk_bytes` | 1 MiB | Slice size when streaming plain JSON from a hit |

Gzip levels measured on the 77 MB WordNet body:

| Level | Size | Time |
|---|---|---|
| 4 | 23.0 MB | 0.8 s |
| 5 | 22.2 MB | 1.3 s |
| 6 | 21.9 MB | 1.8 s |
| 9 | 21.7 MB | 3.3 s |

## Measured

Setup: Apple M3 with 8 GB, a copy-on-write clone of the demo seed, and the
real resolver and frontend query driven in-process through ASGI. These are
server-side numbers only.

| | EMNLP | WordNet |
|---|---|---|
| Before: stock router, `json.dumps` | 341–407 ms | 2.9–4.2 s |
| orjson router, no cache (local dev) | 257–284 ms | 2.6–2.7 s |
| Read-only miss, first load per container (build + gzip once) | about 1.3 s | about 4.4 s |
| Read-only hit, gzip | under 1 ms | under 1 ms |
| Read-only hit, plain (streamed) | about 40–55 ms | about 80–120 ms |

Python heap peak per request (tracemalloc):

| | EMNLP | WordNet |
|---|---|---|
| Stock | 143 MB | 414 MB |
| orjson, no cache | 145 MB | 386 MB |
| Miss | 172 MB | 439 MB |
| Hit, gzip | 0 MB | 0 MB |
| Hit, plain | 1 MB | 1 MB |

The machine was swapping heavily, so timings varied by about 30% between
runs. The cache stores 12.6 MB for EMNLP and 21.9 MB for WordNet.

## HF Space notes

- **nginx strips `Accept-Encoding` upstream on `/graphql`**
  (`deploy/hf-space/nginx.conf`), so the backend sees identity requests:
  - every hit is streamed as plain JSON, about 1 MB of memory;
  - nginx then gzips it again for each visitor (WordNet: about 1.3 s of CPU
    at level 5).

  Forwarding the header on `/graphql` would let the stored gzip bytes leave
  as they are. That means removing the `proxy_set_header Accept-Encoding ""`
  line and adding `gunzip on;`, so nginx can still decompress for the rare
  client without gzip. This is **not applied**: the strip was added
  deliberately for uncached responses, so it is a decision to make.
- **The first build still blocks the event loop.** Each collection and
  projection is still built once per container, synchronously; WordNet holds
  the server for about 8 s on the Space. Moving the resolver to a worker
  thread behind a semaphore is a separate step.
- **Checking after a deploy:** send the same collection query twice and look
  at the `x-orrery-cache` response header (`miss`, then `hit`):

  ```bash
  curl -s -D - -o /dev/null -H 'Content-Type: application/json' -H 'Accept-Encoding: gzip' \
    -d '{"query":"query Q($n:String!){collection(name:$n,projectionTypes:[\"umap_3d\"]){ids}}","variables":{"n":"emotion"}}' \
    https://<sub>.hf.space/graphql | grep -i x-orrery-cache
  ```

## Tests

- **`unit_tests/test_response_cache.py`**:
  - hit and miss behaviour; gzip and plain serving with streamed chunks;
  - variables are part of the key, and key order is not;
  - single-flighted concurrent misses;
  - never cached: errors, other root fields, aliases, mixed roots, mutations,
    GET, oversized and malformed bodies, over-budget entries, writable
    instances, `enabled: false`;
  - the request body reaches downstream intact;
  - LRU budget behaviour, `iter_gunzip` slicing and truncation, settings
    validation, the shipped config;
  - an end-to-end `collection` query through the real schema on an
    in-memory DuckDB;
  - `main.py` wiring (middleware order, router class, extension).
- **`unit_tests/test_json_encoding.py`**:
  - encoder semantics and fallbacks;
  - router parity with Strawberry's stock router (status and body) for
    success, resolver error, validation error and a malformed body.
- **`unit_tests/test_http_compression.py`** still covers `CompressionSettings`
  on the shared `JsonFileSettings` loader.
