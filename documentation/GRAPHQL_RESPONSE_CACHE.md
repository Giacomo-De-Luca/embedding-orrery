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
| `encode_json_bytes` | `backend/utils/json_encoding.py` | orjson with a `json.dumps` fallback; see "orjson output" below. |
| `RootFieldAllowList` | `backend/utils/graphql_operations.py` | Decides whether an operation is a query whose root selections are all allow-listed fields (by name, not alias). Used by both the middleware (on the query text, before execution) and the extension (on the parsed document, after execution). |
| `ResponseCacheMarkerExtension` | `backend/API/response_cache_extension.py` | Schema extension that marks a result cacheable with the `x-orrery-cacheable` response header (allow-listed query, no errors, cache active). |
| `GraphQLResponseCacheMiddleware`, `ResponseCache`, `InFlight`, `ResponseCacheSettings` | `backend/utils/response_cache.py` | The ASGI middleware, the byte-budgeted LRU, single-flight coordination, and the settings. |
| `compress_gzip` | `backend/utils/http_compression.py` | Shared gzip helper (worker thread above a size threshold), used by `JsonGzipMiddleware` and the cache. |
| `JsonFileSettings` | `backend/utils/json_settings.py` | Shared `from_file` loader plus `require_bool` / `require_int` checks for settings dataclasses. `CompressionSettings` uses it too. |
| Settings | `config/response_cache.json` | See "Settings" below. |

orjson was already installed as a chromadb dependency. It is now declared
directly in `pyproject.toml`.

### orjson output vs `json.dumps`

Same data, different bytes:
- compact separators;
- raw UTF-8 instead of `\uXXXX` escapes;
- NaN/Infinity become `null`. `json.dumps` writes bare `NaN`, which is not
  JSON and makes the browser's `JSON.parse` fail;
- NumPy values, UUIDs and enums are serialized natively.

`datetime`/`date`/`time` values and dataclasses still raise `TypeError`, as
with the stdlib (orjson passthrough options plus the fallback). Strawberry's
WebSocket path keeps `json.dumps`, so an unsanitized DuckDB datetime fails the
same way on both transports.

Payloads orjson refuses but the stdlib accepts (integers beyond 64 bits, lone
surrogates) are encoded entirely by `json.dumps`, including its bare `NaN`.

On both demo collections the cached and orjson responses parse identical to
the stock router's output (checked by the benchmark below).

## Request flow

Middleware order, outermost first: `JsonGzipMiddleware` → CORS → response
cache → GraphQL router.

1. **Pre-filter.** Only `application/json` POSTs of at most
   `max_request_bytes`, not asking for a multipart reply, and whose query
   passes `RootFieldAllowList` enter the cache path. Everything else
   (mutations, other queries, GET, uploads, WebSockets) passes straight
   through, with no locking and no change in concurrency. A hit can never
   answer a request the router itself would reject.
2. **Hit:** no resolver, DuckDB or encoding work. Clients that accept gzip get
   the stored bytes as they are. Others get plain JSON decompressed in 1 MiB
   slices, so a hit never holds the whole uncompressed body.
3. **Miss (the leader):** the router builds the response. If the extension
   marked it, the status is 200 and `Content-Length` lies within
   `[min_body_bytes, max_body_bytes]`, the cache gzips it (level 6, worker
   thread), stores it and replies with `X-Orrery-Cache: miss`. Otherwise the
   response is forwarded unchanged.
4. **Single flight.** Identical requests arriving during a miss wait for the
   leader's decision. If it stored the response, they replay it as hits. If
   it won't store it (errors, too small, too large), they are released as
   soon as that is known and build their own responses concurrently.

The cache is active only when `enabled` is true in the config **and**
`ORRERY_READ_ONLY` is truthy; both are checked per request. Writable
instances (local dev) never cache, because mutations and background jobs
would leave entries stale.

Keys are the SHA-256 of the request's `query`, `variables`, `operationName`
and `extensions`, encoded with sorted keys. Entries live for the process
lifetime in an LRU whose budget counts each entry's gzip bytes plus a fixed
1 KiB overhead.

## Settings (`config/response_cache.json`)

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Master switch (read-only instances only either way) |
| `path` | `/graphql` | GraphQL endpoint path |
| `root_fields` | `["collection"]` | Query root fields whose results may be cached |
| `max_bytes` | 256 MiB | Budget: stored gzip bytes plus per-entry overhead (LRU eviction) |
| `min_body_bytes` | 256 KiB | Smaller JSON bodies are cheap to rebuild: never stored |
| `max_body_bytes` | 256 MiB | Larger JSON bodies are served but never stored (no compression wasted) |
| `max_request_bytes` | 64 KiB | Larger request bodies are never keyed |
| `gzip_level` | 6 | Stored compression level, paid once per entry |
| `stream_chunk_bytes` | 1 MiB | Slice size when streaming plain JSON from a hit |

Without `min_body_bytes`, a scripted client could fill the budget with
millions of tiny entries, such as `null` results for nonexistent collection
names. With it, every entry is a real multi-hundred-KB body.

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
| Before: stock router, `json.dumps` | 287–407 ms | 2.9–4.2 s |
| orjson router, no cache (local dev) | 248–284 ms | 2.5–2.7 s |
| Read-only miss, first load per container (build + gzip once) | about 1.3 s | about 4.3 s |
| Read-only hit, gzip | about 1 ms | under 1 ms |
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
  at the `x-orrery-cache` response header (`miss`, then `hit`). Use a
  collection whose response is over 256 KiB:

  ```bash
  curl -s -D - -o /dev/null -H 'Content-Type: application/json' -H 'Accept-Encoding: gzip' \
    -d '{"query":"query Q($n:String!){collection(name:$n,projectionTypes:[\"umap_3d\"]){ids documents}}","variables":{"n":"acl_abstracts_emnlp_findings"}}' \
    https://<sub>.hf.space/graphql | grep -i x-orrery-cache
  ```

## Tests

- **`unit_tests/test_response_cache.py`**:
  - hits and misses, gzip and streamed plain serving, CORS headers on hits;
  - variables are part of the key, and key order is not;
  - concurrency: identical misses build once; non-allow-listed requests still
    run fully in parallel; waiters build their own responses when the
    leader's response is not stored;
  - never cached: errors, other root fields, aliases, mixed roots, mutations,
    GET, oversized and malformed bodies, bodies outside the size bounds,
    writable instances, `enabled: false`;
  - requests answered as without the cache: non-JSON and multipart;
  - ASGI edge cases: request-body replay, a downstream failure after the start
    message was held (the key is released), a non-body message releasing the
    held response unchanged;
  - LRU budget with per-entry overhead, `InFlight` (decision fan-out, release
    on exit, cancellation isolation), request parsing, `iter_gunzip`,
    settings validation, the shipped config;
  - an end-to-end `collection` query through the real schema on an
    in-memory DuckDB, and `main.py` wiring.
- **`unit_tests/test_graphql_operations.py`**: `RootFieldAllowList` on query
  text and documents (aliases, fragments, mixed roots, operation selection,
  mutations, subscriptions, unparsable text).
- **`unit_tests/test_json_encoding.py`**: encoder semantics, passthrough
  parity, fallbacks, and router parity with Strawberry's stock router (status
  and body).
- **`unit_tests/test_http_compression.py`** covers `CompressionSettings` on
  the shared loader and `compress_gzip` through the middleware.
