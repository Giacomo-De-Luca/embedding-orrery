# Frontend Agent Notes

Follow the root `AGENTS.md` rules. Historical frontend notes also live in
`claude.md`; prefer the root instructions if they conflict.

## Production Docker

- `Dockerfile` builds a standalone Next.js production server with Node 22.
- Docker uses `npm run build:docker`, which runs webpack-backed `next build`.
  Keep this separate from the local `npm run build` script while the app still
  relies on webpack aliasing in `next.config.ts`.
- `build:docker` sets `ORRERY_DOCKER_BUILD=1`, which skips the Next ESLint and
  TypeScript build gates for the current frontend backlog. Keep this Docker-only
  and do not treat it as proof that the frontend is type-clean.
- Published images compile `NEXT_PUBLIC_GRAPHQL_URL=/graphql` and an empty API
  base so GraphQL and uploads remain same-origin behind the Compose gateway.
- When `NEXT_PUBLIC_GRAPHQL_WS_URL` is empty, `endpointUrls.ts` derives `ws` or
  `wss` from `window.location`; explicit development overrides remain supported.
- When public endpoint variables are absent under local `npm run dev`,
  `endpointUrls.ts` defaults GraphQL/WebSocket/uploads to `localhost:8000`.
  Explicitly empty Docker values retain same-origin behavior.
- File uploads must resolve the API base through `endpointUrls.ts`, never a
  hardcoded localhost URL.
- `docker-compose.yml` exposes nginx on port 3000; the standalone frontend
  container is internal to the Compose network.

Detailed Docker behavior is documented in `../documentation/DOCKER.md`.

## Memory-Aware Projection Loading

- `lib/utils/projectionLoadPolicy.ts` requests only the active method/dimension.
- `lib/utils/latestRequestGate.ts` prevents stale projection responses from
  overwriting a newer collection/method/dimension request.
- `lib/utils/projectionMembership.ts` validates ordered-item signatures before
  merging a projection-only response into the loaded core data.
- The initial collection request includes core item data; later projection
  requests set GraphQL `includeCore: false`.
- `lib/utils/visualizationPointBuilder.ts` builds wrappers only for the active
  2D or 3D mode.
- `lib/utils/pointTraceColumns3D.ts` materializes Plotly trace coordinates as
  `Float32Array` and point indices as `Uint32Array`.

Full behavior, tests, and remaining limitations are documented in
`../documentation/FRONTEND_HEAP_REDUCTION.md`.
The headed-Chrome heap/FPS regression harness and safe baseline-comparison
workflow are documented in `../benchmarks/fps/README.md`.
