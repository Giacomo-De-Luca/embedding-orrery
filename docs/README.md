# Orrery documentation site

Nextra 4 (Next.js App Router) docs site. Not part of the app — it is excluded
from Docker builds via the root `.dockerignore` and has its own dependencies.

```bash
cd docs
npm install
npm run dev     # http://localhost:3001 (or next free port)
npm run build   # static build + Pagefind search index
```

## Structure

- `content/` — the pages. **Convention: committed pages are `.mdx`**
  (`index.mdx`, `getting-started.mdx`, plus per-folder `_meta.js` for sidebar
  titles + order); every `.md` file under `content/` is **generated at build
  time** and gitignored wholesale (`content/**/*.md`).
- `scripts/sync-content.mjs` — runs automatically before `dev`/`build` (npm
  `predev`/`prebuild` hooks). Copies `gallery/` → `public/gallery/` and the
  curated public subset of `documentation/*.md` → `content/`, rewriting
  repo-relative links to GitHub URLs (and warning about link shapes it can't
  rewrite). Stale generated pages are wiped on each run. **The canonical
  markdown stays in `documentation/`** — edit it there, not here. To publish
  another page, add it to `PAGES` in the script and to the matching `_meta.js`.
- `app/` — Nextra 4 boilerplate: root layout (navbar/footer/theme) and the
  catch-all route that renders `content/`.

## Deploying (Vercel)

- Root Directory: `docs`
- Enable "Include source files outside of the Root Directory" (the sync script
  reads `../gallery` and `../documentation`) — this is the default for
  monorepos.
- Build command / output: defaults (`npm run build`). The `postbuild` hook
  builds the Pagefind search index.
