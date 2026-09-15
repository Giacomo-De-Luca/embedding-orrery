# Frontend build configuration

## Tailwind resolution and the workspace root

The Next.js application and its dependencies live in `embedding_visualization/`.
`app/globals.css` uses Tailwind v4's `@import "tailwindcss"`, processed by
`@tailwindcss/postcss` from `postcss.config.mjs`.

Next.js infers its workspace root from lockfiles in the application directory
and its ancestors. Adding a repository-level `package-lock.json` made Next.js
16.2.9 choose `EmbeddingVisualisation/` as that root. Local development then
reported `Can't resolve 'tailwindcss' in '/Users/jack/EmbeddingVisualisation'`,
even though Tailwind was installed in the frontend's `node_modules`.

`next.config.ts` pins `turbopack.root` to `__dirname`, the frontend directory.
This also keeps the existing relative Plotly aliases anchored to the frontend.
The local `file:` dependencies are under `embedding_visualization/forked/`,
inside this root. Future linked dependencies outside the frontend would require
revisiting the root configuration.

The CSS import, PostCSS configuration, dependencies, and repository-level npm
files do not need changes for this fix. Restart the development server after
changing the Next.js configuration:

```bash
cd embedding_visualization
npm run dev
```

## Validation

- Before the change, the installed Next.js root detector selected the
  repository-level lockfile, and requesting `/` reproduced the exact Tailwind
  resolution error.
- With the explicit root and both lockfiles still present, requesting `/`
  returned HTTP 200 without the root warning or Tailwind resolution error.
- The served stylesheet returned HTTP 200 and contained generated `.flex` and
  `.bg-background` utilities, with the Tailwind import expanded.
- `npm run build` passed Turbopack compilation, TypeScript checking, and static
  page generation. The sandboxed build stalled; rerunning outside the sandbox
  completed successfully. Next.js emitted an unrelated `metadataBase` warning.
- The required code-quality review reported no actionable findings.

Reference: [Next.js Turbopack root directory configuration](https://nextjs.org/docs/app/api-reference/config/next-config-js/turbopack#root-directory).
