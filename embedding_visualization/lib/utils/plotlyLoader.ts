/**
 * One shared import of the forked Plotly bundle.
 *
 * `plotly.js` resolves to the fork (`forked/plotly.js`, the `file:` dependency
 * in `package.json`, symlinked into `node_modules`): the source build, not
 * `plotly.js-dist-min`, so the fork's
 * patches (the gl3d scene fast path, the scattergl zoom marker style, …) are
 * what runs. It is the Explore page's heaviest download, and both scatter
 * plots used to import it in their own mount effect, which runs only once the
 * collection data has arrived and `DashboardPanel` mounts: a first visit
 * downloaded, parsed and GL-initialised Plotly strictly after the manifest and
 * data queries. `plotlyLoader.preload()` on page mount starts the download
 * while those queries run; the plots then await the same promise through
 * `plotlyLoader.load()`. See `documentation/EXPLORE_FIRST_LOAD.md`.
 */

// The import expression's type carries the synthesized `default` that the
// `typeof import('plotly.js')` type query lacks (@types/plotly.js has no
// default export; `esModuleInterop` adds one at runtime).
const importPlotly = () => import('plotly.js');

type PlotlyModule = Awaited<ReturnType<typeof importPlotly>>;

/** The Plotly library object (`newPlot`, `react`, `toImage`, …). */
export type PlotlyLib = PlotlyModule['default'];

export class PlotlyLoader {
  private pending: Promise<PlotlyLib> | null = null;

  constructor(private readonly importer: () => Promise<PlotlyModule>) {}

  /**
   * The library, from one import shared by every caller. A failed import is
   * forgotten so the next call retries it: a transient chunk-load failure
   * must not leave the plots stuck on a rejected promise.
   */
  load(): Promise<PlotlyLib> {
    if (this.pending === null) {
      const attempt: Promise<PlotlyLib> = this.importer().then(
        (mod) => mod.default,
        (error: unknown) => {
          if (this.pending === attempt) this.pending = null;
          throw error;
        },
      );
      this.pending = attempt;
    }
    return this.pending;
  }

  /** Start the import without waiting for it; a failure is left for the next `load()` to report. */
  preload(): void {
    this.load().catch(() => undefined);
  }
}

/** The app's loader of the forked Plotly bundle. */
export const plotlyLoader = new PlotlyLoader(importPlotly);
