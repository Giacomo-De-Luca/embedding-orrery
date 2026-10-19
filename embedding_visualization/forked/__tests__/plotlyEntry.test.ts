/**
 * The slim plotly entry (fork patch 5 in forked/plotly_changes.md).
 *
 * `plotly.js` resolves to `forked/plotly.js/lib/index.js`, the file that registers the
 * trace types plotly can draw. Upstream registers ~48; the mapbox and map traces load
 * full map libraries as soon as the module is evaluated. The app only creates
 * `scattergl` and `scatter3d` traces, so the entry registers just those.
 *
 * The tests bundle `plotly.js` as the app imports it in Turbopack builds (esbuild, with
 * next.config.ts's `turbopack.resolveAlias`) and read the module graph. The webpack
 * Docker build has no maplibre CSS alias, so it still bundles that one stylesheet, as it
 * did before. The tests also scan the app's sources for the trace types it creates, so a
 * new chart type written as a `type:` string literal fails here instead of silently
 * drawing nothing in the browser.
 */
import { readdirSync, readFileSync } from 'node:fs';
import { builtinModules } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { beforeAll, describe, expect, it } from 'vitest';
import nextConfig from '../../next.config';

const FRONTEND = fileURLToPath(new URL('../..', import.meta.url));
const PLOTLY_DIR = path.join(FRONTEND, 'forked/plotly.js');

/** Every trace type plotly ships: one folder per trace under `src/traces`. */
const PLOTLY_TRACE_TYPES = new Set(readdirSync(path.join(PLOTLY_DIR, 'src/traces')));

/** The map, geo and calendar packages the full entry pulls in. */
const MAP_GEO_CALENDAR_PACKAGE = /^(maplibre-gl|@plotly\/mapbox-gl|world-calendars|d3-geo|d3-geo-projection|topojson-client|@turf\/.+)$/;

/** The module graph of `plotly.js` as the app bundles it. */
class PlotlyBundleGraph {
  private constructor(private readonly inputs: string[]) {}

  /** Bundles `require('plotly.js')` from the frontend root; input paths are relative to it. */
  static async build(): Promise<PlotlyBundleGraph> {
    const result = await build({
      stdin: { contents: "require('plotly.js');", resolveDir: FRONTEND, sourcefile: 'entry.js' },
      absWorkingDir: FRONTEND,
      bundle: true,
      write: false,
      metafile: true,
      platform: 'browser',
      logLevel: 'silent',
      alias: PlotlyBundleGraph.appAliases(),
      // Upstream's image trace requires node's `stream` and `assert`; leaving built-ins
      // external lets an entry that registers it still bundle, so the test reports why.
      external: [...builtinModules],
    });
    return new PlotlyBundleGraph(Object.keys(result.metafile.inputs));
  }

  /** next.config.ts's `turbopack.resolveAlias` (glslify stub, maplibre CSS stub) as absolute paths. */
  private static appAliases(): Record<string, string> {
    const aliases = Object.entries(nextConfig.turbopack?.resolveAlias ?? {});
    return Object.fromEntries(aliases.map(([request, target]) => {
      if (typeof target !== 'string') throw new Error(`unsupported resolveAlias shape for "${request}"`);
      return [request, path.resolve(FRONTEND, target)];
    }));
  }

  /** npm packages with at least one module in the bundle. */
  packages(): Set<string> {
    const names = this.inputs.map((input) => input.match(/node_modules\/((?:@[^/]+\/)?[^/]+)/)?.[1]);
    return new Set(names.filter((name): name is string => name !== undefined));
  }

  has(file: string): boolean {
    return this.inputs.includes(file);
  }
}

/** Trace types registered by plotly's entry file. Plotly's core registers `scatter` itself. */
class PlotlyEntry {
  static readonly FILE = path.join(PLOTLY_DIR, 'lib/index.js');
  static readonly CORE_TRACES = ['scatter'];

  static registeredTraces(): Set<string> {
    // Comments stripped, so a commented-out require doesn't count as registered.
    const source = readFileSync(PlotlyEntry.FILE, 'utf8').replace(/\/\*[\s\S]*?\*\/|\/\/.*$/gm, '');
    const required = [...source.matchAll(/require\(['"]\.\/([\w-]+)['"]\)/g)].map((match) => match[1]);
    return new Set(required.filter((name) => PLOTLY_TRACE_TYPES.has(name)));
  }
}

/**
 * Plotly trace types the app creates: string literals in the value of a `type:` key
 * (quoted keys and ternaries included) in its app/ and lib/ sources. A type held in a
 * variable or built at runtime is invisible to the scan.
 */
class AppTraceTypes {
  static readonly ROOTS = ['app', 'lib'];
  /** Tests, and `lib/references` (vendored reference code the app never imports). */
  static readonly SKIP = /(^|\/)(__tests__|node_modules)\/|^lib\/references\//;

  /** Trace type → the source files that create it (relative to the frontend root). */
  static scan(): Map<string, string[]> {
    const found = new Map<string, string[]>();
    for (const root of AppTraceTypes.ROOTS) {
      for (const file of readdirSync(path.join(FRONTEND, root), { recursive: true, encoding: 'utf8' })) {
        const relative = path.join(root, file);
        if (!/\.[jt]sx?$/.test(relative) || AppTraceTypes.SKIP.test(relative)) continue;
        const source = readFileSync(path.join(FRONTEND, relative), 'utf8');
        for (const [, value] of source.matchAll(/['"]?\btype['"]?\s*:\s*([^,;}\n]+)/g)) {
          for (const [, type] of value.matchAll(/['"`]([a-z0-9]+)['"`]/g)) {
            if (!PLOTLY_TRACE_TYPES.has(type)) continue;
            const files = found.get(type) ?? [];
            if (!files.includes(relative)) files.push(relative);
            found.set(type, files);
          }
        }
      }
    }
    return found;
  }
}

let graph: PlotlyBundleGraph;
let appTraces: Map<string, string[]>;

beforeAll(async () => {
  appTraces = AppTraceTypes.scan();
  graph = await PlotlyBundleGraph.build();
}, 60_000);

describe('slim plotly entry', () => {
  it('registers every trace type the app creates', () => {
    // Scanner sanity: the two scatter plots must be found.
    expect([...appTraces.keys()]).toEqual(expect.arrayContaining(['scattergl', 'scatter3d']));

    const available = new Set([...PlotlyEntry.registeredTraces(), ...PlotlyEntry.CORE_TRACES]);
    const unregistered = [...appTraces]
      .filter(([type]) => !available.has(type))
      .map(([type, files]) => `${type} (created in ${files.join(', ')}; register it in forked/plotly.js/lib/index.js)`);
    expect(unregistered).toEqual([]);
  });

  it('registers no trace type the app does not create', () => {
    const unused = [...PlotlyEntry.registeredTraces()]
      .filter((type) => !appTraces.has(type))
      .map((type) => `${type} (unused: drop it from the entry, or if the app picks this type at runtime, write it as a type: '${type}' literal)`);
    expect(unused).toEqual([]);
  });

  it('bundles none of the map, geo or calendar libraries', () => {
    const offending = [...graph.packages()].filter((name) => MAP_GEO_CALENDAR_PACKAGE.test(name));
    expect(offending).toEqual([]);
  });

  it('still bundles the 2D and 3D scatter renderers', () => {
    for (const file of [
      'forked/plotly.js/src/traces/scattergl/index.js',
      'forked/plotly.js/src/traces/scatter3d/index.js',
      'forked/plotly.js/stackgl_modules/index.js',
    ]) {
      expect(graph.has(file), file).toBe(true);
    }
    expect(graph.packages()).toContain('regl-scatter2d');
  });
});
