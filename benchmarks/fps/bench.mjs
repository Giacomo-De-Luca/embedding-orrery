// FPS + memory benchmark for the embedding visualization platform.
//
// Drives real Chrome (Metal GPU) over the live app, simulates continuous
// camera-drag interaction, and records requestAnimationFrame frame times
// plus JS-heap / process-RSS / system-RAM telemetry.
//
// Synthetic targets (synthetic_250k, synthetic_500k, synthetic_1m, or any
// synthetic_<n>[k|m]) never touch the database: the GetCollectionData
// GraphQL response is intercepted and replaced with generated Gaussian
// clusters, so everything downstream of the network — Apollo parsing,
// point transforms, Plotly WebGL — is the real platform code path.
//
// Usage: node bench.mjs [3d|2d|3d-nebula] [collection ...]
//   node bench.mjs 3d                       # full ladder (real + synthetic)
//   node bench.mjs 2d emotion synthetic_1m  # specific targets
//
// Prerequisites: backend on :8000, frontend on :3000, Google Chrome installed.
// Keep the driven Chrome window visible — Chrome throttles rAF when occluded.
import { chromium } from 'playwright-core';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import fs from 'node:fs';

const pExecFile = promisify(execFile);

const BASE = 'http://localhost:3000';
const PASSES = ['3d', '2d', '3d-nebula', '3d-topics'];
const PASS = PASSES.includes(process.argv[2]) ? process.argv[2] : '3d';
const MODE = PASS === '2d' ? '2d' : '3d';
const NEBULA = PASS === '3d-nebula';
const COLOR_TOPICS = NEBULA || PASS === '3d-topics';
// Opt-in renderer heap raise (MB), e.g. BENCH_HEAP_MB=7168 for 1M attempts on big machines.
const HEAP_MB = Number(process.env.BENCH_HEAP_MB) || 0;
const DRAG_MS = 8000;
const SETTLE_MS = 4000;
const SYN_CLUSTERS = Number(process.env.BENCH_SYN_CLUSTERS) || 50;
const RESULTS_DIR = new URL('./results/', import.meta.url).pathname;
const RESULTS_PATH = `${RESULTS_DIR}results_${PASS}.json`;
const BENCH_MARKER = '--bench-run-id=fpsbench';

const LADDER = [
  { name: 'emotion', expected: 1000 },
  { name: 'ag_news', expected: 10000 },
  { name: 'nrc_colours', expected: 19537 },
  { name: 'Concreteness-Ratings', expected: 39954 },
  { name: 'imdb', expected: 50000 },
  { name: 'gemma_29_65k', expected: 65536 },
  { name: 'lacan_sentences_gemini_document', expected: 153772 },
  { name: 'wordnet_senses_full', expected: 212478 },
  { name: 'synthetic_250k', expected: 250000, synthetic: true },
  { name: 'synthetic_500k', expected: 500000, synthetic: true },
  // synthetic_1m is deliberately NOT in the default ladder: it exceeds
  // Chrome's ~3.5-4 GB tab-heap cap with the current load-everything
  // architecture and OOM-crashes the browser. Run it explicitly when wanted:
  //   node bench.mjs 3d synthetic_1m   (optionally BENCH_HEAP_MB=7168)
];

function parseTargets() {
  const requested = process.argv.slice(3);
  if (!requested.length) return LADDER;
  return requested.map((arg) => {
    const known = LADDER.find((c) => c.name === arg);
    if (known) return known;
    const m = arg.match(/^synthetic_(\d+)(k|m)?$/i);
    if (m) {
      const mult = m[2]?.toLowerCase() === 'm' ? 1e6 : m[2] ? 1e3 : 1;
      return { name: arg.toLowerCase(), expected: +m[1] * mult, synthetic: true };
    }
    console.error(`unknown target: ${arg}`);
    process.exit(1);
  });
}

const VIZ_PREFS = JSON.stringify({
  state: {
    method: 'umap', mode: MODE, nebulaMode: NEBULA,
    showClusterLabels: false, showAllClusterLabels: false,
    showAxes: false, pointOpacity: 1.0, distanceMetric: 'COSINE',
  },
  version: 0,
});

// ---------------------------------------------------------------------------
// Synthetic data (GraphQL interception — nothing written to the database)
// ---------------------------------------------------------------------------

function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Full GetCollectionData response body for n points in Gaussian clusters. */
function buildSyntheticBody(n, clusters) {
  const rand = mulberry32(42);
  const gauss = () => {
    let u = 0, v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const centers = Array.from({ length: clusters }, () => [
    rand() * 14 - 7, rand() * 14 - 7, rand() * 14 - 7,
  ]);
  // Build each field as a JSON fragment directly — avoids holding n JS objects
  // AND their serialization in memory at once (matters at 1M points).
  const ids = new Array(n), docs = new Array(n), meta = new Array(n);
  const p3 = new Array(n), p2 = new Array(n);
  for (let i = 0; i < n; i++) {
    const c = i % clusters;
    const [cx, cy, cz] = centers[c];
    const x = (cx + gauss() * 0.9).toFixed(3);
    const y = (cy + gauss() * 0.9).toFixed(3);
    const z = (cz + gauss() * 0.9).toFixed(3);
    ids[i] = `"s${i}"`;
    docs[i] = `"Synthetic point ${i} (cluster ${c})"`;
    meta[i] = `{"topic_id":${c},"topic_label":"Cluster ${c < 10 ? '0' + c : c}"}`;
    p3[i] = `[${x},${y},${z}]`;
    p2[i] = `[${x},${y}]`;
  }
  const metadata = JSON.stringify({
    totalItems: n, embeddingDim: 384, timestamp: 'synthetic',
    pca2dVariance: null, pca3dVariance: null,
    sourceDataset: 'synthetic-benchmark', sourceSplit: null, sourceFile: null,
    hasProjections: true, embeddingProvider: 'synthetic',
    embeddingModel: 'synthetic', embeddingPrompt: null,
    fieldAnalysis: null, saeModelId: null, saeId: null,
  });
  return `{"data":{"collection":{"ids":[${ids.join(',')}],` +
    `"documents":[${docs.join(',')}],` +
    `"itemMetadata":[${meta.join(',')}],` +
    `"availableFields":["topic_id","topic_label"],` +
    `"pca2d":null,"pca3d":null,` +
    `"umap2d":[${p2.join(',')}],"umap3d":[${p3.join(',')}],` +
    `"metadata":${metadata}}}}`;
}

/**
 * GraphQL interception, installed for EVERY run:
 * - GetCollections → strips each collection's saved default_color_scheme so
 *   every baseline run loads uncolored/single-trace (a saved scheme splits
 *   points into one WebGL trace per category — not comparable across
 *   collections; coloring is exercised deliberately by 3d-topics/3d-nebula).
 *   For synthetic targets the generated entry is appended (page.tsx only
 *   loads a URL collection that exists in the list).
 * - Synthetic only: GetCollectionData for the target → generated payload;
 *   topics/probes/activations queries for it → benign empties.
 * Everything else passes through to the real backend.
 */
async function installRoutes(context, target, synBody) {
  await context.route('**/graphql', async (route) => {
    try {
      await handleGraphqlRoute(route, target, synBody);
    } catch {
      // Page/browser died mid-request — never let a route handler rejection
      // escape (it would take down the whole Node process).
      await route.continue().catch(() => {});
    }
  });
}

async function handleGraphqlRoute(route, target, synBody) {
  let payload = {};
  try { payload = JSON.parse(route.request().postData() || '{}'); } catch { /* not JSON */ }
  const op = payload.operationName;
  const vars = payload.variables || {};

    if (op === 'GetCollections') {
      const resp = await route.fetch();
      const json = await resp.json();
      if (json.data?.collections) {
        for (const col of json.data.collections) {
          if (col.metadata) delete col.metadata.default_color_scheme;
        }
        if (synBody) {
          json.data.collections.push({
            name: target.name,
            count: target.expected,
            metadata: { embedding_dim: 384, timestamp: 'synthetic', has_projections: true },
          });
        }
      }
      return route.fulfill({ response: resp, body: JSON.stringify(json) });
    }
    if (synBody) {
      if (op === 'GetCollectionData' && vars.name === target.name) {
        return route.fulfill({ contentType: 'application/json', body: synBody });
      }
      if (vars.collectionName === target.name) {
        const empties = {
          GetCollectionTopics: { collectionTopics: null },
          GetCollectionProbes: { collectionProbes: [] },
          HasDocumentActivations: { hasDocumentActivations: false },
        };
        if (op in empties) {
          return route.fulfill({
            contentType: 'application/json',
            body: JSON.stringify({ data: empties[op] }),
          });
        }
      }
    }
  return route.continue();
}

// ---------------------------------------------------------------------------
// Memory telemetry
// ---------------------------------------------------------------------------

async function psTable() {
  const { stdout } = await pExecFile('ps', ['-axo', 'pid=,ppid=,rss=,args='], {
    maxBuffer: 32 * 1024 * 1024,
  });
  return stdout.split('\n').map((l) => {
    const m = l.trim().match(/^(\d+)\s+(\d+)\s+(\d+)\s+(.*)$/);
    return m && { pid: +m[1], ppid: +m[2], rssKb: +m[3], args: m[4] };
  }).filter(Boolean);
}

async function findBenchChromeRoot() {
  const rows = await psTable();
  const roots = rows.filter((r) => r.args.includes(BENCH_MARKER) && !r.args.includes('--type='));
  if (!roots.length) throw new Error('bench Chrome root process not found');
  // Newest launch wins — orphans from previously crashed harness runs may linger.
  return Math.max(...roots.map((r) => r.pid));
}

/** RSS of the launched Chrome's process tree, split by role. MB. */
async function chromeMem(rootPid) {
  const rows = await psTable();
  const children = new Map();
  rows.forEach((r) => {
    if (!children.has(r.ppid)) children.set(r.ppid, []);
    children.get(r.ppid).push(r);
  });
  const tree = [];
  const stack = [rootPid];
  const byPid = new Map(rows.map((r) => [r.pid, r]));
  while (stack.length) {
    const pid = stack.pop();
    const row = byPid.get(pid);
    if (row) tree.push(row);
    for (const c of children.get(pid) || []) stack.push(c.pid);
  }
  const mb = (kb) => Math.round(kb / 1024);
  const renderers = tree.filter((r) => r.args.includes('--type=renderer'));
  const gpu = tree.find((r) => r.args.includes('--type=gpu-process'));
  return {
    chromeTotalMB: mb(tree.reduce((a, r) => a + r.rssKb, 0)),
    rendererMB: renderers.length ? mb(Math.max(...renderers.map((r) => r.rssKb))) : 0,
    gpuProcessMB: gpu ? mb(gpu.rssKb) : 0,
  };
}

/** System-wide RAM. macOS: available = free + speculative + inactive + purgeable. GB. */
async function systemMem() {
  const [{ stdout: vm }, { stdout: memsize }] = await Promise.all([
    pExecFile('vm_stat'),
    pExecFile('sysctl', ['-n', 'hw.memsize']),
  ]);
  const pageSize = +(vm.match(/page size of (\d+) bytes/)?.[1] ?? 16384);
  const pages = (k) => +(vm.match(new RegExp(`${k}:\\s+(\\d+)`))?.[1] ?? 0);
  const gb = (p) => +(p * pageSize / 1024 ** 3).toFixed(2);
  const free = pages('Pages free') + pages('Pages speculative');
  const avail = free + pages('Pages inactive') + pages('Pages purgeable');
  return {
    totalGB: +(Number(memsize) / 1024 ** 3).toFixed(1),
    freeGB: gb(free),
    availableGB: gb(avail),
  };
}

/**
 * usedJSHeapSize includes not-yet-collected garbage, so readings depend on GC
 * timing (observed: baseline 500k "3.2 GB" vs nebula 500k "0.55 GB" on the
 * same data — the nebula pipeline's allocations forced a major GC first).
 * Pass a CDP session to force a full GC before sampling → true live heap.
 * Do NOT force GC for mid-drag samples — the GC pause would distort FPS.
 */
async function jsHeap(page, cdp) {
  const read = () => page.evaluate(() => {
    const m = performance.memory;
    return m ? {
      heapUsedMB: Math.round(m.usedJSHeapSize / 1048576),
      heapTotalMB: Math.round(m.totalJSHeapSize / 1048576),
      heapLimitMB: Math.round(m.jsHeapSizeLimit / 1048576),
    } : null;
  });
  if (!cdp) return read();
  // Record pre-GC too — the delta shows the GC fired and how much was garbage.
  const pre = await read();
  await cdp.send('HeapProfiler.collectGarbage').catch(() => {});
  const post = await read();
  return post ? { ...post, heapPreGcMB: pre?.heapUsedMB } : pre;
}

async function memSnapshot(page, rootPid, cdp = null) {
  const [heap, chrome, sys] = await Promise.all([
    jsHeap(page, cdp).catch(() => null),
    chromeMem(rootPid).catch(() => null),
    systemMem().catch(() => null),
  ]);
  return { ...heap, ...chrome, ...sys };
}

// ---------------------------------------------------------------------------
// FPS measurement
// ---------------------------------------------------------------------------

function stats(deltasRaw) {
  // Drop the first few frames (ramp-in) and any absurd outliers from tab switches.
  const deltas = deltasRaw.slice(5).filter((d) => d > 0 && d < 5000);
  if (deltas.length < 20) return null;
  const sorted = [...deltas].sort((a, b) => a - b);
  const mean = deltas.reduce((a, b) => a + b, 0) / deltas.length;
  const median = sorted[Math.floor(sorted.length / 2)];
  const p95 = sorted[Math.floor(sorted.length * 0.95)];
  const worst1pct = sorted.slice(Math.floor(sorted.length * 0.99));
  const low1 = worst1pct.reduce((a, b) => a + b, 0) / worst1pct.length;
  return {
    frames: deltas.length,
    fpsMean: +(1000 / mean).toFixed(1),
    fpsMedian: +(1000 / median).toFixed(1),
    fps1pctLow: +(1000 / low1).toFixed(1),
    frameMsMean: +mean.toFixed(1),
    frameMsP95: +p95.toFixed(1),
    frameMsMax: +sorted[sorted.length - 1].toFixed(1),
  };
}

async function measureIdleRefresh(page) {
  return page.evaluate(async () => {
    const deltas = [];
    let last = performance.now();
    await new Promise((res) => {
      const tick = (t) => {
        deltas.push(t - last); last = t;
        if (deltas.length < 90) requestAnimationFrame(tick); else res();
      };
      requestAnimationFrame(tick);
    });
    deltas.splice(0, 10);
    return 1000 / (deltas.reduce((a, b) => a + b, 0) / deltas.length);
  });
}

// Runs in the page: counts plotted points, -1 while the plot isn't ready.
const countPoints = ({ mode }) => {
  const gd = document.querySelector('.js-plotly-plot');
  if (!gd || !gd._fullLayout || !gd.data) return -1;
  if (mode === '3d') {
    const scene = gd._fullLayout.scene && gd._fullLayout.scene._scene;
    if (!scene || !scene.glplot) return -1;
    return gd.data.reduce((n, t) => n + (t.type === 'scatter3d' && t.x ? t.x.length : 0), 0);
  }
  return gd.data.reduce((n, t) => n + ((t.type === 'scattergl' || t.type === 'scatter') && t.x ? t.x.length : 0), 0);
};

async function runOne(browser, rootPid, target) {
  const { name, expected } = target;
  const pageErrors = [];
  const result = { collection: name, expected, pass: PASS, synthetic: !!target.synthetic };
  let context = null;
  try {
    context = await browser.newContext({ viewport: { width: 1600, height: 950 } });
    await context.addInitScript((prefs) => {
      window.localStorage.setItem('viz-preferences', prefs);
    }, VIZ_PREFS);

    let synBody = null;
    if (target.synthetic) {
      console.log(`  generating ${expected.toLocaleString()} synthetic points…`);
      synBody = buildSyntheticBody(expected, SYN_CLUSTERS);
      console.log(`  payload ${(synBody.length / 1048576).toFixed(0)} MB`);
    }
    await installRoutes(context, target, synBody);

    const page = await context.newPage();
    page.on('pageerror', (e) => pageErrors.push(String(e).slice(0, 200)));
    const cdp = await context.newCDPSession(page);

    const t0 = Date.now();
    const colorBy = COLOR_TOPICS ? '&colorBy=topic_label' : '';
    await page.goto(`${BASE}/?collection=${encodeURIComponent(name)}${colorBy}`, {
      waitUntil: 'domcontentloaded', timeout: 60000,
    });

    // Wait until the plot exists and holds (almost) all points.
    await page.waitForFunction(
      (arg) => {
        const gd = document.querySelector('.js-plotly-plot');
        if (!gd || !gd._fullLayout || !gd.data) return false;
        let n;
        if (arg.mode === '3d') {
          const scene = gd._fullLayout.scene && gd._fullLayout.scene._scene;
          if (!scene || !scene.glplot) return false;
          n = gd.data.reduce((a, t) => a + (t.type === 'scatter3d' && t.x ? t.x.length : 0), 0);
        } else {
          n = gd.data.reduce((a, t) => a + ((t.type === 'scattergl' || t.type === 'scatter') && t.x ? t.x.length : 0), 0);
        }
        return n >= arg.want;
      },
      { mode: MODE, want: Math.floor(expected * 0.95) },
      { timeout: 300000, polling: 500 },
    );
    result.loadSeconds = +((Date.now() - t0) / 1000).toFixed(1);

    // gl3d emits plotly_click on ANY frame where the held-button pick lands
    // within 5px of a point (forked scene.js:459) — over dense clouds an orbit
    // drag fires mid-drag point selections whose fly-to animation pollutes the
    // FPS window. Swallow the event; hover picking and relayout stay live.
    await page.evaluate(() => {
      const gd = document.querySelector('.js-plotly-plot');
      if (gd && typeof gd.emit === 'function') {
        const orig = gd.emit.bind(gd);
        gd.emit = (name, ...args) => (name === 'plotly_click' ? undefined : orig(name, ...args));
      }
    });

    if (NEBULA) {
      // Give the haze overlay time to build its sprites before settling.
      await page.waitForTimeout(2000);
    }
    await page.waitForTimeout(SETTLE_MS);
    result.renderedPoints = await page.evaluate(countPoints, { mode: MODE });
    result.memPostLoad = await memSnapshot(page, rootPid, cdp);

    // Locate the WebGL canvas to drag on.
    const canvas = page.locator('.js-plotly-plot canvas').first();
    const box = await canvas.boundingBox();
    if (!box) throw new Error('no canvas bounding box');
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;

    // Start frame-time recording.
    await page.evaluate(() => {
      window.__bench = { deltas: [], running: true, last: performance.now() };
      const tick = (t) => {
        const b = window.__bench;
        b.deltas.push(t - b.last); b.last = t;
        if (b.running) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });

    // Continuous drag: orbit (3D) / pan (2D), sinusoidal path, ~120 moves/s.
    await page.mouse.move(cx, cy);
    await page.mouse.down();
    const ampX = Math.min(320, box.width * 0.3);
    const ampY = Math.min(140, box.height * 0.18);
    const t1 = Date.now();
    let midMemPromise = null;
    while (Date.now() - t1 < DRAG_MS) {
      const elapsed = Date.now() - t1;
      if (!midMemPromise && elapsed > DRAG_MS / 2) {
        // Fire-and-collect-later so sampling doesn't pause the drag.
        midMemPromise = memSnapshot(page, rootPid).catch(() => null);
      }
      const ph = elapsed / 1000;
      await page.mouse.move(
        cx + ampX * Math.sin(ph * 1.4),
        cy + ampY * Math.sin(ph * 0.9),
      );
      await new Promise((r) => setTimeout(r, 5));
    }
    await page.mouse.up();

    const deltas = await page.evaluate(() => {
      window.__bench.running = false;
      return window.__bench.deltas;
    });
    result.stats = stats(deltas);
    result.memMidDrag = midMemPromise ? await midMemPromise : null;
    result.memPostDrag = await memSnapshot(page, rootPid, cdp);
    result.pageErrors = pageErrors.slice(0, 5);
    await page.screenshot({ path: `${RESULTS_DIR}shot_${PASS}_${name}.png` });
  } catch (e) {
    let msg = String(e).slice(0, 300);
    if (/closed|crash/i.test(msg)) {
      msg += ' [tab or browser crashed — usually JS-heap/system-memory exhaustion]';
    }
    result.error = msg;
    result.pageErrors = pageErrors.slice(0, 5);
  } finally {
    // The context (or whole browser) may already be dead — never let cleanup
    // throw or hang.
    if (context) {
      await Promise.race([
        context.close().catch(() => {}),
        new Promise((r) => setTimeout(r, 10000)),
      ]);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fs.mkdirSync(RESULTS_DIR, { recursive: true });
const targets = parseTargets();

// An OOM-thrashed Chrome can hang Playwright calls indefinitely (observed:
// browser.close() never returning after the 1M tab crash). Never let a
// rejection or a hung call take the harness down — log and keep going.
process.on('unhandledRejection', (err) => {
  console.error('  unhandled rejection (continuing):', String(err).slice(0, 200));
});

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const TARGET_WATCHDOG_MS = 600000;

const LAUNCH_ARGS = ['--window-size=1680,1050', '--window-position=40,40', BENCH_MARKER];
if (HEAP_MB) LAUNCH_ARGS.push(`--js-flags=--max-old-space-size=${HEAP_MB}`);

const launchBrowser = () => chromium.launch({
  channel: 'chrome',
  headless: false,
  args: LAUNCH_ARGS,
});

/** Close politely with a timeout, then SIGKILL whatever bench Chrome remains. */
async function closeBrowserHard(b) {
  await Promise.race([b.close().catch(() => {}), sleep(15000)]);
  try {
    const rows = await psTable();
    for (const row of rows.filter((r) => r.args.includes(BENCH_MARKER))) {
      try { process.kill(row.pid, 'SIGKILL'); } catch { /* already gone */ }
    }
  } catch { /* best effort */ }
}

let browser = await launchBrowser();
let rootPid = await findBenchChromeRoot();

// Calibrate the display refresh ceiling on a blank page.
const calPage = await browser.newPage();
await calPage.goto('about:blank');
const idleRefresh = await measureIdleRefresh(calPage);
await calPage.close();

const sysBaseline = await systemMem();
console.log(`display refresh ceiling ≈ ${idleRefresh.toFixed(0)} Hz`);
console.log(`system RAM baseline: ${sysBaseline.freeGB} GB free / ${sysBaseline.availableGB} GB available of ${sysBaseline.totalGB} GB`);

// Merge with any existing results for this pass: re-running a subset of
// targets updates those entries without clobbering the rest of the ladder.
let existingRuns = [];
try { existingRuns = JSON.parse(fs.readFileSync(RESULTS_PATH, 'utf8')).runs || []; } catch { /* first run */ }
const runsByName = new Map(existingRuns.map((r) => [r.collection, r]));

const results = {
  pass: PASS, idleRefreshHz: +idleRefresh.toFixed(1), sysBaseline, runs: [],
};
for (const target of targets) {
  console.log(`\n=== ${target.name} (${target.expected.toLocaleString()} pts, ${PASS}) ===`);
  if (!browser.isConnected()) {
    console.log('  browser is gone (crashed on a previous target) — relaunching Chrome');
    await closeBrowserHard(browser);
    browser = await launchBrowser();
    rootPid = await findBenchChromeRoot();
  }
  let r = await Promise.race([
    runOne(browser, rootPid, target),
    sleep(TARGET_WATCHDOG_MS).then(() => null),
  ]);
  if (!r) {
    console.log(`  watchdog fired after ${TARGET_WATCHDOG_MS / 60000} min — force-restarting Chrome`);
    await closeBrowserHard(browser);
    browser = await launchBrowser();
    rootPid = await findBenchChromeRoot();
    r = {
      collection: target.name, expected: target.expected, pass: PASS,
      synthetic: !!target.synthetic, error: 'watchdog timeout — run hung (browser unresponsive)',
    };
  }
  runsByName.set(r.collection, r);
  results.runs = [...runsByName.values()].sort((a, b) => (a.expected || 0) - (b.expected || 0));
  fs.writeFileSync(RESULTS_PATH, JSON.stringify(results, null, 2));
  if (r.error) console.log(`  ERROR: ${r.error}`);
  else if (!r.stats) console.log('  too few frames recorded');
  else {
    const heap = r.memPostLoad || {};          // GC-forced → true live heap
    const m = r.memMidDrag || r.memPostDrag || {};
    console.log(
      `  load ${r.loadSeconds}s | rendered ${r.renderedPoints.toLocaleString()} | ` +
      `FPS mean ${r.stats.fpsMean} / median ${r.stats.fpsMedian} / 1% low ${r.stats.fps1pctLow} | ` +
      `p95 ${r.stats.frameMsP95}ms`,
    );
    console.log(
      `  live heap ${heap.heapUsedMB ?? '?'} MB | renderer ${m.rendererMB ?? '?'} MB | ` +
      `gpu-proc ${m.gpuProcessMB ?? '?'} MB | sys free ${m.freeGB ?? '?'} GB / avail ${m.availableGB ?? '?'} GB`,
    );
  }
}

await closeBrowserHard(browser);
console.log(`\nresults → ${RESULTS_PATH}`);
process.exit(0);
