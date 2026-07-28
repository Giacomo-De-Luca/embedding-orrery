// Ad-hoc probe: does the 2D scatter plot's modebar navigation actually work?
// Run from benchmarks/fps (uses its playwright-core).
import { chromium } from 'playwright-core';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const out = [];
const say = (...a) => { const s = a.join(' '); out.push(s); console.log(s); };

const browser = await chromium.launch({
  channel: 'chrome',
  headless: false,
  args: ['--window-size=1500,950', '--window-position=20,20'],
});
const page = await browser.newPage({ viewport: { width: 1400, height: 880 } });

await page.addInitScript(() => {
  localStorage.setItem('viz-preferences', JSON.stringify({
    state: { mode: '2d', method: 'umap', nebulaMode: false, showAxes: false, pointOpacity: 1 },
    version: 0,
  }));
  localStorage.setItem('orrery.demo-intro.v1', 'seen');
  localStorage.setItem('orrery.demo-mobile-notice.v1', 'seen');
});

await page.goto('http://localhost:3000/', { waitUntil: 'domcontentloaded' });

// Wait for the 2D plot to be live (scattergl scene created)
await page.waitForFunction(() => {
  const gd = document.querySelector('.js-plotly-plot');
  return !!(gd && gd._fullLayout && gd._fullLayout.xaxis && gd._fullData && gd._fullData.length);
}, null, { timeout: 120000 });
await sleep(3000);

// Instrument: record plotly events
await page.evaluate(() => {
  const gd = document.querySelector('.js-plotly-plot');
  window.__ev = { selected: 0, deselect: 0, click: 0, selecting: 0, lastSelectedCount: null };
  gd.on('plotly_selected', (e) => { window.__ev.selected++; window.__ev.lastSelectedCount = e && e.points ? e.points.length : 0; });
  gd.on('plotly_selecting', () => { window.__ev.selecting++; });
  gd.on('plotly_deselect', () => { window.__ev.deselect++; });
  gd.on('plotly_click', () => { window.__ev.click++; });
});

const snap = () => page.evaluate(() => {
  const gd = document.querySelector('.js-plotly-plot');
  const btns = [...document.querySelectorAll('.modebar-btn')].map((b, i) => ({
    i,
    name: b.getAttribute('data-title') || b.className,
    attr: b.getAttribute('data-attr'),
    val: b.getAttribute('data-val'),
    active: b.classList.contains('active'),
  }));
  const selCounts = (gd._fullData || []).map((t) => (t.selectedpoints ? t.selectedpoints.length : null));
  return {
    dragmode: gd._fullLayout.dragmode,
    layoutPropDragmode: gd.layout.dragmode,
    btns,
    anyActive: btns.some((b) => b.active),
    tracesWithSelection: selCounts.filter((c) => c !== null).length,
    selCounts,
    nTraces: (gd._fullData || []).length,
    ev: window.__ev,
  };
});

const s0 = await snap();
say('--- initial ---');
say('dragmode:', s0.dragmode, '| traces:', s0.nTraces);
say('modebar buttons:', JSON.stringify(s0.btns));

// helper to click the Nth modebar button
const clickBtn = async (i) => {
  await page.evaluate((idx) => document.querySelectorAll('.modebar-btn')[idx].click(), i);
  await sleep(400);
};

// buttons: 0 pan, 1 box-select, 2 lasso, 3 reset, 4 camera (per build2DModeBarButtons)
await clickBtn(2);
const s1 = await snap();
say('--- after clicking lasso button (idx 2) ---');
say('dragmode:', s1.dragmode, '| gd.layout.dragmode:', s1.layoutPropDragmode, '| any modebar button shows active:', s1.anyActive);

// Perform a lasso drag across the middle of the plot
const box = await page.evaluate(() => {
  const gd = document.querySelector('.js-plotly-plot');
  const r = gd.getBoundingClientRect();
  return { x: r.x, y: r.y, w: r.width, h: r.height };
});
const cx = box.x + box.w / 2, cy = box.y + box.h / 2;
await page.mouse.move(cx - 120, cy - 120);
await page.mouse.down();
for (const [dx, dy] of [[120, -120], [120, 120], [-120, 120], [-120, -118]]) {
  await page.mouse.move(cx + dx, cy + dy, { steps: 12 });
  await sleep(60);
}
await page.mouse.up();
await sleep(1200);

const s2 = await snap();
say('--- after lasso drag ---');
say('plotly_selecting fired:', s2.ev.selecting, '| plotly_selected fired:', s2.ev.selected, '| points in selection:', s2.ev.lastSelectedCount);
say('traces carrying selectedpoints:', s2.tracesWithSelection, '/', s2.nTraces);

// Click a single point while still in lasso mode
const pt = await page.evaluate(() => {
  const gd = document.querySelector('.js-plotly-plot');
  const fl = gd._fullLayout, r = gd.getBoundingClientRect();
  const tr = (gd._fullData || []).find((t) => t.x && t.x.length > 20);
  if (!tr) return null;
  const i = Math.floor(tr.x.length / 2);
  return { x: r.x + fl._size.l + fl.xaxis.l2p(tr.x[i]), y: r.y + fl._size.t + fl.yaxis.l2p(tr.y[i]) };
});
if (pt) {
  const before = (await snap()).ev.click;
  await page.mouse.move(pt.x, pt.y);
  await sleep(300);
  await page.mouse.down(); await sleep(60); await page.mouse.up();
  await sleep(800);
  const after = (await snap()).ev.click;
  say('--- single click on a point while in lasso mode ---');
  say('plotly_click fired:', after > before, `(count ${before} -> ${after})`);
}

// Now switch back to pan and see whether the selection dimming survives
await clickBtn(0);
await sleep(600);
const s3 = await snap();
say('--- after clicking pan button ---');
say('dragmode:', s3.dragmode, '| traces still carrying selectedpoints:', s3.tracesWithSelection);

// Re-enter lasso, then force a layout-memo recompute by resizing the window
await clickBtn(2);
say('--- dragmode before resize:', (await snap()).dragmode);
await page.setViewportSize({ width: 1200, height: 760 });
await sleep(1500);
const s4 = await snap();
say('--- after window resize (layout memo recomputes) ---');
say('dragmode:', s4.dragmode, '| gd.layout.dragmode:', s4.layoutPropDragmode);

// Also: does a theme toggle / any other layout-dep change reset it? try box-select then resize
await clickBtn(1);
say('--- dragmode after box-select click:', (await snap()).dragmode);
await page.setViewportSize({ width: 1340, height: 840 });
await sleep(1500);
say('--- dragmode after another resize:', (await snap()).dragmode);

await page.screenshot({ path: '/tmp/probe_2d_nav.png' });
await browser.close();
console.log('\n=== SUMMARY ===\n' + out.join('\n'));
