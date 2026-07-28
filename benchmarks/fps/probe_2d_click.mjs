// Probe: do point clicks work in pan mode? What does plotly_click carry?
import { chromium } from 'playwright-core';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = [];
const say = (...a) => { const s = a.join(' '); log.push(s); console.log(s); };

const browser = await chromium.launch({ channel: 'chrome', headless: false, args: ['--window-size=1500,950'] });
const page = await browser.newPage({ viewport: { width: 1400, height: 880 } });
await page.addInitScript(() => {
  localStorage.setItem('viz-preferences', JSON.stringify({ state: { mode: '2d', nebulaMode: false, showAxes: false }, version: 0 }));
});
await page.goto('http://localhost:3000/', { waitUntil: 'domcontentloaded' });
await page.waitForFunction(() => {
  const gd = document.querySelector('.js-plotly-plot');
  return !!(gd && gd._fullLayout && gd._fullLayout.xaxis && gd._fullData && gd._fullData.length);
}, null, { timeout: 120000 });
await sleep(3500);

await page.evaluate(() => {
  const gd = document.querySelector('.js-plotly-plot');
  window.__c = [];
  gd.on('plotly_click', (e) => {
    window.__c.push((e.points || []).map((p) => ({
      trace: p.data.name || p.data.mode || p.data.type,
      hasCustomdata: !!(p.customdata && typeof p.customdata === 'object'),
    })));
  });
  window.__h = [];
  gd.on('plotly_hover', (e) => {
    window.__h.push((e.points || []).map((p) => ({
      trace: p.data.name || p.data.mode || p.data.type,
      hasCustomdata: !!(p.customdata && typeof p.customdata === 'object'),
    })));
  });
});

const geom = await page.evaluate(() => {
  const gd = document.querySelector('.js-plotly-plot');
  const r = gd.getBoundingClientRect(), fl = gd._fullLayout;
  return { x: r.x, y: r.y, w: r.width, h: r.height, l: fl._size.l, t: fl._size.t, dragmode: fl.dragmode, nTraces: gd._fullData.length };
});
say('dragmode at start:', geom.dragmode, '| traces:', geom.nTraces);

// find dense screen positions of real points
const pts = await page.evaluate(() => {
  const gd = document.querySelector('.js-plotly-plot');
  const fl = gd._fullLayout, r = gd.getBoundingClientRect();
  const tr = (gd._fullData || []).filter((t) => t.x && t.x.length > 50).sort((a, b) => b.x.length - a.x.length)[0];
  if (!tr) return [];
  const out = [];
  for (const i of [10, 300, 900, 1500, 2500]) {
    if (i >= tr.x.length) continue;
    out.push({ i, x: r.x + fl._size.l + fl.xaxis.l2p(tr.x[i]), y: r.y + fl._size.t + fl.yaxis.l2p(tr.y[i]) });
  }
  return out;
});
say('candidate points:', pts.length);

const clickAt = async (p, label) => {
  const before = await page.evaluate(() => window.__c.length);
  await page.mouse.move(p.x, p.y); await sleep(350);
  const hovered = await page.evaluate(() => window.__h[window.__h.length - 1] || null);
  await page.mouse.down(); await sleep(50); await page.mouse.up();
  await sleep(900);
  const after = await page.evaluate(() => ({ n: window.__c.length, last: window.__c[window.__c.length - 1] || null }));
  say(`${label}: hover=${JSON.stringify(hovered)} clickFired=${after.n > before} payload=${JSON.stringify(after.last)}`);
  return after.n > before;
};

say('--- PAN MODE, no search active ---');
let fired = 0;
for (const [k, p] of pts.entries()) if (await clickAt(p, `click#${k}`)) fired++;
say(`clicks that fired plotly_click: ${fired}/${pts.length}`);

// Did the app react? (selected point / similar items table)
const reacted = await page.evaluate(() => ({
  table: !!document.body.innerText.includes('Similar Items'),
  traces: document.querySelector('.js-plotly-plot')._fullData.length,
}));
say('app reacted — Similar Items table:', reacted.table, '| traces now:', reacted.traces);

// Now with glow layers present, click a point again
say('--- PAN MODE, after a search (glow + constellation traces exist) ---');
const pts2 = await page.evaluate(() => {
  const gd = document.querySelector('.js-plotly-plot');
  const fl = gd._fullLayout, r = gd.getBoundingClientRect();
  const tr = (gd._fullData || []).filter((t) => t.x && t.x.length > 50).sort((a, b) => b.x.length - a.x.length)[0];
  if (!tr) return [];
  return [200, 1200, 2200].filter((i) => i < tr.x.length)
    .map((i) => ({ i, x: r.x + fl._size.l + fl.xaxis.l2p(tr.x[i]), y: r.y + fl._size.t + fl.yaxis.l2p(tr.y[i]) }));
});
let fired2 = 0;
for (const [k, p] of pts2.entries()) if (await clickAt(p, `post-search click#${k}`)) fired2++;
say(`clicks that fired: ${fired2}/${pts2.length}`);

// Click directly on a glow / highlighted point
say('--- click on a HIGHLIGHTED point (glow layers stack there) ---');
const glowPt = await page.evaluate(() => {
  const gd = document.querySelector('.js-plotly-plot');
  const fl = gd._fullLayout, r = gd.getBoundingClientRect();
  const glow = (gd._fullData || []).find((t) => /glow|Glow/.test(t.name || '') || (t.marker && t.marker.size > 20 && t.x && t.x.length < 200));
  if (!glow || !glow.x || !glow.x.length) return null;
  const i = 0;
  return { name: glow.name, x: r.x + fl._size.l + fl.xaxis.l2p(glow.x[i]), y: r.y + fl._size.t + fl.yaxis.l2p(glow.y[i]) };
});
if (glowPt) { say('glow trace name:', glowPt.name); await clickAt(glowPt, 'click-on-highlighted'); }
else say('(no glow trace found)');

// trace inventory
const inv = await page.evaluate(() => document.querySelector('.js-plotly-plot')._fullData.map((t, i) => ({
  i, name: t.name, mode: t.mode, n: t.x ? t.x.length : 0, hasCustomdata: !!t.customdata,
})));
say('trace inventory:', JSON.stringify(inv));

await page.screenshot({ path: '/tmp/probe_2d_click.png' });
await browser.close();
console.log('\n=== SUMMARY ===\n' + log.join('\n'));
