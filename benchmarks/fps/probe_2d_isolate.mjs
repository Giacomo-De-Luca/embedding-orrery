// Probe: do clicks still work in 2D after isolating a single topic/cluster?
import { chromium } from 'playwright-core';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = [];
const say = (...a) => { const s = a.join(' '); log.push(s); console.log(s); };

const BASE = process.argv[2] || 'http://localhost:3000/';

const browser = await chromium.launch({ channel: 'chrome', headless: process.env.HEADED !== '1', args: ['--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1400, height: 880 } });
page.on('console', (m) => { const t = m.text(); if (/Point clicked|error/i.test(t)) say(`[console.${m.type()}]`, t.slice(0, 200)); });

await page.addInitScript(() => {
  localStorage.setItem('viz-preferences', JSON.stringify({ state: { mode: '2d', nebulaMode: false, showAxes: false }, version: 0 }));
  localStorage.setItem('orrery.demo-intro.v1', 'seen');
  localStorage.setItem('orrery.demo-mobile-notice.v1', 'seen');
  localStorage.setItem('orrery.demo-tour.v1', 'seen');
});

const url = BASE + (BASE.includes('?') ? '&' : '?') + 'colorBy=topic_label';
say('###', url);
await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 180000 });
await sleep(7000);
await page.waitForFunction(() => {
  const gd = document.querySelector('.js-plotly-plot');
  return !!(gd && gd._fullLayout && gd._fullLayout.xaxis && gd._fullData && gd._fullData.length);
}, null, { timeout: 240000 });
await sleep(4000);

await page.evaluate(() => {
  const gd = document.querySelector('.js-plotly-plot');
  window.__c = 0; window.__hov = null;
  gd.on('plotly_click', () => { window.__c++; });
  gd.on('plotly_hover', (e) => {
    const p = e.points && e.points[0];
    window.__hov = p ? { trace: p.data.name, cd: !!(p.customdata && typeof p.customdata === 'object'), op: p.data.marker && p.data.marker.opacity } : null;
  });
  gd.on('plotly_unhover', () => { window.__hov = null; });
});

const traceInfo = () => page.evaluate(() => {
  const gd = document.querySelector('.js-plotly-plot');
  return gd._fullData.map((t) => ({ name: t.name, n: t.x ? t.x.length : 0, op: t.marker && t.marker.opacity, color: t.marker && t.marker.color }));
});
say('traces before isolation:', JSON.stringify((await traceInfo()).slice(0, 6)), '… total', (await traceInfo()).length);

// hover-confirmed click helper: sweep grid until we hover a point of the given trace name (or any)
async function clickOn(wantTrace, label) {
  const geo = await page.evaluate(() => {
    const gd = document.querySelector('.js-plotly-plot');
    const r = gd.getBoundingClientRect(), fl = gd._fullLayout;
    return { x: r.x, y: r.y, w: r.width, h: r.height, l: fl._size.l, t: fl._size.t, rr: fl._size.r, b: fl._size.b };
  });
  for (let gy = 0.12; gy < 0.92; gy += 0.035) {
    for (let gx = 0.12; gx < 0.92; gx += 0.035) {
      const x = geo.x + geo.l + (geo.w - geo.l - geo.rr) * gx;
      const y = geo.y + geo.t + (geo.h - geo.t - geo.b) * gy;
      await page.mouse.move(x, y); await sleep(60);
      const h = await page.evaluate(() => window.__hov);
      if (!h || !h.cd) continue;
      if (wantTrace && h.trace !== wantTrace) continue;
      const before = await page.evaluate(() => window.__c);
      await page.mouse.down(); await sleep(50); await page.mouse.up();
      await sleep(1600);
      const after = await page.evaluate(() => window.__c);
      const reacted = await page.evaluate(() => document.body.innerText.includes('Similar Items'));
      say(`${label}: hovered "${h.trace}" (opacity ${h.op}) -> plotly_click ${after > before ? 'FIRED' : 'DROPPED'} | app reacted: ${reacted}`);
      return { fired: after > before, trace: h.trace, reacted };
    }
  }
  say(`${label}: never landed on a hoverable point`);
  return null;
}

say('--- baseline: click any point, nothing isolated ---');
const base = await clickOn(null, 'baseline click');

// Open the legend / find a clickable topic row and isolate it
say('--- isolating a single topic via the legend ---');
const legendRows = await page.$$('[role="button"]');
say('elements with role=button on page:', legendRows.length);
let isolated = null;
for (const el of legendRows) {
  const txt = (await el.innerText().catch(() => '')).trim().split('\n')[0];
  if (!txt || txt.length > 40) continue;
  await el.click().catch(() => {});
  await sleep(2500);
  const after = await traceInfo();
  const nonGray = after.filter((t) => t.color && t.color !== '#9ca3af');
  if (nonGray.length && nonGray.length < after.length) { isolated = { txt, after, nonGray }; break; }
}
if (!isolated) { say('could not isolate a topic via legend rows'); }
else {
  say(`isolated "${isolated.txt}" -> ${isolated.nonGray.length} coloured trace(s) of ${isolated.after.length}`);
  say('coloured traces:', JSON.stringify(isolated.nonGray));
  say('--- click a point INSIDE the isolated cluster ---');
  await clickOn(isolated.nonGray[0].name, 'isolated-cluster click');
  say('--- click a point in a MUTED cluster ---');
  const muted = isolated.after.find((t) => t.color === '#9ca3af' && t.n > 5);
  if (muted) await clickOn(muted.name, 'muted-cluster click');
}

const listeners = await page.evaluate(() => {
  const ev = document.querySelector('.js-plotly-plot')._ev;
  const o = {}; for (const n of ev.eventNames()) o[n] = ev.listeners(n).length; return o;
});
say('listeners at end:', JSON.stringify(listeners));

await page.screenshot({ path: '/tmp/probe_2d_isolate.png' });
await browser.close();
console.log('\n=== SUMMARY ===\n' + log.join('\n'));
