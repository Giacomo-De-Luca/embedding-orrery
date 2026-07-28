// Probe v2: isolate ONE topic cleanly, then click the isolated cluster's own points by coordinate.
import { chromium } from 'playwright-core';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = [];
const say = (...a) => { const s = a.join(' '); log.push(s); console.log(s); };

const BASE = process.argv[2] || 'http://localhost:3000/';
const browser = await chromium.launch({ channel: 'chrome', headless: process.env.HEADED !== '1', args: ['--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1400, height: 880 } });
page.on('console', (m) => { const t = m.text(); if (/Point clicked/i.test(t)) say('  [app]', t.slice(0, 120)); });
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
  return !!(gd && gd._fullLayout && gd._fullLayout.xaxis && gd._fullData && gd._fullData.length > 1);
}, null, { timeout: 240000 });
await sleep(4000);

await page.evaluate(() => {
  const gd = document.querySelector('.js-plotly-plot');
  window.__c = 0; window.__hovN = 0; window.__hov = null;
  gd.on('plotly_click', () => { window.__c++; });
  gd.on('plotly_hover', (e) => { window.__hovN++; const p = e.points[0]; window.__hov = { trace: p.data.name, cd: !!(p.customdata && typeof p.customdata === 'object') }; });
  gd.on('plotly_unhover', () => { window.__hov = null; });
});

const state = () => page.evaluate(() => {
  const gd = document.querySelector('.js-plotly-plot');
  const fl = gd._fullLayout;
  return {
    n: gd._fullData.length,
    xr: fl.xaxis.range.map((v) => +v.toFixed(2)),
    yr: fl.yaxis.range.map((v) => +v.toFixed(2)),
    traces: gd._fullData.map((t) => ({ name: t.name, n: t.x ? t.x.length : 0, op: t.marker && t.marker.opacity, gray: t.marker && t.marker.color === '#9ca3af' })),
  };
});

// click points of a named trace, by its own coordinates
async function clickTrace(name, label, howMany = 3) {
  const spots = await page.evaluate((nm) => {
    const gd = document.querySelector('.js-plotly-plot');
    const fl = gd._fullLayout, r = gd.getBoundingClientRect();
    const t = gd._fullData.find((d) => d.name === nm);
    if (!t || !t.x || !t.x.length) return [];
    const out = [];
    const step = Math.max(1, Math.floor(t.x.length / 6));
    for (let i = 0; i < t.x.length && out.length < 6; i += step) {
      const px = r.x + fl._size.l + fl.xaxis.l2p(t.x[i]);
      const py = r.y + fl._size.t + fl.yaxis.l2p(t.y[i]);
      const inside = px > r.x && px < r.x + r.width && py > r.y && py < r.y + r.height;
      out.push({ px, py, inside });
    }
    return out;
  }, name);
  const usable = spots.filter((s) => s.inside).slice(0, howMany);
  say(`${label}: "${name}" has ${spots.length} sampled points, ${usable.length} on screen`);
  let fired = 0, hovered = 0;
  for (const s of usable) {
    await page.mouse.move(s.px - 4, s.py - 4); await sleep(120);
    await page.mouse.move(s.px, s.py); await sleep(450);
    const h = await page.evaluate(() => window.__hov);
    if (h && h.cd) hovered++;
    const before = await page.evaluate(() => window.__c);
    await page.mouse.down(); await sleep(50); await page.mouse.up();
    await sleep(1500);
    const after = await page.evaluate(() => window.__c);
    if (after > before) fired++;
    say(`   at (${Math.round(s.px)},${Math.round(s.py)}): hover=${h ? h.trace : 'NONE'} click=${after > before ? 'FIRED' : 'DROPPED'}`);
  }
  say(`${label}: hovered ${hovered}/${usable.length}, clicks fired ${fired}/${usable.length}`);
  return { hovered, fired, n: usable.length };
}

const s0 = await state();
say(`before isolation: ${s0.n} traces, x=${JSON.stringify(s0.xr)} y=${JSON.stringify(s0.yr)}`);
const target = s0.traces.filter((t) => t.n >= 15 && t.name && t.name !== 'Unclustered')[0];
say('target topic:', target.name, `(${target.n} points)`);
await clickTrace(target.name, 'BEFORE isolation', 2);

// isolate exactly that topic by clicking its legend row once
say('--- isolating via its legend row (single click) ---');
const rows = await page.$$('[role="button"]');
let clicked = false;
for (const el of rows) {
  const txt = (await el.innerText().catch(() => '')).trim();
  if (txt.startsWith(target.name)) { await el.click(); clicked = true; break; }
}
say('legend row clicked:', clicked);
await sleep(3000);
const s1 = await state();
say(`after isolation: ${s1.n} traces, x=${JSON.stringify(s1.xr)} y=${JSON.stringify(s1.yr)}`);
const coloured = s1.traces.filter((t) => !t.gray && t.n > 5);
say('non-gray traces:', JSON.stringify(coloured.slice(0, 5)));
say(`axis range changed: ${JSON.stringify(s0.xr) !== JSON.stringify(s1.xr) || JSON.stringify(s0.yr) !== JSON.stringify(s1.yr)}`);

await clickTrace(target.name, 'AFTER isolation (inside the selected cluster)', 3);
const grayOne = s1.traces.find((t) => t.gray && t.n > 20);
if (grayOne) await clickTrace(grayOne.name, 'AFTER isolation (a muted cluster)', 2);

const listeners = await page.evaluate(() => { const ev = document.querySelector('.js-plotly-plot')._ev; const o = {}; for (const n of ev.eventNames()) o[n] = ev.listeners(n).length; return o; });
say('listeners:', JSON.stringify(listeners));
say('total plotly_hover events seen:', await page.evaluate(() => window.__hovN));

await page.screenshot({ path: '/tmp/probe_2d_isolate2.png' });
await browser.close();
console.log('\n=== SUMMARY ===\n' + log.join('\n'));
