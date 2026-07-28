// Probe: click drop-rate in pan mode, fast vs slow, and correlation with re-renders.
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
  window.__c = 0; window.__replot = 0;
  gd.on('plotly_click', () => { window.__c++; });
  gd.on('plotly_afterplot', () => { window.__replot++; });
});

const pointAt = (i) => page.evaluate((idx) => {
  const gd = document.querySelector('.js-plotly-plot');
  const fl = gd._fullLayout, r = gd.getBoundingClientRect();
  const tr = (gd._fullData || []).filter((t) => t.x && t.x.length > 500).sort((a, b) => b.x.length - a.x.length)[0];
  if (!tr) return null;
  const k = (idx * 137) % tr.x.length;
  return { x: r.x + fl._size.l + fl.xaxis.l2p(tr.x[k]), y: r.y + fl._size.t + fl.yaxis.l2p(tr.y[k]) };
}, i);

async function burst(label, gapMs, n) {
  await page.evaluate(() => { window.__c = 0; window.__replot = 0; });
  let attempted = 0;
  for (let i = 0; i < n; i++) {
    const p = await pointAt(i + 1);
    if (!p) continue;
    await page.mouse.move(p.x, p.y);
    await sleep(120);
    await page.mouse.down(); await sleep(40); await page.mouse.up();
    attempted++;
    await sleep(gapMs);
  }
  await sleep(1500);
  const r = await page.evaluate(() => ({ c: window.__c, replot: window.__replot }));
  say(`${label}: ${r.c}/${attempted} clicks registered  (replots during burst: ${r.replot})`);
  return { fired: r.c, attempted };
}

say('--- pan mode, plot idle between clicks (2.5s gap) ---');
await burst('slow', 2500, 6);
say('--- pan mode, clicking while the previous search is still landing (250ms gap) ---');
await burst('fast', 250, 8);
say('--- pan mode, no gap at all ---');
await burst('rapid', 0, 8);

await browser.close();
console.log('\n=== SUMMARY ===\n' + log.join('\n'));
