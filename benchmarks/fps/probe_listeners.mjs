// Is the app's own plotly_click handler actually attached? Compare demo vs local.
import { chromium } from 'playwright-core';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = [];
const say = (...a) => { const s = a.join(' '); log.push(s); console.log(s); };

const URL = process.argv[2];
const MODE = process.argv[3] || '2d';

const browser = await chromium.launch({ channel: 'chrome', headless: process.env.HEADED !== '1', args: ['--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1400, height: 880 } });
await page.addInitScript((mode) => {
  localStorage.setItem('viz-preferences', JSON.stringify({ state: { mode, nebulaMode: false, showAxes: false }, version: 0 }));
  localStorage.setItem('orrery.demo-intro.v1', 'seen');
  localStorage.setItem('orrery.demo-mobile-notice.v1', 'seen');
  localStorage.setItem('orrery.demo-tour.v1', 'seen');
}, MODE);

say(`### ${URL} (mode=${MODE})`);
await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 180000 });
await sleep(6000);
await page.waitForFunction((mode) => {
  const gd = document.querySelector('.js-plotly-plot');
  if (!gd || !gd._fullLayout || !gd._fullData || !gd._fullData.length) return false;
  return mode === '2d' ? !!gd._fullLayout.xaxis : !!gd._fullLayout.scene;
}, MODE, { timeout: 240000 });

const counts = async (when) => {
  const r = await page.evaluate(() => {
    const gd = document.querySelector('.js-plotly-plot');
    const ev = gd._ev;
    const names = ev && ev.eventNames ? ev.eventNames() : [];
    const out = {};
    for (const n of names) out[n] = ev.listeners(n).length;
    return { names: out, hasEv: !!ev };
  });
  say(`${when}: listeners = ${JSON.stringify(r.names)}`);
  return r;
};

await sleep(2000);
await counts('right after plot ready');
await sleep(6000);
await counts('after 6s settle');

// Now click a point and see whether the app reacts (table appears)
const pt = await page.evaluate(() => {
  const gd = document.querySelector('.js-plotly-plot');
  const fl = gd._fullLayout, r = gd.getBoundingClientRect();
  if (!fl.xaxis) return null;
  const tr = (gd._fullData || []).filter((t) => t.x && t.x.length > 30).sort((a, b) => b.x.length - a.x.length)[0];
  if (!tr) return null;
  const i = Math.floor(tr.x.length / 3);
  return { x: r.x + fl._size.l + fl.xaxis.l2p(tr.x[i]), y: r.y + fl._size.t + fl.yaxis.l2p(tr.y[i]) };
});
if (pt) {
  await page.mouse.move(pt.x, pt.y); await sleep(500);
  await page.mouse.down(); await sleep(50); await page.mouse.up();
  await sleep(3000);
  const reacted = await page.evaluate(() => ({
    table: document.body.innerText.includes('Similar Items'),
    sidebarHasPoint: document.body.innerText.includes('Selected Point') || document.body.innerText.includes('Selected point'),
  }));
  say('after a real click -> Similar Items table:', reacted.table, '| selected-point panel:', reacted.sidebarHasPoint);
  await counts('after the click');
}

await page.screenshot({ path: `/tmp/probe_listeners_${MODE}_${URL.includes('localhost') ? 'local' : 'demo'}.png` });
await browser.close();
console.log('\n=== SUMMARY ===\n' + log.join('\n'));
