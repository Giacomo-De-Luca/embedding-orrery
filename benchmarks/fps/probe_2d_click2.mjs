// Probe: hover-confirmed clicks only. Isolates "cursor is on a point but the click does nothing".
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
  window.__c = 0; window.__hov = null;
  gd.on('plotly_click', () => { window.__c++; });
  gd.on('plotly_hover', (e) => {
    const p = e.points && e.points[0];
    window.__hov = p ? { trace: p.data.name || p.data.mode, cd: !!(p.customdata && typeof p.customdata === 'object') } : null;
  });
  gd.on('plotly_unhover', () => { window.__hov = null; });
});

// Find a screen position that is definitely over a hoverable point, by probing a grid.
async function findHoverablePoint(skip = 0) {
  const geo = await page.evaluate(() => {
    const gd = document.querySelector('.js-plotly-plot');
    const r = gd.getBoundingClientRect(), fl = gd._fullLayout;
    return { x: r.x, y: r.y, w: r.width, h: r.height, l: fl._size.l, t: fl._size.t, rr: fl._size.r, b: fl._size.b };
  });
  let found = 0;
  for (let gy = 0.25; gy < 0.8; gy += 0.06) {
    for (let gx = 0.2; gx < 0.85; gx += 0.05) {
      const x = geo.x + geo.l + (geo.w - geo.l - geo.rr) * gx;
      const y = geo.y + geo.t + (geo.h - geo.t - geo.b) * gy;
      await page.mouse.move(x, y);
      await sleep(90);
      const h = await page.evaluate(() => window.__hov);
      if (h && h.cd) {
        if (found++ < skip) continue;
        return { x, y, trace: h.trace };
      }
    }
  }
  return null;
}

async function hoverConfirmedClick(label) {
  const p = await findHoverablePoint(0);
  if (!p) { say(`${label}: could not land on any hoverable point`); return null; }
  const before = await page.evaluate(() => window.__c);
  await page.mouse.down(); await sleep(40); await page.mouse.up();
  await sleep(1200);
  const after = await page.evaluate(() => window.__c);
  say(`${label}: hovering "${p.trace}" -> click ${after > before ? 'REGISTERED' : 'DROPPED'}`);
  return { p, fired: after > before };
}

say('--- first click on a fresh plot ---');
const a = await hoverConfirmedClick('click 1');

say('--- click the SAME pixel again (point is now the selected one) ---');
if (a) {
  for (let i = 2; i <= 4; i++) {
    const before = await page.evaluate(() => window.__c);
    await page.mouse.move(a.p.x - 3, a.p.y - 3); await sleep(150);
    await page.mouse.move(a.p.x, a.p.y); await sleep(400);
    const h = await page.evaluate(() => window.__hov);
    await page.mouse.down(); await sleep(40); await page.mouse.up();
    await sleep(1200);
    const after = await page.evaluate(() => window.__c);
    say(`click ${i} on same point: hover=${JSON.stringify(h)} -> ${after > before ? 'REGISTERED' : 'DROPPED'}`);
  }
}

say('--- clicks on fresh points, plot fully idle each time ---');
let ok = 0, tries = 0;
for (let i = 0; i < 5; i++) {
  const r = await hoverConfirmedClick(`fresh click ${i + 1}`);
  if (r) { tries++; if (r.fired) ok++; }
  await sleep(1200);
}
say(`hover-confirmed clicks that registered: ${ok}/${tries}`);

await page.screenshot({ path: '/tmp/probe_2d_click2.png' });
await browser.close();
console.log('\n=== SUMMARY ===\n' + log.join('\n'));
