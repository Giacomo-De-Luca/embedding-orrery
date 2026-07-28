// Probe the LIVE HF Space demo: do point clicks work, and what element is under the cursor?
import { chromium } from 'playwright-core';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = [];
const say = (...a) => { const s = a.join(' '); log.push(s); console.log(s); };

const URL = process.argv[2] || 'https://giacomodeluca-orrery-demo.hf.space';
const MODE = process.argv[3] || '2d';

const browser = await chromium.launch({ channel: 'chrome', headless: process.env.HEADED !== '1', args: ['--window-size=1500,950', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1400, height: 880 } });
page.on('console', (m) => say(`[console.${m.type()}]`, m.text().slice(0, 300)));
page.on('pageerror', (e) => say('[pageerror]', String(e).slice(0, 300)));
page.on('requestfailed', (r) => say('[requestfailed]', r.url().slice(0, 120), r.failure()?.errorText));
page.on('response', async (r) => {
  if (!/graphql/i.test(r.url())) return;
  const status = r.status();
  let body = '';
  try { body = (await r.text()).slice(0, 400); } catch { body = '<unreadable>'; }
  const op = /"errors"/.test(body) ? 'HAS ERRORS' : 'ok';
  if (status !== 200 || op !== 'ok') say(`[graphql ${status} ${op}]`, body);
});

await page.addInitScript((mode) => {
  localStorage.setItem('viz-preferences', JSON.stringify({ state: { mode, nebulaMode: false, showAxes: false }, version: 0 }));
  localStorage.setItem('orrery.demo-intro.v1', 'seen');
  localStorage.setItem('orrery.demo-mobile-notice.v1', 'seen');
  localStorage.setItem('orrery.demo-tour.v1', 'seen');
}, MODE);

say(`### ${URL}  (mode=${MODE})`);
await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 180000 });

// dismiss any dialog that still shows
await sleep(6000);
for (const sel of ['button:has-text("Explore on my own")', 'button:has-text("Skip")', '[role="dialog"] button[aria-label="Close"]']) {
  const el = await page.$(sel);
  if (el) { await el.click().catch(() => {}); say('dismissed dialog via', sel); await sleep(1000); }
}

await page.waitForFunction((mode) => {
  const gd = document.querySelector('.js-plotly-plot');
  if (!gd || !gd._fullLayout || !gd._fullData || !gd._fullData.length) return false;
  return mode === '2d' ? !!gd._fullLayout.xaxis : !!gd._fullLayout.scene;
}, MODE, { timeout: 240000 });
await sleep(4000);

const info = await page.evaluate(() => {
  const gd = document.querySelector('.js-plotly-plot');
  return {
    dragmode: gd._fullLayout.dragmode,
    sceneDragmode: gd._fullLayout.scene ? gd._fullLayout.scene.dragmode : null,
    nTraces: gd._fullData.length,
    traces: gd._fullData.map((t) => ({ name: t.name, mode: t.mode, n: t.x ? t.x.length : 0 })),
  };
});
say('dragmode:', info.dragmode, '| scene.dragmode:', info.sceneDragmode, '| traces:', info.nTraces);
say('traces:', JSON.stringify(info.traces));

await page.evaluate(() => {
  const gd = document.querySelector('.js-plotly-plot');
  window.__c = 0; window.__hov = null; window.__hovN = 0;
  gd.on('plotly_click', () => { window.__c++; });
  gd.on('plotly_hover', (e) => {
    window.__hovN++;
    const p = e.points && e.points[0];
    window.__hov = p ? { trace: p.data.name || p.data.mode, cd: !!(p.customdata && typeof p.customdata === 'object') } : null;
  });
  gd.on('plotly_unhover', () => { window.__hov = null; });
});

// Sweep a grid; at each stop report the topmost DOM element and whether plotly hovered.
const geo = await page.evaluate(() => {
  const gd = document.querySelector('.js-plotly-plot');
  const r = gd.getBoundingClientRect();
  return { x: r.x, y: r.y, w: r.width, h: r.height };
});
say(`plot rect: x=${Math.round(geo.x)} y=${Math.round(geo.y)} w=${Math.round(geo.w)} h=${Math.round(geo.h)}`);

let hits = 0, clicksFired = 0, attempts = 0;
const topEls = new Map();
for (let gy = 0.15; gy < 0.9; gy += 0.08) {
  for (let gx = 0.15; gx < 0.9; gx += 0.08) {
    const x = geo.x + geo.w * gx, y = geo.y + geo.h * gy;
    await page.mouse.move(x, y);
    await sleep(80);
    const r = await page.evaluate(([px, py]) => {
      const el = document.elementFromPoint(px, py);
      const desc = el ? `${el.tagName.toLowerCase()}${el.className && typeof el.className === 'string' ? '.' + el.className.trim().split(/\s+/).slice(0, 3).join('.') : ''}` : 'none';
      return { desc, hov: window.__hov };
    }, [x, y]);
    topEls.set(r.desc, (topEls.get(r.desc) || 0) + 1);
    if (r.hov && r.hov.cd) {
      hits++;
      if (attempts < 6) {
        attempts++;
        const before = await page.evaluate(() => window.__c);
        await page.mouse.down(); await sleep(50); await page.mouse.up();
        await sleep(1500);
        const after = await page.evaluate(() => window.__c);
        if (after > before) clicksFired++;
        say(`  click on "${r.hov.trace}" under <${r.desc}> -> ${after > before ? 'REGISTERED' : 'DROPPED'}`);
      }
    }
  }
}
say(`grid stops that hovered a real point: ${hits}`);
say(`hover-confirmed clicks registered: ${clicksFired}/${attempts}`);
say('topmost element under cursor across the grid:', JSON.stringify([...topEls.entries()].slice(0, 8)));

const reacted = await page.evaluate(() => ({
  table: document.body.innerText.includes('Similar Items'),
  hoverN: window.__hovN,
  clicks: window.__c,
}));
say('plotly_hover events:', reacted.hoverN, '| plotly_click events:', reacted.clicks, '| Similar Items table opened:', reacted.table);

await page.screenshot({ path: `/tmp/probe_demo_${MODE}.png` });
await browser.close();
console.log('\n=== SUMMARY ===\n' + log.join('\n'));
