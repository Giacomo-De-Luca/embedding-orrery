// Probe 2: once a lasso selection exists, what clears it?
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
await sleep(3000);

const state = () => page.evaluate(() => {
  const gd = document.querySelector('.js-plotly-plot');
  return {
    dragmode: gd._fullLayout.dragmode,
    selTraces: (gd._fullData || []).filter((t) => t.selectedpoints).length,
    selCount: (gd._fullData || []).reduce((a, t) => a + (t.selectedpoints ? t.selectedpoints.length : 0), 0),
    nTraces: (gd._fullData || []).length,
    hasOutline: !!document.querySelector('.select-outline'),
  };
});
const clickBtn = async (i) => { await page.evaluate((idx) => document.querySelectorAll('.modebar-btn')[idx].click(), i); await sleep(400); };

const box = await page.evaluate(() => { const r = document.querySelector('.js-plotly-plot').getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; });
const cx = box.x + box.w / 2, cy = box.y + box.h / 2;
const lassoDrag = async () => {
  await page.mouse.move(cx - 130, cy - 130); await page.mouse.down();
  for (const [dx, dy] of [[130, -130], [130, 130], [-130, 130], [-130, -128]]) { await page.mouse.move(cx + dx, cy + dy, { steps: 12 }); await sleep(60); }
  await page.mouse.up(); await sleep(1200);
};

await clickBtn(2); await lassoDrag();
say('after lasso:', JSON.stringify(await state()));

// 1) double-click while still in lasso mode
await page.mouse.move(cx + 300, cy + 250); await page.mouse.dblclick(cx + 300, cy + 250); await sleep(1000);
say('after dblclick IN lasso mode:', JSON.stringify(await state()));

// re-select, then switch to pan and try to recover from there
await lassoDrag();
say('re-selected:', JSON.stringify(await state()));
await clickBtn(0);
say('after switching to pan:', JSON.stringify(await state()));
await page.mouse.dblclick(cx + 300, cy + 250); await sleep(1200);
say('after dblclick IN pan mode:', JSON.stringify(await state()));

// pan drag in pan mode (dragbox clearAndResetSelect path)
await page.mouse.move(cx, cy); await page.mouse.down();
await page.mouse.move(cx + 60, cy + 40, { steps: 10 }); await page.mouse.up(); await sleep(1000);
say('after a pan drag:', JSON.stringify(await state()));

await page.screenshot({ path: '/tmp/probe_2d_recover.png' });
await browser.close();
console.log('\n=== SUMMARY ===\n' + log.join('\n'));
