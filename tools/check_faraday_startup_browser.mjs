import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdir, writeFile, stat } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium, webkit } from 'playwright';

const engineName = process.env.FARADAY_BROWSER || 'chromium';
const engine = { chromium, webkit }[engineName];
assert(engine, `Unknown browser engine: ${engineName}`);

const root = resolve(process.argv[2] || '.');
const out = resolve(process.env.FARADAY_QA_ARTIFACTS || 'faraday-browser-evidence');
await mkdir(out, { recursive: true });
const errors = [];
const results = [];
const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname === '/favicon.ico') { res.writeHead(204); res.end(); return; }
    const path = resolve(root, '.' + decodeURIComponent(url.pathname));
    if (path !== root && !path.startsWith(root + sep)) { res.writeHead(403); res.end(); return; }
    const target = (await stat(path)).isDirectory() ? resolve(path, 'index.html') : path;
    const type = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.wasm': 'application/wasm' }[extname(target)] || 'application/octet-stream';
    res.writeHead(200, { 'Content-Type': type });
    res.end(await readFile(target));
  } catch (error) { res.writeHead(404); res.end(String(error)); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
let browser, activePage, activeScenario, activePageErrors;
try {
  browser = await engine.launch({ headless: true });
  for (const scenario of [
    { name: 'folder-http', url: origin + '/index.html', viewport: { width: 1440, height: 1000 } },
    { name: 'source-http', url: origin + '/cymatic.html', viewport: { width: 1440, height: 1000 } },
    { name: 'folder-file', url: pathToFileURL(resolve(root, 'index.html')).href, viewport: { width: 1440, height: 1000 } },
    { name: 'standalone-file', url: pathToFileURL(resolve(root, 'faraday-cell-standalone.html')).href, viewport: { width: 1440, height: 1000 } },
    { name: 'folder-mobile', url: origin + '/index.html', viewport: { width: 390, height: 844 } }
  ]) {
    const context = await browser.newContext({ viewport: scenario.viewport });
    const page = await context.newPage();
    const pageErrors = [];
    activePage = page;
    activeScenario = scenario.name;
    activePageErrors = pageErrors;
    page.on('pageerror', error => pageErrors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') pageErrors.push(message.text()); });
    await page.goto(scenario.url, { waitUntil: 'load', timeout: 30000 });
    await page.waitForFunction(() => typeof simT !== 'undefined' && simT > 0.5 && state && !renderError, undefined, { timeout: 30000 });
    assert.match(await page.title(), /Faraday|cymatic/i);
    assert.equal(await page.locator('#c').isVisible(), true);
    const initial = await page.evaluate(() => {
      const canvas = document.getElementById('c');
      const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      let min = 255, max = 0;
      for (let i = 0; i < pixels.length; i += 4) { min = Math.min(min, pixels[i]); max = Math.max(max, pixels[i]); }
      return { simT, running, renderError, renderEngine, useSolver, expression: state.expression, spread: max - min };
    });
    assert.match(await page.locator('#startupStatus').innerText(), /Canvas rendering started/);
    assert.equal(initial.running, true);
    assert.equal(initial.renderError, null);
    assert.equal(initial.renderEngine, 'cpu');
    assert.equal(initial.useSolver, false);
    assert(initial.expression > 0);
    assert(initial.spread > 10, `Canvas pixel spread ${initial.spread} is unexpectedly flat`);
    const initialPixels = await page.locator('#c').evaluate(canvas => canvas.toDataURL());
    await page.waitForFunction(t => simT > t + 0.3, initial.simT);
    await page.waitForFunction(pixels => document.getElementById('c').toDataURL() !== pixels, initialPixels);
    results.push({ name: scenario.name + '-canvas-changes', passed: true });
    await page.screenshot({ path: resolve(out, `${scenario.name}-running.png`), fullPage: true });
    await page.locator('#play').click();
    const paused = await page.evaluate(() => ({ running, simT }));
    assert.equal(paused.running, false);
    await page.waitForTimeout(600);
    assert.equal(await page.evaluate(() => simT), paused.simT);
    const resumeWallStart = await page.evaluate(() => performance.now());
    await page.locator('#play').click();
    await page.waitForFunction(t => running && simT > t, paused.simT);
    const resumed = await page.evaluate(() => simT);
    const resumeWallEnd = await page.evaluate(() => performance.now());
    assert(resumed - paused.simT <= (resumeWallEnd - resumeWallStart) / 1000 + 0.1, `Resume included paused wall time: ${(resumed - paused.simT).toFixed(3)} s`);
    await page.locator('#reset').click();
    await page.waitForFunction(() => simT < 0.5);
    assert.equal(await page.evaluate(() => running), true);
    await page.locator('#fnum').fill('56');
    await page.locator('#fnum').press('Tab');
    await page.waitForFunction(() => freq === 56 && !dirty && state);
    assert.equal(await page.locator('#f').inputValue(), '56');
    assert.equal(await page.evaluate(() => renderError), null);
    await page.screenshot({ path: resolve(out, `${scenario.name}-changed-frequency.png`), fullPage: true });
    assert.deepEqual(pageErrors, [], `${scenario.name} runtime errors`);
    results.push({ name: scenario.name, initial, paused, resumed, errors: pageErrors, passed: true });
    await context.close();
    activePage = null;
  }
  const preview = await browser.newContext({ javaScriptEnabled: false });
  const previewPage = await preview.newPage();
  await previewPage.goto(origin + '/index.html');
  assert.match(await previewPage.locator('#startupStatus').innerText(), /Not started/);
  await previewPage.screenshot({ path: resolve(out, 'javascript-disabled-notice.png'), fullPage: true });
  results.push({ name: 'javascript-disabled-preview', passed: true });
  await preview.close();
} catch (error) {
  errors.push(String(error.stack || error));
  if (activePage) {
    try {
      await activePage.screenshot({ path: resolve(out, `${activeScenario}-failure.png`), fullPage: true });
      await writeFile(resolve(out, `${activeScenario}-failure.html`), await activePage.content());
      errors.push({ scenario: activeScenario, url: activePage.url(), browserErrors: activePageErrors });
    } catch (captureError) { errors.push('Failure capture: ' + String(captureError)); }
  }
  process.exitCode = 1;
} finally {
  if (browser) await browser.close();
  await new Promise(resolve => server.close(resolve));
  await writeFile(resolve(out, 'results.json'), JSON.stringify({ engine: engineName, scope: 'modal CPU startup and playback; not nonlinear solver or GPU validation', results, errors }, null, 2));
  console.log(JSON.stringify({ engine: engineName, scope: 'modal CPU startup and playback; not nonlinear solver or GPU validation', results, errors }, null, 2));
}
