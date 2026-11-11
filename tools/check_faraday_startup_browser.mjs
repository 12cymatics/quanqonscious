import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdir, writeFile, stat } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium, webkit } from 'playwright';

const engineName = process.env.FARADAY_BROWSER || 'chromium';
const engine = { chromium, webkit }[engineName];
assert(engine, `Unknown browser engine: ${engineName}`);
// Explicit HTTP scope is useful on managed browsers that deny file:// URLs.
// The default still exercises both file scenarios and fails if they cannot load.
const transport = process.env.FARADAY_QA_TRANSPORT || 'file';
assert(['file', 'http'].includes(transport), `Unknown transport: ${transport}`);

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
  const executablePath = process.env.FARADAY_BROWSER_EXECUTABLE;
  browser = await engine.launch({ headless: true, ...(executablePath ? { executablePath } : {}) });
  const fileScenarios = transport === 'file' ? [
    { name: 'folder-file', url: pathToFileURL(resolve(root, 'index.html')).href, viewport: { width: 1440, height: 1000 } },
    { name: 'standalone-file', url: pathToFileURL(resolve(root, 'faraday-cell-standalone.html')).href, viewport: { width: 1440, height: 1000 } }
  ] : [
    { name: 'standalone-http', url: origin + '/faraday-cell-standalone.html', viewport: { width: 1440, height: 1000 } }
  ];
  for (const scenario of [
    { name: 'folder-http', url: origin + '/index.html', viewport: { width: 1440, height: 1000 } },
    { name: 'source-http', url: origin + '/cymatic.html', viewport: { width: 1440, height: 1000 } },
    ...fileScenarios,
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
    const defaultState = await page.evaluate(() => ({ experimentMode, rim, useSolver, view,
      running, solverRequested, drawn: FARADAY_STARTUP.drawn, failed: FARADAY_STARTUP.failed }));
    assert.equal(defaultState.experimentMode, 'reference');
    assert.equal(defaultState.rim, 'free');
    assert.equal(defaultState.useSolver, false);
    assert.equal(defaultState.view, 'sand');
    assert.equal(defaultState.running, true);
    assert.equal(defaultState.solverRequested, false);
    assert.equal(defaultState.failed, false);
    assert.equal(await page.locator('#viewSeg [data-v="sand"]').isEnabled(), true);
    assert.equal(await page.locator('#play').isEnabled(), true);
    assert.equal(await page.locator('#sprinkleBtn').isEnabled(), true);
    assert.equal(await page.locator('#solverOverlay').isVisible(), false);
    results.push({ name: scenario.name + '-sand-on-entry', defaultState, passed: true });
    assert.equal(await page.locator('#c').isVisible(), true);
    const initial = await page.evaluate(() => {
      const canvas = document.getElementById('c');
      const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      let min = 255, max = 0;
      for (let i = 0; i < pixels.length; i += 4) { min = Math.min(min, pixels[i]); max = Math.max(max, pixels[i]); }
      return { simT, running, renderError, renderEngine, useSolver, expression: state.expression, spread: max - min };
    });
    assert.equal(await page.evaluate(() => FARADAY_STARTUP.drawn && !FARADAY_STARTUP.failed), true);
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
    // WebKit may acknowledge the input command before its click handler runs.
    // Observe the actual state transition; a missed click must still time out.
    await page.waitForFunction(() => running === false
      && document.getElementById('play').getAttribute('aria-pressed') === 'false'
      && document.getElementById('play').textContent === 'paused', undefined, { timeout:5000 });
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
    await page.locator('#workspaceSeg [data-workspace="ns"]').click();
    await page.waitForFunction(() => useSolver && !dirty && CELL3D.clocks().initial);
    const initialCondition = await page.evaluate(() => ({ experimentMode, rim, useSolver, solverRequested,
      clocks: CELL3D.clocks(), source: CELL3D.snapshot().provenance,
      signature: state.visualSignature, fieldPresent: ETA.some(v => v !== 0) }));
    assert.equal(initialCondition.experimentMode, 'general');
    assert.equal(initialCondition.rim, 'pinned');
    assert.equal(initialCondition.useSolver, true);
    assert.equal(initialCondition.solverRequested, false);
    assert.equal(initialCondition.clocks.have, false);
    assert.equal(initialCondition.clocks.physT, 0);
    assert.equal(initialCondition.clocks.stepsDone, 0);
    assert.equal(initialCondition.source.fieldKind, 'initial-condition');
    assert.equal(initialCondition.signature, null);
    assert.equal(initialCondition.fieldPresent, true);
    assert.equal(await page.locator('#solverOverlay').isVisible(), false);
    assert.match(await page.locator('#solverStatus').innerText(), /initial|not evolved/i);
    results.push({ name: scenario.name + '-explicit-ns-initial-condition', initialCondition, passed: true });
    await page.screenshot({ path: resolve(out, `${scenario.name}-ns-initial-condition.png`), fullPage: true });
    await page.locator('#workspaceSeg [data-workspace="sand"]').click();
    await page.waitForFunction(() => !useSolver && !dirty && view === 'sand');
    assert.equal(await page.evaluate(() => freq), 56, 'Returning to Sand discarded its frequency');
    assert.equal(await page.locator('#viewSeg [data-v="sand"]').isEnabled(), true);
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
  const scope = 'immediate moving Sand and playback; explicit pinned NS initial-condition preview with no invented evolution; workspace settings retained; not nonlinear integration or GPU validation'
    + (transport === 'http' ? '; explicit HTTP scope, file:// is not validated' : '; includes file:// startup');
  await writeFile(resolve(out, 'results.json'), JSON.stringify({ engine: engineName, transport, scope, results, errors }, null, 2));
  console.log(JSON.stringify({ engine: engineName, transport, scope, results, errors }, null, 2));
}
