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
    const pending = await page.evaluate(async () => {
      let exportError = '';
      try { await exportImage(129, () => {}); } catch (error) { exportError = error.message; }
      return { experimentMode, experiment: !!state.experiment, rim, useSolver, solverRequested,
        haveFrame: CELL3D.clocks().have, comparePaper, visualSignature: state.visualSignature,
        fieldIsZero: [ETA, IMF, NX, NY].every(a => a.every(v => v === 0)), exportError };
    });
    assert.equal(pending.experimentMode, 'general');
    assert.equal(pending.experiment, true);
    assert.equal(pending.rim, 'pinned');
    assert.equal(pending.useSolver, true);
    assert.equal(pending.solverRequested, false);
    assert.equal(pending.haveFrame, false);
    assert.equal(pending.comparePaper, false);
    assert.equal(pending.visualSignature, null);
    assert.equal(pending.fieldIsZero, true, 'Pending nonlinear mode must not substitute a modal field');
    assert.match(pending.exportError, /No nonlinear solver frame/);
    assert.equal(await page.locator('#solverOverlay').isVisible(), true);
    assert.match(await page.locator('#solverOverlay').innerText(), /not started/i);
    results.push({ name: scenario.name + '-general-pinned-pending', pending, passed: true });

    // Exercise the original playback fixture only after explicitly choosing it.
    await page.locator('#configurationMode').selectOption('reference');
    await page.locator('#srcSeg [data-s="modal"]').click();
    await page.locator('#rimSeg [data-r="free"]').click();
    await page.waitForFunction(() => experimentMode === 'reference' && !useSolver
      && !dirty && state.expression > 0 && !renderError);
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
  const sorted = {};
  for (const grid of [{ name: 'grid-base', scale: 1, size: 381 }, { name: 'grid-fine', scale: 2, size: 571 }]) {
    const context = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: grid.scale });
    const page = await context.newPage();
    const pageErrors = [];
    activePage = page;
    activeScenario = grid.name;
    activePageErrors = pageErrors;
    page.on('pageerror', error => pageErrors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') pageErrors.push(message.text()); });
    await page.goto(origin + '/index.html', { waitUntil: 'load', timeout: 30000 });
    await page.waitForFunction(() => typeof simT !== 'undefined' && state && !renderError, undefined, { timeout: 30000 });
    const sizes = await page.evaluate(() => ({ GR, NP, canvas: document.getElementById('c').width, ratio: window.devicePixelRatio }));
    assert.deepEqual(sizes, { GR: grid.size, NP: Math.round(60000*grid.size/381), canvas: grid.size, ratio: grid.scale });
    const bessel = await page.evaluate(() => {
      let compared = 0, different = 0;
      for (const m of [0, 1, 2, 5, 9, 11, 52, 79, 144, 158]) for (let i = 0; i <= 200; i++) {
        const z = i === 0 ? 0 : i*2.4742 + (m % 7)*0.013;
        const want = FARADAY_KERNEL.besselPair(m, z);
        besselInto(m, z);
        compared++;
        if (!Object.is(want[0], BESSEL[0]) || !Object.is(want[1], BESSEL[1])) different++;
      }
      return { compared, different };
    });
    assert.deepEqual(bessel, { compared: 2010, different: 0 });
    await page.locator('#configurationMode').selectOption('reference');
    await page.locator('#srcSeg [data-s="modal"]').click();
    await page.locator('#rimSeg [data-r="free"]').click();
    await page.locator('#fnum').fill('4392');
    await page.locator('#fnum').press('Tab');
    await page.waitForFunction(() => freq === 4392 && !dirty && state && state.render && !useSolver);
    const drawn = await page.evaluate(() => ({ states: state.states.length, drawn: state.render.drawn, gated: state.render.gated,
      coarsest: Math.min(...state.states.map(s => s.pxPerWave)), gate: RES_GATE,
      overlay: getComputedStyle(document.getElementById('solverOverlay')).display }));
    assert(drawn.states > 0 && drawn.drawn === drawn.states && drawn.gated === 0, `${grid.name}: ${JSON.stringify(drawn)}`);
    assert(drawn.coarsest >= drawn.gate, `${grid.name}: ${drawn.coarsest} px per wavelength`);
    assert.equal(drawn.overlay, 'none');
    sorted[grid.name] = await page.evaluate(() => {
      running = false;
      const atGrains = () => { let sum = 0; for (let p = 0; p < NP; p++) sum += Math.abs(ETA[(PY[p] | 0)*GR + (PX[p] | 0)]); return sum/NP; };
      let all = 0, cells = 0;
      for (let i = 0; i < NSZ; i++) if (COVER[i] === 1) { all += Math.abs(ETA[i]); cells++; }
      const mean = all/cells;
      sprinkle();
      const fresh = atGrains()/mean;
      for (let i = 0; i < 120; i++) stepGrains(1);
      return { fresh, settled: atGrains()/mean, pxPerWave: Math.min(...state.states.map(s => s.pxPerWave)) };
    });
    assert(sorted[grid.name].fresh > 0.9 && sorted[grid.name].fresh < 1.1, `${grid.name}: fresh grains ${sorted[grid.name].fresh}`);
    assert(sorted[grid.name].settled < sorted[grid.name].fresh, `${grid.name}: ${JSON.stringify(sorted[grid.name])}`);
    if (grid.scale === 1) {
      await page.locator('#rimSeg [data-r="pinned"]').click();
      await page.waitForFunction(() => !dirty && state.render && state.render.drawn === 0 && state.states.length > 0);
      await page.waitForFunction(() => getComputedStyle(document.getElementById('solverOverlay')).display !== 'none');
      const notice = await page.locator('#solverOverlay').innerText();
      const best = await page.evaluate(() => Math.max(...state.states.map(s => s.pxPerWave)));
      assert.match(notice, /^not drawn · [0-9.]+ px\/λ · 4 needed · grid 381$/);
      assert.equal(notice.split(' · ')[1], `${best.toFixed(1)} px/λ`);
      assert(best < 4, `pinned 4392 Hz on the base grid: ${best} px per wavelength`);
    }
    await page.screenshot({ path: resolve(out, `${grid.name}-4392hz.png`), fullPage: true });
    assert.deepEqual(pageErrors, [], `${grid.name} runtime errors`);
    results.push({ name: grid.name, sizes, drawn, sorted: sorted[grid.name], passed: true });
    await context.close();
    activePage = null;
  }
  assert(sorted['grid-fine'].settled < sorted['grid-fine'].fresh/2, `fine grid at 4392 Hz: ${JSON.stringify(sorted['grid-fine'])}`);
  assert(sorted['grid-fine'].settled < sorted['grid-base'].settled, `fine against base at 4392 Hz: ${JSON.stringify(sorted)}`);
  assert(sorted['grid-fine'].pxPerWave > sorted['grid-base'].pxPerWave);
  results.push({ name: 'grid-fine-sorts-better', sorted, passed: true });
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
  const scope = 'general pinned nonlinear startup with no substitute field or export; explicit reference modal CPU playback; not nonlinear integration or GPU validation'
    + (transport === 'http' ? '; explicit HTTP scope, file:// is not validated' : '; includes file:// startup');
  await writeFile(resolve(out, 'results.json'), JSON.stringify({ engine: engineName, transport, scope, results, errors }, null, 2));
  console.log(JSON.stringify({ engine: engineName, transport, scope, results, errors }, null, 2));
}
