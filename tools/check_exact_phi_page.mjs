/* Browser gate for vedic_v18.51.1_exact_phi.html.

   tools/check_exact_phi_kernel.mjs holds the formulas to account under node.
   That does not open the page, and the page is where the formulas are used.
   The version this gate was written against had a kernel every node check
   passed, and in a browser none of its presets did anything: the worker never
   finished a step, a 26-second synchronous audit froze the page, and the
   picture's Wheeler terms displaced the surface by about 2000 units on a
   torus of radius 180 while the browser silently clamped every colour above
   255. None of that is
   visible from node.

   What it holds the page to, in a real Chromium:
     it loads           the first exact state arrives, nothing throws
     every control      each preset, composition, ALL ON / OFF, RESET and
                        PAUSE / RUN reaches the worker, and the worker runs
                        exactly the set and composition the control chose
     the surface        is drawn, every point finite and in front of the
                        camera, a torus by construction at the default gains
     no clamping        every colour written for the GPU is already in [0, 1]
                        and every depth in [-1, 1], so neither the code nor the
                        browser has anything to clamp
     certified          the badge reaches FAITHFUL on a preset

   It drives Chromium over the DevTools protocol: no dependency, no install.
   It REFUSES rather than skipping when there is no browser.

   Run: node tools/check_exact_phi_page.mjs
   Env: CHROME=/path/to/chrome  overrides binary discovery.
*/
import { spawn } from 'node:child_process';
import { statSync, mkdtempSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { tmpdir } from 'node:os';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const PAGE = pathToFileURL(join(ROOT, 'vedic_v18.51.1_exact_phi.html')).href;

let pass = 0;
const failures = [];
function ok(cond, label, detail) {
  if (cond) { pass++; console.log('  ok   ' + label); }
  else { failures.push(label + (detail ? '  -- ' + detail : '')); console.log('  FAIL ' + label + (detail ? '  -- ' + detail : '')); }
}
const section = t => console.log('\n' + t);
const sleep = ms => new Promise(r => setTimeout(r, ms));

const CANDIDATES = [
  process.env.CHROME, process.env.CHROMIUM,
  '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
  '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium', '/usr/bin/chromium-browser',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
].filter(Boolean);
const BROWSER = CANDIDATES.find(p => { try { return statSync(p).isFile(); } catch { return false; } });
if (!BROWSER) throw new Error('check_exact_phi_page: no Chromium or Chrome binary found. Tried:\n  ' +
  CANDIDATES.join('\n  ') + '\nSet CHROME=/path/to/chrome. This refuses rather than skipping: a pass ' +
  'without a browser would be a pass that checked nothing.');

/* ---- DevTools protocol, as in dns/check-page.mjs ------------------------ */
async function launch() {
  const profile = mkdtempSync(join(tmpdir(), 'exactphi-'));
  const port = 9800 + Math.floor(Math.random() * 150);
  /* --enable-unsafe-swiftshader: runners have no GPU, and Chrome no longer
     falls back to its software GPU for WebGL unless told it may. */
  const proc = spawn(BROWSER, ['--headless=new', '--no-sandbox', '--disable-dev-shm-usage',
    '--enable-unsafe-swiftshader', '--allow-file-access-from-files', '--window-size=1600,1000',
    '--force-device-scale-factor=1', `--user-data-dir=${profile}`, `--remote-debugging-port=${port}`,
    'about:blank'], { stdio: ['ignore', 'pipe', 'pipe'] });
  let err = '';
  proc.stderr.on('data', d => { err += d; });
  let wsUrl = null;
  for (let i = 0; i < 160 && !wsUrl; i++) {
    try { const r = await fetch(`http://127.0.0.1:${port}/json/version`); if (r.ok) wsUrl = (await r.json()).webSocketDebuggerUrl; }
    catch { /* not listening yet */ }
    if (!wsUrl) await sleep(250);
  }
  if (!wsUrl) throw new Error(`check_exact_phi_page: ${BROWSER} never opened a debugging port.\n${err}`);
  const ws = new WebSocket(wsUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = e => rej(new Error('ws: ' + e.message)); });
  let n = 0;
  const pending = new Map(), listeners = [];
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.id !== undefined) {
      const p = pending.get(m.id); pending.delete(m.id);
      if (m.error) p.rej(new Error(JSON.stringify(m.error))); else p.res(m.result);
    } else for (const f of listeners) f(m);
  };
  const send = (method, params = {}, sessionId) => new Promise((res, rej) => {
    const id = ++n; pending.set(id, { res, rej }); ws.send(JSON.stringify({ id, method, params, sessionId }));
  });
  const close = () => { try { ws.close(); } catch {} try { proc.kill('SIGKILL'); } catch {} try { rmSync(profile, { recursive: true, force: true }); } catch {} };
  return { send, listeners, close };
}

const browser = await launch();
try {
  const { targetId } = await browser.send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await browser.send('Target.attachToTarget', { targetId, flatten: true });
  const errors = [];
  browser.listeners.push(m => {
    if (m.sessionId !== sessionId) return;
    if (m.method === 'Runtime.exceptionThrown')
      errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
    if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error')
      errors.push('console.error: ' + m.params.args.map(a => a.value ?? a.description ?? a.type).join(' '));
  });
  const cmd = (m, p) => browser.send(m, p, sessionId);
  const evaluate = async (expression, awaitPromise = false) => {
    const r = await cmd('Runtime.evaluate', { expression, returnByValue: true, awaitPromise });
    if (r.exceptionDetails) throw new Error('page eval threw: ' + (r.exceptionDetails.exception?.description || r.exceptionDetails.text));
    return r.result.value;
  };
  const waitFor = async (expression, ms) => {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      try { if (await evaluate(expression)) return true; } catch { /* page still loading */ }
      await sleep(200);
    }
    return false;
  };
  await cmd('Runtime.enable');
  await cmd('Page.enable');
  await cmd('Emulation.setDeviceMetricsOverride', { width: 1600, height: 1000, deviceScaleFactor: 1, mobile: false });
  await cmd('Page.navigate', { url: PAGE });

  section('load');
  const t0 = Date.now();
  const first = await waitFor(`typeof KERNEL_VIEW !== 'undefined' && KERNEL_VIEW.state !== null`, 60000);
  ok(first, `the first exact state arrives (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
  const tests = await waitFor(`KERNEL_VIEW.selfTestsOk === true`, 60000);
  ok(tests, 'the kernel self-tests pass on the audit worker');
  ok(await evaluate(`KERNEL_VIEW.workerError === null`), 'every worker started', await evaluate(`String(KERNEL_VIEW.workerError)`));

  /* one frame drawn and read back in the same task, before the buffer is
     presented: what the GPU was given, and what it drew */
  const frame = () => evaluate(`(() => {
    projectTorus(); renderTorus();
    const gl = SURFACE_GL.gl, out = { drawn: SURFACE_STATUS.drawn, why: SURFACE_STATUS.why, quads: PROJECTION.quads };
    if (!gl) return out;
    const px = new Uint8Array(4 * W * H);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px);
    let covered = 0;
    for (let k = 3; k < px.length; k += 4) if (px[k] > 0) covered++;
    out.covered = covered / (W * H);
    const D = SURFACE_GL.data, nv = PROJECTION.quads * 6;
    let badCol = 0, badDepth = 0, badPos = 0;
    for (let v = 0; v < nv; v++) {
      const b = v * 6;
      if (!(Number.isFinite(D[b]) && Number.isFinite(D[b + 1]))) badPos++;
      if (!(D[b + 2] >= -1 && D[b + 2] <= 1)) badDepth++;
      for (let c = 3; c < 6; c++) if (!(D[b + c] >= 0 && D[b + c] <= 1)) badCol++;
    }
    let badPt = 0;
    for (let k = 0; k < N_PTS; k++) if (!(Number.isFinite(PT.sx[k]) && Number.isFinite(PT.sy[k]) && Number.isFinite(PT.depth[k]))) badPt++;
    Object.assign(out, { badCol, badDepth, badPos, badPt, behind: PROJECTION.behind, turn: PROJECTION.turn,
                         torus: PROJECTION.torusByConstruction, pole: PROJECTION.pole, nv });
    return out;
  })()`);

  section('the surface');
  const f0 = await frame();
  ok(f0.drawn, 'the surface is drawn', f0.why);
  ok(f0.quads > 1000 && f0.covered > 0.02, `${f0.quads} quads cover ${(100 * f0.covered).toFixed(1)}% of the view`);
  ok(f0.badPt === 0, 'every surface point is finite', `${f0.badPt} are not`);
  ok(f0.behind === 0 && f0.turn < 1e-9, 'no point behind the camera, and the camera turns are rotations');
  ok(f0.torus === true, 'the default gains make the surface a torus by construction (surface_is_torus)');

  section('no clamping');
  ok(f0.badCol === 0, `all ${f0.nv * 3} colour components written for the GPU are in [0, 1]`, `${f0.badCol} outside`);
  ok(f0.badDepth === 0, 'every depth written for the GPU is in [-1, 1]', `${f0.badDepth} outside`);
  ok(f0.badPos === 0, 'every vertex position is finite', `${f0.badPos} are not`);

  section('every control reaches the kernel');
  const applied = async (label, want) => {
    const got = await waitFor(`KERNEL_VIEW.appliedVersion === KERNEL_VIEW.configVersion && KERNEL_VIEW.state && !KERNEL_VIEW.halted`, 30000);
    const s = await evaluate(`({ active: KERNEL_VIEW.appliedActive.slice().sort((a, b) => a - b).join(','),
                                 comp: KERNEL_VIEW.appliedComposition, halted: KERNEL_VIEW.halted && KERNEL_VIEW.halted.message })`);
    const okSet = want.active === undefined || s.active === want.active;
    const okComp = want.comp === undefined || s.comp === want.comp;
    ok(got && okSet && okComp, `${label}: the worker ran ${want.comp || s.comp} [${s.active || 'none'}]`,
       `applied=${got} active=${s.active} comp=${s.comp} halted=${s.halted}`);
  };
  const presets = await evaluate(`Object.entries(PRESETS).map(([id, p]) => [id, p.comp, Object.keys(p.set).filter(k => p.set[k] > 0).map(Number).sort((a, b) => a - b).join(',')])`);
  ok(presets.length === 10, `${presets.length} presets`);
  for (const [id, comp, set] of presets) {
    await evaluate(`document.getElementById(${JSON.stringify(id)}).click()`);
    await applied(id, { comp, active: set });
    const fr = await frame();
    ok(fr.drawn && fr.badCol === 0 && fr.badDepth === 0 && fr.badPt === 0 && fr.behind === 0,
       `${id}: drawn, finite, nothing to clamp`, JSON.stringify({ drawn: fr.drawn, why: fr.why, badCol: fr.badCol, badDepth: fr.badDepth, badPt: fr.badPt }));
  }
  for (const comp of ['PARALLEL', 'SERIES', 'SYMMETRIC_CONCURRENT']) {
    await evaluate(`document.querySelector('#comp-ctrls button[data-comp="${comp}"]').click()`);
    await applied(`composition ${comp}`, { comp });
  }
  await evaluate(`document.getElementById('btn-none').click()`);
  await applied('ALL OFF', { active: '' });
  await evaluate(`document.getElementById('btn-all').click()`);
  await applied('ALL ON', { active: Array.from({ length: 29 }, (_, i) => i + 1).join(',') });
  const before = await evaluate(`KERNEL_VIEW.state.frame`);
  await evaluate(`document.getElementById('btn-reset').click()`);
  ok(await waitFor(`KERNEL_VIEW.state.frame < ${before}`, 20000), 'RESET starts the run again from the seed');
  await evaluate(`document.getElementById('btn-run').click()`);
  await sleep(1500);
  const paused = await evaluate(`KERNEL_VIEW.state.frame`);
  await sleep(1500);
  ok(await evaluate(`KERNEL_VIEW.state.frame`) === paused, 'PAUSE stops the stepping');
  await evaluate(`document.getElementById('btn-run').click()`);
  ok(await waitFor(`KERNEL_VIEW.state.frame > ${paused}`, 20000), 'RUN resumes it');

  section('certified');
  await evaluate(`document.getElementById('pre-wormhole').click()`);
  const faithful = await waitFor(`document.getElementById('faithfulBadge').className.includes(' ok')`, 90000);
  ok(faithful, 'the badge reaches FAITHFUL on the wormhole preset',
     await evaluate(`document.getElementById('faithfulBadge').textContent`));

  section('errors');
  ok(errors.length === 0, 'no exception and no console.error', errors.slice(0, 3).join(' | '));
} finally {
  browser.close();
}
console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) { for (const f of failures) console.log('  ' + f); process.exit(1); }
