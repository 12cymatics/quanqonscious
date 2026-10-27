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
                        toward exactly the set and composition the control
                        chose
     the pace           one step a second, each drawn in order; a formula
                        switched on rises to its strength over the glide's
                        steps (10 unless changed), then holds while the run
                        goes on
     the surface        is drawn, every point finite and in front of the
                        camera
     no clamping        the colour formula runs past 255 and each channel goes
                        through the tone curve instead of being cut off: every
                        colour written for the GPU is already in [0, 1] and
                        every depth in [-1, 1], so neither the code nor the
                        browser has anything to clamp, and the curve keeps the
                        order of the values it is given
     certified          the badge reaches FAITHFUL on a preset
     the torus          the flux map and the lattice on the surface show the
                        worker's exact A and chi as the gate reads them
     the fields         B by Biot-Savart integrated numerically here and E by
                        Coulomb at the plane's samples; the magnetic pressure
                        |B|^2 / 2 on its grid, its places and differences bit
                        for bit; the GPU's ray sums through the grid against
                        the same sums here; the view the scene read where the
                        haze says; no line and no arrow drawn
     motion             the kernel is paced, every step is drawn, the surface
                        moves to each state over its step interval and the
                        framing eases: no strobe, no jump
     framing            the page's framing step, fed a size that swings, does
                        not move back once out; on a fresh page every frame
                        keeps the law: still in the band, toward the size
                        and never past it outside it
     exports            the step file and the operator audit name the build
                        that wrote them; the step carries what replays it

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
  const presetFromSeed = async id => {
    const e0 = await evaluate(`KERNEL_VIEW.epoch`);
    await evaluate(`document.getElementById(${JSON.stringify(id)}).click()`);
    return waitFor(`KERNEL_VIEW.appliedEpoch > ${e0} && KERNEL_VIEW.state && !KERNEL_VIEW.halted`, 30000);
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
  /* by default: one step a second, the formulas rising over a glide of 10 */
  await waitFor(`KERNEL_VIEW.state.frame >= 1 && DISPLAY.started && DISPLAY.dur > 0`, 60000);
  const g0 = await evaluate(`({ slider: document.getElementById('amp-glide').value, glide: GLIDE_STEPS, frame: KERNEL_VIEW.state.frame, t: performance.now() })`);
  await sleep(5000);
  const g1 = await evaluate(`({ frame: KERNEL_VIEW.state.frame, t: performance.now(), dur: DISPLAY.dur, interval: STEP_INTERVAL_MS })`);
  const sps0 = (g1.frame - g0.frame) / ((g1.t - g0.t) / 1000);
  ok(g0.slider === '10' && g0.glide === 10 && g1.interval === 1000 && g1.dur === 1000 && sps0 >= 0.8 && sps0 <= 1.1,
     `by default the run takes one step a second (${sps0.toFixed(2)} a second over 5 s) over a glide of 10 steps`, JSON.stringify({ g0, g1 }));
  const setGlide = s => evaluate(`(() => { const el = document.getElementById('amp-glide'); el.value = '${s}'; el.dispatchEvent(new Event('input')); return GLIDE_STEPS; })()`);
  /* the rest of the checks run at a glide of 1 step: every formula at its setting from the first step */
  await setGlide(1);

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
    /* the surface is indexed: each visible quad is two triangles of the
       vertex grid, so what the GPU draws is every vertex the indices name */
    const D = SURFACE_GL.data, F = SURFACE_GL.FLOATS, IX = SURFACE_GL.index, nIdx = SURFACE_GL.nIdx, NVX = D.length / F;
    const used = new Uint8Array(NVX);
    let badIdx = 0;
    for (let i = 0; i < nIdx; i++) { const v = IX[i]; if (v < NVX) used[v] = 1; else badIdx++; }
    let badCol = 0, badDepth = 0, badPos = 0, nv = 0, white = 0;
    for (let v = 0; v < NVX; v++) {
      if (!used[v]) continue;
      nv++;
      const b = v * F;
      if (!(Number.isFinite(D[b]) && Number.isFinite(D[b + 1]))) badPos++;
      if (!(D[b + 2] >= -1 && D[b + 2] <= 1)) badDepth++;
      for (let c = 3; c < 6; c++) if (!(D[b + c] >= 0 && D[b + c] <= 1)) badCol++;
      if (D[b + 3] > 0.85 && D[b + 4] > 0.85 && D[b + 5] > 0.85) white++;
    }
    out.nIdx = nIdx; out.badIdx = badIdx; out.white = nv ? white / nv : 0;
    let badPt = 0;
    for (let k = 0; k < N_PTS; k++) if (!(Number.isFinite(PT.sx[k]) && Number.isFinite(PT.sy[k]) && Number.isFinite(PT.depth[k]))) badPt++;
    Object.assign(out, { badCol, badDepth, badPos, badPt, behind: PROJECTION.behind, turn: PROJECTION.turn,
                         pole: PROJECTION.pole, nv });
    return out;
  })()`);

  section('the surface');
  const f0 = await frame();
  ok(f0.drawn, 'the surface is drawn', f0.why);
  ok(f0.quads > 1000 && f0.covered > 0.02, `${f0.quads} quads cover ${(100 * f0.covered).toFixed(1)}% of the view`);
  ok(f0.badPt === 0, 'every surface point is finite', `${f0.badPt} are not`);
  ok(f0.behind === 0 && f0.turn < 1e-9, 'no point behind the camera, and the camera turns are rotations');
  ok(f0.nIdx === f0.quads * 6 && f0.badIdx === 0, `each of the ${f0.quads} visible quads is two triangles of the vertex grid, every index inside it`,
     `${f0.nIdx} indices, ${f0.badIdx} outside`);

  section('no clamping');
  ok(f0.badCol === 0, `all ${f0.nv * 3} colour components written for the GPU are in [0, 1]`, `${f0.badCol} outside`);
  ok(f0.badDepth === 0, 'every depth written for the GPU is in [-1, 1]', `${f0.badDepth} outside`);
  /* the map onto [DEPTH_LO, DEPTH_LO + DEPTH_SPAN] leaves room below for the
     lift that puts glyphs in front of the surface: mapped onto
     [-0.999, 0.999], the nearest lifted glyph was at -1.003 and was clipped */
  const dm = await evaluate(`({ lo: DEPTH_LO, span: DEPTH_SPAN, lift: LAT_BIAS })`);
  ok(dm.lo - dm.lift >= -1 && dm.lo + dm.span <= 1 && dm.lift >= 0 && dm.lift <= 0.004 && dm.span > 0,
     `the depth map leaves room for the glyph lift: [${dm.lo}, ${(dm.lo + dm.span).toFixed(3)}] less ${dm.lift} stays in [-1, 1] (ndc_depth_lift_mem)`);
  ok(f0.badPos === 0, 'every vertex position is finite', `${f0.badPos} are not`);
  const tc = await evaluate(`(() => {
    let mid = true, order = true, range = true, prev = -Infinity;
    for (let v = -3000; v <= 3000; v += 0.5) {
      const t = tone(v);
      if (v >= 24 && v <= 231 && t !== v) mid = false;
      if (t < prev) order = false;
      if (!(t >= 0 && t <= 255)) range = false;
      prev = t;
    }
    return { mid, order, range, above: tone(300) > tone(260) && tone(260) > tone(240) };
  })()`);
  ok(tc.mid, 'the tone curve leaves the colour formula unchanged on [24, 231] (tone_mid)');
  ok(tc.order && tc.above, 'a larger value never comes out darker, and values past 255 stay distinct (tone_mono)');
  ok(tc.range, 'every toned value is within [0, 255] (tone_mem)');
  /* A colour is toned with its hue kept. The gate's own low knee,
     24 e^((v - 24)/24) below 24 and v above, from DisplayMap.lean; the
     ratios of the three channels after toning must be the ratios of their low
     knees, and a grey must come out as tone itself. Through tone channel by
     channel (what this replaced) a bright colour's channels all reach the
     high knee together and it goes white. */
  const hk = await evaluate(`(() => {
    const lk = v => v < 24 ? 24 * Math.exp((v - 24) / 24) : v, out = new Float64Array(3);
    let range = true, ratio = 0, grey = 0, n = 0, bright = true;
    for (let r = -400; r <= 3000; r += 85) for (let g = -400; g <= 3000; g += 97) for (let b = -400; b <= 3000; b += 113) {
      toneRGB(r, g, b, out); n++;
      if (!(out[0] > 0 && out[0] <= 1 && out[1] > 0 && out[1] <= 1 && out[2] > 0 && out[2] <= 1)) range = false;
      const L = [lk(r), lk(g), lk(b)];
      for (const [i, j] of [[0, 1], [1, 2], [0, 2]]) ratio = Math.max(ratio, Math.abs(out[i] / out[j] / (L[i] / L[j]) - 1));
    }
    for (let v = -400; v <= 3000; v += 7) { toneRGB(v, v, v, out); const t = tone(v) / 255; grey = Math.max(grey, ...[0, 1, 2].map(c => Math.abs(out[c] / t - 1))); }
    /* a brighter colour of the same hue is never darker, and past 255 still brighter */
    const a = new Float64Array(3), c = new Float64Array(3);
    for (let s = 1; s < 40; s++) { toneRGB(300 * s / 10, 120 * s / 10, 40 * s / 10, a); toneRGB(300 * (s + 1) / 10, 120 * (s + 1) / 10, 40 * (s + 1) / 10, c); if (!(c[0] >= a[0])) bright = false; }
    toneRGB(2400, 960, 320, a);
    return { n, range, ratio, grey, bright, hot: Array.from(a) };
  })()`);
  /* (0, 1) in the reals; in doubles the high knee reaches 255 itself once
     24 e^(-(m - 231)/24) is below half an ulp of 255 (m from 1068, measured),
     so 1 is reached and must be allowed -- and nothing above it. This check
     first found toneRGB(3000, 3000, 3000) = 1.0000000000000002: one ulp past
     1, for 4.4% of bright colours, hidden from the surface check above by the
     conversion to 32-bit floats. */
  ok(hk.range, `every toned colour is in (0, 1] in each channel, 1 only where the knee rounds to 255 (toneRGB_mem, ${hk.n} colours)`);
  ok(hk.ratio < 1e-12, 'toning keeps the ratios of the channels\' low knees: the hue is kept (toneRGB_ratio)', `worst ${hk.ratio.toExponential(2)}`);
  ok(hk.grey < 1e-12, 'a grey is toned as tone itself (toneRGB_grey)', `worst ${hk.grey.toExponential(2)}`);
  /* never darker (highKnee_mono is <=): in doubles two values past ~1100
     both round to 255, so strictly brighter cannot be required */
  ok(hk.bright && hk.hot[2] < 0.5 * hk.hot[0], 'a brighter colour of one hue is never darker, and far past 255 an orange is still orange, not white (highKnee_mono)', JSON.stringify(hk.hot));

  section('the glide');
  await setGlide(5);
  await presetFromSeed('pre-damper');
  await waitFor(`KERNEL_VIEW.state.frame === 2`, 30000);
  const gl = await evaluate(`(() => {
    const st = VISUAL_CERTIFICATE.status(), p = PRESETS['pre-damper'].set, f = KERNEL_VIEW.state.frame;
    return { f, ok: RAMP_VIEW.ok, cert: st.ramp, rising: RAMP_VIEW.rising, row: glideWords(),
             rows: RAMP_VIEW.rows.map(r => [r.id, Q.toString(r.s), Q.toString(r.to), r.k, r.n]),
             want: Object.keys(p).map(Number).sort((a, b) => a - b).map(id => { const to = Q.mk(BigInt(p[id]), 100n);
               return [id, Q.toString(Q.mul(to, Q.mk(BigInt(Math.min(f, 5)), 5n))), Q.toString(to), Math.min(f, 5), 5]; }) };
  })()`);
  ok(gl.ok && gl.cert && JSON.stringify(gl.rows) === JSON.stringify(gl.want) && gl.f < 5 && gl.rising === gl.want.length && gl.row.startsWith(`${gl.want.length} of ${gl.want.length} rising`),
     `at a glide of 5, step ${gl.f} runs each formula at ${gl.f}/5 of its setting, and the glide row says so (${gl.row})`, JSON.stringify(gl));
  await waitFor(`KERNEL_VIEW.state.frame >= 6`, 30000);
  const hd0 = await evaluate(`({ f: KERNEL_VIEW.state.frame, ok: RAMP_VIEW.ok, rising: RAMP_VIEW.rising, held: RAMP_VIEW.rows.length === 3 && RAMP_VIEW.rows.every(r => Q.eq(r.s, r.to)), row: glideWords() })`);
  await sleep(3000);
  const hd1 = await evaluate(`({ f: KERNEL_VIEW.state.frame, ok: RAMP_VIEW.ok, held: RAMP_VIEW.rows.length === 3 && RAMP_VIEW.rows.every(r => Q.eq(r.s, r.to)) })`);
  ok(hd0.ok && hd0.held && hd0.rising === 0 && hd0.row.startsWith('all 3 held') && hd1.ok && hd1.held && hd1.f - hd0.f >= 2,
     `past the glide each formula is held at its setting and the run goes on (step ${hd0.f} to ${hd1.f} in 3 s)`, JSON.stringify({ hd0, hd1 }));
  await setGlide(1);

  section('every control reaches the kernel');
  const applied = async (label, want) => {
    const got = await waitFor(`KERNEL_VIEW.appliedVersion === KERNEL_VIEW.configVersion && KERNEL_VIEW.state && KERNEL_VIEW.appliedComposition !== null && !KERNEL_VIEW.halted`, 30000);
    const s = await evaluate(`({ active: KERNEL_VIEW.appliedSet.slice().sort((a, b) => a - b).join(','),
                                 comp: KERNEL_VIEW.appliedComposition, halted: KERNEL_VIEW.halted && KERNEL_VIEW.halted.message })`);
    const okSet = want.active === undefined || s.active === want.active;
    const okComp = want.comp === undefined || s.comp === want.comp;
    ok(got && okSet && okComp, `${label}: the worker ran ${want.comp || s.comp} set to [${s.active || 'none'}]`,
       `applied=${got} set=${s.active} comp=${s.comp} halted=${s.halted}`);
  };
  const presets = await evaluate(`Object.entries(PRESETS).map(([id, p]) => [id, p.comp, Object.keys(p.set).filter(k => p.set[k] > 0).map(Number).sort((a, b) => a - b).join(',')])`);
  ok(presets.length === 12, `${presets.length} presets`);
  const titles = await evaluate(`Object.keys(PRESETS).map(id => [id, document.getElementById(id).title === presetTitle(PRESETS[id])])`);
  ok(titles.every(([, a]) => a), 'each preset button\'s title is built from its definition', JSON.stringify(titles.filter(([, a]) => !a)));
  const whites = [];
  for (const [id, comp, set] of presets) {
    await evaluate(`document.getElementById(${JSON.stringify(id)}).click()`);
    await applied(id, { comp, active: set });
    const fr = await frame();
    ok(fr.drawn && fr.badCol === 0 && fr.badDepth === 0 && fr.badPt === 0 && fr.behind === 0,
       `${id}: drawn, finite, nothing to clamp`, JSON.stringify({ drawn: fr.drawn, why: fr.why, badCol: fr.badCol, badDepth: fr.badDepth, badPt: fr.badPt }));
    whites.push([id, fr.white]);
  }
  /* Toned channel by channel, 42 to 50% of the drawn surface had all three
     channels above 0.85 (measured on six presets). */
  const worstWhite = whites.reduce((m, w) => w[1] > m[1] ? w : m, ['', 0]);
  ok(worstWhite[1] < 0.05, `on no preset is more than 5% of the drawn surface near white (worst ${worstWhite[0]}, ${(100 * worstWhite[1]).toFixed(2)}%)`);
  await presetFromSeed('pre-damper');
  for (const comp of ['PARALLEL', 'SERIES', 'SYMMETRIC_CONCURRENT']) {
    await evaluate(`document.querySelector('#comp-ctrls button[data-comp="${comp}"]').click()`);
    await applied(`composition ${comp}`, { comp });
  }
  await evaluate(`document.getElementById('btn-none').click()`);
  await applied('ALL OFF', { active: '' });
  const v0 = await evaluate(`KERNEL_VIEW.configVersion`);
  await evaluate(`document.getElementById('btn-all').click()`);
  await sleep(1000);
  const ao = await evaluate(`({ ids: activeSutras().map(a => a.id).join(','), half: activeSutras().every(a => a.strength.n * 2n === a.strength.d),
                               comp: COMPOSITION, cfg: KERNEL_VIEW.configVersion })`);
  const all29 = Array.from({ length: 29 }, (_, i) => i + 1).join(',');
  ok(ao.ids === all29 && ao.half && ao.comp === 'SYMMETRIC_CONCURRENT' && ao.cfg === v0 + 1,
     'ALL ON sends all 29 at 50% to the worker as one setting', JSON.stringify(ao));
  const dp = await evaluate(`({ comp: PRESETS['pre-damper'].comp, active: Object.keys(PRESETS['pre-damper'].set).map(Number).sort((a, b) => a - b).join(',') })`);
  await evaluate(`document.getElementById('pre-damper').click()`);
  await applied('a preset chosen while ALL ON is still calculating replaces that calculation', dp);
  const r0 = await evaluate(`KERNEL_VIEW.appliedEpoch`);
  await evaluate(`document.getElementById('btn-reset').click()`);
  ok(await waitFor(`KERNEL_VIEW.appliedEpoch > ${r0} && KERNEL_VIEW.state.frame >= 2 && KERNEL_VIEW.hist.frame.every((f, i) => f === i)`, 20000),
     'RESET starts the run again from the seed', await evaluate(`KERNEL_VIEW.hist.frame.join(',')`));
  await evaluate(`document.getElementById('btn-run').click()`);
  await sleep(400);
  const p0 = await evaluate(`KERNEL_VIEW.configVersion`);
  await evaluate(`document.getElementById('pre-checksum').click()`);
  await sleep(2500);
  const p1 = await evaluate(`({ cfg: KERNEL_VIEW.configVersion, applied: KERNEL_VIEW.appliedVersion, frame: KERNEL_VIEW.state.frame, inFlight: KERNEL_VIEW.inFlight, queue: KERNEL_VIEW.queue.length })`);
  ok(p1.cfg === p0 + 1 && p1.applied === p1.cfg && p1.frame === 0 && !p1.inFlight && p1.queue === 0, 'PAUSE holds a new setting at its seed, uncalculated', JSON.stringify(p1));
  await evaluate(`document.getElementById('btn-run').click()`);
  ok(await waitFor(`KERNEL_VIEW.state.frame >= 2 && KERNEL_VIEW.appliedVersion === KERNEL_VIEW.configVersion`, 20000), 'RUN goes on from it, a step a second');

  section('certified');
  await presetFromSeed('pre-damper');
  const faithful = await waitFor(`document.getElementById('faithfulBadge').className.includes(' ok')`, 90000);
  ok(faithful, 'the badge reaches FAITHFUL on the damper preset',
     await evaluate(`document.getElementById('faithfulBadge').textContent`));

  section('dynamo and Maya');
  /* The magnetism memory and Maya are shown as the kernel computed them, the
     three new controls reach the kernel, each preset sets them as it says,
     and the drawn field is the true field times Maya's M = observed / true. */
  const gcdN = (a, b) => { a = Math.abs(a); b = Math.abs(b); while (b) [a, b] = [b, a % b]; return a; };
  const qstr = (n, d) => { const g = gcdN(n, d) || 1; return d / g === 1 ? String(n / g) : `${n / g}/${d / g}`; };
  const setSlider = (id, v) => evaluate(`(() => { const el = document.getElementById(${JSON.stringify(id)}); el.value = ${JSON.stringify(String(v))}; el.dispatchEvent(new Event('input')); })()`);
  const kp = () => evaluate(`(() => { const p = KERNEL_VIEW.params; return p ? { omega: Q.toString(p.vortex), drive: Q.toString(p.zpeSource.a), chi: Q.toString(p.polarisation) } : null; })()`);
  const appliedNow = () => waitFor(`KERNEL_VIEW.appliedVersion === KERNEL_VIEW.configVersion && KERNEL_VIEW.state && !KERNEL_VIEW.halted`, 30000);
  await setSlider('amp-omega', 15); await appliedNow();
  await setSlider('amp-drive', 6); await appliedNow();
  await setSlider('amp-polar', 25); await appliedNow();
  const k1 = await kp();
  ok(k1 && k1.omega === '3/2' && k1.drive === '3/2' && k1.chi === '5/2', 'vortex omega, drive and polarisation reach the kernel as exact rationals', JSON.stringify(k1));
  await evaluate(`document.getElementById('btn-vortex').click()`); await appliedNow();
  const k2 = await kp();
  ok(k2 && k2.omega === '-3/2', 'VORTEX reverses omega', JSON.stringify(k2));
  const pdefs = await evaluate(`Object.entries(PRESETS).map(([id, p]) => [id, p.omega, p.drive, p.chi])`);
  let presetParams = true, which = '';
  for (const [id, om, dr, ch] of pdefs) {
    await evaluate(`document.getElementById(${JSON.stringify(id)}).click()`); await appliedNow();
    const k = await kp();
    if (!k || k.omega !== qstr(om, 10) || k.drive !== qstr(dr, 4) || k.chi !== qstr(ch, 10)) { presetParams = false; which = id + ' ' + JSON.stringify(k); }
  }
  ok(presetParams, `every preset sets vortex omega, drive and polarisation as it defines them (${pdefs.length})`, which);
  await presetFromSeed('pre-damper'); await appliedNow();
  await waitFor(`KERNEL_VIEW.hist.frame.length >= 3 && KERNEL_VIEW.state.frame >= 2`, 60000);
  const dy = await evaluate(`(() => {
    const s = VISUAL_CERTIFICATE.status(), H = KERNEL_VIEW.hist, n = H.frame.length, d = KERNEL_VIEW.dyn;
    updateDynamoHud();
    return { maya: s.maya, dynamo: s.dynamo, n, contiguous: H.frame.every((f, i) => f === i),
             flux: document.getElementById('dyn-flux').textContent, wantFlux: fmtE(FAITHFUL.qNum(d.fluxEnergy)),
             phaseRow: document.getElementById('maya-phase').textContent, wantPhase: FAITHFUL.qNum(d.phase).toFixed(5),
             line: document.getElementById('dynLine').textContent, last: shownStep(), closed: d.closed };
  })()`);
  ok(dy.maya && dy.dynamo && dy.closed, 'the Maya and dynamo channels hold on a running preset, its flux closing on every cube', JSON.stringify(dy));
  ok(dy.contiguous && dy.n >= 3, `the timeline holds every step from the seed, in order (${dy.n})`);
  ok(dy.flux === dy.wantFlux && dy.phaseRow.startsWith(dy.wantPhase), 'the flux and Maya rows show the kernel\'s values', `${dy.flux} vs ${dy.wantFlux}; ${dy.phaseRow} vs ${dy.wantPhase}`);
  ok(dy.line.startsWith('step ' + dy.last + ' '), 'the line under the picture names the step shown', dy.line);
  const strip = await evaluate(`(() => {
    const cw = W / devicePixelRatio, chh = H / devicePixelRatio;
    ctx.clearRect(0, 0, cw, chh); drawDynamoTimeline(ctx, cw, chh);
    const y = Math.floor((chh - 118) * devicePixelRatio), h = Math.floor(100 * devicePixelRatio);
    const img = ctx.getImageData(0, y, Math.floor(cw * devicePixelRatio), h).data;
    let lit = 0; for (let k = 3; k < img.length; k += 4) if (img[k] > 0) lit++;
    return lit / (img.length / 4);
  })()`);
  ok(strip > 0.05, `the dynamo timeline is drawn (${(100 * strip).toFixed(1)}% of its strip painted)`);
  /* observed = M x true: project with M as it is and with M = 1; every
     surface point's amplitude differs by exactly the factor M */
  const obs = await evaluate(`(() => {
    const M = MAYA_VIEW.M;
    projectTorus(); const a = Float64Array.from(PT.field);
    MAYA_VIEW.M = 1; projectTorus(); const b = Float64Array.from(PT.field); MAYA_VIEW.M = M; projectTorus();
    let worst = 0, n = 0;
    for (let k = 0; k < a.length; k++) if (b[k] !== 0) { worst = Math.max(worst, Math.abs(a[k] / b[k] - M)); n++; }
    return { M, worst, n };
  })()`);
  ok(obs.n > 1000 && obs.worst < 1e-12 && obs.M > 1, `the field is drawn at Maya's M = ${obs.M.toFixed(6)} times its true value at every point (${obs.n} points, worst ${obs.worst.toExponential(1)})`);
  const before2 = await evaluate(`KERNEL_VIEW.appliedEpoch`);
  await evaluate(`document.getElementById('btn-reset').click()`);
  ok(await waitFor(`KERNEL_VIEW.appliedEpoch > ${before2} && KERNEL_VIEW.state.frame >= 1 && KERNEL_VIEW.hist.frame.every((f, i) => f === i)`, 20000),
     'RESET starts the timeline again from the seed', await evaluate(`KERNEL_VIEW.hist.frame.join(',') + ' at step ' + KERNEL_VIEW.state.frame`));

  section('the engine on the torus');
  /* The flux map and the lattice on the surface draw the magnetism memory
     where it is on the torus (TorusLattice.lean). Paused, the picture rests on
     one state; the worker's exact A and chi for that state are fetched and the
     gate reads them on the torus itself: positions from tessVertexToTorus's
     angles, circulations by walking each face's corners. */
  await presetFromSeed('pre-damper'); await appliedNow();
  await waitFor(`KERNEL_VIEW.state.frame >= 2`, 60000);
  await evaluate(`document.getElementById('btn-run').click()`);
  await sleep(1500);
  await waitFor(`!KERNEL_VIEW.inFlight && KERNEL_VIEW.queue.length === 0 && DISPLAY.tau === 1`, 20000);
  await sleep(400);
  const tor = await evaluate(`new Promise(res => {
    const id = 1e9 + Math.floor(Math.random() * 1e6);
    EXPORT_WAIT.set(id, m => {
      const P = m.payload, q = x => Q.mk(BigInt(x.n), BigInt(x.d)), A = P.magA.map(q), chi = P.magChi.map(q);
      const quarter = t => ((Math.round((t - Math.PI / 4) / (Math.PI / 2)) % 4) + 4) % 4;
      const pos = Array.from({ length: 16 }, (_, k) => { const t = tessVertexToTorus(TESS_V[k], 0); return [quarter(t.theta), quarter(t.phi)]; });
      const step4 = (a, b) => (b - a + 4) % 4 === 1 ? 1 : -1;
      const side = (u, w) => { const a = Math.log2(u ^ w), base = Math.min(u, w), e = KERNEL.MAG_EDGES.findIndex(([v, b]) => v === base && b === a);
                               return u < w ? A[e] : Q.neg(A[e]); };
      const hole = [], tube = [], cells = [];
      for (let a = 0; a < 4; a++) for (let b = a + 1; b < 4; b++) for (let v = 0; v < 16; v++) if (!(v & ((1 << a) | (1 << b)))) {
        const cs = [v, v ^ (1 << a), v ^ (1 << a) ^ (1 << b), v ^ (1 << b)];
        const x = cs.reduce((t, u, k) => Q.add(t, side(u, cs[(k + 1) % 4])), Q.ZERO);
        const th = cs.map(k => pos[k][0]), ph = cs.map(k => pos[k][1]);
        if (ph.every(z => z === ph[0])) hole[ph[0]] = step4(th[0], th[1]) > 0 ? x : Q.neg(x);
        else if (th.every(z => z === th[0])) tube[th[0]] = step4(ph[0], ph[1]) > 0 ? x : Q.neg(x);
        else {
          const u = cs.map(k => [(pos[k][0] - th[0] + 5) % 4 - 1, (pos[k][1] - ph[0] + 5) % 4 - 1]);
          let area = 0; for (let k = 0; k < 4; k++) area += u[k][0] * u[(k + 1) % 4][1] - u[(k + 1) % 4][0] * u[k][1];
          const lo = xs => { const m = [...new Set(xs)]; return (m[1] - m[0] + 4) % 4 === 1 ? m[0] : m[1]; };
          cells[lo(th) * 4 + lo(ph)] = area > 0 ? x : Q.neg(x);
        }
      }
      const flow = KERNEL.MAG_EDGES.map(([v, a, h], e) => {
        const af = Q.sub(A[e], Q.mul(Q.mk(1n, 10n), Q.sub(chi[h], chi[v])));
        const d = (a === 0 || a === 3) ? step4(pos[v][0], pos[h][0]) : step4(pos[v][1], pos[h][1]);
        return d > 0 ? af : Q.neg(af); });
      const num = FAITHFUL.qNum, LV = LATTICE_VIEW, D = LATTICE_DRAWN;
      const eqv = (xs, ys) => xs.length === ys.length && xs.every((x, i) => num(x) === ys[i]);
      /* the surface sample each vertex is drawn at is where tessVertexToTorus puts it */
      const G = torusGeometry(), wrap = t => ((t % TAU) + TAU) % TAU;
      const nodeOk = Array.from({ length: 16 }, (_, k) => {
        const t = tessVertexToTorus(TESS_V[k], 0), i = LAT_TH(KERNEL.ringThetaOf(k)), j = LAT_PH(KERNEL.ringPhiOf(k));
        return Math.abs(wrap(G.theta[i] - t.theta + 1e-9) - 1e-9) < 1e-12 && Math.abs(wrap(G.phi[j] - t.phi + 1e-9) - 1e-9) < 1e-12; });
      res({ frame: P.frame, shown: LV.frame, ok: LV.ok, holeOk: eqv(hole, LV.to.hole), tubeOk: eqv(tube, LV.to.tube),
            cellOk: eqv(cells, LV.to.cells), flowOk: eqv(flow, LV.to.flow), nodes: nodeOk.every(Boolean),
            rest: D.tau === 1 && [...D.hole].every((x, i) => x === LV.to.hole[i]) && [...D.cells].every((x, i) => x === LV.to.cells[i]) &&
                  [...D.flow].every((x, i) => x === LV.to.flow[i]),
            hole: LV.to.hole, tube: LV.to.tube, strips: LV.strips, anyFlux: LV.to.hole.some(x => x !== 0) && LV.to.cells.some(x => x !== 0) });
    });
    KERNEL_WORKER.postMessage({ type: 'export', id });
  })`, true);
  ok(tor.frame === tor.shown && tor.ok && tor.strips === 8, `paused, the picture shows the worker's state (step ${tor.frame}) and all 8 strip balances hold`, JSON.stringify({ f: tor.frame, s: tor.shown, ok: tor.ok, strips: tor.strips }));
  ok(tor.nodes, 'every vertex is drawn at the surface sample where tessVertexToTorus puts it (node_grid_theta, node_grid_phi)');
  ok(tor.holeOk && tor.tubeOk && tor.anyFlux, 'the flux shown through the hole and along the tube is the circulation of the state\'s A round each loop, read here from its corners');
  ok(tor.cellOk, 'each cell\'s flux is shown turning the way the state\'s A circulates round it (cell_orient_area)');
  ok(tor.flowOk, 'each edge\'s flow is the state\'s whole potential along it, signed towards +theta or +phi');
  ok(tor.rest, 'at rest the drawn values are the state\'s exactly (blend_one)');
  /* the flux map is painted, and each loop's bar has the colour of its sign */
  const fm = await evaluate(`new Promise(res => requestAnimationFrame(() => requestAnimationFrame(() => {
    const dpr = devicePixelRatio, F = FLUX_MAP, c = F.cell, X = r => F.x + r * c, Y = s => F.y + (4 - s) * c;
    const px = (x, y) => Array.from(ctx.getImageData(Math.round(x * dpr), Math.round(y * dpr), 1, 1).data);
    const amber = p => p[3] > 0 && p[0] > p[2] + 60, cyan = p => p[3] > 0 && p[2] > p[0] + 60;
    const H = LATTICE_DRAWN.hole, T = LATTICE_DRAWN.tube;
    const holeCol = [0, 1, 2, 3].every(s => (H[s] >= 0 ? amber : cyan)(px(X(4) + 14, Y(s))));
    const by = Y(4) - 6;
    const tubeCol = [0, 1, 2, 3].every(r => (T[r] >= 0 ? amber : cyan)(px(X(r), T[r] >= 0 ? by - 1 : by - 9)));
    const img = ctx.getImageData(Math.round((F.x - 20) * dpr), Math.round((F.y - 44) * dpr), Math.round((4 * c + 96) * dpr), Math.round((4 * c + 80) * dpr)).data;
    /* glyph pixels, not the panel's dark background: measured 11% on the dynamo preset */
    let lit = 0; for (let k = 0; k < img.length; k += 4) if (Math.max(img[k], img[k + 1], img[k + 2]) > 90) lit++;
    /* the arrows and arcs as drawn, against the gate's own reading of which
       way each points: an edge is along theta when its ends differ in
       tessVertexToTorus's theta, and +theta is right, +phi up */
    const quarter = t => ((Math.round((t - Math.PI / 4) / (Math.PI / 2)) % 4) + 4) % 4;
    const pos = Array.from({ length: 16 }, (_, k) => { const t = tessVertexToTorus(TESS_V[k], 0); return [quarter(t.theta), quarter(t.phi)]; });
    const Gl = F.glyphs, Dd = LATTICE_DRAWN;
    const glyphs = [0, 1, 2, 3].every(s => Gl.hole[s] === (H[s] >= 0 ? 1 : -1) && Gl.tube[s] === (T[s] >= 0 ? -1 : 1)) &&
      Gl.cells.length === 16 && Gl.cells.every((ccw, c) => ccw === (Dd.cells[c] >= 0)) &&
      KERNEL.MAG_EDGES.every(([v, , h], e) => { const th = pos[v][0] !== pos[h][0], f = Dd.flow[e];
        const want = th ? [f >= 0 ? 1 : -1, 0] : [0, f >= 0 ? -1 : 1];
        return Gl.edges[e][0] === want[0] && Gl.edges[e][1] === want[1]; });
    res({ drawn: F.drawn, holeCol, tubeCol, glyphs, lit: lit / (img.length / 4),
          rows: [document.getElementById('dyn-hole').textContent, document.getElementById('dyn-strips').textContent] });
  })))`, true);
  ok(fm.drawn && fm.lit > 0.04, `the flux map's glyphs are drawn (${(100 * fm.lit).toFixed(1)}% of its panel bright)`);
  ok(fm.holeCol && fm.tubeCol, 'each loop\'s bar on the flux map is amber for flux circulating towards +theta (+phi) and cyan against');
  ok(fm.glyphs, 'every arrow on the flux map points the way its flux flows, +theta right and +phi up, and every cell\'s arc turns the way its flux circulates');
  ok(fm.rows[1] === '8 of 8 hold exactly' && fm.rows[0].split(/\s+/).length === 4, 'the panel rows show the four loop fluxes and the balances', JSON.stringify(fm.rows));
  /* the lattice on the surface: off until asked, then every glyph is drawn */
  const lat0 = await evaluate(`({ on: LATTICE_SHOW.on, verts: LATTICE_DRAWN.verts, why: LATTICE_DRAWN.surfaceWhy })`);
  await evaluate(`document.getElementById('btn-lattice').click()`);
  await sleep(300);
  const lat1 = await evaluate(`(() => {
    const quarter = t => ((Math.round((t - Math.PI / 4) / (Math.PI / 2)) % 4) + 4) % 4;
    const pos = Array.from({ length: 16 }, (_, k) => { const t = tessVertexToTorus(TESS_V[k], 0); return [quarter(t.theta), quarter(t.phi)]; });
    const D = LATTICE_DRAWN;
    /* each edge is drawn leaving the vertex its flow leaves: a step of +1 round
       its ring from there when the flow is towards +theta or +phi */
    const leaves = KERNEL.MAG_EDGES.every(([v, , h], e) => { const u = D.edgeFrom[e], w = u === v ? h : v, k = pos[u][0] !== pos[w][0] ? 0 : 1;
      return (u === v || u === h) && (((pos[w][k] - pos[u][k] + 4) % 4 === 1) === (D.flow[e] >= 0)); });
    return { on: LATTICE_SHOW.on, verts: D.verts, e: D.nEdges, c: D.nCells, l: D.nLoops, n: D.nNodes, leaves,
             label: document.getElementById('btn-lattice').textContent }; })()`);
  ok(!lat0.on && lat0.verts === 0 && lat0.why === 'hidden', 'the lattice on the surface is off until asked for');
  ok(lat1.on && lat1.verts > 1000 && lat1.e === 32 && lat1.c === 16 && lat1.l === 8 && lat1.n === 16 && lat1.label === 'LATTICE ON',
     'LATTICE draws all 32 edges, 16 cells, 8 loops and 16 vertices on the surface', JSON.stringify(lat1));
  ok(lat1.leaves, 'on the surface each edge is drawn from the vertex its flow leaves');
  await evaluate(`document.getElementById('btn-lattice').click()`);
  await sleep(300);
  ok(await evaluate(`!LATTICE_SHOW.on && LATTICE_DRAWN.verts === 0`), 'and clicked again, takes it off');
  /* withheld when its certificate does not hold */
  await evaluate(`document.getElementById('btn-lattice').click()`);
  await evaluate(`LATTICE_VIEW.ok = false; LATTICE_VIEW.why = 'withheld by the gate'`);
  await sleep(300);
  const wh = await evaluate(`({ map: FLUX_MAP.drawn, verts: LATTICE_DRAWN.verts, ready: LATTICE_DRAWN.ready, why: FLUX_MAP.why })`);
  await evaluate(`LATTICE_VIEW.ok = true; LATTICE_VIEW.why = ''; document.getElementById('btn-lattice').click()`);
  ok(!wh.map && wh.verts === 0 && !wh.ready && wh.why === 'withheld by the gate', 'without its certificate neither the flux map nor the lattice is drawn, and the reason is kept', JSON.stringify(wh));
  const fx = await evaluate(`(() => {
    VISUAL_CERTIFICATE.status(); projectTorus();
    const F = FLUX_SHAPE, D = LATTICE_DRAWN, G = torusGeometry(), T4 = TH_SEG / 4, P4 = PH_SEG / 4, T8 = TH_SEG / 8, P8 = PH_SEG / 8;
    const mod4 = k => ((k % 4) + 4) % 4;
    const at = (k, span) => { const b = Math.floor(k / span); return [mod4(b), mod4(b + 1), (k - b * span) / span]; };
    let worst = 0, n = 0, maxAbs = 0;
    for (let i = 0; i < TH_SEG; i += 7) for (let j = 0; j < PH_SEG; j += 5) {
      const [r0, r1, ft] = at(i - T4, T4), [s0, s1, fp] = at(j - P4, P4), c = D.cells, m = D.mCell;
      const want = m > 0 ? ((1 - ft) * (1 - fp) * c[r0 * 4 + s0] + ft * (1 - fp) * c[r1 * 4 + s0] + (1 - ft) * fp * c[r0 * 4 + s1] + ft * fp * c[r1 * 4 + s1]) / m : 0;
      worst = Math.max(worst, Math.abs(F.cell[i * PH_SEG + j] - want)); n++;
    }
    for (let j = 0; j < PH_SEG; j++) { const [a, b, f] = at(j - P8, P4); worst = Math.max(worst, Math.abs(F.hole[j] - (D.mLoop > 0 ? ((1 - f) * D.hole[a] + f * D.hole[b]) / D.mLoop : 0))); }
    for (let i = 0; i < TH_SEG; i++) { const [a, b, f] = at(i - T8, T4); worst = Math.max(worst, Math.abs(F.tube[i] - (D.mLoop > 0 ? ((1 - f) * D.tube[a] + f * D.tube[b]) / D.mLoop : 0))); }
    for (const a of [F.cell, F.hole, F.tube]) for (let k = 0; k < a.length; k++) maxAbs = Math.max(maxAbs, Math.abs(a[k]));
    const with1 = Float64Array.from(PT.wx), save = { ...AMP };
    AMP.fluxBulge = 0; AMP.fluxRing = 0; AMP.fluxTwist = 0; projectTorus();
    let moved = 0; for (let k = 0; k < N_PTS; k++) if (PT.wx[k] !== with1[k]) moved++;
    Object.assign(AMP, save); projectTorus();
    const lv = LATTICE_VIEW.ok; LATTICE_VIEW.ok = false; const cert = VISUAL_CERTIFICATE.status().flux; projectTorus(); const off = { ok: FLUX_SHAPE.ok, why: FLUX_SHAPE.why };
    LATTICE_VIEW.ok = lv; VISUAL_CERTIFICATE.status(); projectTorus();
    return { ok: F.ok, node: F.node, worst, n, maxAbs, moved, nPts: N_PTS, cert, off, mCell: D.mCell, mLoop: D.mLoop, row: document.getElementById('dyn-shape').textContent };
  })()`);
  ok(fx.ok && fx.node === 0 && fx.worst < 1e-12 && fx.n > 1000, `the surface's flux shape is the bilinear share of the 16 cell fluxes and the linear share of the 8 loop fluxes, recomputed here at ${fx.n} cell samples and every loop sample, and reproduces each cell and loop at its own place (bilinear_node, lerp_node; worst ${fx.worst.toExponential(1)})`, JSON.stringify(fx));
  ok(fx.maxAbs <= 1 && fx.mCell > 0 && fx.mLoop > 0, `every share lies in [-1, 1] (cell_share_bilinear_le, loop_share_lerp_le; largest ${fx.maxAbs.toFixed(6)})`);
  ok(fx.moved > fx.nPts / 2, `the flux moves the surface: with its three gains at zero, ${fx.moved} of ${fx.nPts} points are placed elsewhere`);
  ok(!fx.cert && !fx.off.ok, 'without the lattice certificate the flux does not shape the surface', JSON.stringify(fx.off));
  await evaluate(`document.getElementById('btn-run').click()`);

  section('the fields around the construct');
  /* The magnetic and electric fields of the construct's sources, and the
     energy flow between them (FieldsAround.lean). Paused on one landed state
     of the damper preset, the gate takes what the page asked the field worker
     (sources, their places, the radius) and what it answered, and recomputes
     the answer independently: B by integrating Biot-Savart numerically along
     each straight piece (Gauss-Legendre, panels finer the nearer the point),
     not by the worker's closed form; E by Coulomb; S = E x B. */
  await presetFromSeed('pre-damper'); await appliedNow();
  await waitFor(`KERNEL_VIEW.state.frame >= 2`, 60000);
  await evaluate(`document.getElementById('btn-run').click()`);
  await sleep(1500);
  await waitFor(`!KERNEL_VIEW.inFlight && KERNEL_VIEW.queue.length === 0 && DISPLAY.tau === 1`, 20000);
  const fieldsLanded = await waitFor(`FIELDS.cur && !FIELDS.busy && FIELDS.cur.frame === KERNEL_VIEW.state.frame && performance.now() - FIELDS.t0 > FIELDS.fadeMs`, 30000);
  const fd = await evaluate(`(() => {
    VISUAL_CERTIFICATE.status(); updateDynamoHud(); projectTorus(); renderTorus();
    const F = FIELDS, C = F.cur, rq = C.req, src = KERNEL_VIEW.dyn.sources, num = FAITHFUL.qNum;
    let badOv = 0;
    for (let v = 0; v < OV.n; v++) { const o = v * 7, d = OV.d;
      if (!(Number.isFinite(d[o]) && Number.isFinite(d[o + 1]) && d[o + 2] >= -1 && d[o + 2] <= 1)) badOv++;
      for (let c = 3; c < 7; c++) if (!(d[o + c] >= 0 && d[o + c] <= 1)) badOv++; }
    for (let v = 0; v < PLANE.nV; v++) { const o = v * 7, d = PLANE.d;
      if (!(Number.isFinite(d[o]) && Number.isFinite(d[o + 1]) && d[o + 2] >= -1 && d[o + 2] <= 1)) badOv++;
      for (let c = 3; c < 7; c++) if (!(d[o + c] >= 0 && d[o + c] <= 1)) badOv++; }
    for (let i = 0; i < PLANE.nI; i++) if (!(PLANE.i[i] < PLANE.nV)) badOv++;
    /* where the surface puts each vertex, read from tessVertexToTorus's angles */
    const quarter = t => ((Math.round((t - Math.PI / 4) / (Math.PI / 2)) % 4) + 4) % 4;
    const at = Array.from({ length: 16 }, (_, k) => { const t = tessVertexToTorus(TESS_V[k], 0);
      const i = TH_SEG / 8 + TH_SEG / 4 * quarter(t.theta), j = PH_SEG / 8 + PH_SEG / 4 * quarter(t.phi), idx = i * PH_SEG + j;
      return [PT.wx[idx], PT.wy[idx], PT.wz[idx]]; });
    const pick = [];
    for (let g = 0; g < C.NR * C.NS; g += 7) pick.push(g);
    return { cert: VISUAL_CERTIFICATE.status().fields, ok: F.ok, why: F.why, drawWhy: F.drawWhy, drawn: F.drawn, nOv: OV.n, nPl: PLANE.nV, badOv,
             frame: KERNEL_VIEW.state.frame, curFrame: C.frame, reqFrame: rq && rq.frame,
             rows: ['fld-current', 'fld-charge', 'fld-haze', 'fld-mforce', 'fld-eforce'].map(id => document.getElementById(id).textContent),
             hazeDrawn: HAZE.drawn, hazeWhy: HAZE.why,
             haze: { id: C.id, N: C.hazeN, R: C.hazeR, req: rq.hazeR, reqN: rq.hazeN, maxNorm: PROJECTION.maxNorm, range: C.pRange, onPiece: C.onPiece },
             J: src.current.map(num), Qc: src.charge.map(num), kirchhoff: src.kirchhoff, neutral: src.neutral,
             at, edges: KERNEL.MAG_EDGES.map(([v, , h]) => [v, h]),
             req: { nodes: Array.from(rq.nodes), paths: Array.from(rq.paths), J: Array.from(rq.J), Q: Array.from(rq.Q), R: rq.R, nseg: rq.nseg },
             pick, plane: pick.map(g => [C.plane[g * 4], C.plane[g * 4 + 1], C.plane[g * 4 + 2], C.planeVal[g * 3], C.planeVal[g * 3 + 1], C.planeVal[g * 3 + 2]]),
             singular: C.singular,
             mForce: Array.from(C.mForce), eForce: Array.from(C.eForce), eNet: C.eNet, eSum: C.eSum, mSingular: C.mSingular, eSingular: C.eSingular,
             /* each stored energy's place in its range, as the plane is coloured */
             places: (() => { const lo = C.planeRange, V = C.planeVal; let out = 0, n = 0;
               for (let g = 0; g < C.NR * C.NS; g++) { if (!V[g * 3 + 2]) continue;
                 for (const k of [0, 2]) { const u = V[g * 3 + k / 2]; if (!(u > 0)) continue; n++;
                   const t = (Math.log(u) - lo[k]) / (lo[k + 1] - lo[k]); if (!(t >= 0 && t <= 1)) out++; } }
               return { out, n }; })() };
  })()`);
  ok(fieldsLanded && fd.curFrame === fd.frame && fd.reqFrame === fd.frame, `the fields drawn are those of the state on the screen (step ${fd.frame})`);
  ok(fd.cert && fd.ok && fd.kirchhoff && fd.neutral, 'the fields channel holds: the kernel decided both conservation laws on the sources', fd.why);
  ok(fd.drawn && fd.drawn.plane > 1000 && fd.hazeDrawn && Object.keys(fd.drawn).join() === 'plane',
     `the fields are drawn: ${fd.drawn && fd.drawn.plane} plane cells and the magnetic pressure seen through, and nothing else -- no line, no arrow`, fd.drawWhy + ' ' + fd.hazeWhy);
  ok(fd.badOv === 0, `every overlay and plane vertex written for the GPU is finite, its depth in [-1, 1] and its colour and opacity in [0, 1], every plane index a vertex (${fd.nOv} + ${fd.nPl}; mix_mem, level_mem)`, `${fd.badOv} outside`);
  ok(fd.places.out === 0 && fd.places.n > 1000, `every stored energy on the plane is inside its own range: its place is in [0, 1] (plot_mem; ${fd.places.n} values)`, `${fd.places.out} outside`);
  const sameArr = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
  ok(sameArr(fd.req.J, fd.J) && sameArr(fd.req.Q, fd.Qc), 'the worker was given the kernel\'s current on every edge and charge at every vertex');
  const pos = fd.Qc.filter(q => q > 0).length, neg = fd.Qc.filter(q => q < 0).length;
  ok(fd.rows[1] === `${pos} positive, ${neg} negative · they sum to 0` && fd.rows[0].endsWith('conserved at all 16 vertices'),
     'the rows show the kernel\'s sources', fd.rows.join(' | '));
  ok(fd.at.every((p, k) => p.every((x, c) => x === fd.req.nodes[k * 3 + c])), 'each charge sits where the surface puts its vertex');
  const NS1 = fd.req.nseg + 1, PP = fd.req.paths;
  ok(fd.edges.every(([v, h], e) => [0, 1, 2].every(c => PP[(e * NS1) * 3 + c] === fd.req.nodes[v * 3 + c] && PP[(e * NS1 + fd.req.nseg) * 3 + c] === fd.req.nodes[h * 3 + c])),
     'each edge\'s current runs along the surface from the vertex it leaves to the one it enters, the way J counts it');
  /* the gate's own fields */
  const GL8 = [[-0.9602898564975363, 0.1012285362903763], [-0.7966664774136267, 0.2223810344533745], [-0.5255324099163290, 0.3137066458778873], [-0.1834346424956498, 0.3626837833783620],
               [0.1834346424956498, 0.3626837833783620], [0.5255324099163290, 0.3137066458778873], [0.7966664774136267, 0.2223810344533745], [0.9602898564975363, 0.1012285362903763]];
  const segQuad = (I, ax, ay, az, dx, dy, dz, x, y, z, acc) => {
    const L2 = dx * dx + dy * dy + dz * dz, L = Math.sqrt(L2);
    const ts = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy + (z - az) * dz) / L2));
    const s = Math.hypot(x - ax - ts * dx, y - ay - ts * dy, z - az - ts * dz) / L;
    const br = [0, 1];
    if (ts > 0 && ts < 1) br.push(ts);
    for (const side of [1, -1]) { let w = s / 8; for (let t = ts + side * w; t > 0 && t < 1; w *= 1.5, t = ts + side * w) br.push(t); }
    br.sort((p, q) => p - q);
    for (let i = 0; i + 1 < br.length; i++) { const t0 = br[i], h = br[i + 1] - t0; if (!(h > 0)) continue;
      for (const [xi, w] of GL8) { const t = t0 + (xi + 1) / 2 * h, rx = x - ax - t * dx, ry = y - ay - t * dy, rz = z - az - t * dz, r2 = rx * rx + ry * ry + rz * rz;
        const f = I * w * h / 2 / (r2 * Math.sqrt(r2));
        acc[0] += f * (dy * rz - dz * ry); acc[1] += f * (dz * rx - dx * rz); acc[2] += f * (dx * ry - dy * rx); } }
  };
  const Bq = (x, y, z) => {
    const acc = [0, 0, 0];
    for (let e = 0; e < 32; e++) { const I = fd.req.J[e]; if (I === 0) continue;
      for (let k = 0; k < fd.req.nseg; k++) {
        const o = (e * NS1 + k) * 3, ax = PP[o], ay = PP[o + 1], az = PP[o + 2], dx = PP[o + 3] - ax, dy = PP[o + 4] - ay, dz = PP[o + 5] - az;
        if (!(dx * dx + dy * dy + dz * dz > 0)) continue;
        segQuad(I, ax, ay, az, dx, dy, dz, x, y, z, acc);
      } }
    return acc;
  };
  const Eq = (x, y, z) => { let ex = 0, ey = 0, ez = 0;
    for (let v = 0; v < 16; v++) { const q = fd.req.Q[v], dx = x - fd.req.nodes[v * 3], dy = y - fd.req.nodes[v * 3 + 1], dz = z - fd.req.nodes[v * 3 + 2], r2 = dx * dx + dy * dy + dz * dz;
      const f = q / (r2 * Math.sqrt(r2)); ex += f * dx; ey += f * dy; ez += f * dz; }
    return [ex, ey, ez]; };
  const crs = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2], nrm = a => Math.sqrt(dot(a, a));
  const tl0 = Date.now();
  let worstB = 0, worstE = 0, nPlane = 0, onPlane = true;
  for (const [x, y, z, uE, uB, has] of fd.plane) {
    if (y !== 0 || Math.hypot(x, z) > fd.req.R * (1 + 1e-6)) onPlane = false;
    if (!has) continue;
    nPlane++;
    const B = Bq(x, y, z), E = Eq(x, y, z);
    worstB = Math.max(worstB, Math.abs(uB / (dot(B, B) / 2) - 1));
    worstE = Math.max(worstE, Math.abs(uE / (dot(E, E) / 2) - 1));
  }
  ok(onPlane && nPlane > 1000, `the plane samples lie on the equatorial plane inside the ball (${nPlane} checked, ${fd.singular} on a source)`);
  ok(worstB < 1e-4, `the magnetic energy density at every sample is Biot-Savart integrated numerically (segment_field_integral; worst ${worstB.toExponential(1)})`);
  ok(worstE < 1e-4, `the electric energy density at every sample is Coulomb's (worst ${worstE.toExponential(1)})`);
  const grab = async (name, n) => {
    const bytes = new Uint8Array(4 * n);
    for (let o = 0; o < n; o += 65536) {
      const b64 = await evaluate(`(() => { const a = ${name}, u = new Uint8Array(a.buffer, a.byteOffset + ${o} * 4, Math.min(65536, a.length - ${o}) * 4);
        let s = ''; for (let i = 0; i < u.length; i += 8192) s += String.fromCharCode.apply(null, u.subarray(i, i + 8192)); return btoa(s); })()`);
      bytes.set(Buffer.from(b64, 'base64'), 4 * o);
    }
    return new Float32Array(bytes.buffer);
  };
  const HZ = fd.haze, HN = HZ.N, HR = HZ.R, hh = 2 * HR / (HN - 1), NH = HN * HN * HN;
  HZ.pB = await grab('FIELDS.cur.pB', NH); HZ.g = await grab('FIELDS.cur.haze', 4 * NH);
  ok(await evaluate(`FIELDS.cur.id`) === HZ.id && HZ.pB.length === NH && HZ.g.length === 4 * NH, 'the pressure grid read here is the one on the screen, whole');
  ok(HN === 40 && HZ.reqN === 40 && HR === HZ.req && HR === 1.3 * HZ.maxNorm && HZ.pB.length === NH && HZ.g.length === 4 * NH && HZ.onPiece === 0,
     `the pressure is taken on ${HN}^3 nodes over the cube of half-side ${HR.toFixed(1)}, 1.3 times the farthest surface point, none of them on a current-carrying piece`);
  let worstP = 0, nP = 0;
  for (let g = 0; g < NH; g += 211) {
    const i = g % HN, j = Math.floor(g / HN) % HN, k = Math.floor(g / (HN * HN)), B = Bq(-HR + i * hh, -HR + j * hh, -HR + k * hh);
    worstP = Math.max(worstP, Math.abs(HZ.pB[g] / (dot(B, B) / 2) - 1)); nP++;
  }
  ok(worstP < 1e-4, `the stored pressure at ${nP} nodes spread over the grid is |B|^2 / 2 with B Biot-Savart's integrated here (haze_pressure_nonneg, haze_stress_trace; worst ${worstP.toExponential(1)})`);
  let lo = Infinity, hi = -Infinity;
  for (const p of HZ.pB) if (p > 0) { const L = Math.log(p); if (L < lo) lo = L; if (L > hi) hi = L; }
  let tBad = 0, tMin = Infinity, tMax = -Infinity, gBad = 0, gWide = 0;
  for (let g = 0; g < NH; g++) {
    const t = HZ.g[g * 4 + 3], want = Math.fround(HZ.pB[g] > 0 && hi > lo ? (Math.log(HZ.pB[g]) - lo) / (hi - lo) : 0);
    if (t !== want || !(t >= 0 && t <= 1)) tBad++;
    tMin = Math.min(tMin, t); tMax = Math.max(tMax, t);
  }
  for (let k = 0; k < HN; k++) for (let j = 0; j < HN; j++) for (let i = 0; i < HN; i++) {
    const g = (k * HN + j) * HN + i, at = [i, j, k], st = [1, HN, HN * HN];
    for (let c = 0; c < 3; c++) {
      const side = at[c] === 0 || at[c] === HN - 1, a = at[c] === 0 ? g : g - st[c], b = at[c] === HN - 1 ? g : g + st[c];
      const d = HZ.g[g * 4 + c];
      if (d !== Math.fround((HZ.g[b * 4 + 3] - HZ.g[a * 4 + 3]) / (side ? 1 : 2))) gBad++;
      if (Math.abs(d) > (side ? 1 : 0.5)) gWide++;
    }
  }
  ok(lo === HZ.range[0] && hi === HZ.range[1] && tBad === 0 && tMin === 0 && tMax === 1,
     `every node's place in the log range of the stored pressures is in [0, 1], taken again here bit for bit, the range's ends attained (plot_mem, plot_log_mono)`, `${tBad} differ; range ${HZ.range} against ${lo}, ${hi}`);
  ok(gBad === 0 && gWide === 0, `every difference along every axis at every node is the stored places' central difference inside and one-sided on the faces, bit for bit, within 1/2 and 1 a step (haze_central_le, haze_side_le)`, `${gBad} differ, ${gWide} too wide`);
  const f16 = x => { if (x === 0) return 0; const a = Math.abs(x); let e = Math.floor(Math.log2(a)); if (Math.pow(2, e) > a) e--; if (Math.pow(2, e + 1) <= a) e++;
    const q = Math.pow(2, Math.max(e, -14) - 10), r = a / q, f = Math.floor(r), n = r - f > 0.5 || (r - f === 0.5 && f % 2 === 1) ? f + 1 : f; return Math.sign(x) * n * q; };
  const H16 = Float64Array.from(HZ.g, f16);
  const gpu = await evaluate(`(() => {
    projectTorus(); renderTorus();
    const zk = OV.zk; OV.zk = 0; const h = hazeFrame(); SURFACE_GL.finish(h, W / devicePixelRatio, H / devicePixelRatio); OV.zk = zk;
    const r = SURFACE_GL.readHaze(), c = PROJECTION.cam, out = [];
    for (let a = 1; a < 12; a++) for (let b = 1; b < 8; b++) { const i = Math.floor(a * r.w / 12), j = Math.floor(b * r.h / 8), o = (j * r.w + i) * 4;
      out.push([i, j, r.px[o], r.px[o + 1], r.px[o + 2]]); }
    return { w: r.w, h: r.h, cssW: W / devicePixelRatio, cssH: H / devicePixelRatio, cam: c, steps: HAZE.steps, hw: HAZE.w, slots: h && [h.q0, h.q1], id: FIELDS.cur.id, tex: SURFACE_GL.tex3Id.slice(), out };
  })()`);
  const cR = gpu.cam, colM = (x1, y2, z2) => { const y = y2 * cR.cRX + z2 * cR.sRX, z1 = -y2 * cR.sRX + z2 * cR.cRX; return [x1 * cR.cRY + z1 * cR.sRY, y, -x1 * cR.sRY + z1 * cR.cRY]; };
  const M0 = colM(1, 0, 0), M1 = colM(0, 1, 0), M2 = colM(0, 0, 1);
  const tri = (x) => {
    if (x.some(v => Math.abs(v) > HR)) return [0, 0, 0, 0];
    const u = x.map(v => ((v / (2 * HR) + 0.5) * (HN - 1) / HN + 0.5 / HN) * HN - 0.5);
    const i0 = u.map(v => Math.floor(v)), f = u.map((v, c) => v - i0[c]), out = [0, 0, 0, 0];
    for (let dz = 0; dz < 2; dz++) for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
      const ii = [i0[0] + dx, i0[1] + dy, i0[2] + dz].map(v => Math.min(HN - 1, Math.max(0, v)));
      const wgt = (dx ? f[0] : 1 - f[0]) * (dy ? f[1] : 1 - f[1]) * (dz ? f[2] : 1 - f[2]), g = (ii[2] * HN + ii[1]) * HN + ii[0];
      for (let c = 0; c < 4; c++) out[c] += wgt * H16[g * 4 + c];
    }
    return [out[0] * (HN - 1) / (2 * HR), out[1] * (HN - 1) / (2 * HR), out[2] * (HN - 1) / (2 * HR), out[3]];
  };
  let worstG = 0, worstV = 0, scaleG = 0, through = 0;
  const cpu = gpu.out.map(([i, j]) => {
    const css = [(i + 0.5) / gpu.w * gpu.cssW, (1 - (j + 0.5) / gpu.h) * gpu.cssH], kk = cR.D * cR.scale;
    const a = (css[0] - cR.cx) / kk, b = (css[1] - cR.cy) / kk;
    const O = [0, 1, 2].map(c => -cR.D * M2[c]), d = [0, 1, 2].map(c => a * M0[c] + b * M1[c] + M2[c]);
    const qa = dot(d, d), qb = dot(O, d), disc = qb * qb - qa * (dot(O, O) - HR * HR);
    let u0 = 0, u1 = -1;
    if (disc > 0) { u0 = Math.max(0, (-qb - Math.sqrt(disc)) / qa); u1 = (-qb + Math.sqrt(disc)) / qa; }
    const G = [0, 0, 0]; let P = 0;
    if (u1 > u0) { through++; const du = (u1 - u0) / gpu.steps, ds = du * nrm(d);
      for (let s2 = 0; s2 < gpu.steps; s2++) { const uu = u0 + (s2 + 0.5) * du, x = [0, 1, 2].map(c => O[c] + uu * d[c]), v = tri(x);
        for (let c = 0; c < 3; c++) G[c] += v[c] * ds; P += v[3] * ds; } }
    return [dot(G, M0), dot(G, M1), P / (2 * HR)];
  });
  for (const v of cpu) scaleG = Math.max(scaleG, Math.abs(v[0]), Math.abs(v[1]));
  gpu.out.forEach(([, , gx, gy, gv], n) => { worstG = Math.max(worstG, Math.abs(gx - cpu[n][0]), Math.abs(gy - cpu[n][1])); worstV = Math.max(worstV, Math.abs(gv - cpu[n][2])); });
  ok(gpu.hw === 1 && gpu.slots && gpu.tex[gpu.slots[1]] === gpu.id && through > 30 && scaleG > 0.05 && worstG < 4e-3 * scaleG && worstV < 4e-3,
     `at ${gpu.out.length} pixels of the haze, ${through} of whose rays cross the ball, the GPU's sums along the ray of the trilinearly read gradient and place are those taken here from the same half-precision grid: the view moved toward the higher pressure (haze_trilinear_mem, haze_ray_sum_le, haze_turn_norm, haze_chord_le; worst ${(worstG / scaleG).toExponential(1)} of the largest, glow ${worstV.toExponential(1)}; haze_veil_mem)`,
     JSON.stringify({ hw: gpu.hw, slots: gpu.slots, tex: gpu.tex, id: gpu.id, through, scaleG, worstG, worstV }));
  const PV = await evaluate(`FIELDS.prev && { id: FIELDS.prev.id, N: FIELDS.prev.hazeN, R: FIELDS.prev.hazeR, slot: SURFACE_GL.tex3Id.indexOf(FIELDS.prev.id) }`);
  let blendOk = false, worstBl = Infinity, nBl = 0, blWhy = 'no earlier field state in a grid slot';
  if (PV && PV.slot >= 0 && PV.N === HN) {
    const G0 = Float64Array.from(await grab('FIELDS.prev.haze', 4 * NH), f16), R0 = PV.R;
    const layers = [0, 7, 19, 31, 39];
    const bl = await evaluate(`(() => {
      const S = SURFACE_GL, gl = S.gl, q1 = S.tex3Id.indexOf(FIELDS.cur.id);
      projectTorus(); renderTorus();
      S.finish({ q0: ${PV.slot}, q1, w: 0.375 }, W / devicePixelRatio, H / devicePixelRatio);
      const out = [], buf = new Float32Array(${HN} * ${HN} * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, S.mixFB);
      for (const k of ${JSON.stringify(layers)}) {
        gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, S.tex3[2], 0, k);
        gl.readPixels(0, 0, ${HN}, ${HN}, gl.RGBA, gl.FLOAT, buf);
        for (let j = 1; j < ${HN}; j += 6) for (let i = 2; i < ${HN}; i += 6) { const o = (j * ${HN} + i) * 4; out.push([i, j, k, buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]); }
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return { out, mixN: S.mixN, R1: S.tex3R[q1] };
    })()`);
    worstBl = 0;
    const tri0 = x => {
      if (x.some(v => Math.abs(v) > R0)) return [0, 0, 0, 0];
      const u = x.map(v => ((v / (2 * R0) + 0.5) * (HN - 1) / HN + 0.5 / HN) * HN - 0.5), i0 = u.map(Math.floor), f = u.map((v, c) => v - i0[c]), o4 = [0, 0, 0, 0];
      for (let dz = 0; dz < 2; dz++) for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        const ii = [i0[0] + dx, i0[1] + dy, i0[2] + dz].map(v => Math.min(HN - 1, Math.max(0, v)));
        const wg = (dx ? f[0] : 1 - f[0]) * (dy ? f[1] : 1 - f[1]) * (dz ? f[2] : 1 - f[2]), g = (ii[2] * HN + ii[1]) * HN + ii[0];
        for (let c = 0; c < 4; c++) o4[c] += wg * G0[g * 4 + c];
      }
      return o4;
    };
    for (const [i, j, k, ...got] of bl.out) {
      const x = [i, j, k].map(v => -HR + v * 2 * HR / (HN - 1)), v0 = tri0(x), g = (k * HN + j) * HN + i, w = 0.375;
      const want = [0, 1, 2].map(c => (1 - w) * v0[c] * (HR / R0) + w * H16[g * 4 + c]).concat([(1 - w) * v0[3] + w * H16[g * 4 + 3]]);
      for (let c = 0; c < 4; c++) worstBl = Math.max(worstBl, Math.abs(got[c] - want[c]));
      nBl++;
    }
    blendOk = bl.mixN === HN && bl.R1 === HR && nBl > 100 && worstBl < 2e-3;
    blWhy = JSON.stringify({ mixN: bl.mixN, R1: bl.R1, HR, R0, nBl, worstBl });
  }
  ok(blendOk, `while a new state fades in, the grid the haze reads is the two states' grids blended node by node, the older read at the newer one's nodes and its differences taken to the newer grid's step (blend_mem, haze_rescale; ${nBl} nodes, worst ${worstBl.toExponential(1)})`, blWhy);
  const comp = await evaluate(`(() => {
    projectTorus(); renderTorus();
    const gl = SURFACE_GL.gl, S = SURFACE_GL, scene = new Uint8Array(4 * W * H), view = new Uint8Array(4 * W * H);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, view);
    gl.bindFramebuffer(gl.FRAMEBUFFER, S.sceneFB); gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, scene); gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const hz = S.readHaze(), cssW = W / devicePixelRatio, cssH = H / devicePixelRatio;
    const bil = (img, w, h, x, y, c, n) => { const X = Math.min(w - 1, Math.max(0, x)), Y = Math.min(h - 1, Math.max(0, y)), x0 = Math.floor(X), y0 = Math.floor(Y), fx = X - x0, fy = Y - y0;
      const at = (i, j) => img[(Math.min(h - 1, j) * w + Math.min(w - 1, i)) * n + c];
      return (1 - fy) * ((1 - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1 - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1)); };
    let checked = 0, worst = 0, moved = 0, diff = 0;
    for (let y = 3; y < H; y += 7) for (let x = 3; x < W; x += 7) {
      const hu = (x + 0.5) / W * hz.w - 0.5, hv = (y + 0.5) / H * hz.h - 0.5;
      const gx = bil(hz.px, hz.w, hz.h, hu, hv, 0, 4), gy = bil(hz.px, hz.w, hz.h, hu, hv, 1, 4), gv = bil(hz.px, hz.w, hz.h, hu, hv, 2, 4);
      const dx = HAZE.gain * gx / cssW * W, dy = -HAZE.gain * gy / cssH * H;
      if (Math.hypot(dx, dy) < 1) continue;
      moved++;
      const va = HAZE.veil * gv;
      for (let c = 0; c < 4; c++) {
        const sc = bil(scene, W, H, x + dx, y + dy, c, 4) / 255, col = c < 3 ? HAZE.col[c] : 1;
        const want = col * va + sc * (1 - va), got = view[(y * W + x) * 4 + c] / 255;
        worst = Math.max(worst, Math.abs(got - want));
        if (Math.abs(scene[(y * W + x) * 4 + c] - view[(y * W + x) * 4 + c]) > 8) diff++;
      }
      checked++;
    }
    FIELDS.show = false; projectTorus(); renderTorus();
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, view);
    let covered = 0; for (let k = 3; k < view.length; k += 4) if (view[k] > 0) covered++;
    const off = { drawn: HAZE.drawn, viaScene: S.viaScene, covered: covered / (W * H) };
    FIELDS.show = true; projectTorus(); renderTorus();
    return { checked, moved, worst, diff, off };
  })()`);
  ok(comp.moved > 500 && comp.worst <= 4 / 255 && comp.diff > 0,
     `at ${comp.moved} pixels the haze moves the view by more than a pixel, and each shows the scene read where the haze says, under its glow, within ${Math.round(comp.worst * 255)} of 255`, JSON.stringify(comp));
  ok(!comp.off.drawn && comp.off.viaScene === false && comp.off.covered > 0.02, 'with the fields off there is no haze: the surface is drawn straight onto the view', JSON.stringify(comp.off));
  ok(fd.rows[2].startsWith(`${HN}³ nodes · highest `) && fd.rows[2].endsWith('seen through'), 'the row tells the pressure grid and that it is seen through', fd.rows[2]);
  /* the forces on the construct: each recomputed here, the magnetic one by
     integrating Biot-Savart numerically over every other piece (a piece's
     own field is zero on its own line, axis_integrand_zero), the electric one
     by Coulomb over the other charges */
  const pieces = [];
  for (let e = 0; e < 32; e++) { const I = fd.req.J[e]; if (I === 0) continue;
    for (let k = 0; k < fd.req.nseg; k++) { const o = (e * NS1 + k) * 3, a = [PP[o], PP[o + 1], PP[o + 2]], c = [PP[o + 3], PP[o + 4], PP[o + 5]];
      const L = Math.hypot(c[0] - a[0], c[1] - a[1], c[2] - a[2]); if (L > 0) pieces.push({ e, k, I, a, c, L }); } }
  const Brest = (x, y, z, skip) => {
    const acc = [0, 0, 0];
    for (const P of pieces) { if (P === skip) continue;
      segQuad(P.I, P.a[0], P.a[1], P.a[2], P.c[0] - P.a[0], P.c[1] - P.a[1], P.c[2] - P.a[2], x, y, z, acc); }
    return acc;
  };
  let worstM = 0, worstPerp = 0, unmatched = 0;
  const MF = fd.mForce, nM = MF.length / 6;
  for (let k = 0; k < MF.length; k += 6) {
    const mid = [MF[k], MF[k + 1], MF[k + 2]], f = [MF[k + 3], MF[k + 4], MF[k + 5]];
    const P = pieces.find(P => Math.hypot((P.a[0] + P.c[0]) / 2 - mid[0], (P.a[1] + P.c[1]) / 2 - mid[1], (P.a[2] + P.c[2]) / 2 - mid[2]) <= 1e-9 * fd.req.R);
    if (!P) { unmatched++; continue; }
    const u = [(P.c[0] - P.a[0]) / P.L, (P.c[1] - P.a[1]) / P.L, (P.c[2] - P.a[2]) / P.L], B = Brest(mid[0], mid[1], mid[2], P), g = crs(u, B).map(x => P.I * x);
    worstM = Math.max(worstM, Math.hypot(f[0] - g[0], f[1] - g[1], f[2] - g[2]) / (Math.abs(P.I) * nrm(B)));
    worstPerp = Math.max(worstPerp, Math.abs(dot(f, u)) / (nrm(f) || 1), Math.abs(dot(f, B)) / ((nrm(f) * nrm(B)) || 1));
  }
  ok(nM + fd.mSingular === pieces.length && nM > 50 && unmatched === 0,
     `there is a magnetic force on the middle of each of the ${pieces.length} pieces carrying current (${nM}, and ${fd.mSingular} whose middle lies on another piece)`);
  ok(worstM < 1e-4, `each is I u x B with B Biot-Savart's from every other piece, integrated here (axis_integrand_zero; worst ${worstM.toExponential(1)} of |I||B|)`);
  ok(worstPerp < 1e-9, `each is across both its current and the field (lorentz_perp; worst ${worstPerp.toExponential(1)})`);
  const EF = fd.eForce, nE = EF.length / 6, charged = fd.req.Q.filter(q => q !== 0).length;
  let worstE2 = 0;
  const gNet = [0, 0, 0];
  let gSum = 0;
  for (let k = 0; k < EF.length; k += 6) {
    const v = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].find(v => fd.req.nodes[v * 3] === EF[k] && fd.req.nodes[v * 3 + 1] === EF[k + 1] && fd.req.nodes[v * 3 + 2] === EF[k + 2] && fd.req.Q[v] !== 0);
    if (v === undefined) { unmatched++; continue; }
    const g = [0, 0, 0];
    for (let w = 0; w < 16; w++) { if (w === v || fd.req.Q[w] === 0) continue;
      const d = [0, 1, 2].map(c => fd.req.nodes[v * 3 + c] - fd.req.nodes[w * 3 + c]), r = nrm(d);
      for (let c = 0; c < 3; c++) g[c] += fd.req.Q[v] * fd.req.Q[w] * d[c] / (r * r * r); }
    worstE2 = Math.max(worstE2, Math.hypot(EF[k + 3] - g[0], EF[k + 4] - g[1], EF[k + 5] - g[2]) / (nrm(g) || 1));
    for (let c = 0; c < 3; c++) gNet[c] += g[c]; gSum += nrm(g);
  }
  ok(nE + fd.eSingular === charged && unmatched === 0, `there is an electric force on each of the ${charged} charges (${nE}, and ${fd.eSingular} sharing a point with another)`);
  ok(worstE2 < 1e-9, `each is q E with E Coulomb's from the other charges, summed here (worst ${worstE2.toExponential(1)})`);
  ok(nrm(fd.eNet) <= 1e-12 * fd.eSum && nrm(gNet) <= 1e-12 * gSum,
     `the electric forces on the charges sum to zero (coulomb_newton3; net ${(nrm(fd.eNet) / fd.eSum).toExponential(1)} of their sum, ${(nrm(gNet) / gSum).toExponential(1)} here)`);
  ok(fd.rows[3].startsWith(`${nM} pieces`) && fd.rows[4].startsWith(`${nE} charges`), 'the rows count the forces', fd.rows.slice(3).join(' | '));
  console.log(`       (the gate's own fields took ${((Date.now() - tl0) / 1000).toFixed(1)} s)`);
  /* the toggle, and nothing drawn without the certificate */
  await evaluate(`document.getElementById('btn-fields').click()`);
  const off = await evaluate(`(() => { projectTorus(); renderTorus(); return { show: FIELDS.show, drawn: FIELDS.drawn, haze: HAZE.drawn, label: document.getElementById('btn-fields').textContent }; })()`);
  await evaluate(`document.getElementById('btn-fields').click()`);
  /* shown again, nothing is drawn until the fields of the state on the
     screen have been computed afresh */
  const blank = await evaluate(`(() => { projectTorus(); renderTorus(); return FIELDS.drawn.plane + (HAZE.drawn ? 1 : 0); })()`);
  const back = await waitFor(`FIELDS.cur && FIELDS.cur.frame === KERNEL_VIEW.state.frame && !FIELDS.busy`, 20000);
  const on = await evaluate(`(() => { projectTorus(); renderTorus(); return { show: FIELDS.show, drawn: FIELDS.drawn, haze: HAZE.drawn, label: document.getElementById('btn-fields').textContent }; })()`);
  ok(!off.show && off.label === 'FIELDS OFF' && off.drawn.plane === 0 && !off.haze &&
     blank === 0 && back && on.show && on.label === 'FIELDS ON' && on.drawn.plane > 0 && on.haze,
     'FIELDS takes the fields off and puts fresh ones back', JSON.stringify({ off, blank, back, on }));
  await evaluate(`FIELDS.ok = false; FIELDS.why = 'withheld by the gate'`);
  await sleep(600);
  const fw = await evaluate(`(() => { const cert = VISUAL_CERTIFICATE.status().fields; updateDynamoHud(); projectTorus(); renderTorus(); return { cert, drawn: FIELDS.drawn, haze: HAZE.drawn,
    rows: ['fld-current', 'fld-charge'].map(id => document.getElementById(id).textContent) }; })()`);
  await evaluate(`FIELDS.ok = true; FIELDS.why = ''`);
  ok(!fw.cert && fw.drawn.plane === 0 && !fw.haze && fw.rows[0] === 'withheld' && fw.rows[1] === 'withheld by the gate',
     'without its certificate nothing of the fields is drawn, and the reason is shown', JSON.stringify(fw));
  await evaluate(`document.getElementById('btn-run').click()`);
  await evaluate(`document.getElementById('btn-fields').click()`);
  const q0 = await evaluate(`({ sent: FIELDS.sent, v: KERNEL_VIEW.appliedVersion })`);
  await evaluate(`document.getElementById('pre-checksum').click()`);
  const landed = await waitFor(`KERNEL_VIEW.appliedVersion > ${q0.v} && KERNEL_VIEW.state.frame >= 3`, 20000);
  const q1 = await evaluate(`({ sent: FIELDS.sent, v: KERNEL_VIEW.appliedVersion })`);
  await evaluate(`document.getElementById('btn-run').click()`);
  await waitFor(`!KERNEL_VIEW.inFlight && KERNEL_VIEW.queue.length === 0 && DISPLAY.tau === 1`, 20000);
  await evaluate(`document.getElementById('btn-fields').click()`);
  const resumed = await waitFor(`FIELDS.frame === KERNEL_VIEW.appliedVersion + ':' + KERNEL_VIEW.state.frame && FIELDS.cur && !FIELDS.busy`, 20000);
  await evaluate(`document.getElementById('btn-run').click()`);
  ok(landed && q1.sent === q0.sent && resumed, 'with the fields off nothing is computed for the steps that run; on again, they are computed for the state on the screen',
     JSON.stringify({ q0, q1, resumed }));

  section('motion');
  /* One step a second, every step drawn in order, the surface moving to each
     state over its second and landing on it exactly, the next move starting
     where the last ended. */
  await presetFromSeed('pre-damper');
  await sleep(3000);
  const mo = await evaluate(`new Promise(res => {
    const t0 = performance.now(), f0 = KERNEL_VIEW.state.frame, bad0 = DISPLAY.bad, n0 = DISPLAY.frames;
    const tr0 = DISPLAY.transitions, la0 = DISPLAY.landed;
    let prev = Float64Array.from(DISPLAY_FIELD), prevAt = DISPLAY.at, prevT0 = DISPLAY.t0, prevBody = PROJECTION.frameBody;
    let worstShare = 0, worstZoom = 0, frames = 0, interior = 0, maxQueue = 0, wrongDur = 0;
    const starts = [];
    function tick() {
      const now = performance.now();
      let jump = 0, span = 0;
      for (let i = 0; i < DISPLAY_FIELD.length; i++) {
        jump = Math.max(jump, Math.abs(DISPLAY_FIELD[i] - prev[i]));
        span = Math.max(span, Math.abs(fieldGrid[i] - FROM_FIELD[i]));
      }
      if (DISPLAY.t0 === prevT0 && span > 0 && DISPLAY.dur > 0)
        worstShare = Math.max(worstShare, jump / span - (DISPLAY.at - prevAt) / DISPLAY.dur);
      if (DISPLAY.tau > 0 && DISPLAY.tau < 1) interior++;
      maxQueue = Math.max(maxQueue, KERNEL_VIEW.queue.length);
      if (DISPLAY.t0 !== prevT0) { starts.push(DISPLAY.t0); if (DISPLAY.dur !== STEP_INTERVAL_MS) wrongDur++; }
      if (prevBody > 0) worstZoom = Math.max(worstZoom, Math.abs(PROJECTION.frameBody - prevBody) / prevBody);
      prev = Float64Array.from(DISPLAY_FIELD); prevAt = DISPLAY.at; prevT0 = DISPLAY.t0; prevBody = PROJECTION.frameBody; frames++;
      if (now - t0 < 8000) requestAnimationFrame(tick);
      else {
        const pr = KERNEL_VIEW.projected.slice(-8), gaps = pr.slice(1).filter((f, i) => f - pr[i] !== 1).length;
        const spacing = starts.slice(1).map((x, i) => x - starts[i]);
        res({ sps: (KERNEL_VIEW.state.frame - f0) / ((now - t0) / 1000), gaps, projected: pr.join(' '),
              bad: DISPLAY.bad - bad0, checked: DISPLAY.frames - n0, worstShare, worstZoom, frames, interior, maxQueue, wrongDur,
              spacing: spacing.map(x => x.toFixed(3)).join(' '), worstSpacing: spacing.reduce((m, x) => Math.max(m, Math.abs(x - STEP_INTERVAL_MS)), 0),
              transitions: DISPLAY.transitions - tr0, landed: DISPLAY.landed - la0, landedNow: DISPLAY.landedThis });
      }
    }
    requestAnimationFrame(tick);
  })`, true);
  ok(mo.sps >= 0.8 && mo.sps <= 1.1, `the run takes one step a second (${mo.sps.toFixed(2)} a second over 8 s)`);
  ok(mo.wrongDur === 0 && mo.transitions >= 6 && mo.worstSpacing <= 1e-6,
     `each step is drawn over exactly 1 s and the next starts where the last ended (${mo.spacing} ms apart)`);
  ok(mo.gaps === 0, `every step is drawn, in order (${mo.projected})`);
  ok(mo.maxQueue <= 1, `the kernel is never more than one step ahead of the picture (at most ${mo.maxQueue} waiting)`);
  ok(mo.landed >= mo.transitions - (mo.landedNow ? 0 : 1), `every move lands exactly on its state before the next begins (${mo.landed} of ${mo.transitions})`);
  ok(mo.bad === 0 && mo.checked > 10, `every frame draws a point on the segment between two exact states, and each state exactly at its end (${mo.checked} frames)`);
  ok(mo.worstShare <= 0.01, `within a move, each frame draws the share of the change that its share of the second calls for (worst excess ${mo.worstShare.toFixed(4)})`);
  ok(mo.interior >= 0.3 * mo.frames, `the surface moves between steps: ${mo.interior} of ${mo.frames} frames drawn mid-move`);
  ok(mo.worstZoom <= 0.1, `the framing eases: never more than 10% between frames (worst ${(100 * mo.worstZoom).toFixed(2)}%)`);
  await setGlide(10);
  await presetFromSeed('pre-damper');
  await waitFor(`KERNEL_VIEW.state.frame >= 3 && KERNEL_VIEW.state.frame <= 6`, 20000);
  await setGlide(20);
  const gv = await evaluate(`KERNEL_VIEW.configVersion`);
  await waitFor(`KERNEL_VIEW.appliedVersion === ${gv}`, 20000);
  const rg = await evaluate(`({ ok: RAMP_VIEW.ok, rows: RAMP_VIEW.rows.map(r => ({ id: r.id, s: Q.toString(r.s), k: r.k, n: r.n })), dur: DISPLAY.dur })`);
  await setGlide(1);
  ok(rg.ok && rg.rows.length === 3 && rg.rows.every(r => r.n === 20 && r.k >= 1 && r.k <= 3) && rg.dur === 1000,
     'a glide changed mid-way goes on from where each strength stands over the new number of steps, still one step a second', JSON.stringify(rg));

  section('exports');
  /* saveJSON is replaced by a recorder, so the gate reads exactly the file the
     page would offer. Both files name the build that wrote them; the step
     carries what replays it, and the audit says which arguments each matrix
     belongs to. Checked inside the page: the audit is about 12 MB. */
  await evaluate(`saveJSON = async (payload, filename) => { (window.__saved = window.__saved || []).push({ filename, json: JSON.stringify(payload, jsonReplacer) }); return 1; }`);
  const fp = await evaluate(`SOURCE_FINGERPRINT`);
  await evaluate(`document.getElementById('btn-export').click()`);
  const gotStep = await waitFor(`(window.__saved || []).some(s => s.filename.includes('_step_'))`, 30000);
  const st = gotStep ? await evaluate(`(() => { const j = JSON.parse(window.__saved.find(s => s.filename.includes('_step_')).json), L = j.last;
    const q = x => Q.mk(BigInt(x.n), BigInt(x.d)), st = a => (Array.isArray(a) ? a : a.psi || a.values).map(q);
    const E = a => st(a).reduce((t, x) => Q.add(t, Q.mul(x, x)), Q.ZERO);
    let magKeeps = null, rotKeeps = null;
    try {
      const P = KERNEL_VIEW.params, K = KERNEL, store = p => p.map(K.storeQ), same = (x, y) => x.length === y.length && x.every((v, i) => Q.eq(v, y[i]));
      const B0 = st(L.afterMSTVQCoupling), G = st(L.afterMagnetism), M = st(L.afterMSTVQ), T = st(L.afterTGCR);
      const tau = K.magTau(L.magAIn.map(q), L.magChiIn.map(q), P), tT = K.tgcrTau(M, P);
      const eB = B0.reduce((t, x) => Q.add(t, Q.mul(x, x)), Q.ZERO), eM = M.reduce((t, x) => Q.add(t, Q.mul(x, x)), Q.ZERO);
      const en = p => p.reduce((t, x) => Q.add(t, Q.mul(x, x)), Q.ZERO);
      magKeeps = same(K.edgeRotate(B0, tau, store), G) && Q.eq(en(K.edgeRotate(B0, tau)), eB);
      rotKeeps = same(K.edgeRotate(M, tT, store), T) && Q.eq(en(K.edgeRotate(M, tT)), eM);
    } catch (e) { magKeeps = 'unreadable: ' + e.message; }
    return { schema: j.schema, fp: j.manifest && j.manifest.sourceFingerprint, params: !!L.params && 'appliedH' in L.params,
             dielectricIn: Array.isArray(L.dielectricIn) && L.dielectricIn.length === 16, stepError: !!L.stepError && 'delta' in L.stepError,
             magKeeps, rotKeeps,
             lambda: Array.isArray(L.lambdaIn) && L.lambdaIn.length === 4 && Array.isArray(L.lambdaOut) && L.lambdaOut.length === 4,
             records: Array.isArray(L.sutraRecord) && L.sutraRecord.length === L.activeSutras.length * (L.composition === 'SYMMETRIC_CONCURRENT' ? 2 : 1) || (Array.isArray(L.sutraRecord) ? L.sutraRecord.length + ' records for ' + L.activeSutras.length + ' ' + L.composition : 'none'),
             medium: typeof L.mayaPhaseIn === 'object' && Array.isArray(L.magAIn) && L.magAIn.length === 32 &&
                     Array.isArray(L.magChiIn) && L.magChiIn.length === 16 && typeof j.mayaPhase === 'object' &&
                     Array.isArray(j.magA) && j.magA.length === 32 && Array.isArray(j.magChi) && j.magChi.length === 16,
             maya: !!L.maya && 'factor' in L.maya && 'phase' in L.maya }; })()`) : {};
  ok(gotStep && st.schema === 'vedic-step-export/5' && st.fp === fp, `the step export names this build (${st.schema}, fingerprint ${st.fp})`);
  ok(st.params && st.dielectricIn && st.stepError,
     'the step export carries its parameters, starting dielectric and error budget', JSON.stringify(st));
  ok(st.magKeeps === true && st.rotKeeps === true, 'the exported magnetism and TGCR stages are replayed here from the exported state and memory, bit for bit, and each rotation before storing keeps the energy exactly (rot_pair_energy)', JSON.stringify({ magKeeps: st.magKeeps, rotKeeps: st.rotKeeps }));
  ok(st.lambda === true && st.records === true, 'the step export carries lambda in and out and one record per formula applied', JSON.stringify({ lambda: st.lambda, records: st.records }));
  ok(st.medium === true && st.maya === true, 'the step export carries the medium the step read and left (Maya phase, A, chi) and Maya\'s report');
  await evaluate(`document.getElementById('btn-exact-operator-audit').click()`);
  const gotAudit = await waitFor(`(window.__saved || []).some(s => s.filename.includes('_formula_audit_'))`, 300000);
  const au = gotAudit ? await evaluate(`(() => { const j = JSON.parse(window.__saved.find(s => s.filename.includes('_formula_audit_')).json), F = j.audit.formulas, C = j.audit.corpus;
    const spec = KERNEL.SUTRA_ARG_SPEC;
    let decided = 0, failed = [], refused = 0;
    for (const c of C) for (const r of c.sutras) { if (r.refused) { refused++; continue; } if (r.ok !== true) failed.push('S' + r.id + ' on corpus ' + c.corpus); decided += r.claims.length; }
    return { schema: j.schema, fp: j.manifest && j.manifest.sourceFingerprint, n: F.length,
             args: F.every(f => Array.isArray(f.arguments) && f.arguments.length === spec[f.id].length && f.arguments.every((a, i) => a.name === spec[f.id][i].name)),
             corpora: C.length, decided, failed, refused }; })()`) : {};
  ok(gotAudit && au.schema === 'vedic-operator-audit/2' && au.fp === fp && au.n === 29, `the formula audit names this build (fingerprint ${au.fp}) and covers the 29 formulas`);
  ok(au.args === true, 'it lists each formula\'s operands by name, as the kernel declares them');
  ok(au.corpora > 0 && au.decided > 0 && au.failed.length === 0, `on ${au.corpora} test states every statement decided holds (${au.decided} decided, ${au.refused} refused)`, (au.failed || []).slice(0, 4).join('; '));

  section('framing');
  /* The framing must not breathe. It zooms out within FRAME_OUT_MS when the
     surface grows past it, back in over FRAME_IN_MS once the surface is
     inside FRAME_HOLD of it, and holds still between (frame_hold).

     This section used to count the frame's turns over 8 s each of CYMATIC,
     CROWN and OCTAD and require at most 0.2 a second. That expectation was
     wrong: the count follows what the field does in the window, not the
     framing. Measured with the framing unchanged, the same gate read 0.08,
     0.25 and 0.25 on three runs: in the two it failed, CROWN was caught at
     steps 60 to 84, where the surface's size jumps from about 1860 to 3155
     and back, and a frame that keeps it in view has to follow it out and
     in. Counting turns of the frame against turns of the size did not
     separate the framings either (with the hold 1/18 to 3/11, two lags and
     no hold 3/17 to 5/17, one lag 4/14 to 8/18). So the property is held
     where it can be decided: the page's own framing step, fed a size that
     swings, must not move back once out; and on a real run, every frame
     must keep the law, holding while the size is in the band. */
  const fs = await evaluate(`(() => {
    const run = step => { let F = 1000; const s = [];
      for (let i = 0; i < 300; i++) { const b = 1000 * (1 + 0.06 * Math.sin(2 * Math.PI * i * 100 / 3000)); F = step(F, b, 100); s.push(F); }
      let back = 0; for (let i = 31; i < s.length; i++) if (s[i] < s[i - 1]) back++;
      return { back, lo: Math.min(...s.slice(30)), hi: Math.max(...s.slice(30)) }; };
    return { page: run(frameStep), oneLag: run((F, b, dt) => F + (b - F) * (1 - Math.exp(-dt / 1200))) };
  })()`);
  ok(fs.page.back === 0 && fs.oneLag.back > 50,
     `fed a size swinging 6% either way every 3 s, the frame once out never moves back (frame_hold; one lag of 1.2 s moves back in ${fs.oneLag.back} of 269 steps)`, JSON.stringify(fs));
  await cmd('Page.navigate', { url: PAGE });
  ok(await waitFor(`typeof KERNEL_VIEW !== 'undefined' && KERNEL_VIEW.state !== null && KERNEL_VIEW.selfTestsOk === true`, 90000), 'the page loads again for a fresh view');
  await setGlide(1);
  let lawFrames = 0, lawBroken = [], turns = 0, turnSecs = 0;
  const turnsBy = [];
  for (const id of ['pre-damper', 'pre-amplifier']) {
    await presetFromSeed(id);
    await sleep(6000);
    const br = await evaluate(`new Promise(res => {
      const t0 = performance.now(), s = [], b = [];
      function tick() {
        s.push(PROJECTION.frameBody); b.push(PROJECTION.body);
        if (performance.now() - t0 < 8000) requestAnimationFrame(tick);
        else {
          /* each frame keeps the law: in the band it holds; past the frame it
             moves out toward the size, not past it; inside the band it moves in
             toward the size, not past it */
          let broken = [];
          for (let i = 1; i < s.length; i++) {
            const F = s[i - 1], G = s[i], B = b[i];
            const holds = B >= FRAME_HOLD * F && B <= F ? G === F
                        : B > F ? G >= F && G <= B * (1 + 1e-12)
                        : G <= F && G >= B * (1 - 1e-12);
            if (!holds && broken.length < 3) broken.push({ i, F, G, B });
          }
          let n = 0, dir = 0, ext = s[0];
          for (let i = 1; i < s.length; i++) {
            const v = s[i];
            if (dir >= 0) { if (v > ext) ext = v; else if (ext - v > 0.01 * ext) { if (dir > 0) n++; dir = -1; ext = v; } }
            else { if (v < ext) ext = v; else if (v - ext > 0.01 * ext) { n++; dir = 1; ext = v; } }
            if (dir === 0 && v !== s[0]) dir = v > s[0] ? 1 : -1;
          }
          res({ frames: s.length - 1, broken, rev: n, secs: (performance.now() - t0) / 1000 });
        }
      }
      requestAnimationFrame(tick);
    })`, true);
    lawFrames += br.frames; lawBroken.push(...br.broken.map(x => ({ id, ...x })));
    turns += br.rev; turnSecs += br.secs; turnsBy.push(`${id.slice(4)} ${br.rev}`);
  }
  ok(lawBroken.length === 0 && lawFrames > 100,
     `on a fresh page every one of ${lawFrames} frames over DAMPER and AMPLIFIER keeps the framing law (frame_hold; it turned ${(turns / turnSecs).toFixed(2)} times a second, ${turnsBy.join(', ')}, following the size where it truly jumps)`,
     JSON.stringify(lawBroken));

  section('on a phone');
  await cmd('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 3, mobile: true });
  await cmd('Page.navigate', { url: PAGE });
  await waitFor(`typeof KERNEL_VIEW !== 'undefined' && KERNEL_VIEW.state !== null && document.getElementById('amp-glide')`, 90000);
  await setGlide(1);
  const phoneUp = await waitFor(`typeof KERNEL_VIEW !== 'undefined' && KERNEL_VIEW.state !== null && KERNEL_VIEW.state.frame >= 1 && KERNEL_VIEW.selfTestsOk === true && SURFACE_STATUS.drawn`, 90000);
  const ph = await evaluate(`(() => { projectTorus(); renderTorus();
    const right = document.getElementById('right'), r = right.getBoundingClientRect();
    return { w: innerWidth, collapsed: right.classList.contains('collapsed'), panelLeft: r.left, charts: CHARTS.on, flux: FLUX_MAP.drawn,
             surface: SURFACE_STATUS.drawn, quads: PROJECTION.quads, label: document.getElementById('btn-charts').textContent }; })()`);
  ok(phoneUp && ph.w === 390 && ph.collapsed && ph.panelLeft >= 390 && !ph.charts && ph.label === 'CHARTS OFF' && ph.surface && ph.quads > 1000,
     'on a phone the page opens on the torus alone: the controls closed off the screen and the charts off', JSON.stringify(ph));
  await evaluate(`document.getElementById('toggle-right').click()`);
  await sleep(100);
  await waitFor(`document.getElementById('right').getAnimations().length === 0`, 10000);
  const ph2 = await evaluate(`(() => { const r = document.getElementById('right').getBoundingClientRect(), d = getComputedStyle(document.getElementById('dynLine')).display;
    return { left: r.left, width: r.width, dyn: d }; })()`);
  await evaluate(`document.getElementById('toggle-right').click(); document.getElementById('btn-charts').click()`);
  await sleep(600);
  const ph3 = await evaluate(`({ charts: CHARTS.on, flux: FLUX_MAP.drawn, label: document.getElementById('btn-charts').textContent })`);
  ok(ph2.left === 0 && ph2.width === 390 && ph2.dyn === 'none' && ph3.charts && ph3.label === 'CHARTS ON' && ph3.flux,
     'opened, the controls fill the screen with nothing drawn over them; CHARTS puts the charts back', JSON.stringify({ ph2, ph3 }));

  section('errors');
  ok(errors.length === 0, 'no exception and no console.error', errors.slice(0, 3).join(' | '));
} finally {
  browser.close();
}
console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) { for (const f of failures) console.log('  ' + f); process.exit(1); }
