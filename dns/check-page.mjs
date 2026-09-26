/* Browser gate for cymatic.html.

   Every other suite here runs the physics under node. Nothing ran the PAGE,
   and the page is where it is used. That gap cost exactly what a missing gate
   costs: wiring the Navier-Stokes panel in, I left it calling config() at
   construction time, before the renderer's first recompute() had resolved a
   state. config() dereferenced a null state, the TypeError left the top-level
   script, and the three calls at the end of the file -- sprinkle(),
   recompute(), requestAnimationFrame(loop) -- never ran. The page loaded to a
   blank canvas. Every node suite stayed green, because none of them opens the
   page. Two measurements from that run, kept because they are what this file
   exists to notice:

     before the fix   state === null, canvases blank, one uncaught TypeError
     after the fix    k = 1149.1654751695562 m^-1, both canvases drawn,
                      |mu| = 1.02787755, Floquet growth 3.0521 s^-1 against
                      the renderer's Mathieu 3.2397 s^-1, in 8.3 s

   The same run found a second defect neither of us would have seen from the
   source: the page declared no character set. Served or opened without one a
   browser decodes it as windows-1252, and every micron, s^-1, epsilon and mu
   on the deck came back mangled -- the panel's own heading read "|I^1/4|". One
   <meta charset> fixed it, and the check below is what keeps it fixed.

   It drives a real Chromium over the DevTools protocol. Node 22 has a global
   WebSocket and a built-in http server, so this needs no dependency and no
   install -- but it does need a browser, and it REFUSES rather than skipping
   when there is none. A gate that quietly passes on the machine without the
   binary is the bypass this repository's testing rules forbid.

   Run: node dns/check-page.mjs
   Env: CHROME=/path/to/chrome  overrides binary discovery.
*/
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { createReadStream, existsSync, readFileSync, statSync, mkdtempSync,
         rmSync, writeFileSync } from 'node:fs';
import { dirname, join, normalize, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';
import { createRequire } from 'node:module';

const here = dirname(fileURLToPath(import.meta.url));
const REPO = join(here, '..');
const require = createRequire(import.meta.url);

let pass = 0; const failures = [];
function ok(cond, label, detail){
  if (cond){ pass++; console.log('  ok   ' + label); }
  else { failures.push(`${label}${detail ? '  -- ' + detail : ''}`);
         console.log('  FAIL ' + label + (detail ? '  -- ' + detail : '')); }
}
function rel(got, want, decades, label){
  if (!Number.isFinite(got)) return ok(false, label, `got ${got}`);
  const denom = Math.abs(want) > 0 ? Math.abs(want) : 1;
  const err = Math.abs(got - want)/denom;
  ok(err <= Math.pow(10, -decades), label,
     `got ${got}, want ${want}, relative error ${err.toExponential(3)} > 1e-${decades}`);
}
const section = t => console.log('\n' + t);

/* ---- the browser ------------------------------------------------------- */
const CANDIDATES = [
  process.env.CHROME, process.env.CHROMIUM,
  '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
  '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium', '/usr/bin/chromium-browser',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
].filter(Boolean);
const BROWSER = CANDIDATES.find(p => { try { return statSync(p).isFile(); } catch { return false; } });
if (!BROWSER) throw new Error(
  'check-page: no Chromium or Chrome binary found. Tried:\n  '
  + CANDIDATES.join('\n  ')
  + '\nSet CHROME=/path/to/chrome. This refuses rather than skipping: the page '
  + 'defects this file exists to catch are invisible to every other suite here, '
  + 'so a pass without a browser would be a pass that checked nothing.');

/* ---- the repository, over http ----------------------------------------- */
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
                '.json': 'application/json; charset=utf-8', '.css': 'text/css; charset=utf-8' };
/* No charset is sent for .html on purpose. The page must declare its own: a
   header supplied here would mask exactly the mojibake measured above, which
   is what a reader opening the file from disk sees. */
const HTML_TYPE = 'text/html';
function serve(root){
  const srv = createServer((req, res) => {
    const rel = decodeURIComponent(req.url.split('?')[0]);
    const p = join(root, normalize(rel).replace(/^(\.\.[/\\])+/, ''));
    /* The browser asks for /favicon.ico unprompted. Answering 204 keeps the
       harness from manufacturing a resource error the page did not cause. */
    if (rel === '/favicon.ico'){ res.writeHead(204); return res.end(); }
    if (!p.startsWith(root) || !existsSync(p) || !statSync(p).isFile()){
      res.writeHead(404); return res.end('not found');
    }
    const e = extname(p);
    res.writeHead(200, { 'Content-Type': e === '.html' ? HTML_TYPE : (TYPES[e] || 'application/octet-stream') });
    createReadStream(p).pipe(res);
  });
  return new Promise(r => srv.listen(0, '127.0.0.1', () => r({ srv, port: srv.address().port })));
}

/* ---- DevTools protocol ------------------------------------------------- */
class Browser {
  static async launch(){
    const profile = mkdtempSync(join(tmpdir(), 'checkpage-'));
    const port = 9400 + Math.floor(Math.random()*400);
    const proc = spawn(BROWSER, [
      '--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu',
      '--hide-scrollbars', '--window-size=1400,1100', '--force-device-scale-factor=1',
      `--user-data-dir=${profile}`, `--remote-debugging-port=${port}`, 'about:blank'
    ], { stdio: ['ignore', 'pipe', 'pipe'] });
    let err = '';
    proc.stderr.on('data', d => { err += d; });
    let wsUrl = null;
    for (let i = 0; i < 160 && !wsUrl; i++){
      try {
        const r = await fetch(`http://127.0.0.1:${port}/json/version`);
        if (r.ok) wsUrl = (await r.json()).webSocketDebuggerUrl;
      } catch { /* not listening yet */ }
      if (!wsUrl) await new Promise(r => setTimeout(r, 250));
    }
    if (!wsUrl) throw new Error(`check-page: ${BROWSER} never opened a debugging port.\n${err}`);
    const ws = new WebSocket(wsUrl);
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = e => rej(new Error('ws: ' + e.message)); });
    return new Browser(proc, ws, profile);
  }
  constructor(proc, ws, profile){
    this.proc = proc; this.ws = ws; this.profile = profile;
    this.n = 0; this.pending = new Map(); this.listeners = [];
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id !== undefined){
        const p = this.pending.get(m.id); this.pending.delete(m.id);
        if (m.error) p.rej(new Error(JSON.stringify(m.error))); else p.res(m.result);
      } else for (const f of this.listeners) f(m);
    };
  }
  send(method, params = {}, sessionId){
    const id = ++this.n;
    return new Promise((res, rej) => {
      this.pending.set(id, { res, rej });
      this.ws.send(JSON.stringify({ id, method, params, sessionId }));
    });
  }
  close(){
    try { this.ws.close(); } catch {}
    try { this.proc.kill('SIGKILL'); } catch {}
    try { rmSync(this.profile, { recursive: true, force: true }); } catch {}
  }
}

class Page {
  static async open(browser, url){
    const { targetId } = await browser.send('Target.createTarget', { url: 'about:blank' });
    const { sessionId } = await browser.send('Target.attachToTarget', { targetId, flatten: true });
    const page = new Page(browser, sessionId);
    browser.listeners.push(m => {
      if (m.sessionId !== sessionId) return;
      if (m.method === 'Runtime.consoleAPICalled')
        page.console.push(m.params.type + ': ' + m.params.args
          .map(a => a.value ?? a.description ?? a.type).join(' '));
      if (m.method === 'Runtime.exceptionThrown')
        page.errors.push(m.params.exceptionDetails.exception?.description
          || m.params.exceptionDetails.text);
      if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error')
        page.logErrors.push(m.params.entry.text + ' @' + (m.params.entry.url || ''));
    });
    await page.cmd('Runtime.enable');
    await page.cmd('Log.enable');
    await page.cmd('Page.enable');
    await page.cmd('Page.navigate', { url });
    return page;
  }
  constructor(browser, sessionId){
    this.browser = browser; this.sessionId = sessionId;
    this.console = []; this.errors = []; this.logErrors = [];
  }
  cmd(m, p){ return this.browser.send(m, p, this.sessionId); }
  async eval(expression, awaitPromise = false){
    const r = await this.cmd('Runtime.evaluate',
      { expression, returnByValue: true, awaitPromise });
    if (r.exceptionDetails) throw new Error('page eval threw: '
      + (r.exceptionDetails.exception?.description || r.exceptionDetails.text));
    return r.result.value;
  }
  json(expression){ return this.eval(`JSON.stringify(${expression})`).then(JSON.parse); }
  async waitFor(expression, ms, what){
    const t0 = Date.now();
    for (;;){
      if (await this.eval(expression)) return (Date.now() - t0)/1000;
      if (Date.now() - t0 > ms) throw new Error(`check-page: timed out after ${ms} ms waiting for ${what}`);
      await new Promise(r => setTimeout(r, 250));
    }
  }
  async shot(path){
    const r = await this.cmd('Page.captureScreenshot', { format: 'png' });
    writeFileSync(path, Buffer.from(r.data, 'base64'));
  }
}

/* ---- the panel's solver settings, pinned -------------------------------
   The node reference below must solve the SAME box the page does, without
   asking the page what that is -- otherwise it would agree with a wrong
   mapping. These four are the panel's discretisation choices, pinned here and
   asserted against the page source, so changing them in the page fails this
   gate loudly instead of silently comparing two different problems. The
   physical numbers (k, depth, drive, acceleration, water) come from the
   renderer's own resolved state and are mapped here independently. */
const GRID = { nr: 28, nz: 14, rStretch: 2.2, zStretch: 2.2 };
const GRID_SRC = 'nr: 28, nz: 14';
const GRID_SRC2 = 'rStretch: 2.2, zStretch: 2.2';
const PAGE_SRC = readFileSync(join(REPO, 'cymatic.html'), 'utf8');

/* the drive period must divide into a whole number of steps at this dt for the
   node and browser runs to be the same map; floquet() rounds, so both round
   identically and this is only asserted to have a sane value. */
const FLOQUET = require(join(REPO, 'dns', 'faraday-floquet.js'));
const DISC = require(join(REPO, 'dns', 'faraday-disc.js'));
const KERNEL = require(join(REPO, 'faraday', 'kernel.js'));

/* The box the panel would solve, mapped here from the renderer's own resolved
   state rather than read back off the page. If the page's own mapping were wrong
   the two would disagree, which is the point. */
function configFrom(s){
  return { nr: GRID.nr, nz: GRID.nz,
           rStretch: GRID.rStretch, zStretch: GRID.zStretch,
           R: s.cellDiameterMm/2000, h: s.depthMm/1000,
           rho: s.rho, nu: s.nu, gamma: s.sigma,
           m: s.modeM, k: s.modeK,
           omegaD: 2*Math.PI*s.freq, accel: s.accelAssumed,
           contact: s.rim === 'pinned' ? 'pinned' : 'free' };
}

function seedFor(c){
  const rf = DISC.gradeToEnd(c.nr, c.R, c.rStretch);
  const eta0 = new Float64Array(c.nr);
  for (let i = 0; i < c.nr; i++)
    eta0[i] = KERNEL.besselJ(c.m, c.k*0.5*(rf[i] + rf[i+1]));
  return eta0;
}

/* ---- run --------------------------------------------------------------- */
const { srv, port } = await serve(REPO);
const browser = await Browser.launch();
const built = join(tmpdir(), 'cymatic-standalone-checkpage.html');
let nodeRef = null;          // {muMax, growth} from the node solver
let servedMu = null;         // |mu| the served page reported

async function checkPage(url, name, expectInline){
  section(`${name}  (${url.replace(/^http:\/\/127\.0\.0\.1:\d+/, '')})`);
  const page = await Page.open(browser, url);
  await page.waitFor('typeof state === "object" && state !== null', 30000,
                     'the renderer to resolve a state');

  /* 1. the page declares its own character set */
  const cs = await page.json(`{
    characterSet: document.characterSet,
    deck: document.body.innerText,
    label: document.querySelector('#dnsRun') ? document.querySelector('#dnsRun')
             .closest('.pnl, section, div').textContent.slice(0, 200) : '' }`);
  ok(cs.characterSet === 'UTF-8',
     'the page declares UTF-8 rather than leaving the browser to guess',
     `document.characterSet = ${cs.characterSet}`);
  ok(cs.deck.includes('µm') && !cs.deck.includes('Âµ'),
     'micron survives decoding', `no clean µm in the rendered text`);
  ok(cs.deck.includes('s⁻¹') && !cs.deck.includes('sâ'),
     'inverse seconds survives decoding');

  /* 2. the top-level script ran to the end */
  const s = await page.json(`{
    wavenumber: state.wavenumber, onsetGrowth: state.onset.growth,
    accelAssumed: state.accelAssumed, responseHz: state.responseHz,
    rho: state.water.rho, nu: state.water.nu, sigma: state.water.sigma,
    freq: freq, depthMm: depth, cellDiameterMm: state.cellDiameterMm,
    rim: rim,
    modeM: state.nearestTheoryModes[0].m, modeK: state.nearestTheoryModes[0].k,
    modeN: state.nearestTheoryModes[0].n,
    grains: (() => { let n = 0; for (let i = 0; i < NP; i++) if (PX[i] !== 0) n++; return n; })(),
    grainTotal: NP,
    frames: typeof lastBuild === 'number' ? lastBuild : null }`);
  for (const [k, v] of Object.entries(s))
    if (k !== 'grains' && k !== 'grainTotal' && k !== 'frames' && k !== 'rim')
      ok(Number.isFinite(v) && v !== 0, `state.${k} is a finite nonzero number`, `${k} = ${v}`);
  ok(s.rim === 'free' || s.rim === 'pinned', 'the contact line control has a value',
     String(s.rim));
  /* PX is a Float32Array of zeros until sprinkle() places the grains, and
     sprinkle() is the first of the three statements that follow the panel at
     the end of the file. If the panel throws, PX is still all zeros. */
  ok(s.grains === s.grainTotal,
     'sprinkle() ran, so the statements after the panel were reached',
     `${s.grains} of ${s.grainTotal} grains placed`);
  ok(s.frames >= 0, 'the animation loop is running', `lastBuild = ${s.frames}`);

  /* 3. both canvases actually drew */
  const px = await page.json(`[...document.querySelectorAll('canvas')].map(cv => {
    const g = cv.getContext('2d');
    const d = g.getImageData(0, 0, cv.width, cv.height).data;
    let nz = 0; const seen = new Set();
    for (let i = 0; i < d.length; i += 4){
      if (d[i] || d[i+1] || d[i+2]) nz++;
      seen.add((d[i] << 16) | (d[i+1] << 8) | d[i+2]);
    }
    return { id: cv.id, w: cv.width, h: cv.height, nz, distinct: seen.size };
  })`);
  ok(px.length >= 2, `the page has both canvases`, `found ${px.length}`);
  for (const c of px){
    ok(c.nz > 0.02*c.w*c.h, `canvas #${c.id} is not blank`,
       `${c.nz} non-black of ${c.w*c.h} pixels`);
    ok(c.distinct > 8, `canvas #${c.id} carries a field, not a flat fill`,
       `${c.distinct} distinct colours`);
  }

  /* 4. the panel's inputs are the renderer's state, mapped */
  ok(PAGE_SRC.includes(GRID_SRC) && PAGE_SRC.includes(GRID_SRC2),
     'the page still uses the discretisation this gate solves against',
     `cymatic.html must contain "${GRID_SRC}" and "${GRID_SRC2}"`);
  const c = configFrom(s);
  const cells = await page.json(`(() => {
    const o = {};
    for (const d of document.querySelectorAll('#dnsInput > div'))
      o[d.querySelector('.k').textContent] = d.querySelector('.v').textContent;
    return o; })()`);
  ok(Object.keys(cells).length === 9, 'the panel shows its nine inputs',
     `${Object.keys(cells).length} cells: ${Object.keys(cells).join(' | ')}`);
  rel(parseFloat(cells['cell radius R']), c.R*1000, 4,
      'the radius shown is half the cell diameter, in mm');
  rel(parseFloat(cells['depth h']), c.h*1000, 3, 'the depth shown is the layer depth in mm');
  rel(parseFloat(cells['drive ω_d']), c.omegaD/(2*Math.PI), 3, 'the drive shown is ω_d/2π');
  rel(parseFloat(cells['acceleration a']), c.accel, 3, 'the acceleration shown is the assumed drive');
  rel(parseFloat(cells['ν']), c.nu, 3, 'the viscosity shown is the water model’s ν');
  rel(parseFloat(cells['γ surface tension']), c.gamma, 4,
      'the surface tension shown is the water model’s σ');
  ok(parseInt(cells['azimuthal mode m'], 10) === c.m,
     'the azimuthal mode shown is the renderer’s own dominant mode',
     `${cells['azimuthal mode m']} against m = ${c.m}`);
  ok(cells['azimuthal mode m'].includes(`${2*c.m}-fold`),
     'and it is labelled with the fold count that mode produces',
     cells['azimuthal mode m']);
  ok(cells['contact line'] === c.contact,
     'the contact line shown is the one the control selects', cells['contact line']);
  ok(cells['grid'].startsWith(`${c.nr} × ${c.nz}`),
     'the grid shown is the grid solved', cells['grid']);

  /* 5. inline versus fetched solver source */
  const tags = await page.json(`[...document.querySelectorAll('script')].map(t => ({
    id: t.id, src: t.getAttribute('src') || '', inline: t.textContent.trim().length }))`);
  const dns = tags.filter(t => ['dnsSolver', 'dnsDisc', 'dnsFloquet'].includes(t.id));
  ok(dns.length === 3, 'all three solver scripts are present and carry their ids',
     dns.map(t => t.id).join(','));
  for (const t of dns){
    if (expectInline)
      ok(t.inline > 1000 && t.src === '',
         `#${t.id} carries its source inline in the single-file build`,
         `src=${t.src} inline=${t.inline}`);
    else
      ok(t.src !== '' && t.inline === 0,
         `#${t.id} is loaded from ${t.src} in the checkout`,
         `src=${t.src} inline=${t.inline}`);
  }
  if (expectInline)
    ok(tags.every(t => t.src === ''), 'the single-file build fetches no script at all',
       tags.filter(t => t.src).map(t => t.src).join(','));

  /* 6. the panel solves, and its answer is the solver's answer */
  await page.eval(`document.getElementById('dnsRun').click(), true`);
  const secs = await page.waitFor(
    `document.getElementById('dnsOut').innerHTML !== '' `
    + `|| /did not run/.test(document.getElementById('dnsNote').textContent)`,
    420000, 'the Floquet solve to finish');
  const res = await page.json(`(() => {
    const o = {};
    for (const d of document.querySelectorAll('#dnsOut > div'))
      o[d.querySelector('.k').textContent] = d.querySelector('.v').textContent;
    return { cells: o, note: document.getElementById('dnsNote').textContent,
             btn: document.getElementById('dnsRun').textContent }; })()`);
  ok(!/did not run/.test(res.note), 'the worker ran rather than refusing', res.note.slice(0, 200));
  ok(res.btn === 'run', 'the button returns to rest afterwards', res.btn);
  const mu = parseFloat(res.cells['|μ| largest Floquet multiplier']);
  const growth = parseFloat(res.cells['Floquet growth (disc)']);
  ok(Number.isFinite(mu) && mu > 0, 'the panel reports a positive multiplier', String(mu));

  if (!nodeRef){
    /* computed here, from the state probed out of the page and the mapping
       written in configFrom -- not by asking the page what to solve. */
    const t0 = Date.now();
    nodeRef = FLOQUET.floquetDisc({ ...c, eta0: seedFor(c) });
    console.log(`       node reference: |mu| = ${nodeRef.muMax.toFixed(10)} +/- `
      + `${nodeRef.muSpread.toExponential(2)}, growth ${nodeRef.growth.toFixed(6)} `
      + `s^-1, krylov ${nodeRef.krylov}, ${((Date.now()-t0)/1000).toFixed(1)} s`);
  }
  /* Eight decimals is what the panel prints, so that is what can be compared;
     the underlying agreement is tighter. Any real disagreement -- a different
     cell, a different mode number, a dropped factor -- moves the first digits.
     This is NOT the cluster width: both sides run the same code on the same
     inputs, so they must agree to rounding, whatever the width of the physical
     answer. */
  rel(mu, nodeRef.muMax, 7, `the panel's |μ| is the solver's |μ| (${mu})`);
  rel(growth, nodeRef.growth, 4, `the panel's growth rate is ln|μ| over the drive period`);
  ok(Math.abs(growth - Math.log(mu)/(2*Math.PI/c.omegaD)) < 5e-4*Math.abs(growth),
     'the growth shown is ln|μ| over one drive period, recomputed from the |μ| shown',
     `${growth} vs ${Math.log(mu)/(2*Math.PI/c.omegaD)}`);
  const mathieu = parseFloat(res.cells['deck (Mathieu) growth']);
  rel(mathieu, s.onsetGrowth, 4, 'the Mathieu figure beside it is the deck’s own onset growth');
  const diff = parseFloat(res.cells['the two disagree by']);
  rel(diff, nodeRef.growth - s.onsetGrowth, 2,
      `the disagreement shown is the difference of the two growth rates (${diff} s^-1)`);
  ok(res.cells['the two disagree by'].includes('in sign')
       === (Math.sign(nodeRef.growth) !== Math.sign(s.onsetGrowth)),
     'and it says so when the two disagree in sign, which at the default state they do',
     res.cells['the two disagree by']);
  ok(res.cells['verdict'].includes('stable'),
     'the verdict names a side', res.cells['verdict']);
  const layers = res.cells['cells inside δ'];
  ok(/floor [\d.]+ · surface [\d.]+ · rim [\d.]+/.test(layers),
     'the panel reports how much of each Stokes layer the grid actually resolved, '
     + 'rather than claiming the damping without it', layers);
  console.log(`       panel: |mu| ${mu}, growth ${growth} s^-1 against the deck's `
    + `Mathieu ${mathieu} s^-1, ${diff} s^-1 apart, in ${secs.toFixed(1)} s`);
  console.log(`       ${layers}`);
  if (servedMu === null) servedMu = mu;
  else ok(mu === servedMu,
    'the single-file build and the checkout report the same multiplier to the digit',
    `${mu} vs ${servedMu}`);

  /* 7. the panel's inputs follow the controls
     This is what the construction-time call could not do: showInput ran once,
     at load, so every cell went stale the moment the drive moved. */
  const before = parseFloat(cells['drive ω_d']);
  await page.eval(`freq = ${Math.round(s.freq) + 40}; dirty = true; true`);
  await page.waitFor(
    `parseFloat([...document.querySelectorAll('#dnsInput > div')]`
    + `.find(d => d.querySelector('.k').textContent === 'drive ω_d')`
    + `.querySelector('.v').textContent) !== ${before}`, 15000,
    'the panel to follow the drive control');
  const after = await page.eval(
    `parseFloat([...document.querySelectorAll('#dnsInput > div')]`
    + `.find(d => d.querySelector('.k').textContent === 'drive ω_d')`
    + `.querySelector('.v').textContent)`);
  rel(after, Math.round(s.freq) + 40, 3,
      'moving the drive moves the box the panel would solve', `${before} then ${after}`);

  /* 8. nothing threw
     The font stylesheet is the one allowed failure: it is a remote Google
     Fonts request, this suite runs with no network by design, and a missing
     webfont changes no number on the page. Every other error counts. */
  const fontOnly = t => /fonts\.(googleapis|gstatic)\.com/.test(t);
  ok(page.errors.length === 0, 'no uncaught exception reached the page',
     page.errors.join(' | ').slice(0, 400));
  ok(page.console.filter(t => /^error/.test(t)).length === 0,
     'nothing logged an error to the console',
     page.console.filter(t => /^error/.test(t)).join(' | ').slice(0, 400));
  ok(page.logErrors.filter(t => !fontOnly(t)).length === 0,
     'no resource failed to load but the webfont',
     page.logErrors.filter(t => !fontOnly(t)).join(' | ').slice(0, 400));

  await page.shot(join(tmpdir(), `check-page-${name.replace(/\W+/g, '-')}.png`));
  return page;
}

try {
  await checkPage(`http://127.0.0.1:${port}/cymatic.html`, 'the checkout page', false);

  const { buildStandalone } = await import(join(REPO, 'faraday', 'build-standalone.mjs'));
  writeFileSync(built, buildStandalone());
  await checkPage(`file://${built}`, 'the single-file build', true);
} finally {
  browser.close();
  srv.close();
  try { rmSync(built, { force: true }); } catch {}
}

console.log('\n' + '-'.repeat(66));
if (failures.length){
  console.log(`${pass} passed, ${failures.length} FAILED\n`);
  for (const f of failures) console.log('  FAIL  ' + f);
  process.exit(1);
}
console.log(`${pass} checks passed, 0 failed`);
