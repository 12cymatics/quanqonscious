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
    /* THE GPU IS ENABLED, and --disable-gpu is gone, because the page now has a GPU
       renderer and a browser with no GPU could not test it: with --disable-gpu,
       WebGL2 is simply absent. Measured in this container -- --disable-gpu gives no
       webgl2 context at all; without it, one is there.

       Chrome has no working --enable-gpu switch: using the GPU is the default, and
       what has to be done is to stop disabling it. --enable-unsafe-swiftshader is
       there for the machines with no GPU at all -- GitHub's runners, and the
       container this was written in. On those Chrome no longer falls back to its
       software GPU, SwiftShader, for WebGL unless told it may, so without this flag
       CI would have no WebGL2 and the GPU section would refuse. On a machine WITH a
       GPU the flag changes nothing and the real one is used. The section reports
       which renderer it got, so a pass on SwiftShader is never mistaken for a pass
       on hardware. */
    const proc = spawn(BROWSER, [
      '--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader',
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
   The node reference below must solve the SAME problem the page does, without
   asking the page what that is -- otherwise it would agree with a wrong
   mapping. The grid is no longer a constant to pin: the panel sizes it to the
   mode, so what is pinned is the SIZING CALL, and the node side runs the same
   search on the same arguments and must land on the same grid. The physical
   numbers (k, depth, drive, acceleration, water) come from the renderer's own
   resolved state and are mapped here independently. */
const GRID_SRC = 'FARADAY_DISC.suggestGrid({';
const GRID_SRC2 = 'omegaResponse: c.omegaD/2, omegaDrive: c.omegaD';
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
  const c = { R: s.cellDiameterMm/2000, h: s.depthMm/1000,
              rho: s.rho, nu: s.nu, gamma: s.sigma,
              m: s.modeM, k: s.modeK,
              omegaD: 2*Math.PI*s.freq, accel: s.accelAssumed,
              contact: s.rim === 'pinned' ? 'pinned' : 'free' };
  /* The same search the panel runs, on the same arguments. If the page sized its
     grid differently the two would solve different problems, and the |mu|
     comparison below would fail rather than quietly comparing them. */
  const g = DISC.suggestGrid({
    m: c.m, k: c.k, R: c.R, h: c.h, rho: c.rho, nu: c.nu, gamma: c.gamma,
    contact: c.contact, omegaResponse: c.omegaD/2, omegaDrive: c.omegaD,
    profile: r => KERNEL.besselJ(c.m, c.k*r) });
  if (g.unreachable) throw new Error(
    'check-page: no grid on the ladder carries the renderer default mode, so the '
    + 'panel will refuse and there is nothing to compare. That is a change in the '
    + 'default state or in the gates, not a browser problem.');
  c.nr = g.nr; c.nz = g.nz; c.rStretch = g.rStretch; c.zStretch = g.zStretch;
  c.grid = g;
  return c;
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

  /* 3. both canvases actually drew.
     NAMED, not every canvas on the page -- and that is load bearing. Taking a 2D
     context is not a read: it CLAIMS the canvas for good, and a canvas holds one kind
     of context for its whole life. This used to walk every canvas, and when the GPU
     canvas #cgl arrived the walk took a 2D context on it before the GPU renderer
     could take WebGL2, so "choosing GPU" then found no WebGL2 context at all and the
     GPU section failed for a reason that was the test's own. #cgl is checked by the
     GPU section instead, through WebGL's own readPixels. */
  const px = await page.json(`['c', 'sc'].map(id => document.getElementById(id)).filter(Boolean).map(cv => {
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
  ok(/sized to the mode when you press run/.test(cells['grid']),
     'before a run the panel says the grid is sized to the mode, rather than '
     + 'naming one it has not measured', cells['grid']);

  /* 5. inline versus fetched solver source */
  const tags = await page.json(`[...document.querySelectorAll('script')].map(t => ({
    id: t.id, src: t.getAttribute('src') || '', inline: t.textContent.trim().length }))`);
  const dns = tags.filter(t =>
    ['dnsSolver', 'dnsDisc', 'dnsDiscWasm', 'dnsFloquet'].includes(t.id));
  ok(dns.length === 4, 'all four solver scripts are present and carry their ids',
     dns.map(t => t.id).join(','));
  /* The compiled period map. Inline base64 in the single-file build; fetched
     beside the page in the checkout, where the tag is empty. */
  const b64 = tags.filter(t => t.id === 'dnsWasmBase64');
  ok(b64.length === 1, 'the page has the tag the compiled period map lands in',
     `${b64.length} tags with id dnsWasmBase64`);
  if (expectInline)
    ok(b64[0].inline > 20000,
       'the single-file build carries the compiled period map inline',
       `${b64[0].inline} characters`);
  else
    ok(b64[0].inline === 0,
       'the checkout leaves it empty and fetches dns/faraday_disc.wasm',
       `${b64[0].inline} characters`);
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
    /* engine 'wasm' here too: the two engines are bit-for-bit identical and
       dns/check-disc.mjs is what asserts that. This gate's question is whether
       the PAGE solves what node solves, so both sides run the same engine and
       the comparison is not paying for the slower one twice. */
    nodeRef = FLOQUET.floquetDisc({ ...c, eta0: seedFor(c), krylov: 16,
                                    engine: 'wasm', grid: undefined });
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
  /* The grid the panel actually solved on, now that it has one, against the grid
     the same search picks here. A page that sized differently would have solved a
     different problem, and the |mu| comparison above would have caught it -- this
     says WHICH way it went wrong when it does. */
  const shownGrid = await page.json(`(() => {
    for (const d of document.querySelectorAll('#dnsInput > div'))
      if (/^grid/.test(d.querySelector('.k').textContent))
        return d.querySelector('.v').textContent;
    return null; })()`);
  ok(shownGrid !== null && shownGrid.startsWith(`${c.nr} × ${c.nz}`),
     `after the run the grid shown is the grid the sizing picks (${c.nr} × ${c.nz})`,
     String(shownGrid));
  ok(/graded r /.test(String(shownGrid)) && String(shownGrid).includes(`z ${c.zStretch}`),
     'and it names the grading it was sized with', String(shownGrid));
  ok(/C\+\+/.test(res.cells['period map ran in'] || ''),
     'the panel says the period map ran in C++, because that is what it asked for',
     String(res.cells['period map ran in']));
  ok(/drift /.test(res.cells['Krylov dimension used'] || ''),
     'and reports how far the leading modulus was still moving',
     String(res.cells['Krylov dimension used']));
  const opErr = await page.json(`(() => {
    for (const d of document.querySelectorAll('#dnsInput > div'))
      if (/surface operator error/.test(d.querySelector('.k').textContent))
        return parseFloat(d.querySelector('.v').textContent);
    return null; })()`);
  ok(Number.isFinite(opErr) && opErr <= 1.0,
     'the surface operator error it reports is within the one per cent it gates on',
     `${opErr} %`);

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

/* ---- the solver actually draws, which loading the page does not prove -------
   Everything above establishes that the page comes up and logs nothing. This turns
   the surface source over to dns/faraday-cell3d.js, waits for the worker's first
   frame, and then checks the thing that matters: that the raster the page renders IS
   the solver's surface and not a leftover modal pattern that happens to be on screen.

   The comparison is made in the page, against the page's own resampling vessel, at
   pixel positions chosen across the disc. If buildSurface had run instead of the
   solver path -- or if the solver path had silently fallen back -- ETA would hold a
   normalised modal sum and the residual would be of order one, not of order the
   normalisation's own rounding. */
async function checkSolverDraws(page){
  section('the surface drawn from the three-dimensional solver');

  const before = await page.json('({ src: document.querySelector("#srcSeg [data-s=\'modal\']")'
    + '.getAttribute("aria-pressed"), useSolver })');
  ok(before.src === 'true' && before.useSolver === false,
     'the page starts on the modal superposition, so the solver is opt-in and nothing '
     + 'about the existing picture changed by adding it',
     JSON.stringify(before));

  await page.eval('document.querySelector("#srcSeg [data-s=\'dns\']").click()');
  const waited = await page.waitFor('CELL3D.clocks().have === true', 60000,
                                    "the worker's first frame");
  const c = await page.json('CELL3D.clocks()');
  console.log(`       first frame after ${waited.toFixed(1)} s: ${c.grid.nr} x ${c.grid.nth} `
    + `x ${c.grid.nz}, ${c.stepsDone} steps, ${c.perStep.toFixed(0)} ms a step, `
    + `physics ${(c.physT*1e3).toFixed(2)} ms in ${c.wall.toFixed(1)} s wall`);
  ok(!c.err, 'the worker starts, steps and posts a frame without erroring',
     String(c.err));
  ok(c.stepsDone > 0 && c.physT > 0,
     'and it advanced the physics clock, so the solver is integrating and not idling',
     `${c.stepsDone} steps, t = ${c.physT}`);

  /* force one build with the solver on, then compare the raster against the vessel */
  await page.eval('recompute()');
  const cmp = await page.json(`(() => {
    const V = CELL3D.vesselFor ? CELL3D.vesselFor() : null;
    let worst = 0, scale = 0, n = 0;
    for (let y = 4; y < GR - 4; y += 17)
      for (let x = 4; x < GR - 4; x += 17){
        const i = y*GR + x; if (COVER[i] <= 0) continue;
        const rho = Math.min(1, RAD[i]);
        const th = ANG[i] < 0 ? ANG[i] + 2*Math.PI : ANG[i];
        const want = CELL3D.etaAtPixel(rho, th);
        worst = Math.max(worst, Math.abs(ETA[i] - want));
        scale = Math.max(scale, Math.abs(want));
        n++;
      }
    return { worst, scale, n, rel: scale > 0 ? worst/scale : Infinity };
  })()`);
  console.log(`       page raster against the solver's own surface at ${cmp.n} pixels: `
    + `worst ${cmp.worst.toExponential(3)} of ${cmp.scale.toExponential(3)}, `
    + `${cmp.rel.toExponential(2)} relative`);
  ok(cmp.n > 50, 'the comparison covers the disc rather than a handful of pixels',
     `${cmp.n} pixels`);
  /* The bound is 1e-6 and the residual is 5.52e-8, which is NOT interpolation error: ETA is
     a Float32Array, and Float32 epsilon is 1.19e-7. What is being measured is the raster's own
     storage precision, so a tighter bound would be asserting something about f32 that f32
     cannot deliver. Anything that drew a different surface would be of order one here. */
  ok(cmp.rel < 1e-6,
     "the page's field IS the solver's surface, normalised -- not a modal pattern left "
     + 'on screen while the deck claims otherwise',
     `${cmp.rel.toExponential(3)} relative over ${cmp.n} pixels`);

  const deck = await page.eval('document.getElementById("ro").textContent');
  ok(/3-D Navier-Stokes solver/.test(deck),
     'and the deck says so, so what is drawn and what is claimed cannot disagree',
     deck.slice(0, 160));
  ok(/physics clock/.test(deck) && /wall clock/.test(deck) && /slower than real time/.test(deck),
     'both clocks and their ratio are on the deck, which is how the page states that a '
     + 'direct simulation of this cell does not run at real time',
     deck.slice(0, 200));

  ok(/JavaScript/.test(deck),
     'the deck names the engine that answered, not the one that was asked for',
     deck.slice(0, 200));

  await page.shot(join(tmpdir(), 'check-page-solver.png'));
}

/* The surface drawn on the GPU, held to the CPU's bytes.
   .
   THE BOUND IS ONE LEVEL, AND IT IS DERIVED, NOT TUNED. Both renderers read the same
   Float32Array rasters, so their inputs are identical. The CPU evaluates the shading
   in double precision and the GPU in single, and the two results differ by parts in
   ten million. Each is then quantised to an 8-bit level. A difference that small
   changes the stored byte only when the value sits within it of a rounding boundary,
   and then by exactly one level -- it cannot reach two. So the assertion is that no
   channel anywhere differs by more than one, and the count of channels that differ
   by one is REPORTED, not bounded: it is a property of where boundaries happen to
   fall, and bounding it would be a tolerance with nothing behind it.
   .
   It is checked at the page's own amplitude and then at full amplitude in both
   phases, because at a low amplitude most pixels sit at the base colour and agree
   trivially; with the texture noise off and on, because the noise is the one input
   the GPU does not read from the same array -- it rebuilds the CPU's per-frame
   random sequence once, as a table; and again after the field is rebuilt, because a
   renderer that uploaded the field once and never again would agree on the first
   frame and draw a stale surface on every one after. */
/* THE VALUES BEFORE ROUNDING, and the bound they are held to.
   .
   The byte bound above cannot see an error smaller than one level. Measured: with one
   shading coefficient changed from 168 to 169 -- a wrong transcription, every optical
   pixel off by up to half a level -- that check stayed GREEN, the one-level count
   rising from about 180 to 106 887 while nothing exceeded one. So the shader also
   writes its values before the 8-bit rounding to a float target, the CPU loop hands
   back the same values in double precision, and the two are compared directly.
   .
   RAW_BOUND IS DERIVED FROM THE GLSL ES 3.00 PRECISION REQUIREMENTS, not from what
   this GPU happens to do. The largest legitimate error is the optical view's specular
   term, pow(Nz, 30) = exp2(30 log2 Nz): log2 is allowed 2^-21 absolute error near 1,
   which times 30 is 1.4e-5 in the exponent, and exp2 of an argument down to -30 is
   allowed (3 + 2*30) ulp; together about 1.4e-5 relative, times the coefficient 96 is
   1.3e-3 before the tone curve. The ring, exp(-s^2) with s^2 up to 39.5, is allowed 82
   ulp, 4.9e-6 relative, times 170 is 8.3e-4. The tone curve's slope is at most 255/132,
   so the two reach 4.1e-3 of a level, and the tone curve's own exp adds 1.4e-4. About
   5e-3 in all, so the bound is 2e-2: four times the worst case the precision rules
   permit any conformant GPU, and twenty-five times below the half level the 168 -> 169
   transcription error moves a pixel. */
const RAW_BOUND = 2e-2;

/* This reducer runs IN THE BROWSER, before JSON can turn NaN/Infinity into null.
   Check both operands and their difference: finite operands can still overflow
   subtraction. Count invalid channels explicitly instead of putting a NaN in a
   maximum, where the next finite value could replace it. Alpha is validated too,
   although only RGB contributes to the existing before-rounding bound. */
function summarizeRawParity(rawCpu, rawGpu){
  if (!rawCpu || !rawGpu || !Number.isSafeInteger(rawCpu.length)
      || rawCpu.length <= 0 || rawCpu.length % 4 || rawCpu.length !== rawGpu.length)
    throw new Error('raw GPU parity requires equal, nonempty RGBA arrays');
  let maxRaw = 0, invalidCount = 0;
  for (let i = 0; i < rawCpu.length; i++){
    const cpu = rawCpu[i], gpu = rawGpu[i], difference = Math.abs(cpu - gpu);
    if (!Number.isFinite(cpu) || !Number.isFinite(gpu) || !Number.isFinite(difference)){
      invalidCount++;
      continue;
    }
    if ((i & 3) !== 3 && difference > maxRaw) maxRaw = difference;
  }
  return { status: invalidCount === 0 ? 'finite' : 'invalid', invalidCount,
    checkedCount: rawCpu.length, maxRaw: invalidCount === 0 ? maxRaw : null };
}

/* Do not coerce a transported null to zero, or trust the status alone. A broken
   producer or transport must fail the same gate as a non-finite raw channel. */
function finiteRawParitySummary(row, expectedCount){
  return !!row && row.status === 'finite' && row.invalidCount === 0
    && Number.isSafeInteger(expectedCount) && expectedCount > 0 && expectedCount % 4 === 0
    && row.checkedCount === expectedCount && Number.isFinite(row.maxRaw) && row.maxRaw >= 0;
}

async function checkRawParityGuards(page, label){
  const cases = await page.json(`(() => {
    const summarize = ${summarizeRawParity.toString()};
    const cpu = new Float64Array(GR*GR*4);
    renderSurfaceCpu(state, cpu);
    const gpu = GPU_SURFACE.readRaw(state);
    const cases = [{ name: 'unchanged rendered field', summary: summarize(cpu, gpu), valid: true }];
    const positions = [['beginning', 0], ['middle', Math.floor(cpu.length / 2)], ['end', cpu.length - 1]];
    for (const side of ['CPU', 'GPU'])
      for (const [valueName, value] of [['NaN', NaN], ['+Infinity', Infinity], ['-Infinity', -Infinity]])
        for (const [position, i] of positions){
          const a = new Float64Array(cpu), b = new Float64Array(gpu);
          (side === 'CPU' ? a : b)[i] = value;
          cases.push({ name: side + ' ' + valueName + ' at ' + position,
            summary: summarize(a, b), valid: false });
        }
    const a = new Float64Array(cpu), b = new Float64Array(gpu);
    a[0] = Number.MAX_VALUE; b[0] = -Number.MAX_VALUE;
    cases.push({ name: 'finite operands whose difference overflows',
      summary: summarize(a, b), valid: false });
    return { count: cpu.length, cases };
  })()`);
  for (const { name, summary, valid } of cases.cases){
    ok(valid ? finiteRawParitySummary(summary, cases.count)
             : summary.status === 'invalid' && summary.invalidCount === 1
               && summary.checkedCount === cases.count && summary.maxRaw === null
               && !finiteRawParitySummary(summary, cases.count),
       `raw GPU parity ${valid ? 'accepts' : 'rejects'} ${name}${label}`, JSON.stringify(summary));
  }
  const good = { status: 'finite', invalidCount: 0, checkedCount: 4, maxRaw: 0 };
  for (const [name, change] of [
    ['null maximum after JSON', { maxRaw: null }],
    ['NaN maximum', { maxRaw: NaN }], ['infinite maximum', { maxRaw: Infinity }],
    ['negative infinite maximum', { maxRaw: -Infinity }], ['string maximum', { maxRaw: '0' }],
    ['missing maximum', { maxRaw: undefined }], ['negative maximum', { maxRaw: -1 }],
    ['invalid status', { status: 'invalid' }], ['missing status', { status: undefined }],
    ['nonzero invalid count', { invalidCount: 1 }], ['null invalid count', { invalidCount: null }],
    ['missing invalid count', { invalidCount: undefined }],
    ['wrong channel count', { checkedCount: 8 }], ['null channel count', { checkedCount: null }]
  ]) ok(!finiteRawParitySummary({ ...good, ...change }, 4),
        `raw GPU parity rejects transported ${name}${label}`);
  ok(!finiteRawParitySummary(null, 4), `raw GPU parity rejects a null summary${label}`);
}

async function gpuParity(page, label){
  const r = await page.json(`(() => {
    const summarizeRawParity = ${summarizeRawParity.toString()};
    const vs0 = state.visualSignature, e0 = state.expression, ps0 = phaseSign, v0 = view;
    const rows = [];
    const rawCpu = new Float64Array(GR*GR*4);
    try {
      for (const amp of ['as is', 'full, phase I', 'full, phase II'])
        for (const texOn of [false, true])
          for (const v of ['sand', 'optical', 'height', 'nodal']){
            if (amp !== 'as is'){ state.expression = 1; phaseSign = amp.endsWith('II') ? -1 : 1; }
            state.visualSignature = vs0 ? Object.assign({}, vs0, { textureConc: texOn ? 0.5 : 0 }) : null;
            view = v;
            renderSurfaceCpu(state, rawCpu);
            const cpu = new Uint8Array(pxl);
            GPU_SURFACE.draw(state);
            const gpu = GPU_SURFACE.readRGBA();
            const rawGpu = GPU_SURFACE.readRaw(state);
            let maxd = 0, n1 = 0, lit = 0;
            for (let i = 0; i < cpu.length; i++){
              const d = Math.abs(cpu[i] - gpu[i]);
              if (d > maxd) maxd = d;
              if (d) n1++;
              if ((i & 3) === 0 && cpu[i] > 40) lit++;
            }
            rows.push({ amp, tex: texOn, view: v, maxd, n1, n: cpu.length, lit,
              ...summarizeRawParity(rawCpu, rawGpu) });
            state.expression = e0; phaseSign = ps0;
          }
    } finally { state.visualSignature = vs0; state.expression = e0; phaseSign = ps0; view = v0; }
    return rows;
  })()`);
  const worst = Math.max(...r.map(x => x.maxd));
  const ones = r.reduce((a, x) => a + x.n1, 0), all = r.reduce((a, x) => a + x.n, 0);
  const invalidRaw = r.filter(x => !finiteRawParitySummary(x, x.n));
  const rawWorst = invalidRaw.length ? null : Math.max(...r.map(x => x.maxRaw));
  console.log(`       ${r.length} renderings ${label}: worst channel difference ${worst}, `
    + `${ones} of ${all} channels one level apart; before rounding, worst `
    + `${rawWorst === null ? 'INVALID' : rawWorst.toExponential(2)} of a level against a bound of ${RAW_BOUND}`);
  for (const x of r.filter(x => x.maxd > 1))
    console.log(`       ${x.amp} / tex ${x.tex} / ${x.view}: max ${x.maxd}`);
  ok(r.length === 24, `twenty-four renderings compared ${label}`, `${r.length}`);
  ok(worst <= 1,
     `the GPU's bytes are the CPU's to within one level in every channel of every view ${label}`,
     `worst ${worst}: ${r.filter(x => x.maxd > 1).map(x => `${x.amp}/${x.tex}/${x.view}=${x.maxd}`).join(', ')}`);
  ok(invalidRaw.length === 0,
     `every CPU/GPU raw value and difference is finite before JSON transport ${label}`,
     JSON.stringify(invalidRaw));
  ok(invalidRaw.length === 0 && Number.isFinite(rawWorst) && rawWorst <= RAW_BOUND,
     `before rounding, the GPU's values are the CPU's to within ${RAW_BOUND} of a level ${label}`
     + ' -- the bound the GLSL precision rules permit, which catches what rounding hides',
     `worst ${rawWorst}: ${r.filter(x => !finiteRawParitySummary(x, x.n) || x.maxRaw > RAW_BOUND).map(x =>
        `${x.amp}/${x.tex}/${x.view}=${String(x.maxRaw)} (${x.status}, invalid ${x.invalidCount})`).join(', ')}`);
  /* And the comparison is not of two blank canvases. */
  const lit = Math.min(...r.filter(x => x.amp !== 'as is').map(x => x.lit));
  ok(lit > 1000, `at full amplitude every view lights more than a thousand pixels ${label}`,
     `fewest ${lit}`);
  return r;
}

async function checkGpuBoundary(page, label){
  const results = await page.json(`(() => {
    const surface = new FARADAY_RENDER_GL.SurfaceGL(document.createElement('canvas'), 2);
    const field = new Float32Array([.1,.2,.3,.4]);
    surface.uploadStatic(new Float32Array(4).fill(1), new Float32Array(4));
    surface.uploadField(field, field, field); surface.uploadBins(new Uint16Array(4));
    const options = {view:'optical',k:1,contrast:1,tex:0,slope:.8}, cases = [];
    const rejects = (name, action) => {
      let reason = ''; try { action(); } catch (e){ reason = String(e.message); }
      cases.push({name,reason,glError:surface.gl.getError()});
    };
    for (const raster of ['uCover','uNoise','uEta','uNx','uNy'])
      for (const [position,i] of [['beginning',0],['middle',2],['end',3]])
        for (const [kind,value] of [['NaN',NaN],['+Infinity',Infinity],['-Infinity',-Infinity]]){
          const data = new Float32Array(field); data[i] = value;
          rejects(raster+' '+kind+' at '+position, () => surface.upload(raster,data));
        }
    for (const uniform of ['k','contrast','tex','slope'])
      for (const [kind,value] of [['NaN',NaN],['+Infinity',Infinity],['-Infinity',-Infinity],['float32 overflow',Number.MAX_VALUE]])
        rejects(uniform+' '+kind, () => surface.draw({...options,[uniform]:value}));
    for (const mode of ['nodal','optical']) for (const value of [0,-1,Number.MIN_VALUE])
      rejects(mode+' nonpositive float32 contrast '+value, () => surface.draw({...options,view:mode,contrast:value}));
    surface.draw(options);
    const raw = surface.readRaw(options);
    const normal = raw.every(Number.isFinite) && surface.readRGBA().some((v,i) => (i&3)!==3 && v>0);
    surface.releaseResources();
    const lose = surface.gl.getExtension('WEBGL_lose_context'); if (lose) lose.loseContext();
    return {cases,normal};
  })()`);
  for (const r of results.cases)
    ok(/finite|float32|positive/i.test(r.reason) && r.glError === 0,
      `renderer boundary refuses ${r.name} before GL submission${label}`, JSON.stringify(r));
  ok(results.normal, `valid rendering remains finite and nonblank after refused inputs${label}`);
}

async function checkGpuContextRecovery(page, label){
  section('explicit drawing recovery after WebGL context loss' + label);
  const shot = async name => {
    if (process.env.PAGE_GATE_SCREENSHOT_DIR) await page.shot(join(process.env.PAGE_GATE_SCREENSHOT_DIR,
      `gpu-${name}-${label.replace(/[^a-z0-9]+/gi, '-') || 'served'}.png`));
  };
  await page.eval(`document.querySelector('#engSeg [data-e="gpu"]').click();
    globalThis.__recoveryOriginal = renderSurface;
    globalThis.__recoveryDraws = {cpu:0,gpu:0};
    renderSurface = function(st){ __recoveryOriginal(st); __recoveryDraws[renderEngine]++; };
    globalThis.__recoveryGL = document.getElementById('cgl').getContext('webgl2');
    globalThis.__recoveryLose = __recoveryGL.getExtension('WEBGL_lose_context');
    globalThis.__recoveryRestored = false;
    document.getElementById('cgl').addEventListener('webglcontextrestored',
      () => { __recoveryRestored = true; }, {once:true}); true`);
  try {
    const supported = await page.eval('!!__recoveryLose');
    ok(supported, `WEBGL_lose_context is available for an actual loss/restoration test${label}`);
    if (!supported) return;
    await page.waitFor('__recoveryDraws.gpu > 0', 15000, 'the selected GPU to draw');
    await page.eval('__recoveryLose.loseContext(); true');
    await page.waitFor('!!renderError && !GPU_SURFACE.ready', 15000, 'visible context loss');
    const lost = await page.json(`({engine:renderEngine,error:renderError,
      message:document.getElementById('renderStatus').textContent,
      hidden:document.getElementById('cgl').style.visibility, cpu:__recoveryDraws.cpu})`);
    ok(lost.engine === 'gpu' && /context/i.test(lost.error)
       && /GPU drawing paused/.test(lost.message) && lost.hidden === 'hidden' && lost.cpu === 0,
       `context loss is visible and withholds the frame without CPU substitution${label}`, JSON.stringify(lost));
    await shot('context-lost');
    await page.eval('document.querySelector(\'#engSeg [data-e="gpu"]\').click(); true');
    ok(await page.eval('renderEngine === "gpu" && !!renderError && !GPU_SURFACE.ready && __recoveryDraws.cpu === 0'),
       `GPU retry refuses the still-lost object instead of accepting it${label}`);
    await page.eval('document.querySelector(\'#engSeg [data-e="cpu"]\').click(); true');
    await page.waitFor('__recoveryDraws.cpu > 0 && !renderError', 15000, 'explicit CPU recovery');
    ok(await page.eval(`renderEngine === 'cpu' && document.getElementById('c').style.display === 'block'
      && document.getElementById('c').style.visibility === 'visible'
      && /CPU drawing/.test(document.getElementById('renderStatus').textContent)`),
       `explicit CPU selection resumes the live animation after loss${label}`);
    await shot('cpu-recovered');
    await page.eval('__recoveryLose.restoreContext(); true');
    await page.waitFor('__recoveryRestored && !__recoveryGL.isContextLost()', 15000, 'the browser context to restore');
    ok(await page.eval('renderEngine === "cpu" && !GPU_SURFACE.ready && !renderError'),
       `restoration keeps CPU selected and waits for explicit GPU reinitialization${label}`);
    const previous = await page.eval('__recoveryDraws.gpu');
    await page.eval('document.querySelector(\'#engSeg [data-e="gpu"]\').click(); true');
    await page.waitFor(`GPU_SURFACE.ready && !renderError && __recoveryDraws.gpu > ${previous}`,
      15000, 'explicit GPU reinitialization to draw the current rasters');
    ok(await page.eval('renderEngine === "gpu" && document.getElementById("cgl").style.visibility === "visible"'),
       `explicit GPU selection rebuilds its resources and resumes drawing${label}`);
    await shot('gpu-restored');
    await gpuParity(page, 'after actual context restoration' + label);
  } finally {
    await page.eval(`renderSurface = __recoveryOriginal;
      document.querySelector('#engSeg [data-e="cpu"]').click(); true`);
  }
}

async function checkGpuRender(page, label){
  section('the surface drawn on the GPU' + label);
  const probe = await page.json('FARADAY_RENDER_GL.renderGlProbe()');
  ok(probe.ok, 'the test browser offers WebGL2, which it did not while it was launched '
     + 'with --disable-gpu', JSON.stringify(probe));
  if (!probe.ok) return;
  /* Said, not asserted: on a machine with no GPU this is SwiftShader, which runs the
     same WebGL2 API and the same ANGLE shader compiler in software. A pass there says
     the shader computes what the CPU computes; it says nothing about GPU speed. */
  console.log(`       renderer: ${probe.renderer}`
    + (/swiftshader/i.test(probe.renderer) ? '  (software -- correctness only, not speed)' : ''));

  await page.eval('document.querySelector("#engSeg [data-e=\'gpu\']").click()');
  const st = await page.json(`({ engine: renderEngine, ready: GPU_SURFACE.ready,
    error: GPU_SURFACE.error, c: document.getElementById('c').style.display,
    cgl: document.getElementById('cgl').style.display })`);
  ok(st.engine === 'gpu' && st.ready && !st.error,
     'choosing GPU starts the WebGL2 renderer', JSON.stringify(st));
  ok(st.cgl === 'block' && st.c === 'none',
     'and the GPU canvas is the one shown, the CPU one hidden', JSON.stringify(st));
  /* Nothing below can run on a renderer that did not start, and the failure above
     already says why; carrying on would only crash on its absence. */
  if (!(st.engine === 'gpu' && st.ready)) return;

  await checkRawParityGuards(page, label);
  await checkGpuBoundary(page, label);
  await gpuParity(page, 'on the modal field');

  /* The field rebuilt: a different drive frequency is a different mode and a
     different surface, so every texture the GPU holds is now stale unless it was
     re-uploaded. */
  const gens = await page.json(`(() => { const g0 = FIELD_GEN, f0 = freq;
    setFreq(f0 + 37); recompute(); const g1 = FIELD_GEN; return { g0, g1, f0, f1: freq }; })()`);
  ok(gens.g1 > gens.g0, 'rebuilding the field moves its generation, which is what the GPU '
     + 'renderer re-uploads on', JSON.stringify(gens));
  await gpuParity(page, 'after the field is rebuilt');
  await page.eval(`setFreq(${gens.f0}); recompute()`);

  await page.eval('recompute()');
  const deck = await page.eval('document.getElementById("ro").textContent');
  ok(/drawn on\s*GPU/.test(deck) && /WebGL2/.test(deck),
     'the deck says the GPU drew it, and through which API', deck.slice(0, 200));
  ok(/main-thread cost/.test(deck), 'and states what drawing cost the main thread',
     deck.slice(0, 200));
  const cost = await page.json('({ cpu: renderCost.cpu, gpu: renderCost.gpu })');
  console.log(`       main-thread cost per frame: CPU ${cost.cpu.toFixed(2)} ms, `
    + `GPU ${cost.gpu.toFixed(2)} ms (submission only; ${/swiftshader/i.test(probe.renderer)
       ? 'software GPU, so not a speed measurement' : 'hardware GPU'})`);

  await page.eval('document.querySelector("#engSeg [data-e=\'cpu\']").click()');
  const back = await page.json(`({ engine: renderEngine,
    c: document.getElementById('c').style.display,
    cgl: document.getElementById('cgl').style.display })`);
  ok(back.engine === 'cpu' && back.c === 'block' && back.cgl === 'none',
     'choosing CPU again draws with the CPU on the CPU canvas', JSON.stringify(back));
  await page.eval('recompute()');
  const deck2 = await page.eval('document.getElementById("ro").textContent');
  ok(/drawn on\s*CPU/.test(deck2), 'and the deck says so', deck2.slice(0, 200));
  await checkGpuContextRecovery(page, label);
}

/* The C++ engine, on the page, drawing the same surface.
   .
   dns/check-cell3d-wasm.mjs is what holds the two engines to identical fields;
   this is a different question and the page is the only place to ask it: does the
   button reach the module, does the module reach the worker, and does what the
   deck claims match what answered. The bytes travel differently in the two cases
   -- fetched from dns/faraday_cell3d.wasm on a served checkout, carried as base64
   in the single file -- so this runs on both. */
async function checkCppEngine(page, inlined){
  section('the C++ engine on the page' + (inlined ? ', from the inlined bytes' : ''));

  const has = await page.json('({ tag: !!document.getElementById("dnsCell3dWasmBase64"),'
    + ' inline: !!globalThis.FARADAY_CELL3D_WASM_BASE64,'
    + ' loader: !!globalThis.FARADAY_CELL3D_WASM })');
  ok(has.loader, 'the loader is on the page', JSON.stringify(has));
  ok(has.tag, 'and so is the tag the single-file build inlines the bytes into',
     JSON.stringify(has));
  ok(has.inline === inlined,
     inlined ? 'the single file carries the bytes inline'
             : 'the served checkout does not, and fetches them instead',
     JSON.stringify(has));

  await page.eval('document.querySelector("#srcSeg [data-s=\'dnscpp\']").click()');
  const waited = await page.waitFor(
    'CELL3D.clocks().have === true && CELL3D.clocks().engine === "cpp"', 90000,
    "the C++ worker's first frame");
  const c = await page.json('CELL3D.clocks()');
  console.log(`       first C++ frame after ${waited.toFixed(1)} s: ${c.grid.nr} x `
    + `${c.grid.nth} x ${c.grid.nz}, ${c.stepsDone} steps, `
    + `${c.perStep.toFixed(0)} ms a step`);
  ok(!c.err, 'the C++ worker starts, steps and posts a frame', String(c.err));
  ok(c.engine === 'cpp' && c.asked === 'cpp',
     'and the engine that ANSWERED is the C++ one, read off its own frames rather '
     + 'than copied from what was requested -- a silent fall back to the '
     + 'JavaScript would be invisible otherwise',
     JSON.stringify({ engine: c.engine, asked: c.asked }));
  ok(c.stepsDone > 0 && c.physT > 0, 'and it advanced the physics clock',
     `${c.stepsDone} steps, t = ${c.physT}`);

  await page.eval('recompute()');
  const cmp = await page.json(`(() => {
    let worst = 0, scale = 0, n = 0;
    for (let y = 4; y < GR - 4; y += 17)
      for (let x = 4; x < GR - 4; x += 17){
        const i = y*GR + x; if (COVER[i] <= 0) continue;
        const rho = Math.min(1, RAD[i]);
        const th = ANG[i] < 0 ? ANG[i] + 2*Math.PI : ANG[i];
        const want = CELL3D.etaAtPixel(rho, th);
        worst = Math.max(worst, Math.abs(ETA[i] - want));
        scale = Math.max(scale, Math.abs(want));
        n++;
      }
    return { worst, scale, n, rel: scale > 0 ? worst/scale : Infinity };
  })()`);
  console.log(`       page raster against the C++ solver's surface at ${cmp.n} pixels: `
    + `${cmp.rel.toExponential(2)} relative`);
  ok(cmp.n > 50, 'the comparison covers the disc', `${cmp.n} pixels`);
  ok(cmp.rel < 1e-6,
     "the page's field IS the C++ solver's surface, to the raster's own f32 precision",
     `${cmp.rel.toExponential(3)} relative over ${cmp.n} pixels`);

  const deck = await page.eval('document.getElementById("ro").textContent');
  ok(/faraday_cell3d\.wasm/.test(deck),
     'and the deck names the module that ran', deck.slice(0, 220));

  /* Back to the JavaScript engine, and the deck must follow. Switching re-inits
     the same worker, and a slice already in flight from the C++ run would
     otherwise be counted into the new one -- which is what the generation stamp on
     every message exists to prevent. */
  await page.eval('document.querySelector("#srcSeg [data-s=\'dns\']").click()');
  await page.waitFor('CELL3D.clocks().have === true && CELL3D.clocks().engine === "js"',
                     90000, 'the JavaScript worker after switching back');
  const back = await page.json('CELL3D.clocks()');
  ok(back.engine === 'js' && !back.err,
     'switching back reaches the JavaScript engine, with the C++ cell released -- '
     + 'the module\'s arena is static, so a second live cell would have been refused',
     JSON.stringify({ engine: back.engine, err: back.err }));

  await page.shot(join(tmpdir(), `check-page-cpp${inlined ? '-inline' : ''}.png`));
}

try {
  const first = await checkPage(`http://127.0.0.1:${port}/cymatic.html`, 'the checkout page', false);
  await checkGpuRender(first, '');
  await checkSolverDraws(first);
  /* and on the solver's field, whose rasters arrive from a different path */
  await first.eval('document.querySelector("#engSeg [data-e=\'gpu\']").click()');
  if (await first.json('GPU_SURFACE.ready')) await gpuParity(first, 'on the Navier-Stokes field');
  else ok(false, 'the GPU renderer is running for the Navier-Stokes parity check',
          await first.json('String(GPU_SURFACE.error)'));
  await first.eval('document.querySelector("#engSeg [data-e=\'cpu\']").click()');
  await checkCppEngine(first, false);

  const { buildStandalone } = await import(join(REPO, 'faraday', 'build-standalone.mjs'));
  writeFileSync(built, buildStandalone());
  const single = await checkPage(`file://${built}`, 'the single-file build', true);
  await checkGpuRender(single, ', in the single-file build');
  await checkCppEngine(single, true);
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
