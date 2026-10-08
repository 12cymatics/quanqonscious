'use strict';

/* The nonlinear three-dimensional cell solver, loaded as WebAssembly, behind the
   same arrays and the same calls as the JavaScript one.

   WHAT THIS IS NOT. It is not an approximation of dns/faraday-cell3d.js and not a
   faster-but-looser mode. dns/faraday_cell3d.cpp is a transcription -- same
   discretisation, same flux forms, same conjugate gradient with the same
   tolerance and cap, same order of operations down to the grouping of each sum,
   double precision throughout, no fused multiply-add, no fast-math. It computes
   no transcendental: the cos in the drive and the tanh in the stability limit are
   evaluated in JavaScript and passed in. The two therefore agree BIT FOR BIT,
   which dns/check-cell3d-wasm.mjs asserts rather than assumes.

   THERE IS NO SILENT FALLBACK. The caller names the engine. If the module cannot
   be loaded this refuses and says why, rather than running the JavaScript under
   the C++ engine's name and reporting a time that means something else.

   The grid is not rebuilt here. A JavaScript FaradayCell3D is constructed for the
   same cell and ITS node arrays are copied in, so the graded-grid code has
   exactly one implementation and the two cannot diverge on it. That is the same
   arrangement dns/faraday-disc-wasm.js uses. */

/* The array table. The keys are the cases in cell3d_ptr/cell3d_len, and the gate
   checks this list against the module rather than trusting it: a renumbering on
   one side would otherwise point JavaScript at the wrong array, and a wrong
   array that still holds plausible numbers is the failure mode this repository
   cares about most. `len` is what the module reports, and a view built one
   element short would read a neighbouring array's first value as its own last --
   which no comparison of the overlapping part can see. */
const CELL3D_ARRAYS = [
  'rf', 'sf', 'rc', 'sc', 'drc', 'dsc', 'drf', 'dsf', 'rx',
  'u', 'v', 'w', 'om', 'p', 'eta',
  'H', 'Hr', 'Hth', 'Hdr', 'Hdth', 'Ht', 'Hx', 'Hxr', 'Hxt',
  'Ex', 'Er', 'Eth', 'Edr', 'Edth', 'Exr', 'Ext',
  'Tx', 'Txr', 'Txt', 'Tr', 'Tth', 'Tdr', 'Tdth',
  /* The working arrays. They carry the JavaScript's names without its leading
     underscore, and `JS_NAME` below maps each one back, so the gate can compare
     them rather than only the state. A projection that agreed on the answer
     while disagreeing on an intermediate would mean one of the two had a
     cancellation the other did not, and that is worth knowing before it matters. */
  'gu', 'gv', 'gw', 'gom', 'div',
  'cgr', 'cgd', 'cgq', 'cgz',
  'lapU', 'lapV', 'lapW', 'advU', 'advV', 'advW',
  'fsr', 'fst', 'fsz', 'kap', 'psurf',
  /* H at every node of each family, which refreshMetric fills and every column
     reconstruction reads. Compared like the rest, so a depth that differed between
     the two engines would be found where it is made, not three operators later. */
  'hcolP', 'hcolU', 'hcolV', 'hcolW',
  /* The pressure solve's preconditioner: the azimuthal Fourier basis, which JavaScript
     evaluates and writes in because the module computes no transcendental, the per-mode
     band Cholesky factors, which JavaScript builds and writes in so the factorisation has
     one implementation, and the two working rows of an application. */
  'dftc', 'dfts', 'pcband', 'pchatc', 'pchats'
];

/* The JavaScript field each module array corresponds to, where the names differ.
   Anything not listed here has the same name on both sides. */
const CELL3D_JS_NAME = {
  gu: '_gu', gv: '_gv', gw: '_gw', gom: '_gom', div: '_div',
  cgr: '_r', cgd: '_d', cgq: '_q', cgz: '_z',
  pcband: '_pcBand', pchatc: '_pcHatC', pchats: '_pcHatS',
  lapU: '_lu', lapV: '_lv', lapW: '_lw', advU: '_au', advV: '_av', advW: '_aw',
  fsr: '_fsr', fst: '_fst', fsz: '_fsz', kap: '_kap', psurf: '_ps',
  hcolP: '_hcolP', hcolU: '_hcolU', hcolV: '_hcolV', hcolW: '_hcolW'
};

/* The arrays JavaScript writes INTO the module: the grid, which it owns, and the
   state, which the caller sets. Everything else the module fills. */
const CELL3D_GRID = ['rf', 'sf', 'rc', 'sc', 'drc', 'dsc', 'drf', 'dsf', 'rx', 'dftc', 'dfts'];
const CELL3D_STATE = ['u', 'v', 'w', 'om', 'p', 'eta', 'Ht'];

const CELL3D_WASM_EXPORTS = [
  'cell3d_init', 'cell3d_ptr', 'cell3d_len', 'cell3d_error', 'cell3d_errorI',
  'cell3d_errorK', 'cell3d_clearError', 'cell3d_arenaUsed',
  'cell3d_refreshMetric',
  'cell3d_divergence', 'cell3d_gradient', 'cell3d_omegaOf', 'cell3d_applyL',
  'cell3d_applyPreconditioner', 'cell3d_solveP', 'cell3d_cgIters',
  'cell3d_cgResidual', 'cell3d_omegaFromW', 'cell3d_wFromOmega',
  'cell3d_maxDivergence',
  'cell3d_buildFamilies', 'cell3d_axisU', 'cell3d_refreshSurfaceFluxes',
  'cell3d_viscous',
  'cell3d_advectTransport', 'cell3d_advectCurvature', 'cell3d_advect',
  'cell3d_curvature', 'cell3d_surfaceArea', 'cell3d_surfaceExcessArea',
  'cell3d_setGravity', 'cell3d_gravity', 'cell3d_surfacePressure', 'cell3d_step',
  'cell3d_stepBegin', 'cell3d_stepExplicit', 'cell3d_stepFinish',
  'cell3d_viscousRows', 'cell3d_advectRows'
];

function decodeBase64(b64){
  if (typeof Buffer !== 'undefined') return new Uint8Array(Buffer.from(b64, 'base64'));
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/* Where the bytes come from, in order: an explicit Uint8Array the caller passes,
   globalThis.FARADAY_CELL3D_WASM_BASE64 which the single-file build inlines, and
   dns/faraday_cell3d.wasm read from disk under node. A browser served from a
   checkout has no synchronous third option, so the single-file build is what
   carries it there; this refuses rather than guessing. */
function cell3dWasmBytes(explicit){
  if (explicit) return explicit;
  if (typeof globalThis !== 'undefined' && globalThis.FARADAY_CELL3D_WASM_BASE64)
    return decodeBase64(globalThis.FARADAY_CELL3D_WASM_BASE64);
  if (typeof require === 'function'){
    try {
      const { readFileSync } = require('fs');
      const { join } = require('path');
      return new Uint8Array(readFileSync(join(__dirname, 'faraday_cell3d.wasm')));
    } catch (e){
      throw new Error(
        'the three-dimensional C++ module could not be read from '
        + `dns/faraday_cell3d.wasm: ${e.message}. Build it with `
        + 'dns/build-wasm-cell3d.sh. This refuses rather than running the '
        + 'JavaScript solver under the C++ engine\'s name.');
    }
  }
  throw new Error(
    'no bytes for the three-dimensional C++ module: pass a Uint8Array, set '
    + 'globalThis.FARADAY_CELL3D_WASM_BASE64 (the single-file build does), or '
    + 'run under node beside dns/faraday_cell3d.wasm.');
}

function instantiateCell3dWasm(explicitBytes){
  const bytes = cell3dWasmBytes(explicitBytes);
  const mod = new WebAssembly.Module(bytes);
  /* The module imports nothing -- no libc, no transcendentals, no environment --
     so an empty import object is the whole contract and there is nothing to get
     wrong between node and a browser. */
  const inst = new WebAssembly.Instance(mod, {});
  const missing = CELL3D_WASM_EXPORTS.filter(n => typeof inst.exports[n] !== 'function');
  if (missing.length) throw new Error(
    `the three-dimensional C++ module is missing ${missing.join(', ')}. It was `
    + 'built from a different source than this loader expects; rebuild with '
    + 'dns/build-wasm-cell3d.sh.');
  if (!(inst.exports.memory instanceof WebAssembly.Memory)) throw new Error(
    'the three-dimensional C++ module exports no memory, so its arrays cannot '
    + 'be read.');
  return inst;
}

/* One instance per module, because the module's arena is static: two cells would
   share it. Constructing a second one while the first is alive is a refusal, not
   a silent aliasing of their states. */
let CELL3D_WASM_LIVE = null;

class FaradayCell3DWasm {
  /* `o` is what FaradayCell3D takes. A JavaScript solver is constructed from it
     and kept: it owns the grid, and the gate compares against it. */
  constructor(o, opts){
    const options = opts || {};
    const JS = options.jsClass || (typeof require === 'function'
      ? require('./faraday-cell3d.js').FaradayCell3D
      : (typeof globalThis !== 'undefined' && globalThis.FARADAY_CELL3D
         && globalThis.FARADAY_CELL3D.FaradayCell3D));
    if (typeof JS !== 'function') throw new Error(
      'the JavaScript solver is not available, and this needs it to build the '
      + 'grid -- the graded-grid code has one implementation on purpose.');

    if (CELL3D_WASM_LIVE) throw new Error(
      'a three-dimensional C++ cell is already live. The module\'s arena is '
      + 'static, so a second cell would share its arrays with the first; call '
      + 'release() on the first one. Refusing rather than aliasing two states.');

    this.js = new JS(o);
    /* THE CLOCK LIVES HERE, not on this.js, and that is not a detail. The drive's
       cos is evaluated in JavaScript against it, so if the clock were a field of
       the JavaScript solver then anything else that stepped that solver -- a gate
       comparing the two, most obviously -- would advance the time this module
       reads and the two would silently run at different instants of the drive.
       Measured while writing the gate: forty driven steps agreed bit for bit
       undriven and diverged to 1e-1 relative driven, entirely from that shared
       counter. pullState() copies it across when the caller wants it there. */
    this.t = this.js.t;
    this.inst = instantiateCell3dWasm(options.bytes);
    const X = this.inst.exports;

    const err = X.cell3d_init(
      this.js.nr, this.js.nth, this.js.nz,
      this.js.contact === 'pinned' ? 1 : 0,
      this.js.R, this.js.h, this.js.rho, this.js.nu, this.js.gamma, this.js.g,
      this.js.dth);
    if (err) throw new Error(this.errorMessage(err));

    this.views = {};
    this.keyOf = {};
    for (let key = 0; key < CELL3D_ARRAYS.length; key++){
      const name = CELL3D_ARRAYS[key];
      const ptr = X.cell3d_ptr(key), len = X.cell3d_len(key);
      if (!(ptr > 0) || !(len > 0)) throw new Error(
        `the module gave pointer ${ptr} and length ${len} for ${name} (key `
        + `${key}). A null pointer is offset zero, which is rf, and would have `
        + 'looked like a grid that had gone wrong.');
      this.views[name] = new Float64Array(X.memory.buffer, ptr, len);
      this.keyOf[name] = key;
    }

    for (const name of CELL3D_GRID) this.views[name].set(this.js[name]);
    /* The family descriptors and the radial bracket tables are made OF the grid,
       so they are built after it is written in and not inside cell3d_init. */
    X.cell3d_buildFamilies();
    const built = X.cell3d_error();
    if (built) throw new Error(this.errorMessage(built));
    /* The pressure solve's preconditioner, factored by the JavaScript solver from a flat
       twin of this cell and copied in like the grid: one implementation of the
       factorisation, and the module only applies it. */
    this.js.buildPreconditioner();
    this.views.pcband.set(this.js._pcBand);
    CELL3D_WASM_LIVE = this;
  }

  errorMessage(code){
    const X = this.inst.exports;
    if (code === 1) return 'the C++ module\'s arena is exhausted: this grid needs '
      + `more than ${X.cell3d_arenaUsed()} doubles. Raise ARENA in `
      + 'dns/faraday_cell3d.cpp and --initial-memory in dns/build-wasm-cell3d.sh.';
    if (code === 2) return 'three-dimensional pressure solve did not converge: '
      + `residual ${X.cell3d_cgResidual().toExponential(3)} after `
      + `${X.cell3d_cgIters()} iterations on a `
      + `${this.js.nr}x${this.js.nth}x${this.js.nz} grid.`;
    if (code === 3){
      const i = X.cell3d_errorI(), k = X.cell3d_errorK();
      return 'the free surface has reached the floor: h + eta <= 0 at '
        + `r = ${this.js.rc[i].toExponential(3)} m, theta = `
        + `${(k*this.js.dth).toFixed(3)} rad. The surface-following coordinate `
        + 'z = sigma*(h + eta) requires a positive depth everywhere; a '
        + 'non-positive one means the layer has broken and there is no '
        + 'single-valued surface to follow. Refusing rather than continuing '
        + 'with an inverted cell, which would return a field.';
    }
    if (code === 6) return 'the C++ module was asked to step without being told '
      + 'this instant\'s effective gravity. The cos in g + a cos(omega_d t) is the '
      + 'one transcendental the physics needs and it is evaluated in JavaScript, '
      + 'because a libm cos and V8\'s cos need not agree on the last bit and the '
      + 'whole step hangs off that number. The module clears it after every step '
      + 'rather than reusing the previous instant\'s.';
    if (code === 7) return 'the C++ module was given a non-positive time step.';
    if (code === 5) return 'famLaplacian: a family offers fewer than four values '
      + 'in sigma and the face derivative needs four. nz >= 4 guarantees them, so '
      + 'this is a descriptor error rather than a grid that is too coarse.';
    return `the C++ module reported error ${code}, which this loader does not `
      + 'know. It was built from a different source than this loader expects.';
  }

  /* Copy the JavaScript solver's state in, so both run on the same numbers. */
  pushState(){
    for (const name of CELL3D_STATE) this.views[name].set(this.js[name]);
    return this;
  }

  refreshMetric(){
    const code = this.inst.exports.cell3d_refreshMetric();
    if (code) throw new Error(this.errorMessage(code));
    return this;
  }

  /* The byte offset of a named array, which is what the module's functions take
     as an argument: a wasm32 pointer IS that offset. Named rather than numbered
     at the call sites below, so a renumbering cannot pass the wrong array. */
  at(name){
    const key = this.keyOf[name];
    if (key === undefined) throw new Error(
      `${name} is not one of the module's arrays: ${CELL3D_ARRAYS.join(', ')}.`);
    return this.inst.exports.cell3d_ptr(key);
  }

  divergence(a = 'u', b = 'v', c = 'om', out = 'div'){
    this.inst.exports.cell3d_divergence(this.at(a), this.at(b), this.at(c), this.at(out));
    return this.views[out];
  }

  gradient(q = 'p'){
    this.inst.exports.cell3d_gradient(this.at(q), this.at('gu'), this.at('gv'),
                                      this.at('gw'));
    return [this.views.gu, this.views.gv, this.views.gw];
  }

  omegaOf(a = 'gu', b = 'gv', c = 'gw', out = 'gom'){
    this.inst.exports.cell3d_omegaOf(this.at(a), this.at(b), this.at(c), this.at(out));
    return this.views[out];
  }

  applyL(q = 'p', out = 'div'){
    this.inst.exports.cell3d_applyL(this.at(q), this.at(out));
    return this.views[out];
  }

  applyPreconditioner(r = 'cgr', z = 'cgz'){
    this.inst.exports.cell3d_applyPreconditioner(this.at(r), this.at(z));
    return this.views[z];
  }

  solveP(rhs = 'div', tol = 1e-12, maxIt = 2000){
    const res = this.inst.exports.cell3d_solveP(this.at(rhs), tol, maxIt);
    const code = this.inst.exports.cell3d_error();
    if (code) throw new Error(this.errorMessage(code));
    return res;
  }

  get cgIters(){ return this.inst.exports.cell3d_cgIters(); }
  get cgResidual(){ return this.inst.exports.cell3d_cgResidual(); }

  axisU(){ this.inst.exports.cell3d_axisU(); return this; }

  refreshSurfaceFluxes(){
    this.inst.exports.cell3d_refreshSurfaceFluxes();
    return [this.views.fsr, this.views.fst, this.views.fsz];
  }

  viscous(hasBc = 1){
    this.inst.exports.cell3d_viscous(hasBc ? 1 : 0);
    const code = this.inst.exports.cell3d_error();
    if (code) throw new Error(this.errorMessage(code));
    return [this.views.lapU, this.views.lapV, this.views.lapW];
  }

  advectTransport(){
    this.inst.exports.cell3d_advectTransport();
    return [this.views.advU, this.views.advV, this.views.advW];
  }
  advectCurvature(){
    this.inst.exports.cell3d_advectCurvature();
    return [this.views.advU, this.views.advV];
  }
  advect(){
    this.inst.exports.cell3d_advect();
    return [this.views.advU, this.views.advV, this.views.advW];
  }

  curvature(){
    this.inst.exports.cell3d_curvature(this.at('kap'));
    return this.views.kap;
  }
  surfaceArea(){ return this.inst.exports.cell3d_surfaceArea(); }
  surfaceExcessArea(){ return this.inst.exports.cell3d_surfaceExcessArea(); }

  /* The drive's cos, evaluated HERE and passed in. The module refuses to step
     without it and clears it afterwards, so a step can never silently run on the
     previous instant's gravity. The clock it reads is this.js.t, which `step`
     advances -- the JavaScript object is the clock even while the C++ holds the
     field. */
  gravityNow(){
    const J = this.js;
    return J.g + J.accel*Math.cos(J.omegaD*this.t);
  }

  surfacePressure(){
    this.inst.exports.cell3d_setGravity(this.gravityNow());
    const code = this.inst.exports.cell3d_surfacePressure(this.at('psurf'));
    if (code) throw new Error(this.errorMessage(code));
    return this.views.psurf;
  }

  step(dt){
    if (!(typeof dt === 'number' && Number.isFinite(dt) && dt > 0)) throw new TypeError(
      `dt = ${dt}: a finite positive time step is required.`);
    this.inst.exports.cell3d_setGravity(this.gravityNow());
    const code = this.inst.exports.cell3d_step(dt);
    if (code) throw new Error(this.errorMessage(code));
    this.t += dt;
    return this;
  }

  /* step() in its three parts, for dns/cell3d-pool.js: the same calls cell3d_step makes, in
     the same order, with the gravity set where step() sets it and the clock advanced where
     step() advances it. */
  stepBegin(dt){
    if (!(typeof dt === 'number' && Number.isFinite(dt) && dt > 0)) throw new TypeError(
      `dt = ${dt}: a finite positive time step is required.`);
    this.inst.exports.cell3d_setGravity(this.gravityNow());
    const code = this.inst.exports.cell3d_stepBegin(dt);
    if (code) throw new Error(this.errorMessage(code));
    return this;
  }
  stepExplicit(i0 = 0, i1 = this.js.nr - 1){
    const code = this.inst.exports.cell3d_stepExplicit(i0, i1);
    if (code) throw new Error(this.errorMessage(code));
    return this;
  }
  stepFinish(dt){
    const code = this.inst.exports.cell3d_stepFinish(dt);
    if (code) throw new Error(this.errorMessage(code));
    this.t += dt;
    return this;
  }

  /* Copy the module's state back into the JavaScript object, so everything that
     is deliberately NOT ported -- the renderer's resamplers, the stability limit
     with its tanh, the energy split, the divergence report -- reads the real
     field. One pass over about a megabyte, against a step that costs tens of
     milliseconds, so the renderer steps in C++ and pulls once a frame. */
  pullState(){
    for (const name of CELL3D_ARRAYS){
      const js = CELL3D_JS_NAME[name] === undefined ? name : CELL3D_JS_NAME[name];
      if (js === null) continue;
      const dst = this.js[js];
      if (dst && dst.length === this.views[name].length) dst.set(this.views[name]);
    }
    this.js.t = this.t;
    return this;
  }

  omegaFromW(){ this.inst.exports.cell3d_omegaFromW(); return this; }
  wFromOmega(){ this.inst.exports.cell3d_wFromOmega(); return this; }
  maxDivergence(){ return this.inst.exports.cell3d_maxDivergence(); }

  release(){ if (CELL3D_WASM_LIVE === this) CELL3D_WASM_LIVE = null; return this; }
}

const FARADAY_CELL3D_WASM = {
  FaradayCell3DWasm, instantiateCell3dWasm, cell3dWasmBytes,
  CELL3D_ARRAYS, CELL3D_GRID, CELL3D_STATE, CELL3D_WASM_EXPORTS, CELL3D_JS_NAME
};

if (typeof module !== 'undefined' && module.exports) module.exports = FARADAY_CELL3D_WASM;
if (typeof globalThis !== 'undefined') globalThis.FARADAY_CELL3D_WASM = FARADAY_CELL3D_WASM;
