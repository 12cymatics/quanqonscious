'use strict';

/* The C++ period map, loaded as WebAssembly, behind the same interface as the
   JavaScript solver's.

   WHAT THIS IS NOT. It is not an approximation of the JavaScript solver and not
   a faster-but-looser mode. dns/faraday_disc.cpp is a transcription of
   dns/faraday-disc.js -- same discretisation, same flux forms, same conjugate
   gradient with the same tolerance and cap, same order of operations, double
   precision throughout, no fused multiply-add, no fast-math. It computes no
   transcendental: the cos in the drive is evaluated in JavaScript into a table
   and passed in. The two therefore agree BIT FOR BIT, which
   dns/check-disc.mjs asserts rather than assumes.

   WHAT IT IS FOR. Time. At the renderer's default working point the cheapest grid
   that both represents the mode and resolves every Stokes layer is 40 x 12, which
   is 2580 steps per drive period; one Krylov-16 solve there takes 79.1 s in
   JavaScript and 33.5 s here, a factor of 2.36. A higher radial mode is dearer
   again. That is the whole purpose: the numbers are identical, the wait is not.

   THERE IS NO SILENT FALLBACK. The caller names the engine and the result says
   which one ran. If the module cannot be loaded this refuses and says so, rather
   than quietly running the JavaScript and reporting a time that means something
   else.

   The grid is not rebuilt here. The JavaScript solver builds it and its node
   arrays are copied in, so the graded-grid code has one implementation. */

const DISC_WASM_EXPORTS = [
  'setup', 'setScalars', 'ptrRf', 'ptrZf', 'ptrRc', 'ptrZc', 'ptrDrc', 'ptrDzc',
  'ptrDrf', 'ptrDzf', 'ptrDrive', 'ptrVecIn', 'ptrVecOut', 'ptrEta',
  'getCgIters', 'getCgResidual', 'getLastError', 'applyPeriodMap', 'stepFromVec',
  'prepare'
];

function decodeBase64(b64){
  if (typeof Buffer !== 'undefined') return new Uint8Array(Buffer.from(b64, 'base64'));
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/* Where the bytes come from, in order:
     an explicit Uint8Array the caller passes,
     globalThis.FARADAY_DISC_WASM_BASE64, which the single-file build inlines,
     dns/faraday_disc.wasm read from disk under node.
   A browser served from a checkout has no synchronous third option, so the
   single-file build is what carries it there; this refuses rather than guessing. */
function discWasmBytes(explicit){
  if (explicit) return explicit;
  if (typeof globalThis !== 'undefined' && globalThis.FARADAY_DISC_WASM_BASE64)
    return decodeBase64(globalThis.FARADAY_DISC_WASM_BASE64);
  if (typeof require === 'function'){
    try {
      const { readFileSync } = require('fs');
      const { join } = require('path');
      return new Uint8Array(readFileSync(join(__dirname, 'faraday_disc.wasm')));
    } catch (e) {
      throw new Error(
        'faraday-disc-wasm: dns/faraday_disc.wasm could not be read '
        + `(${e.message}). Build it with dns/build-wasm.sh.`);
    }
  }
  throw new Error(
    'faraday-disc-wasm: no WebAssembly bytes available. Either pass them in, or '
    + 'load a page whose build inlined FARADAY_DISC_WASM_BASE64. Refusing rather '
    + 'than running the JavaScript solver under this name, which would report a '
    + 'time for work the C++ did not do.');
}

/* Synchronous, so a worker can build it without an await in the hot path.
   WebAssembly.Module on a 19 KB module is well under the size where browsers
   require the async form. */
function createDiscEngine(o){
  o = o || {};
  if (typeof WebAssembly === 'undefined') throw new Error(
    'faraday-disc-wasm: this environment has no WebAssembly.');
  const bytes = discWasmBytes(o.bytes);
  let instance;
  try {
    instance = new WebAssembly.Instance(new WebAssembly.Module(bytes), {});
  } catch (e) {
    throw new Error(
      `faraday-disc-wasm: the module would not compile (${e.message}). It is `
      + `built with 128-bit SIMD, which every browser carrying the workers and `
      + `Float64Array this page already uses also has; if this fires, that is `
      + `what is missing. Refusing rather than running the JavaScript solver `
      + `under this name.`);
  }
  const X = instance.exports;
  for (const name of DISC_WASM_EXPORTS)
    if (typeof X[name] !== 'function') throw new Error(
      `faraday-disc-wasm: the module does not export ${name}. It was built from `
      + `different source than this loader expects.`);

  /* The heap view is rebuilt on each use: the module's memory is fixed at build
     time today, but a view held across a growth would silently detach. */
  const heap = () => new Float64Array(X.memory.buffer);

  return {
    engine: 'wasm',
    byteLength: bytes.length,

    /* Copy a JavaScript solver's grid and the drive table in, and size the
       arena. `steps` is needed up front because the drive table is that long. */
    load(S, steps, dt, accel, omegaD, gBase){
      if (X.setup(S.nr, S.nz, steps) !== 0) throw new Error(
        `faraday-disc-wasm: the module's arena cannot hold a ${S.nr} x ${S.nz} `
        + `grid with ${steps} drive samples. Raise ARENA in `
        + `dns/faraday_disc.cpp and rebuild.`);
      X.setScalars(S.m, S.rho, S.nu, S.gamma, gBase);
      const H = heap();
      const put = (ptr, arr) => H.set(arr, ptr/8);
      put(X.ptrRf(), S.rf);   put(X.ptrZf(), S.zf);
      put(X.ptrRc(), S.rc);   put(X.ptrZc(), S.zc);
      put(X.ptrDrc(), S.drc); put(X.ptrDzc(), S.dzc);
      put(X.ptrDrf(), S.drf); put(X.ptrDzf(), S.dzf);
      /* The drive, sampled at t = n*dt exactly. Accumulating dt instead makes
         the map cover slightly less than one period and was what first broke
         bit-for-bit agreement with the JavaScript. */
      const drive = new Float64Array(steps);
      for (let n = 0; n < steps; n++)
        drive[n] = gBase + accel*Math.cos(omegaD*(n*dt));
      put(X.ptrDrive(), drive);
      /* The pressure operator's diagonal, for the Jacobi preconditioner. It
         depends on the grid and on m, so it is formed after both are in place and
         before any step -- the same point in the sequence as the JavaScript
         solver's constructor, and from the same two applications of the same
         operator, so the two preconditioners are identical and the parity holds. */
      const prep = X.prepare();
      if (prep !== 0) throw new Error(
        `faraday-disc-wasm: the pressure operator's diagonal is not negative `
        + `definite on this ${S.nr} x ${S.nz} grid at m = ${S.m} (code ${prep}). `
        + `It is negative definite by construction, so this means a cell is `
        + `decoupled from the pressure field.`);
      this.steps = steps;
      this.pinned = S.contact === 'pinned' ? 1 : 0;
      return this;
    },

    applyPeriodMap(vec, out, dt, href, uref){
      heap().set(vec, X.ptrVecIn()/8);
      const rc = X.applyPeriodMap(dt, href, uref, this.pinned);
      if (rc === 2) throw new Error(
        `faraday-disc-wasm: the pressure solve did not converge, residual `
        + `${X.getCgResidual().toExponential(3)} after ${X.getCgIters()} `
        + `iterations. Same tolerance and cap as the JavaScript solver, so this `
        + `is the problem and not the port.`);
      if (rc !== 0) throw new Error(`faraday-disc-wasm: error code ${rc}.`);
      out.set(new Float64Array(X.memory.buffer, X.ptrVecOut(), out.length));
      return out;
    },

    /* One step from a state the caller wrote, for the parity gate: a single step
       is where a divergence is still attributable to a term. */
    stepOnce(vec, out, dt, gNow, href, uref){
      heap().set(vec, X.ptrVecIn()/8);
      const rc = X.stepFromVec(dt, gNow, href, uref, this.pinned);
      if (rc !== 0) throw new Error(`faraday-disc-wasm: error code ${rc}.`);
      out.set(new Float64Array(X.memory.buffer, X.ptrVecOut(), out.length));
      return out;
    },

    cgIters(){ return X.getCgIters(); },
    cgResidual(){ return X.getCgResidual(); }
  };
}

const FARADAY_DISC_WASM = { createDiscEngine, discWasmBytes, DISC_WASM_EXPORTS };
if (typeof module !== 'undefined' && module.exports) module.exports = FARADAY_DISC_WASM;
if (typeof globalThis !== 'undefined') globalThis.FARADAY_DISC_WASM = FARADAY_DISC_WASM;
