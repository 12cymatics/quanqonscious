#!/usr/bin/env node
/* Run the three-dimensional nonlinear solver from a terminal, with no browser.
 *
 * The page in cymatic.html draws the same solver through a worker. This exists
 * because a browser is not always to hand, because an ASCII plan view of eta is
 * enough to see whether a pattern is forming, and because the clock ratio is the
 * number that decides what grid is worth asking for on a given machine.
 *
 *     node dns/run-cell3d.mjs
 *     node dns/run-cell3d.mjs --nr 16 --nth 32 --nz 10 --m 4 --accel 12 --periods 4
 *     node dns/run-cell3d.mjs --engine cpp
 *
 * `--engine cpp` runs dns/faraday_cell3d.wasm, the same discretisation transcribed
 * to C++ and compiled freestanding for wasm32. It is not a faster-but-looser mode:
 * dns/check-cell3d-wasm.mjs holds the two to IDENTICAL fields, operator by
 * operator and then over a whole drive period, so the only thing that changes is
 * the wait. Measured here: 3.2x at 10x24x8 and 2.7x at 16x24x10. There is no
 * silent fallback -- if the module cannot be loaded this refuses and says so,
 * rather than running the JavaScript under the C++ engine's name and reporting a
 * time that means something else.
 *
 * Every figure it prints is measured in the run, including the wall clock. It
 * does not estimate.
 *
 * It REFUSES rather than substituting: an unknown flag, a non-numeric value, or a
 * grid the solver rejects stops the run and names the reason. A runner that
 * quietly ran something other than what was asked for would make its own output
 * worthless.
 */

import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const K = require(join(here, '..', 'faraday', 'kernel.js'));
const { FaradayCell3D } = require(join(here, 'faraday-cell3d.js'));
const { FaradayCell3DWasm } = require(join(here, 'faraday-cell3d-wasm.js'));

/* ---- the apparatus ------------------------------------------------------- */
/* A 24.25 mm quartz cell holding 3 mm of water at 20 C: the same constants the
   gate and the page use, so a run here is comparable with both. */
const DEFAULTS = {
  R: 12.125e-3, h: 3e-3,
  rho: 998.2041322005837, nu: 1.0035510043427237e-6,
  gamma: 0.07273614042160757, g: 9.80665,
  nr: 12, nth: 24, nz: 8,
  m: 4, rel: 0.15, accel: 0, freq: 0,
  periods: 1, frames: 6, contact: 'free',
  rStretch: 2.2, zStretch: 2.2, width: 61,
  engine: 'js'
};

const NUMERIC = new Set(['R', 'h', 'rho', 'nu', 'gamma', 'g', 'nr', 'nth', 'nz',
                         'm', 'rel', 'accel', 'freq', 'periods', 'frames',
                         'rStretch', 'zStretch', 'width']);

function parseArgs(argv){
  const o = { ...DEFAULTS };
  for (let i = 0; i < argv.length; i++){
    const a = argv[i];
    if (!a.startsWith('--')) throw new Error(
      `unexpected argument ${JSON.stringify(a)}. Every option is --name value.`);
    const name = a.slice(2);
    if (!Object.prototype.hasOwnProperty.call(DEFAULTS, name)) throw new Error(
      `unknown option --${name}. Known: ${Object.keys(DEFAULTS).join(', ')}.`);
    const raw = argv[++i];
    if (raw === undefined) throw new Error(`--${name} needs a value.`);
    if (NUMERIC.has(name)){
      const v = Number(raw);
      if (!Number.isFinite(v)) throw new Error(
        `--${name} ${JSON.stringify(raw)} is not a finite number.`);
      o[name] = v;
    } else o[name] = raw;
  }
  if (o.engine !== 'js' && o.engine !== 'cpp') throw new Error(
    `--engine ${JSON.stringify(o.engine)}: either 'js' for dns/faraday-cell3d.js `
    + `or 'cpp' for the same discretisation compiled to WebAssembly. The two agree `
    + `bit for bit, which dns/check-cell3d-wasm.mjs asserts, so the choice is `
    + `about time and nothing else.`);
  if (o.contact !== 'free' && o.contact !== 'pinned') throw new Error(
    `--contact ${JSON.stringify(o.contact)}: the contact line is either 'free' or `
    + `'pinned'.`);
  return o;
}

/* ---- the mode -----------------------------------------------------------
   The free-surface modes of a disc with a free contact line have J_m'(k R) = 0.
   jpZerosNearN finds the zeros of J_m' near a given order; the first one above
   m is the lowest radial mode of that azimuthal number. */
const kOf = (m, R) => K.jpZerosNearN(m, m + 1.9, 1).sort((a, b) => a - b)[0]/R;

/* ---- an ASCII plan view of the surface ----------------------------------
   Sampled through etaAt, which is the solver's own surface at an arbitrary
   position -- not a re-read of the stored cell values -- so what is drawn is
   what the renderer would draw. */
const RAMP = ' .:-=+*#%@';
function plan(S, width){
  const rows = Math.max(5, Math.round(width*0.47));
  let peak = 0;
  const z = new Float64Array(rows*width);
  const inside = new Uint8Array(rows*width);
  for (let b = 0; b < rows; b++)
    for (let a = 0; a < width; a++){
      const x = (2*(a + 0.5)/width - 1), y = (2*(b + 0.5)/rows - 1);
      const rr = Math.hypot(x, y);
      if (rr > 1) continue;
      /* etaAt takes a physical radius; the rim column is the solver's own
         boundary value, so sample just inside it rather than exactly on it. */
      const r = Math.min(rr, 1 - 1e-12)*S.R;
      const th = Math.atan2(y, x);
      const e = S.etaAt(r, th);
      z[b*width + a] = e; inside[b*width + a] = 1;
      peak = Math.max(peak, Math.abs(e));
    }
  const out = [];
  for (let b = 0; b < rows; b++){
    let line = '';
    for (let a = 0; a < width; a++){
      if (!inside[b*width + a]){ line += ' '; continue; }
      const t = peak > 0 ? (z[b*width + a]/peak + 1)/2 : 0.5;
      const idx = Math.min(RAMP.length - 1, Math.max(0, Math.round(t*(RAMP.length - 1))));
      line += RAMP[idx];
    }
    out.push(line);
  }
  return { art: out, peak };
}

const mm = x => (x*1e3).toFixed(4);
const sci = x => x.toExponential(4);

function main(){
  const o = parseArgs(process.argv.slice(2));
  const k = kOf(o.m, o.R);
  const omega = Math.sqrt((o.g*k + o.gamma*k*k*k/o.rho)*Math.tanh(k*o.h));
  /* With no drive given, the shaker runs at twice the mode's own frequency --
     the subharmonic resonance Faraday waves live on. */
  const omegaD = o.freq > 0 ? 2*Math.PI*o.freq : 2*omega;

  const cellOpts = {
    nr: o.nr, nth: o.nth, nz: o.nz,
    R: o.R, h: o.h, rho: o.rho, nu: o.nu, gamma: o.gamma, g: o.g,
    accel: o.accel, omegaD, contact: o.contact,
    rStretch: o.rStretch, zStretch: o.zStretch
  };
  /* The C++ engine holds the field and the JavaScript object holds the clock, the
     grid and everything deliberately not ported -- the resampler, the stability
     limit with its tanh, the energy split. `pullState` copies the field across,
     once a frame rather than once a step, so every number printed below comes from
     whichever engine was asked for. */
  const W = o.engine === 'cpp' ? new FaradayCell3DWasm(cellOpts) : null;
  const S = W ? W.js : new FaradayCell3D(cellOpts);

  for (let i = 0; i < S.nr; i++)
    for (let kk = 0; kk < S.nth; kk++)
      S.eta[S.ie(i, kk)] = o.rel*o.h*K.besselJ(o.m, k*S.rc[i])
                                   *Math.cos(o.m*(kk + 0.5)*S.dth);
  S.refreshMetric();
  if (W){ W.pushState(); W.refreshMetric(); }
  const advance = W ? (dt => W.step(dt)) : (dt => S.step(dt));
  const sync = W ? (() => W.pullState()) : (() => S);

  const L = S.stepLimits();
  const dt = S.stableStep();
  const tEnd = o.periods*(2*Math.PI/omega);
  const steps = Math.max(1, Math.round(tEnd/dt));
  const e0 = S.energy();

  console.log(`cell            R = ${mm(o.R)} mm, h = ${mm(o.h)} mm, contact ${o.contact}`);
  console.log(`fluid           rho = ${o.rho.toFixed(4)} kg/m^3, nu = ${sci(o.nu)} m^2/s, `
              + `gamma = ${o.gamma.toFixed(6)} N/m`);
  console.log(`grid            ${S.nr} x ${S.nth} x ${S.nz}  `
              + `(graded r ${o.rStretch}, sigma ${o.zStretch})`);
  console.log(`engine          ${o.engine === 'cpp'
    ? 'dns/faraday_cell3d.wasm, the C++ transcription'
    : 'dns/faraday-cell3d.js'}`);
  console.log(`mode            m = ${o.m}, k = ${k.toFixed(4)} 1/m, `
              + `f = ${(omega/(2*Math.PI)).toFixed(4)} Hz, seed ${(o.rel*100).toFixed(1)}% of h`);
  console.log(`drive           a = ${o.accel} m/s^2 at ${(omegaD/(2*Math.PI)).toFixed(4)} Hz`);
  console.log(`step limits     capillary ${sci(L.capillary)} s, viscous ${sci(L.viscous)} s, `
              + `advective ${sci(L.advective)} s, gravity ${sci(L.gravity)} s`);
  console.log(`taking          dt = ${sci(dt)} s, ${steps} steps to t = ${sci(tEnd)} s `
              + `(${o.periods} mode period${o.periods === 1 ? '' : 's'})`);
  console.log(`energy at t=0   kinetic ${sci(e0.kinetic)} J, hydrostatic ${sci(e0.hydrostatic)} J, `
              + `capillary ${sci(e0.capillary)} J`);

  const every = Math.max(1, Math.floor(steps/Math.max(1, o.frames)));
  const wall0 = process.hrtime.bigint();
  let shown = 0;
  for (let n = 1; n <= steps; n++){
    advance(dt);
    if (n % every === 0 || n === steps){
      sync();
      const wall = Number(process.hrtime.bigint() - wall0)/1e9;
      const { art, peak } = plan(S, o.width);
      const e = S.energy();
      console.log('');
      console.log(`step ${n}/${steps}   t = ${sci(S.t)} s   peak |eta| = ${mm(peak)} mm `
                  + `(${(100*peak/o.h).toFixed(2)}% of h)`);
      console.log(`   energy  total ${sci(e.total)} J  (k ${sci(e.kinetic)} + h ${sci(e.hydrostatic)} `
                  + `+ c ${sci(e.capillary)})   max |div u| = ${sci(S.maxDivergence())}`);
      console.log(`   clocks  physics ${S.t.toFixed(6)} s   wall ${wall.toFixed(3)} s   `
                  + `${(wall/S.t).toFixed(1)}x slower than real time`);
      for (const line of art) console.log('   ' + line);
      shown++;
    }
  }
  const wall = Number(process.hrtime.bigint() - wall0)/1e9;
  console.log('');
  console.log(`done            ${steps} steps, ${shown} frame${shown === 1 ? '' : 's'} shown`);
  console.log(`wall clock      ${wall.toFixed(3)} s for ${S.t.toFixed(6)} s of physics `
              + `= ${(wall/S.t).toFixed(1)}x slower than real time`);
  console.log(`per step        ${(1e3*wall/steps).toFixed(3)} ms`);
}

try { main(); }
catch (e){
  console.error('refused: ' + (e && e.message || e));
  process.exit(1);
}
