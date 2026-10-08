'use strict';

/* S11: the three-dimensional step's explicit terms spread over several threads.

   WHAT IS SPREAD, AND WHAT IS NOT. A step is three parts -- stepBegin, stepExplicit and
   stepFinish, in either engine -- and only the middle one is divided here: the viscous term
   and the advective term, formed row by row from the state alone, with nothing accumulated
   across rows. Each worker forms a contiguous band of radial rows and sends them back; the
   owner copies them in and finishes the step itself. The pressure solve stays on the owner's
   thread, and that is not an oversight: a conjugate gradient needs two global sums every
   iteration, and without shared memory -- which a page opened from a file does not have,
   because SharedArrayBuffer needs cross-origin isolation and a file has no headers to grant
   it -- every one of those sums is a round trip through the message queue. S12b cut the
   solve to seven or eight iterations a step instead, which is what leaves the explicit terms
   most of the step and makes them worth dividing.

   THE ANSWER DOES NOT DEPEND ON THE NUMBER OF WORKERS, and not approximately: each output
   element is formed by exactly the arithmetic the single-threaded step uses, from exactly the
   same state, whichever worker forms it, and the owner does every reduction in the order it
   always did. So the state after N steps on any number of workers is the state after N calls
   of step(), bit for bit. dns/check-cell3d-pool.mjs asserts that with Object.is over every
   array, for both engines, rather than leaving it to this paragraph.

   THE WORKERS ARE HANDED THE STATE, NOT TRUSTED TO REBUILD IT. Every array the explicit terms
   read travels with each request -- the velocity, Omega, the surface rate, the depth and its
   face values and slopes, the column-depth tables -- rather than the worker recomputing the
   metric from eta. Recomputing would be the same arithmetic, but the rate the advection reads
   is the one stepBegin leaves, not the one the last refreshMetric saw, and a worker that
   rebuilt it would be one ordering subtlety away from a different answer. Sending them is
   about 270 kilobytes a step at 16x40x10.

   THE LIST IS EXACTLY WHAT IS READ, AND THAT WAS MEASURED, NOT ASSUMED. It started as every
   state and metric array, 32 of them, and dns/check-cell3d-pool.mjs was run with each left
   out in turn. Fourteen are load bearing -- leave any one out and the states part company on
   the first step, or the worker's step fails outright -- and those are the fourteen below.
   The other eighteen -- eta itself, the depth's centred slopes, eta's and the rate's own
   extended columns, and the pressure family's depth table -- were green when left out,
   because the explicit terms do not read them; they no longer travel.

   A worker that fails is not routed around. The step refuses with the worker's own error,
   and there is no fallback to forming the rows on the owner's thread, which would be the
   single-threaded step under the pool's name.

   Transport is the caller's: a channel is { post(message, transfer), onMessage(fn) }, which a
   browser Worker, a MessagePort and a node worker_threads Worker can each be wrapped as in two
   lines. The worker's side is cell3dPoolServe below. */

/* The arrays stepExplicit reads, by their names on the JavaScript solver -- each one shown
   load bearing by leaving it out. */
const CELL3D_POOL_STATE = [
  'u', 'v', 'w', 'om', 'Ht',
  'H', 'Hr', 'Hth', 'Hx', 'Hxr', 'Hxt',
  '_hcolU', '_hcolV', '_hcolW'
];

/* The arrays it writes, each with the family whose rows it is laid out in. */
const CELL3D_POOL_OUT = [
  ['_lu', 'u'], ['_lv', 'v'], ['_lw', 'w'],
  ['_au', 'u'], ['_av', 'v'], ['_aw', 'w']
];

/* The same arrays by the C++ module's names, where they differ: dns/faraday-cell3d-wasm.js's
   CELL3D_JS_NAME read the other way. */
const CELL3D_POOL_MODULE_NAME = {
  _hcolP: 'hcolP', _hcolU: 'hcolU', _hcolV: 'hcolV', _hcolW: 'hcolW',
  _lu: 'lapU', _lv: 'lapV', _lw: 'lapW', _au: 'advU', _av: 'advV', _aw: 'advW'
};

const CELL3D_POOL_GRID = ['rf', 'sf', 'rc', 'sc', 'drc', 'dsc', 'drf', 'dsf', 'rx'];

/* Either engine, seen the same way: its JavaScript solver (which holds the grid and the
   family descriptors in both cases) and its arrays by the JavaScript names. */
function cell3dPoolView(E){
  if (E && E.views && E.js) return {
    js: E.js,
    arr(name){
      const a = E.views[CELL3D_POOL_MODULE_NAME[name] || name];
      if (!a) throw new Error(`the C++ engine has no array for ${name}.`);
      return a;
    }
  };
  if (E && E.FAM) return {
    js: E,
    arr(name){
      const a = E[name];
      if (!(a instanceof Float64Array)) throw new Error(`the solver has no array ${name}.`);
      return a;
    }
  };
  throw new TypeError('the pool needs a FaradayCell3D or a FaradayCell3DWasm.');
}

/* The contiguous run of one array that holds rows a0..a1 of its family: a family index is
   (a*nth + k)*stride + b, so whole rows are whole runs. */
function cell3dRowRun(js, fam, a0, a1){
  const F = js.FAM[fam], per = js.nth*F.stride;
  return a0 > a1 ? [0, 0] : [a0*per, (a1 + 1)*per];
}

/* Radial rows 0..nr-1 cut into `parts` contiguous bands, as evenly as the count allows. */
function cell3dBands(nr, parts){
  if (!(Number.isInteger(parts) && parts >= 1)) throw new RangeError(
    `the pool needs a whole number of workers, at least one; got ${parts}.`);
  if (parts > nr) throw new RangeError(
    `${parts} workers for ${nr} radial rows: a worker with no row would be idle, and the pool `
    + 'does not pretend otherwise. Use at most as many workers as rows.');
  const out = [];
  for (let p = 0; p < parts; p++)
    out.push([Math.floor(p*nr/parts), Math.floor((p + 1)*nr/parts) - 1]);
  return out;
}

function cell3dPack(view, names){
  let n = 0;
  for (const name of names) n += view.arr(name).length;
  const buf = new Float64Array(n);
  let o = 0;
  for (const name of names){ const a = view.arr(name); buf.set(a, o); o += a.length; }
  return buf;
}

function cell3dUnpack(view, names, buf){
  let n = 0;
  for (const name of names) n += view.arr(name).length;
  if (buf.length !== n) throw new Error(
    `the pool's state message holds ${buf.length} values where this grid's arrays hold ${n}: `
    + 'the two ends were built for different grids.');
  let o = 0;
  for (const name of names){
    const a = view.arr(name);
    a.set(buf.subarray(o, o + a.length)); o += a.length;
  }
}

const cell3dClock = () => (typeof performance !== 'undefined' ? performance : Date).now();

/* THE WORKER'S SIDE. `make(options)` builds the worker's own engine for the cell -- the same
   engine the owner runs, from the same options -- and the owner holds its grid to its own,
   element by element, before the first step. */
function cell3dPoolServe(channel, make){
  let E = null, V = null;
  channel.onMessage(msg => {
    try {
      if (msg.kind === 'init'){
        E = make(msg.options); V = cell3dPoolView(E);
        const grid = {};
        for (const name of CELL3D_POOL_GRID) grid[name] = Float64Array.from(V.arr(name));
        channel.post({ kind: 'ready', id: msg.id, grid });
        return;
      }
      if (msg.kind === 'explicit'){
        if (!E) throw new Error('asked for rows before the cell was built.');
        const t0 = cell3dClock();
        cell3dUnpack(V, CELL3D_POOL_STATE, msg.state);
        const t1 = cell3dClock();
        E.stepExplicit(msg.a0, msg.a1);
        const t2 = cell3dClock();
        const runs = [], transfer = [];
        for (const [name, fam] of CELL3D_POOL_OUT){
          const [lo, hi] = cell3dRowRun(V.js, fam, msg.a0, msg.a1);
          const run = V.arr(name).slice(lo, hi);
          runs.push(run); transfer.push(run.buffer);
        }
        channel.post({ kind: 'rows', id: msg.id, a0: msg.a0, a1: msg.a1, runs,
                       unpackMs: t1 - t0, workMs: t2 - t1 }, transfer);
        return;
      }
      throw new Error(`unknown request ${JSON.stringify(msg.kind)}.`);
    } catch (e){
      channel.post({ kind: 'error', id: msg && msg.id, message: String(e && e.message || e) });
    }
  });
}

/* THE OWNER'S SIDE. `E` is the owner's engine, `channels` one per worker. The owner forms a
   band of rows itself while the workers form theirs, rather than waiting idle: n workers
   give n + 1 bands. Its band is the first, the one holding the axis rows. */
class Cell3DPool {
  constructor(E, channels){
    if (!channels || !channels.length) throw new RangeError(
      'a pool needs at least one worker. The single-threaded step is step(), and the pool '
      + 'does not stand in for it.');
    this.E = E;
    this.V = cell3dPoolView(E);
    this.channels = channels;
    const bands = cell3dBands(this.V.js.nr, channels.length + 1);
    this.ownBand = bands[0];
    this.bands = bands.slice(1);
    this.pending = new Map();
    this.nextId = 1;
    this.ready = false;
    this.lastWaitMs = 0; this.lastWorkMs = []; this.lastUnpackMs = [];
    for (const ch of channels) ch.onMessage(msg => this.receive(msg));
  }

  receive(msg){
    const p = this.pending.get(msg.id);
    if (!p) return;
    this.pending.delete(msg.id);
    if (msg.kind === 'error') p.reject(new Error('a pool worker refused: ' + msg.message));
    else p.resolve(msg);
  }

  request(ch, msg, transfer){
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      ch.post({ ...msg, id }, transfer);
    });
  }

  /* Build each worker's cell and hold its grid to the owner's, element by element. */
  async start(options){
    const replies = await Promise.all(this.channels.map(ch =>
      this.request(ch, { kind: 'init', options })));
    for (let w = 0; w < replies.length; w++)
      for (const name of CELL3D_POOL_GRID){
        const arr = replies[w].grid[name], mine = this.V.arr(name);
        if (!arr || arr.length !== mine.length) throw new Error(
          `pool worker ${w} built ${name} with ${arr && arr.length} values against the `
          + `owner's ${mine.length}: not the same cell.`);
        for (let i = 0; i < arr.length; i++) if (!Object.is(arr[i], mine[i])) throw new Error(
          `pool worker ${w} built ${name}[${i}] = ${arr[i]} against the owner's ${mine[i]}: `
          + 'the same options gave a different grid, so its rows would not be this cell\'s.');
      }
    this.ready = true;
    return this;
  }

  /* One step: begin and finish here, the explicit rows on the workers. */
  async step(dt){
    if (!this.ready) throw new Error('the pool was asked to step before start() finished.');
    const E = this.E, V = this.V;
    E.stepBegin(dt);
    const state = cell3dPack(V, CELL3D_POOL_STATE);
    const t0 = cell3dClock();
    const asked = Promise.all(this.channels.map((ch, w) =>
      this.request(ch, { kind: 'explicit', state, a0: this.bands[w][0], a1: this.bands[w][1] })));
    E.stepExplicit(this.ownBand[0], this.ownBand[1]);
    this.lastOwnMs = cell3dClock() - t0;
    const replies = await asked;
    this.lastWaitMs = cell3dClock() - t0;
    this.lastWorkMs = replies.map(r => r.workMs);
    this.lastUnpackMs = replies.map(r => r.unpackMs);
    for (const r of replies)
      CELL3D_POOL_OUT.forEach(([name, fam], j) => {
        const [lo, hi] = cell3dRowRun(V.js, fam, r.a0, r.a1);
        if (r.runs[j].length !== hi - lo) throw new Error(
          `pool rows ${r.a0}..${r.a1} of ${name} came back with ${r.runs[j].length} values `
          + `where ${hi - lo} were asked for.`);
        V.arr(name).set(r.runs[j], lo);
      });
    E.stepFinish(dt);
    return E;
  }
}

const FARADAY_CELL3D_POOL = {
  Cell3DPool, cell3dPoolServe, cell3dPoolView, cell3dBands, cell3dRowRun,
  cell3dPack, cell3dUnpack,
  CELL3D_POOL_STATE, CELL3D_POOL_OUT, CELL3D_POOL_MODULE_NAME, CELL3D_POOL_GRID
};
if (typeof module !== 'undefined' && module.exports) module.exports = FARADAY_CELL3D_POOL;
if (typeof globalThis !== 'undefined') globalThis.FARADAY_CELL3D_POOL = FARADAY_CELL3D_POOL;
