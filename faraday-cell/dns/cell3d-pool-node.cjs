'use strict';

/* The pool's node transport: worker_threads on both sides of dns/cell3d-pool.js.

   Required from the main thread, it exports spawnNodePool, which starts `parts` workers
   running THIS file and returns a started Cell3DPool over them. Run as one of those workers
   (workerData.cell3dPool is set), it builds the cell the owner asks for, in the engine the
   owner names, and serves rows. The browser has its own transport in cymatic.html; the pool
   itself knows nothing about either. */

const { Worker, isMainThread, parentPort, workerData } = require('node:worker_threads');
const { join } = require('node:path');
const { cell3dPoolServe, Cell3DPool } = require(join(__dirname, 'cell3d-pool.js'));

if (!isMainThread && workerData && workerData.cell3dPool){
  const engine = workerData.engine;
  const make = engine === 'cpp'
    ? o => new (require(join(__dirname, 'faraday-cell3d-wasm.js')).FaradayCell3DWasm)(o)
    : o => new (require(join(__dirname, 'faraday-cell3d.js')).FaradayCell3D)(o);
  cell3dPoolServe({ post: (m, t) => parentPort.postMessage(m, t),
                    onMessage: fn => parentPort.on('message', fn) }, make);
}

/* `E` the owner's engine, built from `options`; `engine` 'js' or 'cpp', which the workers
   must match -- a C++ owner with JavaScript workers would be the right answer from a
   different engine than the one named. */
async function spawnNodePool(E, options, parts, engine){
  if (engine !== 'js' && engine !== 'cpp') throw new Error(
    `engine ${JSON.stringify(engine)}: the workers run 'js' or 'cpp', the owner's own.`);
  const ownerIsCpp = !!(E && E.views && E.js);
  if (ownerIsCpp !== (engine === 'cpp')) throw new Error(
    `the owner is the ${ownerIsCpp ? 'C++' : 'JavaScript'} engine and the workers were asked `
    + `to run ${engine}: both ends of a pool run one engine.`);
  const workers = [];
  for (let p = 0; p < parts; p++)
    workers.push(new Worker(__filename, { workerData: { cell3dPool: true, engine } }));
  const channels = workers.map(w => ({
    post: (m, t) => w.postMessage(m, t), onMessage: fn => w.on('message', fn) }));
  const pool = new Cell3DPool(E, channels);
  pool.workers = workers;
  pool.close = () => Promise.all(workers.map(w => w.terminate()));
  try { await pool.start(options); }
  catch (e){ await pool.close(); throw e; }
  return pool;
}

module.exports = { spawnNodePool };
