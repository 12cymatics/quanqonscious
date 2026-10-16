'use strict';

/* The pool's node transport: worker_threads on both sides of dns/cell3d-pool.js.

   Required from the main thread, it exports spawnNodePool, which starts `parts` workers
   running THIS file and returns a started Cell3DPool over them. Run as one of those workers
   (workerData.cell3dPool is set), it builds the cell the owner asks for, in the engine the
   owner names, and serves rows. The browser has its own transport in cymatic.html; the pool
   itself knows nothing about either. */

const { Worker, isMainThread, parentPort, workerData } = require('node:worker_threads');
const { join } = require('node:path');
const { cell3dPoolServe, Cell3DPool, cell3dPoolView, cell3dBands } =
  require(join(__dirname, 'cell3d-pool.js'));

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
   different engine than the one named.

   Everything the pool would refuse is refused BEFORE a worker is started: a worker started
   and then abandoned keeps the process alive, so a caller that caught the refusal used to be
   left with threads it never asked to keep. And from the first worker started, anything that
   fails terminates every one of them before the error leaves here. */
async function spawnNodePool(E, options, parts, engine){
  if (engine !== 'js' && engine !== 'cpp') throw new Error(
    `engine ${JSON.stringify(engine)}: the workers run 'js' or 'cpp', the owner's own.`);
  const ownerIsCpp = !!(E && E.views && E.js);
  if (ownerIsCpp !== (engine === 'cpp')) throw new Error(
    `the owner is the ${ownerIsCpp ? 'C++' : 'JavaScript'} engine and the workers were asked `
    + `to run ${engine}: both ends of a pool run one engine.`);
  /* the owner's band and one per worker -- the count the pool's constructor holds to */
  cell3dBands(cell3dPoolView(E).js.nr, parts + 1);
  const workers = [];
  const close = () => Promise.all(workers.map(w => w.terminate()));
  try {
    for (let p = 0; p < parts; p++)
      workers.push(new Worker(__filename, { workerData: { cell3dPool: true, engine } }));
    /* A worker that raises outside its own handler, or exits, never answers what it was
       asked: the pool hears of it here and fails those requests with the reason, rather than
       leaving a step awaiting a thread that is gone. An exit the pool's own close() causes
       is reported the same way, so a step asked of a closed pool refuses too. */
    const channels = workers.map(w => ({
      post: (m, t) => w.postMessage(m, t),
      onMessage: fn => w.on('message', fn),
      onFailure: fn => {
        w.on('error', e => fn(`raised: ${e && e.message || e}`));
        w.on('messageerror', e => fn(`sent a message that could not be read: ${e && e.message || e}`));
        w.on('exit', code => fn(`exited, with code ${code}, and will not answer.`));
      } }));
    const pool = new Cell3DPool(E, channels);
    pool.workers = workers;
    pool.close = close;
    await pool.start(options);
    return pool;
  } catch (e){
    await close();
    throw e;
  }
}

module.exports = { spawnNodePool };
