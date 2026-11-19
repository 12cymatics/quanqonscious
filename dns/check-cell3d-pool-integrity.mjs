import assert from 'node:assert/strict';
import test from 'node:test';
import { Worker } from 'node:worker_threads';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
const require=createRequire(import.meta.url);
const {FaradayCell3D}=require('./faraday-cell3d.js');
const {FaradayCell3DWasm}=require('./faraday-cell3d-wasm.js');
const {Cell3DPool}=require('./cell3d-pool.js');
const workerPath=fileURLToPath(new URL('./cell3d-pool-node.cjs',import.meta.url));
const options={nr:6,nth:8,nz:4,R:0.017,h:0.004,rho:875,nu:4e-6,gamma:0.028,g:9.80665,accel:4,omegaD:130,contact:'pinned'};
async function runFault(engine,alter,expected,parts=1){
  const solver=engine==='cpp'?new FaradayCell3DWasm(options):new FaradayCell3D(options);
  const js=solver.js||solver;
  for(let i=0;i<js.nr;i++)for(let k=0;k<js.nth;k++){
    const r=js.rc[i]/js.R;
    js.eta[js.ie(i,k)]=1e-5*r*r*(1-r*r)*Math.cos(2*(k+0.5)*js.dth);
  }
  js.refreshMetric();
  if(solver.pushState){solver.pushState();solver.refreshMetric();}
  const workers=Array.from({length:parts},()=>new Worker(workerPath,{workerData:{cell3dPool:true,engine}}));
  const channels=workers.map((worker,w)=>({
    post:(message,transfer)=>worker.postMessage(message,transfer),
    onMessage:receive=>worker.on('message',message=>{
      const result=message.kind==='rows'?alter(message,w):message;
      if(Array.isArray(result))for(const response of result)receive(response);
      else receive(result);
    }),
    onFailure:fail=>{worker.on('error',fail);worker.on('messageerror',fail);worker.on('exit',code=>fail(`exited ${code}`));}
  }));
  const pool=new Cell3DPool(solver,channels);
  try{
    await pool.start(options);
    const before=solver.t;
    await assert.rejects(pool.step(js.stableStep()),expected);
    assert.equal(solver.t,before,'a rejected worker result must not finish or advance a step');
    await assert.rejects(pool.step(js.stableStep()),/earlier step failed/);
    assert.equal(pool.pending.size,0,'no rejected request remains pending');
  }finally{
    await Promise.all(workers.map(worker=>worker.terminate()));
    if(solver.release)solver.release();
  }
}
const cases=[
  ['empty row range',m=>({...m,a0:1,a1:0,runs:m.runs.map(()=>new Float64Array(0))}),/requested rows/],
  ['shifted row range',m=>({...m,a0:m.a0-1,a1:m.a1-1}),/requested rows/],
  ['missing result array',m=>({...m,runs:m.runs.slice(0,-1)}),/all 6 required arrays/],
  ['non-finite result',m=>{m.runs[0][0]=NaN;return m;},/non-finite/],
  ['wrong response kind',m=>({...m,kind:'ready'}),/expected rows/],
  ['valid rows followed by an unrequested response',m=>[m,{...m,id:m.id+1000}],/unknown request/],
  ['another worker request id',(m,w)=>({...m,id:m.id+(w===0?1:-1)}),/assigned to worker/,2]
];
for(const engine of ['js','cpp'])for(const [label,alter,expected,parts]of cases)
  test(`${engine}: real worker ${label} cannot finish the solver step`,{timeout:20000},()=>runFault(engine,alter,expected,parts));
test('a synchronous transport failure rejects and clears its request',async()=>{
  const solver=new FaradayCell3D(options);
  const pool=new Cell3DPool(solver,[{onMessage(){},post(){throw new Error('transport unavailable');}}]);
  await assert.rejects(pool.start(options),/transport unavailable/);
  assert.equal(pool.pending.size,0);
  await assert.rejects(pool.start(options),/transport unavailable/);
});
test('a valid ready reply followed by an unrequested response cannot start the pool',async()=>{
  const solver=new FaradayCell3D(options);
  const worker=new Worker(workerPath,{workerData:{cell3dPool:true,engine:'js'}});
  const pool=new Cell3DPool(solver,[{
    post:(message,transfer)=>worker.postMessage(message,transfer),
    onMessage:receive=>worker.on('message',message=>{receive(message);receive({...message,id:message.id+1000});}),
    onFailure:fail=>{worker.on('error',fail);worker.on('exit',code=>fail(`exited ${code}`));}
  }]);
  try{
    await assert.rejects(pool.start(options),/unknown request/);
    assert.equal(pool.ready,false);
    assert.equal(pool.pending.size,0);
  }finally{await worker.terminate();}
});
