import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const {FaradayCell3D} = require('./faraday-cell3d.js');
const {FaradayCell3DWasm} = require('./faraday-cell3d-wasm.js');
const options = {nr:4,nth:8,nz:4,R:.015,h:.004,rho:1000,nu:1e-6,gamma:.0728,g:9.80665,accel:0,omegaD:10,contact:'pinned',rStretch:0,zStretch:0};
const fields = ['u','v','w','om','eta','H','Ht','p'];
let passed = 0;
function check(name,run){run();passed++;console.log('ok '+name);}
for(const engine of ['js','wasm']){
  function withCell(run){
    const cell = engine === 'js' ? new FaradayCell3D(options) : new FaradayCell3DWasm(options);
    try {run(cell,engine === 'js' ? cell : cell.views);}
    finally {if(engine === 'wasm')cell.release();}
  }
  check(engine+' exact rest has zero divergence',()=>withCell(cell=>assert.equal(cell.maxDivergence(),0)));
  check(engine+' a finite radial flux is measured',()=>withCell((cell,state)=>{
    const js=engine === 'js'?cell:cell.js;
    state.u[js.iu(1,0,1)]=.01;
    assert(Number.isFinite(cell.maxDivergence()) && cell.maxDivergence()>0);
  }));
  check(engine+' measures current vertical flow without changing authoritative state',()=>withCell((cell,state)=>{
    const js=engine === 'js'?cell:cell.js;
    state.w[js.iw(1,0,1)]=.1;
    const before=Object.fromEntries(fields.map(field=>[field,Array.from(state[field])]));
    const divergence=cell.maxDivergence();
    assert(Number.isFinite(divergence) && divergence>0);
    for(const field of fields)assert.deepEqual(Array.from(state[field]),before[field]);
    cell.omegaFromW();
    assert.equal(cell.maxDivergence(),divergence);
  }));
  for(const field of fields)for(const invalid of [NaN,Infinity,-Infinity]){
    check(engine+' refuses '+field+'='+invalid,()=>withCell((cell,state)=>{
      state[field][0]=invalid;
      assert.throws(()=>cell.maxDivergence(),new RegExp('maxDivergence:.*'+field+'\\['));
      if(engine === 'js'){
        assert.throws(()=>cell.stepLimits(),/stepLimits: non-finite state/);
        assert.throws(()=>cell.energy(),/energy: non-finite state/);
      }
    }));
  }
  check(engine+' refuses a zero layer depth',()=>withCell((cell,state)=>{
    state.H[0]=0;
    assert.throws(()=>cell.maxDivergence(),/maxDivergence:.*(layer depth|state H)/);
  }));
  check(engine+' refuses an invalid volume',()=>withCell((cell,state)=>{
    state.dsc[0]=Infinity;
    assert.throws(()=>cell.maxDivergence(),/maxDivergence: invalid cell volume/);
  }));
  check(engine+' refuses an overflowing finite flux',()=>withCell((cell,state)=>{
    const js=engine === 'js'?cell:cell.js;
    state.u[js.iu(1,0,1)]=1e308;
    assert.throws(()=>cell.maxDivergence(),/maxDivergence:.*divergence/);
  }));
}
check('WASM constructor initializes the supplied JavaScript physical state',()=>{
  class InitializedCell extends FaradayCell3D {
    constructor(o){
      super(o);
      for(let i=0;i<this.nr;i++)for(let k=0;k<this.nth;k++)
        this.eta[this.ie(i,k)]=1e-5*(1-this.rc[i]/this.R)*Math.cos((k+.5)*this.dth);
      this.u[this.iu(1,0,1)]=.01;
      this.t=.3;
      this.refreshMetric();
      this.omegaFromW();
    }
  }
  const cell=new FaradayCell3DWasm(options,{jsClass:InitializedCell});
  try {
    for(const field of fields)assert.deepEqual(Array.from(cell.views[field]),Array.from(cell.js[field]));
    assert.equal(cell.t,cell.js.t);
    assert.equal(cell.maxDivergence(),cell.js.maxDivergence());
  }finally{cell.release();}
});
check('raw WASM traps on invalid state instead of returning zero',()=>{
  const cell=new FaradayCell3DWasm(options);
  try {
    cell.views.u[cell.js.iu(1,0,1)]=NaN;
    assert.throws(()=>cell.inst.exports.cell3d_maxDivergence(),WebAssembly.RuntimeError);
    assert.equal(cell.inst.exports.cell3d_error(),8);
  }finally{cell.release();}
});
console.log(passed+' diagnostic checks passed');
