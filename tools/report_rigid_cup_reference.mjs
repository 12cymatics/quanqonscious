import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import assert from 'node:assert/strict';

const require=createRequire(import.meta.url);
const source=process.env.CYMATICA_SOLVER_PATH||'../dns/faraday-cell3d.js';
const {FaradayCell3D}=require(source.startsWith('.')?source:resolve(source));
const R=24.25e-3/2,volume=2.5e-6,h=volume/(Math.PI*R*R);
const fluid={rho:998.2041322005837,nu:1.0035510043427237e-6,gamma:0.07273614042160757,g:9.80665};
const grids=[[6,16,4],[9,24,6],[12,32,8]],end=0.001;
const initial=(r,theta)=>0.1*h*((r/R)**2*Math.cos(2*theta)+0.5*(r/R)**3*Math.sin(3*theta))*(1-(r/R)**2)**2;
function create(grid,contact){
  const [nr,nth,nz]=grid;
  const S=new FaradayCell3D({nr,nth,nz,R,h,...fluid,accel:0,omegaD:0,contact});
  for(let i=0;i<nr;i++)for(let k=0;k<nth;k++)S.eta[S.ie(i,k)]=initial(S.rc[i],(k+0.5)*S.dth);
  S.refreshMetric();
  return S;
}
function sample(S){
  const out=[];
  for(let i=0;i<16;i++)for(let k=0;k<48;k++)out.push(S.etaAt((i+0.5)*R/16,(k+0.5)*2*Math.PI/48));
  return out;
}
function difference(a,b){return Math.sqrt(a.reduce((s,x,i)=>s+(x-b[i])**2,0)/a.length);}
const contacts=[];
for(const contact of ['free','pinned']){
  const limits=grids.map(g=>create(g,contact).stableStep());
  const steps=Math.ceil(end/(0.5*Math.min(...limits))),dt=end/steps;
  const records=[];
  for(const grid of grids)for(const refinement of [1,2]){
    const S=create(grid,contact),E0=S.energy(),wall0=performance.now();
    for(let n=0;n<steps*refinement;n++){
      const step=dt/refinement;
      assert.ok(step<=S.stableStep(),`evolving step limit violated at ${S.t}`);
      S.step(step);
    }
    const samples=sample(S),energy=S.energy(),divergence=S.maxDivergence();
    assert.ok(samples.every(Number.isFinite)&&Number.isFinite(energy.total)&&Number.isFinite(divergence),'nonfinite evolved output');
    const record={grid,refinement,steps:steps*refinement,dt:dt/refinement,time:S.t,wallMs:performance.now()-wall0,
      energy0:E0,energy,divergence,samples};
    records.push(record);
    console.log(JSON.stringify({type:'run',source,contact,...record}));
  }
  const temporal=grids.map((grid,i)=>({grid,rmsDifferenceM:difference(records[2*i].samples,records[2*i+1].samples)}));
  const spatial=[difference(records[1].samples,records[3].samples),difference(records[3].samples,records[5].samples)];
  const result={contact,dt,steps,temporal,spatialRmsDifferencesM:spatial,spatialDifferenceDecreases:spatial[1]<spatial[0]};
  contacts.push(result);
  console.log(JSON.stringify({type:'convergence',...result}));
}
console.log(JSON.stringify({type:'qualification',source,diameterM:2*R,volumeM3:volume,meanDepthM:h,fluid,
  horizonS:end,initialCondition:'chosen smooth m2+m3 surface, zero velocity; test data, not measurement',
  forcing:'unforced test, peak acceleration zero; no mV calibration inferred',
  meanDepthScope:'volume divided by circular area; meniscus unspecified',
  comparison:'full nonlinear numerical short-time self-convergence with independent common sampling; not paper morphology agreement',
  physicalComparison:'not quantitatively evaluable without measured drive transfer, meniscus/contact-line and initial-state data',
  timeDifferenceScope:'two step sizes at each spatial grid; not an established temporal order or continuum error bound',
  spaceDifferenceScope:'nonuniform refinement ratios; differences reported, no inferred asymptotic order',contacts}));
