import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {resolve} from 'node:path';
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
import {uniformPatternSamples} from './pattern_samples.mjs';
import {createHash} from 'node:crypto';
const require=createRequire(import.meta.url),dir=resolve(process.env.PATTERN_ARTIFACT||'ns-pattern-input');
const out=resolve('ns-pattern-audit');mkdirSync(out,{recursive:true});
const read=name=>JSON.parse(readFileSync(resolve(dir,name),'utf8'));
const lines=name=>readFileSync(resolve(dir,name),'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
const finite=(x,label)=>assert.ok(typeof x==='number'&&Number.isFinite(x),label+' must be a finite number, not null');
const vector=(v,n,label)=>{assert.equal(v.length,n,label+' length');v.forEach((x,i)=>finite(x,label+'['+i+']'));};
const report={sourceRun:process.env.PATTERN_EXPECTED_RUN||'41147436690',sourceCommit:process.env.PATTERN_EXPECTED_COMMIT||'08561f8eed9676aabae50cf7b681346aebccd4f5',scope:'saved-state validation, not per-step validation or continuum convergence'};
let W;
try{
 const config=read('config.json'),grid=read('grid.json'),summary=read('summary.json'),final=read('final-state.json');
 const rawFrames=lines('surface.ndjson'),metrics=lines('metrics.ndjson');
 const {frames,extraTerminalObservation}=uniformPatternSamples(config,summary,rawFrames);
 report.extraTerminalObservation=extraTerminalObservation;
 report.originalStatus=summary.status;report.originalError=summary.error;
 assert.equal(config.commit,report.sourceCommit);assert.equal(config.run,report.sourceRun);
 assert.equal(config.engine,'full nonlinear 3D free-surface Navier-Stokes, freshly built C++/WASM');
 for(const path of ['dns/faraday-cell3d.js','dns/faraday-cell3d-wasm.js','dns/faraday_cell3d.cpp'])assert.equal(createHash('sha256').update(readFileSync(path)).digest('hex'),config.hashes[path],'same solver source '+path);
 report.rebuiltWasmSha256=createHash('sha256').update(readFileSync('dns/faraday_cell3d.wasm')).digest('hex');
 const {nr,nth,h}=config.options,NE=nr*nth;
 assert.equal(rawFrames.length,metrics.length);assert.ok(frames.length>0);
 vector(grid.rf,nr+1,'rf');vector(grid.rc,nr,'rc');
 const weights=grid.rc.map((r,i)=>r*(grid.rf[i+1]-grid.rf[i])*2*Math.PI/nth);
 const area=weights.reduce((a,b)=>a+b,0)*nth;
 let minDepth=Infinity,peak=0,maxSampledCgResidual=0,maxSampledDivergence=0;
 for(let j=0;j<rawFrames.length;j++){
  const frame=rawFrames[j],metric=metrics[j];finite(frame.time,'snapshot time');
  if(j)assert.ok(frame.time>rawFrames[j-1].time,'snapshot times increase');
  assert.equal(frame.time,metric.time);vector(frame.eta,NE,'snapshot eta');
  finite(metric.cgResidual,'sampled CG residual');finite(metric.maxDivergence,'sampled divergence');
  assert.ok(metric.cgResidual<1e-11,'sampled recursive CG residual below source criterion');
  maxSampledCgResidual=Math.max(maxSampledCgResidual,metric.cgResidual);
  maxSampledDivergence=Math.max(maxSampledDivergence,metric.maxDivergence);
  for(const x of frame.eta){minDepth=Math.min(minDepth,h+x);peak=Math.max(peak,Math.abs(x));}
 }
 assert.ok(minDepth>0,'every saved surface has positive depth');
 const tail=frames.slice(-256),power=Array(nth/2+1).fill(0);
 for(const frame of tail)for(let i=0;i<nr;i++)for(let m=0;m<power.length;m++){
  let re=0,im=0;for(let k=0;k<nth;k++){const x=frame.eta[i*nth+k];re+=x*Math.cos(m*k*2*Math.PI/nth)/nth;im-=x*Math.sin(m*k*2*Math.PI/nth)/nth;}
  power[m]+=weights[i]*nth*(re*re+im*im)*(m===0||m===nth/2?1:2)/(area*tail.length);
 }
 let dominant=1;for(let m=2;m<power.length;m++)if(power[m]>power[dominant])dominant=m;
 const rms=rows=>rows.length?Math.sqrt(rows.reduce((sum,f)=>sum+f.eta.reduce((a,x,e)=>a+weights[Math.floor(e/nth)]*x*x,0)/area,0)/rows.length):null;
 const previousRms=rms(frames.slice(-512,-256)),lastRms=rms(tail);
 const envelopeChange=previousRms?Math.abs(lastRms-previousRms)/previousRms:null;
 function recurrence(lag,sign){
  if(frames.length<=lag)return null;
  let a=0,b=0;for(let t=Math.max(lag,frames.length-256);t<frames.length;t++)for(let e=0;e<NE;e++){
   const x=frames[t].eta[e],y=frames[t-lag].eta[e],w=weights[Math.floor(e/nth)];a+=w*(x-sign*y)**2;b+=w*x*x;
  }return b?Math.sqrt(a/b):null;
 }
 const antiPeriodError=recurrence(8,-1),twoPeriodError=recurrence(16,1);
 const angularTotal=power.reduce((a,b)=>a+b,0);finite(angularTotal,'angular power total');
 const noSolverError=summary.status!=='solver-error'&&!summary.error;
 const established=noSolverError&&final.time*config.frequency>=128&&frames.length>=513&&lastRms>10*config.seedRms&&envelopeChange!==null&&envelopeChange<0.05&&twoPeriodError!==null&&twoPeriodError<0.1;
 Object.assign(report,{savedFrames:frames.length,rawSavedFrames:rawFrames.length,minSavedDepthM:minDepth,peakSavedEtaM:peak,maxSampledCgResidual,maxSampledDivergence,
  dominantSignedMode:angularTotal>0?dominant:null,dominantPowerShare:angularTotal>0?power[dominant]/angularTotal:0,angularPower:power,
  previousRms,lastRms,envelopeChange,antiPeriodError,twoPeriodError,establishedTwoPeriodDiagnostic:established,
  establishedSubharmonicDiagnostic:established&&antiPeriodError!==null&&antiPeriodError<0.1,
  opticalComparison:'No optical forward model; signed surface modes are not photographic fold counts.'});
 assert.ok(noSolverError,'solver error: final arrays may be a partially mutated failed step');
 assert.ok(['completed-horizon','partial-wall-budget'].includes(summary.status),'known non-error status');
 const {FaradayCell3DWasm}=require('../dns/faraday-cell3d-wasm.js');W=new FaradayCell3DWasm(config.options);
 for(const name of ['eta','u','v','w']){vector(final[name],W.views[name].length,'final '+name);W.views[name].set(final[name]);}
 finite(final.time,'final time');assert.ok(final.time>=rawFrames.at(-1).time);
 const finalMinDepth=Math.min(...final.eta.map(x=>h+x));assert.ok(finalMinDepth>0);
 W.t=final.time;W.refreshMetric();W.omegaFromW();W.pullState();
 const divergence=W.maxDivergence();finite(divergence,'reconstructed final divergence');
 vector(W.views.H,NE,'reconstructed H');vector(W.views.om,W.js.NW,'reconstructed omega');vector(W.views.div,W.js.NP,'reconstructed flux divergence');
 let independentlyScannedDivergence=0;
 for(let i=0;i<nr;i++)for(let k=0;k<nth;k++)for(let j=0;j<W.js.nz;j++){
  const q=W.views.div[W.js.ip(i,k,j)]/(weights[i]*W.views.H[i*nth+k]*W.js.dsc[j]);finite(q,'volume-normalized divergence');independentlyScannedDivergence=Math.max(independentlyScannedDivergence,Math.abs(q));
 }
 const energy=W.js.energy();Object.entries(energy).forEach(([key,value])=>finite(value,'energy '+key));
 Object.assign(report,{finalMinDepthM:finalMinDepth,finalReconstructedDivergence:divergence,independentlyScannedDivergence,finalEnergy:energy,validation:'saved fields finite and positive-depth; accepted final velocity divergence recomputed, not threshold-certified'});
}catch(e){report.establishedTwoPeriodDiagnostic=false;report.establishedSubharmonicDiagnostic=false;report.validation='failed';report.failure=e.stack||String(e);process.exitCode=1;}
finally{if(W)W.release();writeFileSync(resolve(out,'audit.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));}
