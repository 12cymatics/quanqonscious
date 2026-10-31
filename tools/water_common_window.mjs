import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {patternCase} from './pattern_case.mjs';
import {uniformPatternSamples} from './pattern_samples.mjs';
export const WATER_BATCH_RUN='41219846979';
export const WATER_BATCH_COMMIT='58d5416d49cf4730261079c1deb9c5a1927cf45d';
export const WATER_AUDIT_SUCCESS='saved fields finite and positive-depth; accepted final velocity divergence recomputed, not threshold-certified';
const sourcePaths=['dns/faraday-cell3d.js','dns/faraday-cell3d-wasm.js','dns/faraday_cell3d.cpp'];
const finite=x=>typeof x==='number'&&Number.isFinite(x);
function prepare(record){
 const {config:c,summary:s,audit:a,grid:g,raw,planned:p}=record;
 assert.ok(c&&s&&a&&g&&Array.isArray(raw),'required saved inputs exist');
 assert.equal(c.run,WATER_BATCH_RUN,'expected simulation run');assert.equal(c.commit,WATER_BATCH_COMMIT,'expected simulation commit');
 assert.equal(a.sourceRun,c.run,'audit run');assert.equal(a.sourceCommit,c.commit,'audit commit');assert.equal(a.validation,WATER_AUDIT_SUCCESS,'successful saved-state audit');
 assert.ok(!a.failure&&!s.error&&!a.originalError,'no simulation or audit error');
 assert.ok(['partial-wall-budget','completed-horizon'].includes(s.status),'accepted trajectory status');assert.equal(a.originalStatus,s.status,'audit status');
 assert.equal(c.engine,'full nonlinear 3D free-surface Navier-Stokes, freshly built C++/WASM');
 const expected=patternCase({PATTERN_VOLUME_ML:p.volumeMl,PATTERN_ACCEL_G:p.peakAccelerationG,PATTERN_CONTACT:p.contact}).options;
 assert.deepEqual(c.options,expected,'full planned physics and discretization');
 for(const [key,value] of Object.entries({frequency:56,cycles:72,safety:0.4,seed:1729,seedRms:1e-7,wallSeconds:3600,initialCondition:'common-double-zero-rim-broadband-v1'}))assert.equal(c[key],value,'planned '+key);
 assert.match(c.initialSurfaceSha256,/^[0-9a-f]{64}$/,'initial surface hash');for(const path of sourcePaths)assert.match(c.hashes?.[path]||'',/^[0-9a-f]{64}$/,'source hash '+path);
 const {nr,nth,h,R}=c.options;
 assert.equal(g.nth,nth);assert.equal(g.rf.length,nr+1);assert.equal(g.rc.length,nr);
 g.rf.forEach((x,i)=>assert.ok(finite(x)&&(!i||x>g.rf[i-1]),'ordered finite radial faces'));
 assert.ok(Math.abs(g.rf[0])<=Number.EPSILON&&Math.abs(g.rf.at(-1)-R)<=8*Number.EPSILON*R,'radial extent');
 g.rc.forEach((x,i)=>assert.ok(finite(x)&&x>g.rf[i]&&x<g.rf[i+1],'radial centres inside cells'));
 assert.ok(finite(s.simulatedDriveCycles)&&s.simulatedDriveCycles>=0&&s.simulatedDriveCycles<=72+1e-10,'achieved horizon');
 let previous=-1;
 raw.forEach(f=>{assert.ok(finite(f.time)&&f.time>previous,'strict raw time ordering');previous=f.time;assert.equal(f.eta.length,nr*nth);f.eta.forEach(x=>assert.ok(finite(x)&&h+x>0,'finite positive-depth saved surface'));});
 const {frames,extraTerminalObservation}=uniformPatternSamples(c,s,raw);
 assert.ok(frames.at(-1).time*c.frequency<=s.simulatedDriveCycles+1e-8,'saved sample does not exceed accepted trajectory');
 const initial=Float64Array.from(frames[0].eta);
 assert.equal(createHash('sha256').update(Buffer.from(initial.buffer)).digest('hex'),c.initialSurfaceSha256,'actual initial surface hash');
 const weights=g.rc.map((r,i)=>r*(g.rf[i+1]-g.rf[i])*2*Math.PI/nth),area=weights.reduce((a,b)=>a+b,0)*nth;
 assert.ok(finite(area)&&area>0,'positive quadrature area');
 return {...record,frames,weights,area,extraTerminalObservation,completedWholeCycles:Math.floor((frames.length-1)/8)};
}
function statistics(record,cycle){
 const {config:c,frames,weights,area}=record,{nr,nth}=c.options,last=cycle*8,start=last-255;
 const rms=f=>Math.sqrt(f.eta.reduce((s,x,e)=>s+weights[Math.floor(e/nth)]*x*x,0)/area);
 const result={id:record.id,originalStatus:record.summary.status,achievedDriveCycles:record.summary.simulatedDriveCycles,commonEndpointCycle:cycle,
  initialRmsM:rms(frames[0]),endpointRmsM:rms(frames[last]),phaseSamples:[cycle-1,cycle].map(k=>({driveCycle:k,timeS:frames[k*8].time,rmsM:rms(frames[k*8])})),
  extraTerminalObservation:record.extraTerminalObservation};
 if(start<16)return {...result,windowAvailable:false,reason:'Fewer than 256 common samples plus two periods of lag history; no unequal shortened window substituted.'};
 const power=Array(nth/2+1).fill(0);let square=0;
 for(let j=start;j<=last;j++){
  const f=frames[j];square+=rms(f)**2;
  for(let i=0;i<nr;i++)for(let m=0;m<power.length;m++){
   let re=0,im=0;for(let k=0;k<nth;k++){const x=f.eta[i*nth+k];re+=x*Math.cos(m*k*2*Math.PI/nth)/nth;im-=x*Math.sin(m*k*2*Math.PI/nth)/nth;}
   power[m]+=weights[i]*nth*(re*re+im*im)*(m===0||m===nth/2?1:2)/(area*256);
  }
 }
 function recurrence(lag,sign){let num=0,den=0;for(let j=start;j<=last;j++)for(let e=0;e<nr*nth;e++){const x=frames[j].eta[e],y=frames[j-lag].eta[e],w=weights[Math.floor(e/nth)];num+=w*(x-sign*y)**2;den+=w*x*x;}return den>0?Math.sqrt(num/den):null;}
 let dominant=1;for(let m=2;m<power.length;m++)if(power[m]>power[dominant])dominant=m;
 const total=power.reduce((a,b)=>a+b,0),nonaxisymmetric=power.slice(1).reduce((a,b)=>a+b,0);
 return {...result,windowAvailable:true,windowFirstCycle:start/8,windowLastCycle:cycle,samples:256,windowRmsM:Math.sqrt(square/256),angularPower:power,
  strongestNonaxisymmetricMode:nonaxisymmetric>0?dominant:null,strongestNonaxisymmetricShareOfTotalPower:total>0?power[dominant]/total:0,
  antiPeriodError:recurrence(8,-1),twoPeriodError:recurrence(16,1)};
}
export function commonWaterWindow(records){
 const valid=[],cases=[];
 for(const record of records){try{const prepared=prepare(record);valid.push(prepared);cases.push({id:record.id,eligible:true,originalStatus:record.summary.status,achievedDriveCycles:record.summary.simulatedDriveCycles,completedWholeCycles:prepared.completedWholeCycles});}
 catch(error){cases.push({id:record.id,eligible:false,reason:error.message,originalStatus:record.summary?.status||'missing-output',achievedDriveCycles:record.summary?.simulatedDriveCycles??null});}}
 const report={schema:'water-common-window/1',classification:'Exploratory shorter common window; not replacement for requested 72-cycle completion',targetCycles:72,
  sourceRun:WATER_BATCH_RUN,sourceCommit:WATER_BATCH_COMMIT,cases,includedIds:valid.map(r=>r.id),excludedIds:cases.filter(r=>!r.eligible).map(r=>r.id),commonCycle:null,results:[],
  limitations:['Finite-grid, finite-horizon sensitivity only; no convergence, saturation, threshold or photo-match certification.',
   'Actual temporal samples are one drive period apart; opposite water phase is not presumed.',
   'Missing or failed cases are explicitly excluded from the common window, never silently replaced.']};
 if(!valid.length)return {...report,available:false,reason:'No independently audited valid trajectory.'};
 try{for(const r of valid.slice(1)){
  assert.equal(r.config.initialSurfaceSha256,valid[0].config.initialSurfaceSha256,'common initial surface');
  assert.deepEqual(r.grid,valid[0].grid,'common radial/angular grid');
  for(const path of sourcePaths)assert.equal(r.config.hashes[path],valid[0].config.hashes[path],'same solver source '+path);
 }}catch(error){return {...report,available:false,reason:'Cross-case control mismatch: '+error.message};}
 const cycle=Math.min(...valid.map(r=>r.completedWholeCycles));
 if(cycle<1)return {...report,available:false,reason:'No shared full drive period for two temporal samples.'};
 return {...report,available:true,commonCycle:cycle,allPlannedCasesIncluded:valid.length===records.length,results:valid.map(r=>statistics(r,cycle))};
}
