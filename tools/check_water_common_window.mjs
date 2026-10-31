import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {patternCase} from './pattern_case.mjs';
import {commonWaterWindow,WATER_BATCH_RUN,WATER_BATCH_COMMIT,WATER_AUDIT_SUCCESS} from './water_common_window.mjs';
const paths=['dns/faraday-cell3d.js','dns/faraday-cell3d-wasm.js','dns/faraday_cell3d.cpp'];
function fixture(id,cycles=34,volume=1.5,contact='free',accel=0.5){
 const options=patternCase({PATTERN_VOLUME_ML:volume,PATTERN_ACCEL_G:accel,PATTERN_CONTACT:contact}).options,{nr,nth,R}=options;
 const rf=Array.from({length:nr+1},(_,i)=>R*i/nr),rc=Array.from({length:nr},(_,i)=>(rf[i]+rf[i+1])/2);
 const raw=Array.from({length:cycles*8+1},(_,j)=>({time:j/(8*56),eta:Array.from({length:nr*nth},(_,e)=>Math.SQRT2*1e-7*Math.cos(3*(e%nth)*2*Math.PI/nth)*Math.cos(Math.PI*j/8))}));
 const config={options,run:WATER_BATCH_RUN,commit:WATER_BATCH_COMMIT,engine:'full nonlinear 3D free-surface Navier-Stokes, freshly built C++/WASM',
 frequency:56,cycles:72,safety:0.4,seed:1729,seedRms:1e-7,wallSeconds:3600,initialCondition:'common-double-zero-rim-broadband-v1',
 initialSurfaceSha256:createHash('sha256').update(Buffer.from(Float64Array.from(raw[0].eta).buffer)).digest('hex'),hashes:Object.fromEntries(paths.map(p=>[p,'a'.repeat(64)]))};
 return {id,planned:{volumeMl:volume,peakAccelerationG:accel,contact},config,grid:{rf,rc,nth},raw,
 summary:{status:'partial-wall-budget',error:null,simulatedDriveCycles:cycles+0.2},
 audit:{sourceRun:WATER_BATCH_RUN,sourceCommit:WATER_BATCH_COMMIT,validation:WATER_AUDIT_SUCCESS,originalStatus:'partial-wall-budget',originalError:null}};
}
const a=fixture('free'),b=fixture('pinned',35,2.5,'pinned',1),report=commonWaterWindow([a,b]);
assert.equal(report.available,true);assert.equal(report.commonCycle,34);assert.equal(report.targetCycles,72);assert.equal(report.allPlannedCasesIncluded,true);
assert.deepEqual(report.results.map(r=>r.windowFirstCycle),[2.125,2.125]);assert.deepEqual(report.results.map(r=>r.samples),[256,256]);
for(const r of report.results){assert.equal(r.strongestNonaxisymmetricMode,3);assert.ok(Math.abs(r.strongestNonaxisymmetricShareOfTotalPower-1)<1e-12);assert.ok(r.antiPeriodError<1e-12);assert.ok(r.twoPeriodError<1e-12);assert.ok(Math.abs(r.windowRmsM-Math.SQRT1_2*1e-7)<1e-20);assert.deepEqual(r.phaseSamples.map(f=>f.driveCycle),[33,34]);}
for(const [name,mutate] of [
 ['missing frame',x=>x.raw.splice(100,1)],['duplicate time',x=>x.raw[100].time=x.raw[99].time],
 ['nonfinite eta',x=>x.raw[100].eta[0]=null],['invalid depth',x=>x.raw[100].eta[0]=-x.config.options.h],
 ['missing audit',x=>x.audit={}],['failed audit',x=>x.audit.validation='failed'],['wrong run',x=>x.config.run='other'],
 ['wrong source',x=>x.config.commit='b'.repeat(40)],['wrong safety',x=>x.config.safety=0.2],['wrong physics',x=>x.config.options.nu*=2],
 ['wrong initial field',x=>x.raw[0].eta[0]*=2],['solver failure',x=>x.summary.status='solver-error'],
 ['mismatched audit status',x=>x.audit.originalStatus='completed-horizon'],['out-of-grid centre',x=>x.grid.rc[0]=-1],
 ['ahead of trajectory',x=>x.summary.simulatedDriveCycles=2]
 ]){
 const x=structuredClone(a);mutate(x);const result=commonWaterWindow([x]);assert.equal(result.available,false,name);assert.equal(result.cases[0].eligible,false,name);
}
const changed=structuredClone(b);changed.config.hashes[paths[0]]='b'.repeat(64);assert.match(commonWaterWindow([a,changed]).reason,/Cross-case control mismatch/);
const subset=commonWaterWindow([a,{id:'missing'}]);assert.equal(subset.available,true);assert.equal(subset.allPlannedCasesIncluded,false);assert.deepEqual(subset.excludedIds,['missing']);
const short=commonWaterWindow([fixture('short',2)]);assert.equal(short.commonCycle,2);assert.equal(short.results[0].windowAvailable,false);
const zero=structuredClone(a);for(const f of zero.raw.slice(1))f.eta.fill(0);const zr=commonWaterWindow([zero]).results[0];assert.equal(zr.windowRmsM,0);assert.equal(zr.strongestNonaxisymmetricMode,null);assert.equal(zr.strongestNonaxisymmetricShareOfTotalPower,0);assert.equal(zr.antiPeriodError,null);assert.equal(zr.twoPeriodError,null);
assert.equal(commonWaterWindow([]).available,false);
console.log('Common-window synthetic analytic and negative controls passed; no simulation executed.');
