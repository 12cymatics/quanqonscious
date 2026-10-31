import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {patternCase,seedCommonSurface,commonRadial,REFERENCE_RADIUS_M} from './pattern_case.mjs';
const require=createRequire(import.meta.url),{FaradayCell3D}=require('../dns/faraday-cell3d.js');
const defaultCase=patternCase();assert.ok(Math.abs(defaultCase.options.h-2.5e-6/(Math.PI*REFERENCE_RADIUS_M**2))<1e-17);
assert.ok(Math.abs(patternCase({PATTERN_VOLUME_ML:'1.5'}).options.h-1.5e-6/(Math.PI*REFERENCE_RADIUS_M**2))<1e-17);
assert.equal(patternCase({PATTERN_DEPTH_MM:'3'}).options.h,.003);
assert.equal(patternCase({PATTERN_ACCEL_G:'0'}).options.accel,0);
assert.equal(patternCase({PATTERN_ACCEL_G:'1'}).options.accel,9.80665);
for(const env of [{PATTERN_DEPTH_MM:'3',PATTERN_VOLUME_ML:'2.5'},{PATTERN_VOLUME_ML:'0'},{PATTERN_DEPTH_MM:'NaN'},{PATTERN_ACCEL_G:'-1'},{PATTERN_CONTACT:'fixed'}])assert.throws(()=>patternCase(env));
let reference;
for(const volume of ['1.5','2.5'])for(const accel of ['.5','1'])for(const contact of ['free','pinned']){
 const {options}=patternCase({PATTERN_VOLUME_ML:volume,PATTERN_ACCEL_G:accel,PATTERN_CONTACT:contact,PATTERN_NR:'6',PATTERN_NTH:'32',PATTERN_NZ:'4'});
 const S=new FaradayCell3D(options),meta=seedCommonSurface(S);
 if(reference)assert.deepEqual(S.eta,reference,'same initial surface across depth/drive/contact');else reference=S.eta.slice();
 const rms=Math.sqrt(S.eta.reduce((s,x,e)=>s+x*x*meta.weights[Math.floor(e/S.nth)],0)/meta.area);
 assert.ok(Math.abs(rms-1e-7)<1e-20);assert.ok([...S.u,...S.v,...S.w].every(x=>x===0));
 S.refreshMetric();assert.ok([...S.H].every(x=>Number.isFinite(x)&&x>0));
}
for(let m=1;m<=12;m++)for(const b of [-1,0,1]){
 assert.equal(commonRadial(1,m,b),0);const e=1e-5;
 assert.ok(Math.abs(commonRadial(1-e,m,b))<5*e*e,'double rim zero, not single zero');
}
const A=new FaradayCell3D(defaultCase.options),B=new FaradayCell3D(defaultCase.options);
seedCommonSurface(A,1729);seedCommonSurface(B,1730);assert.notDeepEqual(A.eta,B.eta,'seed actually changes field');
assert.throws(()=>seedCommonSurface(A,-1));assert.throws(()=>seedCommonSurface(A,1,0));
console.log('Explicit depth/volume, drive units, all8 identical compatible initial fields, finite metrics and seed controls passed.');
