// Run from any directory: node /path/to/repository/native/check-fluid-runtime.mjs
// Requires Node, a POSIX shell and a C++20 compiler (CXX, default c++).
// Uses the checked-in WASM oracle. All generated data lives in a temporary
// directory removed on success or failure. Peak native array allocation: 121 MB.
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {mkdtempSync,readFileSync,rmSync,writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname,join,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';

const native=dirname(fileURLToPath(import.meta.url));
const root=resolve(native,'..');
const require=createRequire(import.meta.url);
const {FaradayCell3D}=require('../dns/faraday-cell3d.js');
const {FaradayCell3DWasm,CELL3D_ARRAYS,CELL3D_JS_NAME}=require('../dns/faraday-cell3d-wasm.js');
const kernel=require('../faraday/kernel.js');
const temporary=mkdtempSync(join(tmpdir(),'faraday-native-runtime-'));
const executable=join(temporary,'check-fluid-runtime');
const results=[];
const uint=value=>{const buffer=Buffer.alloc(4);buffer.writeUInt32LE(value);return buffer;};
const real=value=>{const buffer=Buffer.alloc(8);buffer.writeDoubleLE(value);return buffer;};
const bytes=values=>Buffer.from(values.buffer,values.byteOffset,values.byteLength);
const run=args=>execFileSync(executable,args,{encoding:'utf8',timeout:120000});

function compareOutput(path,wasm,name){
  const data=readFileSync(path);
  let offset=0,compared=0;
  assert.equal(data.readDoubleLE(offset),wasm.t,`${name}: physical time`); offset+=8;
  for (const key of CELL3D_ARRAYS){
    const length=data.readUInt32LE(offset); offset+=4;
    assert.equal(length,wasm.views[key].length,`${name}: ${key} length`);
    const actual=data.subarray(offset,offset+length*8),expected=bytes(wasm.views[key]);
    offset+=length*8;
    if (!actual.equals(expected)){
      let maximum=0,count=0,first=-1;
      for (let i=0;i<length;i++){
        const a=actual.readDoubleLE(i*8),b=expected.readDoubleLE(i*8);
        if (a!==b){count++;maximum=Math.max(maximum,Math.abs(a-b));if(first<0)first=i;}
      }
      throw Error(`${name}/${key}: bytes differ; ${count} numeric differences, max ${maximum}, first ${first}`);
    }
    compared+=length;
  }
  assert.equal(offset,data.length,`${name}: trailing output`);
  return compared;
}

function checkCase(name,nr,nth,nz,amplitude){
  const options={R:.015,h:.004,rho:1000,nu:1e-6,gamma:.0728,g:9.80665,
    accel:7,omegaD:2*Math.PI*111,nr,nth,nz,contact:'pinned',rStretch:2.2,zStretch:2.2};
  const js=new FaradayCell3D(options),wasm=new FaradayCell3DWasm(options);
  try {
    let seed=43210;
    const random=()=>{seed=(seed*16807)%2147483647;return seed/2147483647-.5;};
    const kr=kernel.jZeroNear(2,3.9);
    for (let i=0;i<nr;i++) for (let k=0;k<nth;k++)
      js.eta[js.ie(i,k)]=amplitude*js.h*kernel.besselJ(2,kr*js.rc[i]/js.R)*Math.cos(2*(k+.5)*js.dth);
    for (const key of ['u','v','w'])
      for (let j=0;j<js[key].length;j++) js[key][j]=2e-4*random();
    js.refreshMetric();
    for (const key of ['u','v','w','om','p','eta','Ht']) wasm.views[key].set(js[key]);
    wasm.refreshMetric();
    const steps=20,dt=1e-6;
    const chunks=[uint(nr),uint(nth),uint(nz),
      ...[js.R,js.h,js.rho,js.nu,js.gamma,js.g,js.dth].map(real)];
    for (const key of CELL3D_ARRAYS)
      chunks.push(uint(wasm.views[key].length),Buffer.from(bytes(wasm.views[key])));
    chunks.push(uint(steps),real(dt));
    // Share the exact drive values so platform libm differences do not mask a
    // discrepancy in the fluid arithmetic being compared.
    for (let step=0;step<steps;step++){
      chunks.push(real(wasm.gravityNow())); wasm.step(dt); js.step(dt);
    }
    const fixture=join(temporary,name+'.input');
    writeFileSync(fixture,Buffer.concat(chunks));
    for (const key of CELL3D_ARRAYS)
      assert(bytes(js[CELL3D_JS_NAME[key]??key]).equals(bytes(wasm.views[key])),
        `JS/WASM oracle differs: ${name}/${key}`);
    for (const mode of ['full','split','checkpoint']){
      const output=join(temporary,name+'-'+mode+'.output');
      const log=run([fixture,output,mode]);
      const actual=log.split('\n').find(line=>line.startsWith('LIMITS ')).split(' ').slice(1).map(Number);
      const limits=js.stepLimits();
      const expected=['capillary','viscous','advective','gravity',
        'kRadial','kVertical','kAzimuthal'].map(key=>limits[key]);
      expected.push(js.stableStep());
      assert.equal(actual.length,expected.length);
      for (let i=0;i<actual.length;i++)
        assert(Math.abs(actual[i]-expected[i])<=4e-15*Math.abs(expected[i]),
          `${name}/${mode}: stability bound ${i}: native ${actual[i]}, JS ${expected[i]}`);
      const doubles=compareOutput(output,wasm,name+'/'+mode);
      results.push({test:`${name}: ${mode} byte parity and stability bounds`,arrays:67,doubles,steps});
      console.log(`PASS ${name}/${mode}: 67 arrays, ${doubles} doubles, ${steps} steps`);
    }
    for (const mode of ['projection','projection-rigid']){
      const output=join(temporary,name+'-'+mode+'.output');
      console.log(run([fixture,output,mode]).trim());
      results.push({test:`${name}: ${mode}, state isolation, alias refusal, mass symmetry and idempotence`});
    }
  } finally { wasm.release(); }
}

try {
  execFileSync(join(native,'build.sh'),[executable,join(native,'check-fluid-runtime.cpp')],
    {cwd:root,stdio:'inherit',timeout:120000});
  console.log(run(['guards']).trim());
  results.push({test:'dynamic allocation, memory budget, signed indices and singleton recovery'});
  checkCase('graded-driven',8,16,8,.02);
  checkCase('deformed-driven',12,24,10,.15);
  console.log(JSON.stringify({passed:results.length,results},null,2));
} finally { rmSync(temporary,{recursive:true,force:true}); }
