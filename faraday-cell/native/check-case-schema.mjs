import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,rm,stat} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
import {createRequire} from 'node:module';
import {validateCase,estimateResources,assertPreparationBudget,ARRAY_FIELDS,BINARY_MAGIC,MATERIAL_NOTICE} from './case-schema.mjs';
import {prepareState,writePreparedCase,prepareCaseFile,solverOptions} from './prepare-case.mjs';
const require=createRequire(import.meta.url),{FaradayCell3D}=require('../dns/faraday-cell3d.js');
const root=dirname(fileURLToPath(import.meta.url)),folder=await mkdtemp(resolve(tmpdir(),'faraday-native-case-'));
const base=()=>({format:'faraday-native-case',version:1,fluid:{radius:.015,depth:.004,density:1000,dynamicViscosity:.001,surfaceTension:.0728,gravity:9.80665,contactLine:'pinned'},drive:{frequency:111,acceleration:7},grid:{nr:4,nth:8,nz:4,rStretch:0,zStretch:0},initial:{type:'flat'},grains:[],materialProvenance:{kind:'illustrative',description:'Silica-rich spherical grains and density are illustrative, not measured SE Queensland sand.',source:'Declared verification inputs, not sample measurements'},contact:{friction:.5,restitution:.3},resourceBudgetBytes:134217728,duration:1e-5,dt:1e-6,outputInterval:5e-6});
const grain=()=>({id:0,centre:[0,0,.002],radius:.000125,density:2650,velocity:[0,0,0],omega:[0,0,0]});
let passed=0;const check=async(name,fn)=>{await fn();passed++;console.log('ok '+name);};
function allocatedArrays(value,objects=new Set(),buffers=new Set()){
 if(!value||typeof value!=='object'||objects.has(value))return 0;objects.add(value);
 if(ArrayBuffer.isView(value)){if(buffers.has(value.buffer))return 0;buffers.add(value.buffer);return value.buffer.byteLength;}
 return Object.values(value).reduce((n,v)=>n+allocatedArrays(v,objects,buffers),0);
}
try{
await check('all required physical inputs are explicit and aliases/unknown keys are refused',()=>{
 const c=validateCase(base());assert.deepEqual(c,base());
 for(const key of Object.keys(base())){const b=base();delete b[key];assert.throws(()=>validateCase(b),/required/);}
 const b=base();b.grainCount=50;assert.throws(()=>validateCase(b),/not a recognised input/);
 assert.throws(()=>validateCase({...base(),version:2}),/version/);
});
await check('nonfinite values, invalid dimensions and moving contact lines are rejected',()=>{
 for(const value of [NaN,Infinity,-Infinity,'1000',null]){const b=base();b.fluid.density=value;assert.throws(()=>validateCase(b),/finite number/);}
 for(const [key,value] of [['nr',3],['nth',9],['nz',3],['rStretch',-1]]){const b=base();b.grid[key]=value;assert.throws(()=>validateCase(b));}
 const b=base();b.fluid.contactLine='free';assert.throws(()=>validateCase(b),/pinned/);
 const tinyOutput=base();tinyOutput.outputInterval=1e-300;assert.throws(()=>validateCase(tinyOutput),/safe output-count/);
});
await check('sphere state, explicit IDs, density and contact parameters are validated',()=>{
 const b=base();b.grains=[grain()];assert.equal(validateCase(b).grains.length,1);
 for(const mut of [g=>g.radius=0,g=>g.density=1000,g=>g.velocity=[0,0],g=>g.omega[0]=NaN,g=>g.centre[2]=0,g=>g.centre[0]=.015]){const c=base();c.grains=[grain()];mut(c.grains[0]);assert.throws(()=>validateCase(c));}
 const c=base();c.grains=[grain(),{...grain(),centre:[.002,0,.002]}];assert.throws(()=>validateCase(c),/unique/);
 const d=base();d.contact.restitution=1.1;assert.throws(()=>validateCase(d),/<= 1/);
});
await check('overlap is refused while exact touching is accepted without position repair',()=>{
 const b=base();b.grains=[grain(),{...grain(),id:1,centre:[.00025,0,.002]}];assert.deepEqual(validateCase(b).grains,b.grains);
 b.grains[1].centre[0]=.000249;assert.throws(()=>validateCase(b),/overlap/);
});
await check('metadata is explicit and source sample measurements are not inferred',()=>{
 const b=base();delete b.materialProvenance.source;assert.throws(()=>validateCase(b),/required/);
 assert.match(MATERIAL_NOTICE,/not measured/);
});
await check('resource counts match actual canonical typed arrays on three grids',()=>{
 for(const grid of [{nr:4,nth:8,nz:4},{nr:7,nth:12,nz:5},{nr:16,nth:30,nz:12}]){
  const b=base();Object.assign(b.grid,grid);const s=new FaradayCell3D(solverOptions(validateCase(b)));
  assert.equal(estimateResources(b.grid).jsCellArrayBytes,BigInt(allocatedArrays(s)),JSON.stringify(grid));
 }
});
await check('resource/index failures occur before any preconditioner build or output',async()=>{
 const b=base();b.resourceBudgetBytes=1;assert.throws(()=>prepareState(b),/No allocation or coarsening/);
 const big=base();Object.assign(big.grid,{nr:512,nth:4096,nz:256});assert.throws(()=>prepareState(big),/signed-index or band-preconditioner/);
 assert.throws(()=>assertPreparationBudget(validateCase(base()),1),/available memory/);
 const p=resolve(folder,'overbudget.json'),out=resolve(folder,'must-not-exist.fns');await writeFile(p,JSON.stringify(b));await assert.rejects(prepareCaseFile(p,out),/No allocation/);await assert.rejects(stat(out),{code:'ENOENT'});
});
await check('pressure-band resource scaling includes the extra vertical factor',()=>{
 const a=estimateResources({nr:64,nth:128,nz:16}),b=estimateResources({nr:64,nth:128,nz:32});assert(b.pcBandBytes>3n*a.pcBandBytes);assert(b.preparationPeakArrayBytes>2n*b.jsCellArrayBytes);assert(a.nativeFluidArrayBytes>a.jsCellArrayBytes);
});
await check('flat canonical state and stable-step refusal preserve requested physics',()=>{
 const b=base(),p=prepareState(b);assert.equal(p.cell.eta.every(x=>x===0),true);assert.equal(p.cell.u.every(x=>x===0),true);assert(p.stableDt>=b.dt);assert.equal(p.cell._pcReady,true);
 const bad=base();bad.dt=p.stableDt*2;bad.duration=bad.dt*2;assert.throws(()=>prepareState(bad),/not reduced/);
});
await check('pinned Bessel seed is a stored initial field and not a changed runtime equation',()=>{
 const b=base();b.initial={type:'seed',azimuthalOrder:2,amplitude:1e-7};const p=prepareState(b);assert(p.cell.eta.some(x=>x!==0));assert.equal(p.cell.t,0);assert.equal(p.cell.contact,'pinned');assert.deepEqual(p.case.initial,b.initial);
});
await check('portable binary is little endian, complete and preserves all canonical arrays',async()=>{
 const b=base();b.grains=[grain()];b.materialProvenance.description+=' µm';const p=prepareState(b),path=resolve(folder,'case.fns');await writePreparedCase(p,path);const bytes=await readFile(path);
 let at=0;const str=n=>{const s=bytes.subarray(at,at+n).toString('utf8');at+=n;return s;},u32=()=>{const v=bytes.readUInt32LE(at);at+=4;return v;},u64=()=>{const v=bytes.readBigUInt64LE(at);at+=8;return v;},f64=()=>{const v=bytes.readDoubleLE(at);at+=8;return v;};
 assert.equal(str(8),BINARY_MAGIC);assert.equal(bytes.subarray(8,12).toString('hex'),'01000000');assert.deepEqual([u32(),u32(),u32(),u32(),u32()],[1,4,8,4,1]);assert.equal(u64(),BigInt(b.resourceBudgetBytes));
 const scalar=Array.from({length:14},f64);assert.deepEqual(scalar,[.015,.004,1000,.000001,.0728,9.80665,2*Math.PI/8,2*Math.PI*111,7,1e-5,1e-6,5e-6,.5,.3]);
 const n=u32(),a=u32(),jsonBytes=u32();assert.equal(n,1);assert.equal(a,19);assert.equal(at,160);assert.deepEqual(JSON.parse(str(jsonBytes)),b);assert.equal(u64(),0n);assert.deepEqual(Array.from({length:11},f64),[0,0,.002,.000125,2650,0,0,0,0,0,0]);
 for(const [key,name] of ARRAY_FIELDS){assert.equal(u32(),key);assert.equal(u64(),BigInt(p.cell[name].length));for(const value of p.cell[name])assert(Object.is(f64(),value));}assert.equal(at,bytes.length);
});
await check('CLI requires paths, never overwrites input JSON and emits a preparation audit',async()=>{
 const noargs=spawnSync(process.execPath,[resolve(root,'prepare-case.mjs')],{encoding:'utf8'});assert.notEqual(noargs.status,0);assert.match(noargs.stderr,/both --case and --out/);
 const path=resolve(folder,'good.json');await writeFile(path,JSON.stringify(base()));await assert.rejects(prepareCaseFile(path,path),/must not overwrite/);
 const out=resolve(folder,'cli.fns'),run=spawnSync(process.execPath,[resolve(root,'prepare-case.mjs'),'--case',path,'--out',out],{encoding:'utf8'});assert.equal(run.status,0,run.stderr);const report=JSON.parse(run.stdout);assert.equal(report.arrayCount,19);assert.match(report.scope,/grain resolution/);assert.match(report.materialNotice,/not measured/);assert.match(run.stderr,/resource-preflight/);
});
console.log(passed+' passed, 0 failed');
}finally{await rm(folder,{recursive:true,force:true});}
