import { createRequire } from 'node:module';
import { open, rename, rm, readFile } from 'node:fs/promises';
import { resolve, dirname, basename } from 'node:path';
import { pathToFileURL } from 'node:url';
import { freemem } from 'node:os';
import { randomUUID } from 'node:crypto';
import { validateCase, assertPreparationBudget, resourceReport, ARRAY_FIELDS,
  BINARY_MAGIC, BINARY_VERSION, MATERIAL_NOTICE } from './case-schema.mjs';
const require=createRequire(import.meta.url);
const {FaradayCell3D}=require('../dns/faraday-cell3d.js');
const K=require('../faraday/kernel.js');

export function solverOptions(c){return {nr:c.grid.nr,nth:c.grid.nth,nz:c.grid.nz,rStretch:c.grid.rStretch,zStretch:c.grid.zStretch,R:c.fluid.radius,h:c.fluid.depth,rho:c.fluid.density,nu:c.fluid.dynamicViscosity/c.fluid.density,gamma:c.fluid.surfaceTension,g:c.fluid.gravity,accel:c.drive.acceleration,omegaD:2*Math.PI*c.drive.frequency,contact:c.fluid.contactLine};}

export function prepareState(input,{availableMemoryBytes=process.availableMemory?.()??freemem(),progress=()=>{}}={}){
  const c=validateCase(input),estimate=assertPreparationBudget(c,availableMemoryBytes);
  progress({stage:'resource-preflight',...resourceReport(c.grid),resourceBudgetBytes:c.resourceBudgetBytes});
  // No FaradayCell3D (including its flat twin and O(NP*nz) pressure factor)
  // is constructed until both the explicit budget and machine memory pass.
  const cell=new FaradayCell3D(solverOptions(c));
  if(c.initial.type==='seed'){
    const m=c.initial.azimuthalOrder,k=K.jZeroNear(m,m+1.9)/cell.R;
    for(let i=0;i<cell.nr;i++)for(let j=0;j<cell.nth;j++)cell.eta[cell.ie(i,j)]=c.initial.amplitude*K.besselJ(m,k*cell.rc[i])*Math.cos(m*(j+0.5)*cell.dth);
    cell.refreshMetric();
  }
  // Canonical stepLimits already uses g+abs(accel), the worst forcing gravity,
  // and the initial field's advective bound. Future moving-grain and field
  // limits still have to be checked by the native integrator at every step.
  const stableDt=cell.stableStep();
  if(!(Number.isFinite(stableDt)&&stableDt>0))throw new RangeError('canonical initial stableStep is not finite and positive');
  if(c.dt>stableDt)throw new RangeError('dt='+c.dt+' exceeds the canonical initial stableStep='+stableDt+'; the requested step is not reduced');
  progress({stage:'preconditioner',stableDt,requestedDt:c.dt});
  cell.buildPreconditioner();
  const arrays=ARRAY_FIELDS.map(([key,name])=>({key,name,values:cell[name]}));
  for(const a of arrays){if(!(a.values instanceof Float64Array))throw new TypeError('canonical array '+a.name+' is missing');for(const v of a.values)if(!Number.isFinite(v))throw new RangeError('canonical array '+a.name+' contains a nonfinite value');}
  return {case:c,cell,arrays,stableDt,estimate};
}

/* Portable case format, read field-by-field (never a compiler struct):
   magic[8], u32 version,nr,nth,nz,pinned, u64 resourceBudgetBytes,
   14 f64: R,h,rho,nu,gamma,g,dtheta,omegaD,accel,duration,dt,outputInterval,
           contact friction,restitution;
   u32 grainCount,arrayCount,jsonBytes; canonical case JSON UTF-8;
   each grain: u64 id,11 f64 centre[3],radius,density,velocity[3],omega[3];
   each array: u32 key,u64 elementCount, elementCount f64. All integers and
   IEEE-754 doubles are little-endian. No alignment padding or trailing data.
   Data are streamed through 64 KiB, without duplicating the complete state. */
export async function writePreparedCase(prepared,outputPath){
  if(!outputPath)throw new TypeError('an explicit output path is required');
  const c=prepared.case,json=Buffer.from(JSON.stringify(c),'utf8');
  if(json.length>0xffffffff)throw new RangeError('case provenance JSON exceeds the binary format');
  const header=Buffer.alloc(8+5*4+8+14*8+3*4);let at=0;
  header.write(BINARY_MAGIC,at,'ascii');at+=8;
  const u32=v=>{header.writeUInt32LE(v,at);at+=4;},u64=v=>{header.writeBigUInt64LE(BigInt(v),at);at+=8;},f64=v=>{header.writeDoubleLE(v,at);at+=8;};
  for(const v of [BINARY_VERSION,c.grid.nr,c.grid.nth,c.grid.nz,1])u32(v);u64(c.resourceBudgetBytes);
  for(const v of [c.fluid.radius,c.fluid.depth,c.fluid.density,c.fluid.dynamicViscosity/c.fluid.density,c.fluid.surfaceTension,c.fluid.gravity,prepared.cell.dth,2*Math.PI*c.drive.frequency,c.drive.acceleration,c.duration,c.dt,c.outputInterval,c.contact.friction,c.contact.restitution])f64(v);
  for(const v of [c.grains.length,prepared.arrays.length,json.length])u32(v);
  if(at!==header.length)throw new Error('internal binary header size mismatch');
  const destination=resolve(outputPath),temporary=resolve(dirname(destination),'.'+basename(destination)+'.'+randomUUID()+'.partial');
  let file;
  try{
    file=await open(temporary,'wx');
    async function write(bytes){let offset=0;while(offset<bytes.length){const {bytesWritten}=await file.write(bytes,offset,bytes.length-offset);if(!bytesWritten)throw new Error('binary output write made no progress');offset+=bytesWritten;}}
    await write(header);await write(json);
    const record=Buffer.alloc(8+11*8);
    for(const g of c.grains){record.writeBigUInt64LE(BigInt(g.id),0);let x=8;for(const v of [...g.centre,g.radius,g.density,...g.velocity,...g.omega]){record.writeDoubleLE(v,x);x+=8;}await write(record);}
    const chunk=Buffer.alloc(65536),arrayHeader=Buffer.alloc(12);
    for(const a of prepared.arrays){arrayHeader.writeUInt32LE(a.key,0);arrayHeader.writeBigUInt64LE(BigInt(a.values.length),4);await write(arrayHeader);for(let start=0;start<a.values.length;start+=chunk.length/8){const n=Math.min(chunk.length/8,a.values.length-start);for(let j=0;j<n;j++)chunk.writeDoubleLE(a.values[start+j],j*8);await write(chunk.subarray(0,n*8));}}
    await file.sync();await file.close();file=null;await rename(temporary,destination);
    return {path:destination,format:BINARY_MAGIC,version:BINARY_VERSION,arrayCount:prepared.arrays.length,grainCount:c.grains.length,stableDt:prepared.stableDt,requestedDt:c.dt,...resourceReport(c.grid),materialNotice:c.materialProvenance.kind==='illustrative'?MATERIAL_NOTICE:'Material measurements are supplied by the case author; their source is recorded without independent certification.',scope:'Resource preflight covers preparation and native fluid arrays. Coupling workspace, grain resolution, gaps and evolving stability require native runtime checks. No coarse grid or substitute force law was generated.'};
  }catch(error){if(file)await file.close();await rm(temporary,{force:true});throw error;}
}
export async function prepareCaseFile(casePath,outputPath,options={}){
  if(!casePath||!outputPath)throw new TypeError('--case and --out paths are both required');
  if(resolve(casePath)===resolve(outputPath))throw new Error('output must not overwrite the input case JSON');
  const input=JSON.parse(await readFile(casePath,'utf8'));
  return writePreparedCase(prepareState(input,options),outputPath);
}
function parseArgs(argv){const args={};for(let i=0;i<argv.length;i++){const key=argv[i];if(!['--case','--out'].includes(key)||Object.hasOwn(args,key))throw new Error('usage: node native/prepare-case.mjs --case input.json --out prepared.fns');const value=argv[++i];if(!value||value.startsWith('--'))throw new Error(key+' requires a path');args[key]=value;}if(!args['--case']||!args['--out'])throw new Error('both --case and --out are required; no example is run by default');return args;}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href){try{const args=parseArgs(process.argv.slice(2));const result=await prepareCaseFile(args['--case'],args['--out'],{progress:event=>process.stderr.write(JSON.stringify(event)+'\n')});process.stdout.write(JSON.stringify(result,null,2)+'\n');}catch(error){process.stderr.write('Native case preparation refused: '+error.message+'\n');process.exitCode=1;}}
