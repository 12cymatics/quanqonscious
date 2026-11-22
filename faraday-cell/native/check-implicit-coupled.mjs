import assert from 'node:assert/strict';
import {mkdtemp,writeFile,mkdir,rm,readFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname,resolve,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawn,spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {prepareState,writePreparedCase} from './prepare-case.mjs';

const here=dirname(fileURLToPath(import.meta.url));
const out=process.env.FARADAY_NATIVE_QA?resolve(process.env.FARADAY_NATIVE_QA):await mkdtemp(join(tmpdir(),'faraday-implicit-coupled-'));
await mkdir(out,{recursive:true});
const input=JSON.parse(await readFile(join(here,'examples/single-sphere-verification.json'),'utf8'));
input.resourceBudgetBytes=512*1024*1024;
const fixture=join(out,'case.fns'),exe=join(out,'check-implicit-coupled');
await writePreparedCase(prepareState(input),fixture);
// The verification TU includes the production runtime to inspect exact
// internal operators. Link it once; do not also compile fluid-runtime.cpp.
const sourcePaths=['check-implicit-coupled.cpp','fluid-runtime.cpp','rigid-coupler.cpp','rigid-contact.cpp','rigid-interpolation.cpp','../dns/faraday_cell3d.cpp','fluid-runtime.hpp','rigid-coupler.hpp','rigid-contact.hpp','rigid-interpolation.hpp','rigid-types.hpp','memory-budget.hpp'].sort();
const fingerprint=async()=>Object.fromEntries(await Promise.all(sourcePaths.map(async p=>[p,createHash('sha256').update(await readFile(join(here,p))).digest('hex')])));
const sources=await fingerprint();
const compiler=process.env.CXX||'c++';
const buildArgs=['-std=c++20','-O3','-ffp-contract=off','-fno-fast-math','-Wall','-Wextra','-I'+here,join(here,'check-implicit-coupled.cpp'),join(here,'rigid-coupler.cpp'),join(here,'rigid-contact.cpp'),join(here,'rigid-interpolation.cpp'),'-o',exe];
const build=spawnSync(compiler,buildArgs,{encoding:'utf8',maxBuffer:8*1024*1024});
await writeFile(join(out,'build.log'),(build.stdout??'')+(build.stderr??''));
process.stdout.write(build.stdout??'');process.stderr.write(build.stderr??'');assert.equal(build.status,0,'Compilation failed; evidence: '+out);
assert.deepEqual(await fingerprint(),sources,'Numerical source changed while compilation was in progress; rerun from stable sources');
await writeFile(join(out,'compiled-sources.json'),JSON.stringify({compiler,buildArgs,sources,executableSha256:createHash('sha256').update(await readFile(exe)).digest('hex')},null,2));
const args=[fixture];
// The cached factor preconditions the unchanged physical operator. Its full
// state equivalence is checked separately by mode=preconditioner; use it for
// the ordinary three-resolution gate to avoid repeating that costly comparison.
const mode=process.env.FARADAY_IMPLICIT_MODE??(process.env.FARADAY_IMPLICIT_COMPARE_EXPLICIT==='1'?'explicit':process.env.FARADAY_IMPLICIT_SINGLE==='1'?'single':'cached');
assert.ok(['full','single','explicit','cached','preconditioner','above-capillary','above-capillary-single'].includes(mode),'Unknown verification mode');
if(mode!=='full')args.push(mode);
const run=await new Promise((resolveRun,reject)=>{
  const child=spawn(exe,args,{stdio:['ignore','pipe','pipe']});let stdout='',stderr='';
  child.stdout.setEncoding('utf8');child.stderr.setEncoding('utf8');
  child.stdout.on('data',chunk=>{stdout+=chunk;process.stdout.write(chunk);});
  child.stderr.on('data',chunk=>{stderr+=chunk;process.stderr.write(chunk);});
  child.on('error',reject);child.on('close',(status,signal)=>resolveRun({status,signal,stdout,stderr}));
});
await writeFile(join(out,'run.log'),run.stdout+run.stderr);
const entries=(run.stdout??'').split('\n').filter(s=>s.startsWith('{')).map(JSON.parse);
await writeFile(join(out,'results.json'),JSON.stringify({scope:'Finite-amplitude joint no-contact verification',mode,entries},null,2));
assert.equal(run.status,0,'Independent implicit verification failed; evidence: '+out);assert.equal(entries.at(-1)?.passed,true);
assert.equal(entries.filter(x=>x.kind==='finite-amplitude-joint-implicit').length,mode==='explicit'?0:['single','above-capillary-single'].includes(mode)?1:mode==='preconditioner'?2:3);
assert.equal(entries.filter(x=>x.kind==='finite-amplitude-explicit-comparison').length,mode==='explicit'?3:0);
assert.equal(entries.filter(x=>x.kind==='joint-implicit-temporal-refinement').length,['full','cached','above-capillary'].includes(mode)?4:mode==='explicit'?2:0);
assert.equal(entries.filter(x=>x.kind==='projected-preconditioner-state-comparison').length,mode==='preconditioner'?1:0);
assert.equal(entries.filter(x=>x.kind==='coupled-step-bounds').length,mode.startsWith('above-capillary')?1:0);
if(process.env.FARADAY_NATIVE_QA)console.log('Independent joint implicit verification passed. Evidence: '+out);else await rm(out,{recursive:true,force:true});
