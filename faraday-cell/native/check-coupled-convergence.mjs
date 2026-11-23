import assert from 'node:assert/strict';
import {mkdtemp,writeFile,mkdir,rm,readFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname,resolve,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawn,spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {prepareState,writePreparedCase} from './prepare-case.mjs';

const here=dirname(fileURLToPath(import.meta.url));
const out=process.env.FARADAY_NATIVE_QA?resolve(process.env.FARADAY_NATIVE_QA):await mkdtemp(join(tmpdir(),'faraday-convergence-'));
await mkdir(out,{recursive:true});
const input=JSON.parse(await readFile(join(here,'examples/single-sphere-verification.json'),'utf8'));
input.resourceBudgetBytes=512*1024*1024;
// All refinements use the same physical sphere, vessel and fluid. This runner
// varies only dt, with a common conditioning step and common final time.
input.dt=2e-8;input.duration=1.6e-7;input.outputInterval=1.6e-7;
const fixture=join(out,'case.fns'),exe=join(out,'check-coupled-convergence');
await writeFile(join(out,'case.json'),JSON.stringify(input,null,2));
await writePreparedCase(prepareState(input),fixture);
const sourcePaths=['build.sh','check-coupled-convergence.cpp','fluid-runtime.cpp','rigid-coupler.cpp','rigid-contact.cpp','rigid-interpolation.cpp','../dns/faraday_cell3d.cpp','fluid-runtime.hpp','rigid-coupler.hpp','rigid-contact.hpp','rigid-interpolation.hpp','rigid-types.hpp','memory-budget.hpp'].sort();
const fingerprint=async()=>Object.fromEntries(await Promise.all(sourcePaths.map(async p=>[p,createHash('sha256').update(await readFile(join(here,p))).digest('hex')])));
const sources=await fingerprint();
const build=spawnSync(join(here,'build.sh'),[exe,join(here,'check-coupled-convergence.cpp'),join(here,'rigid-coupler.cpp'),join(here,'rigid-contact.cpp'),join(here,'rigid-interpolation.cpp')],{encoding:'utf8',maxBuffer:8*1024*1024});
await writeFile(join(out,'build.log'),(build.stdout??'')+(build.stderr??''));
process.stdout.write(build.stdout??'');process.stderr.write(build.stderr??'');
assert.equal(build.status,0,'Compilation failed; evidence: '+out);
assert.deepEqual(await fingerprint(),sources,'Numerical source changed while compilation was in progress; rerun from stable sources');
await writeFile(join(out,'compiled-sources.json'),JSON.stringify({compiler:process.env.CXX||'c++',sources,executableSha256:createHash('sha256').update(await readFile(exe)).digest('hex')},null,2));
const run=await new Promise((resolveRun,reject)=>{
  const child=spawn(exe,[fixture],{stdio:['ignore','pipe','pipe']});let stdout='',stderr='';
  child.stdout.setEncoding('utf8');child.stderr.setEncoding('utf8');
  child.stdout.on('data',chunk=>{stdout+=chunk;process.stdout.write(chunk);});
  child.stderr.on('data',chunk=>{stderr+=chunk;process.stderr.write(chunk);});
  child.on('error',reject);child.on('close',(status,signal)=>resolveRun({status,signal,stdout,stderr}));
});
await writeFile(join(out,'run.log'),run.stdout+run.stderr);
const entries=(run.stdout??'').split('\n').filter(s=>s.startsWith('{')).map(JSON.parse);
await writeFile(join(out,'results.json'),JSON.stringify({scope:'Fixed-physical-time temporal refinement; no deposition or grid-convergence claim',entries},null,2));
assert.equal(run.status,0,'Independent temporal verification failed; evidence: '+out);
assert.equal(entries.filter(x=>x.kind==='trajectory').length,4);
assert.equal(entries.filter(x=>x.kind==='temporal-refinement').length,3);
assert.equal(entries.at(-1)?.passed,true);
if(process.env.FARADAY_NATIVE_QA)console.log('Independent coupled temporal verification passed. Evidence: '+out);
else await rm(out,{recursive:true,force:true});
