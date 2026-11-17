// Coupling mechanics and transactional acceptance checks. These do NOT establish
// fine-beach-sand deposition, measured-material accuracy or mesh convergence.
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,mkdir,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname,resolve,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
import {prepareState,writePreparedCase} from './prepare-case.mjs';

const here=dirname(fileURLToPath(import.meta.url));
const out=process.env.FARADAY_NATIVE_QA?resolve(process.env.FARADAY_NATIVE_QA):await mkdtemp(join(tmpdir(),'faraday-coupler-'));
await mkdir(out,{recursive:true});
const input={format:'faraday-native-case',version:1,
  fluid:{radius:.0005,depth:.0005,density:1000,dynamicViscosity:.001,surfaceTension:.0728,gravity:9.80665,contactLine:'pinned'},
  drive:{frequency:111,acceleration:7},grid:{nr:20,nth:64,nz:20,rStretch:0,zStretch:0},initial:{type:'flat'},
  grains:[{id:0,centre:[0,0,.00025],radius:.000125,density:2650,velocity:[0,0,0],omega:[0,0,0]}],
  materialProvenance:{kind:'illustrative',description:'Declared 250 micrometre spherical grain in a 1mm-diameter by 0.5mm-depth verification vessel. This is not the user vessel or a measured beach sample.',source:'Numerical mechanics regression fixture; no sample measurement claimed'},
  contact:{friction:.5,restitution:.3},resourceBudgetBytes:512*1024*1024,duration:1e-7,dt:5e-9,outputInterval:1e-7};
const fixture=join(out,'verification-case.fns'),exe=join(out,'check-coupler');
await writeFile(join(out,'verification-case.json'),JSON.stringify(input,null,2));
await writePreparedCase(prepareState(input),fixture);
const build=spawnSync(join(here,'build.sh'),[exe,join(here,'check-coupler.cpp'),join(here,'rigid-coupler.cpp'),join(here,'rigid-contact.cpp'),join(here,'rigid-interpolation.cpp')],{encoding:'utf8',maxBuffer:8*1024*1024});
process.stdout.write(build.stdout??'');process.stderr.write(build.stderr??'');
await writeFile(join(out,'build.log'),(build.stdout??'')+(build.stderr??''));
assert.equal(build.status,0,'Native coupling check compilation failed; evidence: '+out);
const run=spawnSync(exe,[fixture],{encoding:'utf8',maxBuffer:8*1024*1024});
process.stdout.write(run.stdout??'');process.stderr.write(run.stderr??'');
await writeFile(join(out,'check-coupler.log'),(run.stdout??'')+(run.stderr??''));
const entries=(run.stdout??'').split('\n').filter(s=>s.startsWith('{')).map(s=>JSON.parse(s));
await writeFile(join(out,'results.json'),JSON.stringify({scope:'Coupling algebra, rollback and initial/finite-speed motion in the declared small vessel; not hydrodynamic convergence or sand deposition validation',entries},null,2));
assert.equal(run.status,0,'Native coupling regression failed; evidence: '+out);
const summary=entries.find(x=>x.summary);assert(summary&&summary.failed===0&&summary.passed===10,'Missing complete native coupling result');
if(process.env.FARADAY_NATIVE_QA)console.log('Native coupling checks passed. Evidence retained: '+out);
else{await rm(out,{recursive:true,force:true});console.log('Native coupling checks passed. Temporary evidence removed; set FARADAY_NATIVE_QA to retain it.');}
