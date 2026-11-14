import assert from 'node:assert/strict';
import {mkdtemp,rm,readFile,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname,join,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
import {prepareState,writePreparedCase} from './prepare-case.mjs';
import RECORDING from '../faraday/native-recording.js';

const here=dirname(fileURLToPath(import.meta.url));
const dir=await mkdtemp(join(tmpdir(),'faraday-native-cli-'));
const executable=join(dir,'faraday-native'),prepared=join(dir,'case.fns');
const invoke=args=>spawnSync(executable,args,{encoding:'utf8',timeout:60000,maxBuffer:8*1024*1024});
try{
  const build=spawnSync(join(here,'build.sh'),[executable,...['run.cpp','rigid-coupler.cpp','rigid-contact.cpp','rigid-interpolation.cpp'].map(x=>join(here,x))],{encoding:'utf8',timeout:120000});
  assert.equal(build.status,0,build.stderr);
  const c=JSON.parse(await readFile(join(here,'examples/single-sphere-verification.json'),'utf8'));
  Object.assign(c.grid,{nr:4,nth:8,nz:4});c.grains=[];
  c.duration=1e-8;c.dt=5e-9;c.outputInterval=1e-22;
  c.materialProvenance.description='Zero-grain command-line serialization regression; no sediment validation.';
  await writePreparedCase(prepareState(c),prepared);
  const run=invoke([prepared]);assert.equal(run.status,0,run.stderr||String(run.error));
  const events=[...RECORDING.iterate(run.stdout)];
  assert.equal(events.at(-1).type,'complete');
  const frames=events.filter(x=>x.type==='frame').map(x=>x.frame);
  assert.equal(frames.length,3);assert.deepEqual(frames.map(f=>f.time),[0,5e-9,1e-8]);
  assert(frames.every(f=>f.particles.length===0));
  console.log('PASS exact accepted frames with output interval much smaller than dt; no threshold-iteration hang');
  const bytes=await readFile(prepared),tampered=join(dir,'unrepresentable.fns');
  bytes.writeDoubleLE(1e-300,36+11*8);await writeFile(tampered,bytes);
  const refused=invoke([tampered]);assert.equal(refused.status,1);
  const error=JSON.parse(refused.stdout);assert.equal(error.type,'error');assert.match(error.message,/time controls/);
  console.log('PASS native reader independently rejects unrepresentable output schedules');
  const truncated=join(dir,'truncated.fns');await writeFile(truncated,bytes.subarray(0,40));
  const partial=invoke([truncated]);assert.equal(partial.status,1);assert.match(JSON.parse(partial.stdout).message,/Truncated/);
  const usage=invoke([]);assert.equal(usage.status,1);assert.match(JSON.parse(usage.stdout).message,/Usage/);
  console.log('PASS truncated input and missing arguments produce parseable refusal records');
  console.log('3 native CLI checks passed');
}finally{await rm(dir,{recursive:true,force:true});}
