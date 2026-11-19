import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
const require=createRequire(import.meta.url);
const N=require('./native-recording.js'),O=require('../dns/cell3d-observation.js');
const {FaradayCell3D}=require('../dns/faraday-cell3d.js');
let checks=0;
const check=(name,fn)=>{fn();checks++;console.log('  ok  '+name);};
const copy=x=>JSON.parse(JSON.stringify(x));
const lines=records=>records.map(x=>JSON.stringify(x)).join('\n')+'\n';
const events=records=>[...N.iterate(lines(records),{generation:19,maxFlowGlyphs:8})];
function fixture({rStretch=0,zStretch=0}={}){
  // A prescribed serialization fixture. The native executable integration is
  // checked separately below when its actual output path is supplied.
  const S=new FaradayCell3D({nr:4,nth:8,nz:4,R:.02,h:.004,rho:1000,nu:1e-6,gamma:.0728,g:9.80665,accel:0,omegaD:0,contact:'pinned',rStretch,zStretch});
  const grain={id:17,centre:[.003,.002,.002],radius:.0001,density:2650,velocity:[0,0,0],omega:[0,0,0]};
  const c={format:'faraday-native-case',version:1,fluid:{radius:S.R,depth:S.h,density:S.rho,dynamicViscosity:S.rho*S.nu,surfaceTension:S.gamma,gravity:9.80665,contactLine:'pinned'},drive:{frequency:0,acceleration:0},
    grid:{nr:4,nth:8,nz:4,rStretch,zStretch},initial:{type:'flat'},grains:[grain],materialProvenance:{kind:'illustrative',description:'Serialization test sphere',source:'test fixture; not measured beach sand'},contact:{friction:.4,restitution:.5},resourceBudgetBytes:100000000,duration:.0001,dt:.0001,outputInterval:.0001};
  const header={type:'native-case',format:N.FORMAT,version:N.VERSION,caseJson:JSON.stringify(c),nativeFluidBytes:100000};
  const f={type:'frame',time:0,steps:0,initial:true,nr:S.nr,nth:S.nth,nz:S.nz,R:S.R,h:S.h,contact:'pinned',grains:[copy(grain)],diagnostics:{dt:0,slipResidual:0,divergence:0,constraintIterations:0,markers:0,eventRefinements:0,contacts:0}};
  for(const k of ['rf','rc','sf','sc','eta','Ht','u','v','w'])f[k]=Array.from(S[k]);
  const next=copy(f);next.time=c.duration;next.steps=1;next.initial=false;next.grains[0].centre[2]-=.000001;next.grains[0].velocity[2]=-.01;next.diagnostics={dt:c.dt,slipResidual:2e-12,divergence:1e-13,constraintIterations:4,markers:10,eventRefinements:0,contacts:0};
  return {header,initial:f,next,complete:{type:'complete',time:next.time,steps:next.steps},case:c};
}
check('actual frame arrays, physical grain radii and diagnostics share the native clock',()=>{
  const f=fixture(),out=events([f.header,f.initial,f.next,f.complete]);
  assert.deepEqual(out.map(x=>x.type),['header','frame','frame','complete']);
  const a=out[1].frame,b=out[2].frame;
  assert.equal(a.time,0);assert.equal(b.time,f.next.time);assert.equal(b.steps,1);assert.equal(b.generation,19);
  assert(b.particles instanceof Float64Array);assert(b.particleRadii instanceof Float64Array);
  assert.equal(b.particleRadii[0],.0001);assert.equal(b.particles[2],f.next.grains[0].centre[2]);
  assert.equal(b.grainVelocities[2],-.01);assert.equal(b.grainIds[0],17);assert.equal(b.grainDensity[0],2650);
  assert.equal(b.grainProvenance.physicalTime,b.time);assert.equal(b.provenance.physicalTime,b.time);
  assert.equal(b.source.diagnostics.slipResidual,2e-12);assert.equal(b.source.pressureAvailable,false);
  assert.equal(b.grainProvenance.geometry,'spheres');assert.equal(b.tracerProvenance,undefined);
  assert.equal(b.units.position,'m');assert.equal(b.units.radius,'m');
});
check('chunk boundaries including split JSON strings preserve complete records',()=>{
  const f=fixture(),text=lines([f.header,f.initial,f.next,f.complete]);
  const reader=N.createSession({maxFlowGlyphs:0}),out=[];
  for(let i=0;i<text.length;i+=17)out.push(...reader.feed(text.slice(i,i+17)));
  out.push(...reader.finish());assert.equal(out.length,4);assert.equal(reader.status().terminal,'complete');assert.equal(reader.status().closed,true);
  assert.equal(reader.lastFrame().time,f.next.time);
});
check('returned snapshots and provenance cannot mutate the retained native frame',()=>{
  const f=fixture(),s=N.createSession({maxFlowGlyphs:0}),out=[...s.feed(lines([f.header,f.initial,f.next]))];
  out[0].header.case.fluid.density=1;out[2].frame.particles.fill(9);out[2].frame.particleRadii.fill(9);
  out[2].frame.source.diagnostics.dt=9;out[2].frame.grainProvenance.material.kind='measured';
  const a=s.lastFrame();assert.equal(a.particleRadii[0],.0001);assert.equal(a.particles[2],f.next.grains[0].centre[2]);assert.equal(a.source.diagnostics.dt,.0001);
  assert.equal(a.grainProvenance.material.kind,'illustrative');assert.equal(s.header().case.fluid.density,1000);
  a.particles.fill(4);assert.equal(s.lastFrame().particles[0],.003);
});
check('radii, grain count, identity and immutable material properties are validated',()=>{
  for(const change of [f=>{f.grains[0].radius=0;},f=>{f.grains[0].radius=-1;},f=>{f.grains=[];},f=>{f.grains[0].id=18;},f=>{f.grains[0].density=2700;},f=>{f.grains[0].radius=.0002;},f=>{f.grains[0].velocity=[0,1];}]){
    const f=fixture();change(f.next);assert.throws(()=>events([f.header,f.initial,f.next,f.complete]));
  }
});
check('recordings reject backwards, repeated and mismatched clocks',()=>{
  for(const change of [f=>{f.next.time=0;},f=>{f.next.steps=0;},f=>{f.next.initial=true;},f=>{f.complete.time=.00009;},f=>{f.complete.steps=2;},f=>{f.initial.time=.00001;}]){
    const f=fixture();change(f);assert.throws(()=>events([f.header,f.initial,f.next,f.complete]));
  }
});
check('staggered array counts and native grid geometry cannot drift between frames',()=>{
  for(const change of [f=>{f.u.pop();},f=>{f.rf[1]=f.rf[0];},f=>{f.rc[1]*=1.01;},f=>{f.h*=2;},f=>{f.eta[0]=null;},f=>{f.nz=5;}]){
    const f=fixture();change(f.next);assert.throws(()=>events([f.header,f.initial,f.next,f.complete]));
  }
});
check('first-frame coordinates match the declared uniform or graded native mapping',()=>{
  for(const grading of [{rStretch:0,zStretch:0},{rStretch:2.2,zStretch:1.4},{rStretch:4,zStretch:3}]){
    const f=fixture(grading);assert.equal(events([f.header,f.initial,f.next,f.complete]).length,4);
    for(const name of ['rf','rc','sf','sc']){
      const bad=copy(f);bad.initial[name][1]*=.95;bad.next[name][1]*=.95;
      assert.throws(()=>events([bad.header,bad.initial,bad.next,bad.complete]),/declared grid mapping/);
    }
  }
  const f=fixture({rStretch:2.2,zStretch:2.2});
  f.initial.rf[1]+=2*Number.EPSILON*f.case.fluid.radius;f.next.rf[1]=f.initial.rf[1];
  assert.equal(events([f.header,f.initial,f.next,f.complete]).length,4,'roundoff allowance must not rewrite supplied coordinates');
});
check('array and JSON budgets refuse oversized inputs before typed-array conversion',()=>{
  const f=fixture();assert.throws(()=>[...N.iterate(lines([f.header,f.initial]),{maxCells:64})],/cell count/);
  assert.throws(()=>[...N.iterate(lines([f.header,f.initial]),{maxArrayElements:130})],/array limit/);
  assert.throws(()=>[...N.iterate(lines([f.header,f.initial]),{maxFrameBytes:64,maxFlowGlyphs:0})],/byte budget/);
  assert.throws(()=>[...N.iterate(lines([f.header]),{maxLineChars:32})],/before JSON parsing/);
  const c=copy(f.case);c.grid.nr=Number.MAX_SAFE_INTEGER;f.header.caseJson=JSON.stringify(c);
  assert.throws(()=>[...N.iterate(lines([f.header]))],/cell count/);
});
check('observation arrays and attached grain fields also fit the frame byte budget',()=>{
  const f=fixture(),records=lines([f.header,f.initial,f.next,f.complete]);
  const output=[...N.iterate(records,{maxFlowGlyphs:128})].find(e=>e.type==='frame').frame;
  const emittedBytes=Object.values(output).filter(ArrayBuffer.isView).reduce((sum,value)=>sum+value.byteLength,0);
  assert.equal(emittedBytes,5204);
  assert.throws(()=>[...N.iterate(records,{maxFlowGlyphs:128,maxFrameBytes:emittedBytes-1})],/byte budget/);
  assert.equal([...N.iterate(records,{maxFlowGlyphs:128,maxFrameBytes:emittedBytes})].length,4);
  assert.throws(()=>[...N.iterate(records,{maxFlowGlyphs:0,maxArrayElements:200})],/Observed.*array limit/);
});
check('a native error retains the last coherent frame and does not claim completion',()=>{
  const f=fixture(),s=N.createSession({maxFlowGlyphs:0});
  const out=[...s.feed(lines([f.header,f.initial,{type:'error',message:'Unresolved grain gap'}])),...s.finish()];
  assert.equal(out.at(-1).type,'error');assert.equal(s.status().terminal,'error');assert.equal(s.status().failed,true);
  assert.equal(s.lastFrame().time,0);assert.equal(s.status().frames,1);
  const pre=events([{type:'error',message:'Case refused before allocation'}]);assert.equal(pre[0].type,'error');
});
check('truncated, unterminated and extra records fail while preserving accepted frames',()=>{
  const f=fixture(),s=N.createSession({maxFlowGlyphs:0});[...s.feed(lines([f.header,f.initial])+'{"type":"frame"')];
  assert.throws(()=>[...s.finish()],/not complete JSON/);assert.equal(s.lastFrame().time,0);assert.equal(s.status().failed,true);
  assert.throws(()=>events([f.header,f.initial]),/without a complete/);
  assert.throws(()=>events([f.header,f.initial,f.next,f.complete,f.next]),/after/);
});
check('unconsumed generator events close the session instead of silently dropping input',()=>{
  const f=fixture(),s=N.createSession({maxFlowGlyphs:0});
  for(const event of s.feed(lines([f.header,f.initial]))){assert.equal(event.type,'header');break;}
  assert.equal(s.status().closed,true);assert.equal(s.status().failed,true);assert.throws(()=>[...s.feed('')],/closed/);
});
check('unfinished EOF events lock the reader and abandonment closes it as incomplete',()=>{
  const f=fixture(),s=N.createSession({maxFlowGlyphs:0});
  [...s.feed(lines([f.header])+JSON.stringify(f.initial))];
  const eof=s.finish();assert.equal(eof.next().value.type,'frame');
  assert.throws(()=>[...s.feed(lines([f.next]))],/previous.*generator/);
  assert.throws(()=>[...s.finish()],/pending.*events/);
  eof.return();assert.equal(s.status().closed,true);assert.equal(s.status().failed,true);
  assert.match(s.status().error,/EOF was not accepted/);assert.equal(s.lastFrame().time,0);
  assert.throws(()=>[...s.feed(lines([f.next,f.complete]))],/closed/);
  const failed=N.createSession();
  for(const event of failed.feed(lines([{type:'error',message:'Native run refused'}]))){assert.equal(event.type,'error');break;}
  assert.equal(failed.status().closed,true);assert.equal(failed.status().error,'Native run refused');
});
check('header provenance is strict and untrusted strings remain data',()=>{
  const f=fixture();const bad=copy(f.header);bad.version=2;assert.throws(()=>events([bad]),/supported/);
  bad.version=1;bad.caseJson='{}';assert.throws(()=>events([bad]),/required/);
  const c=copy(f.case);c.materialProvenance.source='<img src=x onerror=alert(1)>';f.header.caseJson=JSON.stringify(c);
  const out=events([f.header,f.initial,f.next,f.complete]);assert.equal(out[2].frame.grainProvenance.material.source,c.materialProvenance.source);
  assert.match(out[0].header.scope,/Experimental/);assert.equal(out[0].header.pressureAvailable,false);
});
check('UMD browser module loads with the existing observation provider and no DOM/network',()=>{
  const context=vm.createContext({FARADAY_CELL3D_OBSERVATION:O});
  new vm.Script(require('node:fs').readFileSync(new URL('./native-recording.js',import.meta.url),'utf8')).runInContext(context);
  assert.equal(typeof context.FARADAY_NATIVE_RECORDING.createSession,'function');
});
{
  const f=fixture();f.case.materialProvenance.description='Silica test — μm';f.header.caseJson=JSON.stringify(f.case);
  const bytes=new TextEncoder().encode(lines([f.header,f.initial,f.next,f.complete]));
  async function* chunks(){for(let i=0;i<bytes.length;i+=7)yield bytes.subarray(i,i+7);}
  const out=[];for await(const event of N.iterateStream(chunks(),{maxFlowGlyphs:0}))out.push(event);
  assert.equal(out[0].header.case.materialProvenance.description,'Silica test — μm');assert.equal(out.at(-1).type,'complete');
  checks++;console.log('  ok  streamed UTF-8 split across multibyte characters is decoded exactly');
}
if(process.argv[2]){
  const recording=await readFile(process.argv[2],'utf8'),out=[...N.iterate(recording,{generation:52,maxFlowGlyphs:32})];
  const frames=out.filter(x=>x.type==='frame');assert(frames.length>0,'Native executable did not emit a frame');
  assert(['complete','error'].includes(out.at(-1).type));
  for(const {frame} of frames){assert.equal(frame.source.engine,'native-cpp-recording');assert.equal(frame.generation,52);assert.equal(frame.particles.length/3,frame.particleRadii.length);}
  checks++;console.log('  ok  actual native executable recording: '+frames.length+' frames, '+out.at(-1).type);
}
console.log(checks+' native recording checks passed'+(process.argv[2]?' including native output.':'; pass a native JSONL path to additionally verify executable output.'));
