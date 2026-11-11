import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { performance } from 'node:perf_hooks';
const require=createRequire(import.meta.url);
const T=require('./cell3d-tracers.js');
const OBS=require('./cell3d-observation.js');
const {FaradayCell3D}=require('./faraday-cell3d.js');
let passed=0;
function test(name,fn){fn();passed++;console.log(`  ok   ${name}`);}
function close(a,b,tolerance,message){assert(Math.abs(a-b)<=tolerance,`${message}: ${a} vs ${b}`);}
function frame(time,{R=1,h=2,generation=1,surface,velocity}={}){
  return {R,h,time,generation,
    surfaceAt(x,y,out={}){return Object.assign(out,surface?surface(x,y,time):{H:h,Ht:0,Hx:0,Hy:0});},
    heightAt(r,theta){return this.surfaceAt(r*Math.cos(theta),r*Math.sin(theta)).H;},
    velocityAt(x,y,z,out={}){return Object.assign(out,velocity?velocity(x,y,z,time):{x:0,y:0,z:0});}};
}
const zero=frame(0);
test('deterministic golden-distributed initialization lies inside the SI liquid domain',()=>{
  const a=T.create(zero),b=T.create(zero),q=a.snapshot(),positions=a.physicalPositions(zero);
  assert.equal(q.count,4096);assert.equal(q.seed,0x46524144);assert.deepEqual(q.coordinates,b.snapshot().coordinates);
  for(let p=0;p<q.count;p++){
    const i=3*p;assert(Math.hypot(positions[i],positions[i+1])<zero.R);
    assert(positions[i+2]>0&&positions[i+2]<zero.h);
    close(q.coordinates[i]**2+q.coordinates[i+1]**2,(p+.5)/q.count,5e-16,'equal-area radial stratum');
  }
});
test('analytic initial geometry can seed particles before any velocity grid exists',()=>{
  const initial={R:.015,h:.004,time:0,generation:8,heightAt:(r,th)=>.004+1e-6*Math.cos(th)*r/.015};
  const a=T.create(initial,{count:31}),xyz=a.physicalPositions(initial),q=a.snapshot();
  for(let p=0;p<31;p++){
    const i=3*p,r=Math.hypot(xyz[i],xyz[i+1]);
    close(xyz[i+2],.98*initial.heightAt(r,Math.atan2(xyz[i+1],xyz[i])),1e-18,'initial physical height');
  }
  assert.equal(q.generation,8);
  assert.throws(()=>a.step(initial,{...initial,time:.1},.1),/complete nonlinear velocity/);
});
test('zero flow leaves all coordinates exactly unchanged across physical steps',()=>{
  const a=T.create(zero,{count:129}),before=a.snapshot().coordinates;
  let old=zero;
  for(let j=1;j<=12;j++){const next=frame(j*.01);a.step(old,next,next.time-old.time);old=next;}
  assert.deepEqual(a.snapshot().coordinates,before);assert.equal(a.time,.12);
  assert.equal(a.snapshot().lastStep.refinements,0);
});
test('three-dimensional strain and rotation follow independent analytic trajectories',()=>{
  // Manufactured divergence-free interior field. This is an advection test,
  // not a claim that these fields solve the rigid-wall momentum problem.
  const strain=.4,omega=1.7,duration=.6;
  const options={surface:(_x,_y,t)=>({H:2*Math.exp(strain*t),Ht:strain*2*Math.exp(strain*t),Hx:0,Hy:0}),
    velocity:(x,y,z)=>({x:-strain*x/2-omega*y,y:omega*x-strain*y/2,z:strain*z})};
  const f0=frame(0,options),f1=frame(duration,options);
  const a=T.create(f0,{count:47,relativeTolerance:1e-10}),initial=a.physicalPositions(f0);
  a.step(f0,f1,duration);const actual=a.physicalPositions(f1);
  const c=Math.cos(omega*duration),s=Math.sin(omega*duration),decay=Math.exp(-strain*duration/2);
  for(let p=0;p<a.count;p++){
    const i=3*p,x=initial[i],y=initial[i+1];
    close(actual[i],decay*(x*c-y*s),3e-10,'analytic x');
    close(actual[i+1],decay*(x*s+y*c),3e-10,'analytic y');
    close(actual[i+2],initial[i+2]*Math.exp(strain*duration),3e-12,'analytic z');
  }
  assert(a.snapshot().lastStep.refinements>0);
});
test('uniform column motion preserves sigma and follows the free-surface kinematics',()=>{
  const options={surface:(_x,_y,t)=>({H:2+t,Ht:1,Hx:0,Hy:0}),velocity:(_x,_y,z,t)=>({x:0,y:0,z:z/(2+t)})};
  const f0=frame(0,options),f1=frame(.3,options),a=T.create(f0,{count:41}),initial=a.snapshot();
  a.step(f0,f1,.3);const q=a.snapshot(),xyz=a.physicalPositions(f1);
  assert.deepEqual(q.coordinates,initial.coordinates);
  for(let p=0;p<a.count;p++)close(xyz[3*p+2],.98*2.3,1e-15,'moving column height');
});
test('mapped velocity includes both horizontal surface-slope terms',()=>{
  const u=.001,v=-.0005;
  const options={surface:(x,y)=>({H:2+.1*x-.2*y,Ht:0,Hx:.1,Hy:-.2}),
    velocity:(x,y,z)=>({x:u,y:v,z:z/(2+.1*x-.2*y)*(u*.1-v*.2)})};
  const f0=frame(0,options),f1=frame(.2,options),a=T.create(f0,{count:43}),initial=a.snapshot().coordinates;
  a.step(f0,f1,.2);const q=a.snapshot().coordinates;
  for(let p=0;p<a.count;p++){
    const i=3*p;close(q[i],initial[i]+u*.2,3e-16,'slope x');
    close(q[i+1],initial[i+1]+v*.2,3e-16,'slope y');close(q[i+2],initial[i+2],2e-15,'slope sigma');
  }
});
test('time-varying solved velocity is interpolated over physical time',()=>{
  const options={velocity:(x,y,_z,t)=>({x:-(.8+.3*t)*y,y:(.8+.3*t)*x,z:0})};
  const f0=frame(0,options),f1=frame(.4,options),a=T.create(f0,{count:31,relativeTolerance:1e-10});
  const initial=a.snapshot().coordinates;a.step(f0,f1,.4);const q=a.snapshot().coordinates;
  const angle=.8*.4+.3*.4*.4/2,c=Math.cos(angle),s=Math.sin(angle);
  for(let p=0;p<a.count;p++){
    const i=3*p;close(q[i],initial[i]*c-initial[i+1]*s,2e-10,'time-varying x');
    close(q[i+1],initial[i]*s+initial[i+1]*c,2e-10,'time-varying y');
  }
});
test('a boundary-crossing stage is refined when the true trajectory remains inside',()=>{
  const options={velocity:(x)=>({x:-100*x,y:0,z:0})},f0=frame(0,options),f1=frame(.1,options);
  const a=T.create(f0,{count:19,relativeTolerance:1e-9}),initial=a.snapshot().coordinates;
  a.step(f0,f1,.1);const q=a.snapshot();assert(q.lastStep.refinements>0);
  for(let p=0;p<a.count;p++){
    const i=3*p;close(q.coordinates[i],initial[i]*Math.exp(-10),2e-10,'refined decay');
    assert.equal(q.coordinates[i+1],initial[i+1]);assert.equal(q.coordinates[i+2],initial[i+2]);
  }
});
test('genuine domain escape refuses transactionally without clipping or advancing the clock',()=>{
  const options={velocity:()=>({x:10,y:0,z:0})},f0=frame(0,options),f1=frame(1,options);
  const a=T.create(f0,{count:17,maxRefinements:8}),before=a.snapshot();
  assert.throws(()=>a.step(f0,f1,1),/leave the liquid domain.*no boundary clipping/);
  assert.deepEqual(a.snapshot(),before);
});
test('physical clock, generation, and frame consistency failures are explicit',()=>{
  const a=T.create(zero,{count:7}),before=a.snapshot();
  assert.throws(()=>a.step(zero,frame(.2),.1),/dt does not match/);
  assert.throws(()=>a.step(frame(.1),frame(.2),.1),/old fluid physical times differ/);
  assert.throws(()=>a.step(zero,frame(.1,{generation:2}),.1),/different solver generation/);
  assert.throws(()=>a.physicalPositions(frame(.1)),/matching fluid frame/);
  assert.deepEqual(a.snapshot(),before);
});
test('render batching and snapshot mutation cannot change the physical trajectory',()=>{
  const options={velocity:(x,y)=>({x:-.7*y,y:.7*x,z:0})};
  const f0=frame(0,options),a=T.create(f0,{count:57}),b=T.create(f0,{count:57});let old=f0;
  for(let j=1;j<=20;j++){
    const next=frame(j*.005,options),dt=next.time-old.time;a.step(old,next,dt);b.step(old,next,dt);
    a.physicalPositions(next).fill(NaN);a.snapshot().coordinates.fill(NaN);old=next;
  }
  assert.deepEqual(a.snapshot(),b.snapshot());assert.deepEqual(a.physicalPositions(old),b.physicalPositions(old));
});
test('re-sprinkle is reproducible at the current solved time without advancing it',()=>{
  const a=T.create(zero,{count:53}),next=frame(.1);a.step(zero,next,.1);const before=a.snapshot();
  a.reseed(next,{seed:987});const reseeded=a.snapshot();assert.equal(reseeded.time,.1);assert.equal(reseeded.seed,987);
  assert.notDeepEqual(reseeded.coordinates,before.coordinates);
  const b=T.create(next,{count:53,seed:987});assert.deepEqual(reseeded.coordinates,b.snapshot().coordinates);
});
test('actual staggered solver observations advect tracers without mutating fluid arrays',()=>{
  const S=new FaradayCell3D({nr:6,nth:16,nz:6,R:.015,h:.004,rho:1000,nu:1e-6,gamma:.0728,
    g:9.80665,accel:7,omegaD:2*Math.PI*111,contact:'pinned'});
  for(let i=0;i<S.nr;i++)for(let k=0;k<S.nth;k++)S.eta[S.ie(i,k)]=1e-7*(1-S.rc[i]/S.R)*Math.cos(2*(k+.5)*S.dth);
  S.refreshMetric();const old=OBS.createObserver(S,{generation:4}),a=T.create(old,{count:257});
  const dt=S.stableStep();S.step(dt);const next=OBS.createObserver(S,{generation:4});
  const fields=['eta','Ht','u','v','w'].map(key=>[key,Float64Array.from(S[key])]);
  const started=performance.now();a.step(old,next,dt);const elapsed=performance.now()-started;
  for(const [key,values] of fields)assert.deepEqual(S[key],values);
  const xyz=a.physicalPositions(next);assert(xyz.every(Number.isFinite));assert.equal(a.time,S.t);
  assert(a.snapshot().coordinates.some((v,i)=>v!==T.create(old,{count:257}).snapshot().coordinates[i]));
  console.log(`       actual ${S.nr}×${S.nth}×${S.nz} field, 257 tracers: ${elapsed.toFixed(1)} ms`);
});
console.log(`${passed} tracer checks passed, 0 failed`);
