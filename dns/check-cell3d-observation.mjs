import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
const require=createRequire(import.meta.url);
const {FaradayCell3D}=require('./faraday-cell3d.js');
const {FaradayCell3DWasm}=require('./faraday-cell3d-wasm.js');
const OBS=require('./cell3d-observation.js');

let passed=0;
function check(name,fn){fn();passed++;console.log('  ok  '+name);}
function close(a,b,tol=1e-12){assert(Math.abs(a-b)<=tol,`${a} differs from ${b} by ${Math.abs(a-b)} > ${tol}`);}
function cell(nr=12,nth=32,nz=12){
  return new FaradayCell3D({nr,nth,nz,R:.015,h:.004,rho:1000,nu:1e-6,gamma:.0728,
    g:9.80665,accel:0,omegaD:2*Math.PI*111,contact:'pinned',rStretch:0,zStretch:0});
}
function deform(S){
  for(let i=0;i<S.nr;i++)for(let k=0;k<S.nth;k++){
    const r=S.rc[i]/S.R,th=(k+.5)*S.dth,e=S.ie(i,k);
    S.eta[e]=.2*S.h*r*(1-r*r)*Math.cos(th);
    S.Ht[e]=.3*S.eta[e];
  }
  S.refreshMetric();return S;
}
function fillVelocity(S,fn){
  for(const key of ['u','v','w']){
    const f=S.FAM[key];
    for(let i=0;i<f.rn.length;i++)for(let k=0;k<S.nth;k++)for(let j=0;j<f.sn.length;j++){
      const r=f.rn[i],th=(k+f.thOff)*S.dth,H=S.HatH(r,th),z=f.sn[j]*H;
      const velocity=fn(r*Math.cos(th),r*Math.sin(th),z);
      S[key][f.idx(i,k,j)]=key==='u'?velocity.x*Math.cos(th)+velocity.y*Math.sin(th)
        :key==='v'?-velocity.x*Math.sin(th)+velocity.y*Math.cos(th):velocity.z;
    }
  }
  return S;
}
const norm=v=>Math.hypot(v.x,v.y,v.z);
const subtract=(a,b)=>({x:a.x-b.x,y:a.y-b.y,z:a.z-b.z});

check('zero flow produces exact zero physical vectors and a nonempty full surface mesh',()=>{
  const S=cell(),frame=OBS.createFrame(S,{initial:true,maxFlowGlyphs:97,generation:7});
  assert(frame.flowVectors.every(v=>v===0));assert.equal(frame.maxSpeed,0);
  assert.equal(frame.positions.length,3*(1+(S.nr+1)*S.nth));
  assert.equal(frame.indices.length,3*S.nth*(1+2*S.nr));
  assert(frame.flowPositions.length>0&&frame.flowPositions.length/3<=97);
  assert.equal(frame.generation,7);assert.equal(frame.initial,true);
  assert.equal(frame.flowSampling.totalCells,S.nr*S.nth*S.nz);
  assert.equal(frame.flowSampling.displayOnly,true);
});
check('all surface triangles are positively oriented and the periodic seam is closed',()=>{
  const S=cell(),f=OBS.createFrame(S,{maxFlowGlyphs:0}),P=f.positions;
  const edges=new Map();
  for(let i=0;i<f.indices.length;i+=3){
    const a=f.indices[i],b=f.indices[i+1],c=f.indices[i+2];
    assert(a<P.length/3&&b<P.length/3&&c<P.length/3);
    const area=(P[3*b]-P[3*a])*(P[3*c+1]-P[3*a+1])-(P[3*b+1]-P[3*a+1])*(P[3*c]-P[3*a]);
    assert(area>0);
    for(const [x,y]of[[a,b],[b,c],[c,a]]){const key=[Math.min(x,y),Math.max(x,y)].join(':');edges.set(key,(edges.get(key)||0)+1);}
  }
  const boundary=[...edges.values()].filter(v=>v===1);assert.equal(boundary.length,S.nth);
  assert([...edges.values()].every(v=>v===1||v===2));
});
check('surface vertices preserve every stored polar surface cell and pinned rim',()=>{
  const S=deform(cell()),f=OBS.createFrame(S,{maxFlowGlyphs:0});
  for(let i=0;i<S.nr;i++)for(let k=0;k<S.nth;k++){
    const vertex=1+i*S.nth+k;
    close(f.surfaceDisplacement[vertex],S.eta[S.ie(i,k)],2e-11);
    close(f.positions[3*vertex+2],S.h+S.eta[S.ie(i,k)],3e-10);
  }
  for(let k=0;k<S.nth;k++)close(f.surfaceDisplacement[1+S.nr*S.nth+k],0,0);
});
check('eta and rate reconstruction match native samples and the existing outer-ring interpolation',()=>{
  const S=deform(cell()),O=OBS.createObserver(S);
  for(const r of [S.rc[0],.07*S.R,.35*S.R,.91*S.R,S.R])for(const th of [-.1,.4,3.2,6.7]){
    close(O.etaAt(r,th),S.etaAt(r,th),2e-18);
    close(O.etaDotAt(r,th),S.etaDotAt(r,th),2e-18);
  }
});
check('smooth even azimuthal modes have a regular surface and bounded gradients approaching the axis',()=>{
  const S=cell(8,32,8),amplitude=1e-5;
  for(let i=0;i<S.nr;i++)for(let k=0;k<S.nth;k++){
    const r=S.rc[i],th=(k+.5)*S.dth,e=S.ie(i,k);
    S.eta[e]=amplitude*(r/S.R)**2*Math.cos(2*th);S.Ht[e]=2*S.eta[e];
  }
  S.refreshMetric();const O=OBS.createObserver(S);
  for(const th of [.13,.37,1.1,2.7])for(const fraction of [1e-3,1e-6,1e-9]){
    const r=fraction*S.R,x=r*Math.cos(th),y=r*Math.sin(th),v=O.surfaceAt(x,y);
    const e=O.etaAt(r,th);
    close(v.H,O.heightAt(r,th),1e-18);close(v.Ht,2*e,1e-18);
    assert(Math.hypot(v.Hx,v.Hy)<=2.5*amplitude*r/(S.R*S.R)+1e-16,
      `Spurious singular axis gradient ${JSON.stringify(v)} at r=${r}`);
    close(e,amplitude*(r/S.R)**2*Math.cos(2*th),.03*amplitude*(r/S.R)**2+1e-20);
  }
  close(O.heightAt(0,0),O.surfaceAt(0,0).H,0);
  close(O.volumeDisplacement(0,0,S.h/2),.5*(O.heightAt(0,0)-S.h),1e-19);
  const f=O.frame({maxFlowGlyphs:0,particles:new Float64Array([0,0,S.h/2])});
  close(f.surfaceDisplacement[0],O.etaAt(0,0),1e-25);
});
check('smooth even-mode vertical velocity tends continuously to its axis value',()=>{
  const S=cell(8,32,8);
  fillVelocity(S,(x,y,z)=>({x:0,y:0,z:.01*(x*x-y*y)/(S.R*S.R)*(z/S.h)}));
  const O=OBS.createObserver(S);
  for(const th of [.13,.37,1.1,2.7])for(const fraction of [1e-3,1e-6,1e-9]){
    const r=fraction*S.R,z=.5*S.h,v=O.velocityAt(r*Math.cos(th),r*Math.sin(th),z);
    const expected=.005*fraction*fraction*Math.cos(2*th);
    close(v.z,expected,.03*.005*fraction*fraction+1e-16);
  }
  close(O.velocityAt(0,0,.5*S.h).z,0,1e-16);
});
check('surface slopes are analytic derivatives of the same interpolated height',()=>{
  const S=deform(cell()),O=OBS.createObserver(S),r=.372*S.R,th=.51;
  const x=r*Math.cos(th),y=r*Math.sin(th),a=O.surfaceAt(x,y),delta=1e-8*S.R;
  const value=(xx,yy)=>O.heightAt(Math.hypot(xx,yy),Math.atan2(yy,xx));
  close(a.H,value(x,y),2e-18);
  close(a.Hx,(value(x+delta,y)-value(x-delta,y))/(2*delta),2e-8);
  close(a.Hy,(value(x,y+delta)-value(x,y-delta))/(2*delta),2e-8);
  const axis=O.surfaceAt(0,0);assert(Object.values(axis).every(Number.isFinite));
  close(axis.H,S.h,1e-17);close(axis.Ht,0,1e-17);
});
check('common-physical-height reconstruction is exact for a cubic vertical field on a deformed mesh',()=>{
  const S=deform(cell());fillVelocity(S,(x,y,z)=>({x:0,y:0,z:(z/S.h)**3}));
  const O=OBS.createObserver(S),r=.427*S.R,th=.37,z=.61*S.h;
  close(O.velocityAt(r*Math.cos(th),r*Math.sin(th),z).z,(z/S.h)**3,2e-14);
});
check('Cartesian constant interior flow converges across the staggered angular families',()=>{
  const S=cell(16,96,16),exact={x:.3,y:-.2,z:.1};fillVelocity(S,()=>exact);
  const O=OBS.createObserver(S);
  for(const th of [.15,.72,2.15,5.9]){
    const v=O.velocityAt(.47*S.R*Math.cos(th),.47*S.R*Math.sin(th),.6*S.h);
    assert(norm(subtract(v,exact))/norm(exact)<.0007);
  }
});
check('the axis reconstructs a unique Cartesian vector and finite vertical velocity',()=>{
  const S=cell(12,32,12),exact={x:.3,y:-.2,z:.1};fillVelocity(S,()=>exact);
  const v=OBS.createObserver(S).velocityAt(0,0,.6*S.h);
  close(v.x,exact.x,1e-14);close(v.y,exact.y,1e-14);close(v.z,exact.z,1e-14);
});
check('velocity is continuous through the angular seam',()=>{
  const S=deform(cell(16,64,16));fillVelocity(S,(x,y,z)=>({x:z/S.h,y:.2*z/S.h,z:.1*z/S.h}));
  const O=OBS.createObserver(S),x=.4*S.R,eps=1e-10*S.R;
  assert(norm(subtract(O.velocityAt(x,eps,.5*S.h),O.velocityAt(x,-eps,.5*S.h)))<1e-8);
  close(O.etaAt(.4*S.R,-.2),O.etaAt(.4*S.R,2*Math.PI-.2),1e-18);
});
check('manufactured smooth no-slip velocity shows second-order spatial convergence',()=>{
  const error=n=>{
    const S=deform(cell(n,2*n,n));
    const exact=(x,y,z)=>{const shape=1-(x*x+y*y)/(S.R*S.R),q=z/S.h;
      return{x:.01*shape*q*(1-.2*q),y:.007*shape*q,z:.002*shape*q*q};};
    fillVelocity(S,exact);const O=OBS.createObserver(S);
    let error=0;
    for(const [r,th,q]of[[.31,.45,.41],[.57,1.23,.53],[.72,2.69,.63],[.41,5.42,.33]]){
      const x=r*S.R*Math.cos(th),y=r*S.R*Math.sin(th),z=q*S.h;
      error+=norm(subtract(O.velocityAt(x,y,z),exact(x,y,z)))**2;
    }
    return Math.sqrt(error);
  };
  const a=error(12),b=error(24);assert(a/b>2.8,`Refinement ratio ${a/b} is below second-order expectation`);
});
check('no-slip floor and wall observations are exact boundary values',()=>{
  const S=cell();fillVelocity(S,()=>({x:1,y:2,z:3}));const O=OBS.createObserver(S);
  assert.deepEqual(O.velocityAt(.3*S.R,.2*S.R,0),{x:0,y:0,z:0});
  assert.deepEqual(O.velocityAt(S.R,0,.5*S.h),{x:0,y:0,z:0});
});
check('out-of-domain and nonfinite queries are refused rather than projected or clipped',()=>{
  const O=OBS.createObserver(cell());
  for(const args of [[O.R*1.001,0,O.h/2],[0,0,-1e-9],[0,0,O.h+1e-9],[NaN,0,0]])
    assert.throws(()=>O.velocityAt(...args));
  assert.throws(()=>O.etaAt(-1e-8,0));assert.throws(()=>O.etaAt(O.R*1.01,0));
});
check('observer capture is isolated from live solver mutations and exported snapshot mutations',()=>{
  const S=deform(cell()),O=OBS.createObserver(S,{time:.125,generation:8,steps:17});
  const point=O.etaAt(.5*S.R,.7),snap=O.snapshot();
  S.eta.fill(1);S.u.fill(100);snap.eta.fill(2);snap.u.fill(200);
  close(O.etaAt(.5*S.R,.7),point,0);assert.equal(norm(O.velocityAt(0,0,.5*S.h)),0);
  assert.equal(O.time,.125);assert.equal(O.generation,8);assert.equal(O.steps,17);
});
check('creating a frame does not mutate any solver field',()=>{
  const S=deform(cell()),before=['eta','Ht','u','v','w'].map(k=>Buffer.from(S[k].buffer).toString('hex'));
  OBS.createFrame(S);
  assert.deepEqual(['eta','Ht','u','v','w'].map(k=>Buffer.from(S[k].buffer).toString('hex')),before);
});
check('particle positions remain metric copies and carry sigma-times-elevation display displacement',()=>{
  const S=deform(cell()),O=OBS.createObserver(S),x=.43*S.R,y=.1*S.R,H=O.surfaceAt(x,y).H;
  const input=new Float64Array([x,y,.35*H]),frame=O.frame({particles:input,maxFlowGlyphs:0});
  close(frame.particleDisplacement[0],.35*(H-S.h),2e-11);
  assert.deepEqual(frame.particles,input);input[0]=0;assert.equal(frame.particles[0],x);
  assert.throws(()=>O.frame({particles:new Float64Array([2*S.R,0,S.h/2])}));
});
check('incomplete, nonfinite or dry physical snapshots are refused',()=>{
  const S=cell();assert.throws(()=>OBS.capture(S,{source:{...S,w:new Float64Array(0)}}));
  S.u[0]=NaN;assert.throws(()=>OBS.capture(S));S.u[0]=0;
  S.eta[0]=-S.h;assert.throws(()=>OBS.capture(S));
});
check('analytic initial frame has zero-time provenance and an explicit display-only tessellation',()=>{
  const R=.015,h=.004,at=(rho,theta)=>({Re:1e-5*rho*(1-rho*rho)*Math.cos(theta)});
  const f=OBS.createInitialFrame({R,h,at,generation:9,initialCondition:{kind:'test-analytic'},
    particles:new Float64Array([0,0,h/2])});
  assert.equal(f.time,0);assert.equal(f.steps,0);assert.equal(f.initial,true);assert.equal(f.generation,9);
  assert.equal(f.flowVectors.length,0);assert.equal(f.grid,null);
  assert.equal(f.displayTessellation.selectsSolverGrid,false);
  assert.equal(f.provenance.fieldKind,'initial-condition');close(f.particleDisplacement[0],0,0);
  assert(f.surfaceDisplacement.some(x=>x>0)&&f.surfaceDisplacement.some(x=>x<0));
});
check('JS and C++ evolved observations agree, including direct fresh WASM views with stale JS metrics',()=>{
  const S=deform(cell(8,16,8)),W=new FaradayCell3DWasm(S._options);
  try{
    W.js.eta.set(S.eta);W.js.Ht.set(S.Ht);W.js.refreshMetric();W.pushState();W.refreshMetric();
    for(let j=0;j<4;j++){const dt=S.stableStep();S.step(dt);W.step(dt);}
    // Deliberately do not pullState: source=views must be a coherent observation
    // of the newly evolved module, not of the stale JS wrapper's metric.
    const a=OBS.createFrame(S,{generation:3,steps:4,maxFlowGlyphs:101});
    const b=OBS.createFrame(W.js,{source:W.views,time:W.t,generation:3,steps:4,maxFlowGlyphs:101});
    close(a.time,b.time,0);assert.equal(a.steps,b.steps);assert.equal(a.generation,b.generation);
    for(const key of ['positions','surfaceDisplacement','flowPositions','flowVectors','flowDisplacement']){
      assert.equal(a[key].length,b[key].length);
      let diff=0,scale=0;for(let i=0;i<a[key].length;i++){diff=Math.max(diff,Math.abs(a[key][i]-b[key][i]));scale=Math.max(scale,Math.abs(a[key][i]));}
      assert(diff<=1e-7*Math.max(scale,1e-12),`${key} parity error ${diff} / ${scale}`);
    }
    assert.deepEqual(a.indices,b.indices);assert(a.maxSpeed>0);
  }finally{W.release();}
});
console.log(`${passed} observation checks passed, 0 failed`);
