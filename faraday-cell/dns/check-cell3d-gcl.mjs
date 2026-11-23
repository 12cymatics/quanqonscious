/* Independent moving-grid mass and kinetic-energy identities. These checks use
 * the public energy/operator results and analytic geometry, not the production
 * radial-mass helper. FARADAY_GCL_SOURCE permits falsifying an older kernel. */
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
const require=createRequire(import.meta.url);
const here=dirname(fileURLToPath(import.meta.url));
const {FaradayCell3D}=require(process.env.FARADAY_GCL_SOURCE
  ? resolve(process.env.FARADAY_GCL_SOURCE):resolve(here,'faraday-cell3d.js'));
const options={nr:10,nth:16,nz:8,R:0.012125,h:0.003,rho:998.2,nu:1.004e-6,
  gamma:0.0727,g:9.80665,contact:'pinned'};
let assertions=0;
const check=(condition,message)=>{assert.ok(condition,message);assertions++;};
const max=a=>a.reduce((s,x)=>Math.max(s,Math.abs(x)),0);
const sum=a=>{let s=0,c=0;for(const x of a){const y=x-c,t=s+y;c=(t-s)-y;s=t;}return s;};
function fixture(stretch){
  const S=new FaradayCell3D({...options,rStretch:stretch,zStretch:2.2});
  for(let i=0;i<S.nr;i++)for(let k=0;k<S.nth;k++){
    const x=S.rc[i]/S.R,t=(k+.5)*S.dth;
    S.eta[S.ie(i,k)]=.27*S.h*(1-x*x)*(1+.3*x*x*Math.cos(2*t));
    S.Ht[S.ie(i,k)]=.017*(1-x*x)*(1+.6*x*x*Math.cos(2*t));
  }
  S.refreshMetric();
  for(let i=1;i<S.nr;i++)for(let k=0;k<S.nth;k++)for(let j=0;j<S.nz;j++)
    S.u[S.iu(i,k,j)]=.002*Math.sin((k+.5)*S.dth)*Math.sin(Math.PI*S.sc[j])
      *Math.sin(Math.PI*S.rf[i]/S.R);
  return S;
}
function pressureVolume(S,i,k,j){return S.rc[i]*S.drc[i]*S.dth*S.H[S.ie(i,k)]*S.dsc[j];}
function eachVelocity(S,fn){
  for(let i=1;i<S.nr;i++)for(let k=0;k<S.nth;k++)for(let j=0;j<S.nz;j++)
    fn(0,S.iu(i,k,j),i,k,j,.5*(pressureVolume(S,i-1,k,j)+pressureVolume(S,i,k,j)));
  for(let i=0;i<S.nr;i++)for(let k=0;k<S.nth;k++){
    for(let j=0;j<S.nz;j++)fn(1,S.iv(i,k,j),i,k,j,
      .5*(pressureVolume(S,i,k-1,j)+pressureVolume(S,i,k,j)));
    for(let j=1;j<=S.nz;j++)fn(2,S.iw(i,k,j),i,k,j,
      .5*(pressureVolume(S,i,k,j-1)+(j<S.nz?pressureVolume(S,i,k,j):0)));
  }
}
function transferredD(S,D,c,i,k,j){
  if(c===0)return .5*(D[S.ip(i-1,k,j)]+D[S.ip(i,k,j)]);
  if(c===1)return .5*(D[S.ip(i,k-1,j)]+D[S.ip(i,k,j)]);
  return .5*(D[S.ip(i,k,j-1)]+(j<S.nz?D[S.ip(i,k,j)]:0));
}
function analyticMeshNet(S,c,i,k,j){
  const Q=(a,b,s)=>S.rc[a]*S.drc[a]*S.dth*S.sf[s]*S.Ht[S.ie(a,b)];
  if(c===0)return .5*(Q(i-1,k,j+1)+Q(i,k,j+1))-.5*(Q(i-1,k,j)+Q(i,k,j));
  if(c===1)return .5*(Q(i,k-1,j+1)+Q(i,k,j+1))-.5*(Q(i,k-1,j)+Q(i,k,j));
  return (j<S.nz?.5*(Q(i,k,j)+Q(i,k,j+1)):Q(i,k,j))-.5*(Q(i,k,j-1)+Q(i,k,j));
}
function energyDerivative(S,rate,dt){
  const eta=S.eta.slice(),base=[S.u.slice(),S.v.slice(),S.w.slice()],fields=[S.u,S.v,S.w];
  const sample=s=>{
    for(let n=0;n<eta.length;n++)S.eta[n]=eta[n]+s*S.Ht[n];
    for(let c=0;c<3;c++)for(let n=0;n<fields[c].length;n++)fields[c][n]=base[c][n]+s*rate[c][n];
    S.refreshMetric();return S.energy().kinetic;
  };
  // K is cubic along this line: volume is linear in eta and velocity squared
  // is quadratic. This centered four-sample derivative is exact for a cubic.
  const d=(8*(sample(dt)-sample(-dt))-(sample(2*dt)-sample(-2*dt)))/(12*dt);
  S.eta.set(eta);for(let c=0;c<3;c++)fields[c].set(base[c]);S.refreshMetric();return d;
}
const records=[];
for(const stretch of [0,2.2,4])for(const rateKind of ['varying','constant']){
  const S=fixture(stretch);if(rateKind==='constant')S.Ht.fill(.017);
  const terms=[];
  eachVelocity(S,(c,n,i,k,j)=>terms.push(.5*S.rho*[S.u,S.v,S.w][c][n]**2*analyticMeshNet(S,c,i,k,j)));
  const rate=[new Float64Array(S.NU),new Float64Array(S.NV),new Float64Array(S.NW)];
  const want=sum(terms),got=energyDerivative(S,rate,.02*S.h/max(S.Ht));
  const defect=Math.abs(got-want)/sum(terms.map(Math.abs));
  check(defect<2e-12,`Actual kinetic mass derivative must equal mesh flux: stretch=${stretch}, ${rateKind}, relative defect=${defect}`);
  records.push({kind:'actual-kinetic-mass-GCL',stretch,rateKind,relativeDefect:defect});
}
for(const stretch of [0,2.2,4]){
  const S=fixture(stretch),fields=[S.u,S.v,S.w];
  let state=2511;
  const random=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return (state/4294967296-.5)*.003;};
  eachVelocity(S,(c,n)=>{fields[c][n]=random();});
  S.omegaFromW();
  for(let i=0;i<S.nr;i++)for(let k=0;k<S.nth;k++)S.Ht[S.ie(i,k)]=S.om[S.iw(i,k,S.nz)];
  const D=new Float64Array(S.NP);S.divergence(S.u,S.v,S.om,D);
  const T=[new Float64Array(S.NU),new Float64Array(S.NV),new Float64Array(S.NW)];
  S.advectTransport(...T);
  const terms=[],power=[];
  eachVelocity(S,(c,n,i,k,j,V)=>{
    terms.push(.5*S.rho*fields[c][n]**2*transferredD(S,D,c,i,k,j));
    power.push(S.rho*V*fields[c][n]*T[c][n]);
  });
  const dt=.01*Math.min(S.h/max(S.Ht),...T.map((a,c)=>max(fields[c])/max(a)));
  const derivative=energyDerivative(S,T,dt),scale=sum(power.map(Math.abs));
  const defect=Math.abs(derivative+sum(terms))/scale;
  check(defect<3e-11,`ALE kinetic-energy transport identity: stretch=${stretch}, relative defect=${defect}`);
  const q=Float64Array.from({length:S.NP},()=>random());
  const G=[new Float64Array(S.NU),new Float64Array(S.NV),new Float64Array(S.NW)];
  S.gradient(q,...G);
  const adjoint=[];eachVelocity(S,(c,n,i,k,j,V)=>adjoint.push(V*fields[c][n]*G[c][n]));
  for(let n=0;n<q.length;n++)adjoint.push(q[n]*D[n]);
  const adjointDefect=Math.abs(sum(adjoint))/sum(adjoint.map(Math.abs));
  check(adjointDefect<2e-14,`Pressure gradient/divergence mass adjoint: stretch=${stretch}, relative defect=${adjointDefect}`);
  const C=[new Float64Array(S.NU),new Float64Array(S.NV)];S.advectCurvature(...C);
  const curvature=[];eachVelocity(S,(c,n,i,k,j,V)=>{if(c<2)curvature.push(V*fields[c][n]*C[c][n]);});
  const curvatureDefect=Math.abs(sum(curvature))/sum(curvature.map(Math.abs));
  check(curvatureDefect<2e-14,`Cylindrical curvature transfers zero kinetic power: stretch=${stretch}, defect=${curvatureDefect}`);
  records.push({kind:'ALE-transport-and-adjoints',stretch,energyDefect:defect,adjointDefect,curvatureDefect});
}
console.log(JSON.stringify({passed:true,assertions,records},null,2));
