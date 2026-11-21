import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {dirname,join,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import vm from 'node:vm';
const require=createRequire(import.meta.url),here=dirname(fileURLToPath(import.meta.url));
const V=require('./vedic-polynomial.js'),O=require('./vedic-operators.js');
const oracle=[{"nodes":["0","1/3","2/3","1"],"target":"1/3","left":"0","right":"1","fieldNodes":["0","1/3","2/3","1"],"weights":["-83/2520","19/280","-11/280","11/2520","19/280","31/35","9/280","1/70","-11/280","9/280","13/280","-11/280","11/2520","1/70","-11/280","13/630"]},{"nodes":["-2/5","1/7","3/4","6/5"],"target":"1/7","left":"1/7","right":"3/4","fieldNodes":["-1/3","1/4","4/5","7/5"],"weights":["-84059833275/38326663438336","-173306075/4560504423552","10434597875/3291792666624","-138759701375/147831416119296","206855869119/2673854403584","465194592595/430455774672","-43705927975/229651462656","449990306725/13953475501056","88643525/23638978272","-374493425/16876866699","2476592725/115727085936","-809785225/273536748576","-16330689175/9735147921408","619268542525/79928840686464","-274192687625/47659432955904","-821543876375/2590940082511872"]},{"nodes":["-1","0","1","2"],"target":"0","left":"-1/2","right":"1/2","fieldNodes":["-1/2","1/2","3/2","5/2"],"weights":["-155/48384","523/80640","-49/11520","47/48384","5141/16128","24911/26880","-1651/5376","4967/80640","-307/80640","127/26880","-19/26880","-17/80640","181/241920","-37/80640","-1/2304","1/6912"]}];
let passed=0;
function check(name,fn){fn();passed++;console.log('ok '+name)}
function q(s){const[n,d]=s.split('/');return V.Q(BigInt(n),d===undefined?1n:BigInt(d))}
function close(a,b,scale=1){assert(Number.isFinite(a)&&Number.isFinite(b));assert(Math.abs(a-b)<=2e-11*Math.max(scale,Math.abs(a),Math.abs(b)),`${a} differs from ${b}`)}
for(const[index,record]of oracle.entries())check('independent physical-coordinate symbolic transport integral '+index,()=>{
 const plan=O.compileTransport(record.nodes.map(q),q(record.target),q(record.left),q(record.right),record.fieldNodes.map(q));
 assert.deepEqual(plan.exact.weights.map(V.qstr),record.weights);
 const nodes=record.nodes.map(s=>V.toNumber(q(s))),fnodes=record.fieldNodes.map(s=>V.toNumber(q(s))),target=V.toNumber(q(record.target));
 for(let a=0;a<=3;a++)for(let b=0;b<=3-a;b++){
  const flux=nodes.map(x=>x**a),field=fnodes.map(x=>x**b),targetFlux=target**a;
  close(O.projectedFlux(plan,flux,field,field[1],targetFlux),target**(a+b));
 }
 for(const c of[0,1,-3,1e-18,1e18]){
  const flux=nodes.map(x=>2-x+x*x),targetFlux=2-target+target*target;
  assert.equal(O.projectedFlux(plan,flux,Array(4).fill(c),c,targetFlux),targetFlux*c);
 }
});
check('degree-six product uses its independently derived L2 cubic',()=>{
 const nodes=[V.Q(0),V.Q(1,3),V.Q(2,3),V.Q(1)],samples=nodes.map(x=>V.toNumber(V.pow(x,3n)));
 for(const[index,expected]of[[1,5/1134],[2,47/567]]){
  const plan=O.compileTransport(nodes,nodes[index],V.Q(0),V.Q(1));
  const actual=O.projectedFlux(plan,samples,samples,0,samples[index]);
  close(actual,expected);
  assert(Math.abs(actual-samples[index]**2)>1e-4);
 }
});
check('transport consumes the contracted coefficients and refuses invalid inputs',()=>{
 const plan=O.compileTransport([0,1,2,3],1,0,3),flux=[1,2,3,4],field=[4,3,2,1];
 const before=O.projectedFlux(plan,flux,field,0,2);
 plan.weights[0]+=.125;
 close(O.projectedFlux(plan,flux,field,0,2)-before,.5);
 assert.throws(()=>O.projectedFlux(plan,[1,NaN,3,4],field,0,2),/finite/);
 assert.throws(()=>O.projectedFlux(plan,flux,[1,2,3,Infinity],0,2),/finite/);
 assert.throws(()=>O.projectedFlux(plan,flux,field,0,3),/target|face|flux/i);
 assert.throws(()=>O.compileTransport([0,1,1,2],1,0,2),/distinct/);
 assert.throws(()=>O.compileTransport([0,1,2,3],.5,0,3),/node|target/i);
 assert.throws(()=>O.compileTransport([0,1,2,3],1,1,0),/positive|interval/i);
});
const methods=[
 [1,'ekadhikena','digits'],[2,'nikhilam','poly'],[3,'urdhva','poly'],[4,'paravartya','quotient'],[5,'shunyamCoefficient','aggregate'],[6,'nodalWeights','poly'],[7,'symmetricTranslate','parity'],[8,'purana','offset'],[9,'calana','poly'],[10,'yavadunam','poly'],[11,'vyashti','polynomial'],[12,'sesanyankena','digits'],[13,'sopantyaProductDerivative','poly'],[14,'ekanyunena','poly'],[15,'gunitasamuccaya','q'],[16,'gunakasamuccaya','poly'],[17,'anurupyena','poly'],[18,'shishyateKnownLinear','poly'],[19,'adyamDivide','poly'],[20,'kevalaih','digits'],[21,'osculateEvaluate','q'],[22,'tavadunamCube','poly'],[23,'extendedSquare','poly'],[24,'antyayordasakepi','poly'],[25,'antyayorevaProductSecond','negative_poly'],[26,'coefficientSum','coefficient_sum'],[27,'retainedQuadraticNumber','number'],[28,'reciprocalPolynomial','poly'],[29,'gunitasamuccayaSamuccayagunitah','poly']
];
const opts={nr:4,nth:8,nz:4,R:.012,h:.003,rho:997,nu:1.1e-6,gamma:.072,g:9.81,contact:'pinned',accel:5,omegaD:150,rStretch:1.3,zStretch:1.4};
const fields=['u','v','w','eta','Ht'];
function seed(s){
 for(let i=0;i<s.nr;i++)for(let k=0;k<s.nth;k++)s.eta[s.ie(i,k)]=1e-6*(1-(s.rc[i]/s.R)**2)**2*Math.cos((k+.5)*s.dth);
 for(let i=0;i<=s.nr;i++)for(let k=0;k<s.nth;k++)for(let j=0;j<s.nz;j++){const r=s.rf[i]/s.R,z=s.sc[j];s.u[s.iu(i,k,j)]=1e-4*r*(1-r)*z*(1-z)*(1+.3*z)*Math.cos((k+.5)*s.dth)}
 for(let i=0;i<s.nr;i++)for(let k=0;k<s.nth;k++)for(let j=0;j<s.nz;j++){const r=s.rc[i]/s.R,z=s.sc[j];s.v[s.iv(i,k,j)]=7e-5*r*(1-r)*z*(1-z)*(1-.2*z)*Math.sin(k*s.dth)}
 for(let i=0;i<s.nr;i++)for(let k=0;k<s.nth;k++)for(let j=0;j<=s.nz;j++){const r=s.rc[i]/s.R,z=s.sf[j];s.w[s.iw(i,k,j)]=1e-4*(1-r*r)*z*(1-z/2)*(1+.4*z)*Math.cos(2*(k+.5)*s.dth)}
 s.refreshMetric();s.omegaFromW();return s;
}
function snapshot(s){return Object.fromEntries(fields.map(name=>[name,Float64Array.from(s[name])]))}
function bumpQ(a,sign=1n){let n=a.n*1048576n+sign*a.d,d=a.d*1048576n;let x=n<0n?-n:n,y=d;while(y){const r=x%y;x=y;y=r}return Object.freeze({n:n/x,d:d/x})}
function bumpPoly(a){const b=Array.from(a);b[0]=bumpQ(b[0]);return Object.freeze(b)}
function corrupt(value,kind,args){
 if(kind==='aggregate'){const out=Array.from(value);out[args[1]]=bumpQ(out[args[1]]);return Object.freeze(out)}
 if(kind==='digits'){const a=Array.from(value);a[0]=(a[0]+1)%10;return Object.freeze(a)}
 if(kind==='poly')return bumpPoly(value);
 if(kind==='negative_poly'){const a=Array.from(value);a[0]=bumpQ(a[0],-1n);return Object.freeze(a)}
 if(kind==='q')return bumpQ(value);
 if(kind==='coefficient_sum'){const a=value,b=args[0][args[0].length>1?1:0];let n=a.n*b.d*1048576n+b.n*a.d,d=a.d*b.d*1048576n,x=n<0n?-n:n,y=d;while(y){const r=x%y;x=y;y=r}return Object.freeze({n:n/x,d:d/x})}
 if(kind==='number')return value*(1+1e-4);
 if(kind==='quotient')return Object.freeze({...value,quotient:bumpPoly(value.quotient)});
 if(kind==='polynomial')return Object.freeze({...value,polynomial:bumpPoly(value.polynomial)});
 if(kind==='offset')return Object.freeze({...value,offset:bumpQ(value.offset)});
 if(kind==='parity')return Object.freeze({...value,sum:bumpPoly(value.sum),even:bumpPoly(value.even)});
 throw new Error('Unknown corruption shape '+kind);
}
const sourceSnapshot=new Map;
function loadActual(fault){
 const cache=new Map,observed=new Map;
 function load(path){
  path=resolve(path);if(cache.has(path))return cache.get(path).exports;
  const module={exports:{},__fault(name,value,args){observed.set(name,(observed.get(name)||0)+1);return fault&&name===fault[1]?corrupt(value,fault[2],args):value}};
  cache.set(path,module);if(!sourceSnapshot.has(path))sourceSnapshot.set(path,readFileSync(path,'utf8'));let text=sourceSnapshot.get(path);
  const targeted=path===join(here,'vedic-polynomial.js')?methods.filter(x=>x[0]!==27):path===join(here,'vedic-operators.js')?methods.filter(x=>x[0]===27):[];
  if(targeted.length){const tag='const api=';assert(text.includes(tag),'actual module export declaration');const wrappers=targeted.map(([,name])=>`{const original=${name};${name}=function(...args){return module.__fault('${name}',original(...args),args)}}`).join('');text=text.replace(tag,wrappers+tag)}
  const child=specifier=>specifier.startsWith('.')?load(resolve(dirname(path),specifier)):require(specifier);
  const fn=vm.runInThisContext('(function(exports,require,module,__filename,__dirname){'+text+'\n})',{filename:path});
  fn(module.exports,child,module,path,dirname(path));return module.exports;
 }
 return{load,observed};
}
function execution(fault=null){
 const bundle=loadActual(fault);let s,initial,stage='loading';
 try{
  const {FaradayCell3D}=bundle.load(join(here,'faraday-cell3d.js'));stage='constructing';s=new FaradayCell3D(opts);seed(s);initial=snapshot(s);stage='advancing';
  for(let i=0;i<3;i++)s.step(s.stableStep()*.01);
  s.assertFiniteState('sutra evolution verification');
  return{observed:bundle.observed,state:snapshot(s),initial,time:s.t};
 }catch(error){return{observed:bundle.observed,error,stage,state:s?snapshot(s):null,initial,time:s?.t??0}}
}
const baseline=execution();assert.ifError(baseline.error);assert(baseline.time>0);
assert(fields.some(name=>baseline.state[name].some((v,i)=>!Object.is(v,baseline.initial[name][i]))),'the actual unmodified solver must change its field');
for(const method of methods)check('sutra '+method[0]+' '+method[1]+' is consumed by evolution or rejects corrupt arithmetic',()=>{
 assert((baseline.observed.get(method[1])||0)>0,'the normal production path never computes '+method[1]);
 const changed=execution(method);assert((changed.observed.get(method[1])||0)>0,'no method output was changed');
 if(changed.error){
  assert(!(changed.error instanceof ReferenceError||changed.error instanceof SyntaxError),'verification infrastructure failed');
  assert.match(changed.error.message,/factor|remainder|positive|Gram|cardinal|reciprocal|nonfinite|converge|pivot|distinct|integer|denominator|scale|finite|rational|target|polynomial|coefficient|transport|reproduction/i);
  console.log(JSON.stringify({sutra:method[0],outcome:'explicit arithmetic rejection',stage:changed.stage,error:changed.error.message}));
 }else{
  assert(changed.time>0,'corruption silently prevented time advancement');
  assert(fields.some(name=>changed.state[name].some((v,i)=>!Object.is(v,baseline.state[name][i]))),'sutra '+method[0]+' corrupt output had no effect on any evolved velocity, surface or surface-rate field');
  console.log(JSON.stringify({sutra:method[0],outcome:'evolved field changed',time:changed.time}));
 }
});
const WM=require('./faraday-cell3d-wasm.js'),Actual=require('./faraday-cell3d.js').FaradayCell3D;
const angularOracle=[['65/2016','-253/3360','173/3360','-17/2016','-17/224','179/1120','-19/224','1/1120','-11/1120','533/1120','639/1120','-41/1120','-13/1440','1/480','17/672','-37/2016'],['-37/2016','17/672','1/480','-13/1440','-41/1120','639/1120','533/1120','-11/1120','1/1120','-19/224','179/1120','-17/224','-17/2016','173/3360','-253/3360','65/2016']];
check('periodic frozen-flux momentum agrees with an independent conservative skew matrix',()=>{
 const F=require('./vedic-fluid.js'),s=new Actual(opts),n=s.nth,wrap=k=>(k%n+n)%n;
 const angular=s.polynomialTransport.records.find(r=>r.group==='angular');
 for(let h=0;h<2;h++)assert.deepEqual(angular.projections[h].exact.weights.map(V.qstr),angularOracle[h]);
 const weights=angularOracle.map(row=>row.map(x=>V.toNumber(q(x))*.5));
 const field=Float64Array.from({length:n},(_,k)=>.7+Math.cos(2*Math.PI*k/n)+.3*Math.sin(6*Math.PI*k/n));
 for(let component=0;component<3;component++){
  const values=component===0?s.u:component===1?s.v:s.w,index=component===0?s.iu:component===1?s.iv:s.iw;
  const radial=s.nr+(component===0?1:0),vertical=s.nz+(component===2?1:0);
  for(let i=0;i<radial;i++)for(let k=0;k<n;k++)for(let b=0;b<vertical;b++)values[index.call(s,i,k,b)]=field[k];
 }
 s.fluxR=()=>0;s.fluxS=()=>0;
 for(const variable of[false,true]){
  const supplied=Float64Array.from({length:n},(_,k)=>variable?1+.4*Math.sin(2*Math.PI*k/n)+.2*Math.cos(6*Math.PI*k/n):1.3);
  s.fluxTh=(i,k,b)=>supplied[wrap(k)];
  for(let component=0;component<3;component++){
   const flux=Float64Array.from({length:n},(_,k)=>component===1?.5*(supplied[wrap(k-1)]+supplied[k]):supplied[k]);
   const R=Array.from({length:n},()=>new Float64Array(n));
   for(let face=0;face<n;face++)for(let h=0;h<2;h++)for(let j=0;j<4;j++)for(let k=0;k<4;k++)R[face][wrap(face-2+k)]+=weights[h][j*4+k]*flux[wrap(face-2+h+j)];
   const C=Array.from({length:n},(_,i)=>Float64Array.from({length:n},(_,j)=>R[wrap(i+1)][j]-R[i][j]));
   const divergence=Float64Array.from({length:n},(_,i)=>flux[wrap(i+1)]-flux[i]);
   const expected=Float64Array.from({length:n},(_,i)=>{let value=-.5*divergence[i]*field[i];for(let j=0;j<n;j++)value-=.5*(C[i][j]-C[j][i])*field[j];return value});
   const actual=Float64Array.from({length:n},(_,k)=>F.transportMomentum(s,component,1,k,1));
   for(let k=0;k<n;k++)close(actual[k],expected[k]);
   close(actual.reduce((a,b)=>a+b,0),0);
   let work=0,expectedWork=0;for(let k=0;k<n;k++){work+=field[k]*actual[k];expectedWork-=.5*divergence[k]*field[k]*field[k]}
   close(work,expectedWork);
   if(!variable)close(work,0);
  }
 }
});
function runTransport(engine,mutate){
 let e;
 try{
  e=engine==='cpp'?new WM.FaradayCell3DWasm(opts):new Actual(opts);
  const s=e.js||e;seed(s);if(e.views){e.pushState();e.refreshMetric()}
  const data=e.transportData||s.polynomialTransport.data;
  assert(data instanceof Float64Array&&data.length>0&&data.length%35===0);
  if(mutate)for(let offset=0;offset<data.length;offset+=35)data[offset+3]+=.015625;
  let advection;
  if(e.views){e.advect();advection=['advU','advV','advW'].map(name=>e.views[name].slice())}
  else{s.advect(s._au,s._av,s._aw);advection=[s._au.slice(),s._av.slice(),s._aw.slice()]}
  for(let i=0;i<3;i++)e.step(1e-7);
  const actual=e.views||e;
  return{state:snapshot(actual),time:e.t,advection};
 }finally{if(e?.release)e.release()}
}
const consumed={};
for(const engine of['js','cpp'])check(engine+' live transport table changes momentum and advanced fields',()=>{
 const before=runTransport(engine,false),after=runTransport(engine,true);consumed[engine]={before,after};
 assert(before.time>0&&before.time===after.time);
 assert(before.advection.some((a,k)=>a.some((v,i)=>!Object.is(v,after.advection[k][i]))),'momentum transport ignores its compiled coefficients');
 assert(fields.some(name=>before.state[name].some((v,i)=>!Object.is(v,after.state[name][i]))),'advanced fields ignore the changed transport operator');
});
check('JS and native use identical original and changed transport coefficients',()=>{
 for(const mode of['before','after']){
  const a=consumed.js[mode],b=consumed.cpp[mode];assert.equal(a.time,b.time);
  for(let k=0;k<a.advection.length;k++)assert.deepEqual(a.advection[k],b.advection[k],mode+' advection component '+k);
  for(const name of fields)assert.deepEqual(a.state[name],b.state[name],mode+' '+name);
 }
});
function scaledPressure(engine,amplitude){
 let e;
 try{
  e=engine==='cpp'?new WM.FaradayCell3DWasm(opts):new Actual(opts);
  const s=e.js||e,rhs=Float64Array.from({length:s.NP},(_,i)=>amplitude*(1+.2*Math.sin(i*.37))),out=new Float64Array(s.NP);
  assert(rhs.some(x=>x!==0));
  let pressure;
  if(e.views){e.views.div.set(rhs);e.solveP('div',1e-11,2000);pressure=e.views.p.slice();out.set(e.applyL('p','cgq'))}
  else{s.solveP(rhs,1e-11,2000);pressure=s.p.slice();s.applyL(s.p,out)}
  assert(pressure.every(Number.isFinite)&&pressure.some(x=>x!==0),'a nonzero representable pressure correction was replaced with zero');
  const scale=Math.max(...rhs.map(Math.abs));let numerator=0,denominator=0;
  for(let i=0;i<rhs.length;i++){const r=(rhs[i]-out[i])/scale,b=rhs[i]/scale;numerator+=r*r;denominator+=b*b}
  const residual=Math.sqrt(numerator/denominator);assert(Number.isFinite(residual)&&residual<1e-11,'independently recomputed pressure residual '+residual);
  return{pressure,residual};
 }finally{if(e?.release)e.release()}
}
for(const amplitude of[1e-300,1e200])check('nonzero scaled pressure RHS is solved without norm underflow or overflow: '+amplitude,()=>{
 const a=scaledPressure('js',amplitude),b=scaledPressure('cpp',amplitude);
 assert.deepEqual(a.pressure,b.pressure,'JS/native scaled pressure solution');
 assert.equal(a.residual,b.residual,'JS/native independently measured residual');
});
console.log(passed+' Vedic evolution groups passed');
