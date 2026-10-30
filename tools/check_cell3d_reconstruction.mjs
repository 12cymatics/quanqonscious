import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';

const require = createRequire(import.meta.url);
const source = process.env.CYMATICA_SOLVER_PATH || '../dns/faraday-cell3d.js';
const { FaradayCell3D } = require(source.startsWith('.') ? source : resolve(source));
const proto = FaradayCell3D.prototype;
let checks = 0;
function check(condition, label){ checks++; assert.ok(condition, label); }
function near(a, b, scale, label){ check(Math.abs(a-b) <= 2e-11*Math.max(1,scale), `${label}: ${a} versus ${b}`); }
function oldValue(S, f, fam, a, k, z, lev, bracket){
  let aa=a, kk=k, sign=1;
  if(a<0){ aa=-1-a; kk=k+(S.nth>>1); sign=fam.axisSign; }
  const sn=fam.sn, nJ=sn.length, col=aa*S.nth+S.kw(kk), base=col*fam.stride;
  const ss=z/fam.Hcol[col], n=Math.min(4,nJ);
  if(nJ===1)return sign*f[base];
  let j0;
  if(bracket){let lo=0,hi=nJ-1;while(hi-lo>1){const m=(lo+hi)>>1;if(sn[m]<=ss)lo=m;else hi=m;}j0=lo-1;}
  else j0=(lev===undefined?1:lev)-1;
  if(j0+n>nJ)j0=nJ-n;
  if(j0<0)j0=0;
  let v=0;
  for(let i=0;i<n;i++){let L=1;for(let m=0;m<n;m++)if(m!==i)L*=(ss-sn[j0+m])/(sn[j0+i]-sn[j0+m]);v+=f[base+j0+i]*L;}
  return sign*v;
}
function fixture(nJ, graded, sign){
  const S={nth:8,kw:k=>((k%8)+8)%8,colKnotSlope:proto.colKnotSlope};
  const sn=Float64Array.from({length:nJ},(_,j)=>nJ===1?0.5:graded?(j/(nJ-1))**2:j/(nJ-1));
  const fam={sn,stride:nJ+2,axisSign:sign,Hcol:Float64Array.from({length:24},(_,i)=>0.7+i/19)};
  const f=new Float64Array(24*fam.stride);
  return {S,fam,f};
}
function value(F,a,k,sigma,br=true,lev=2){
  const aa=a<0?-1-a:a,kk=a<0?k+4:k,col=aa*8+F.S.kw(kk);
  return proto.colValueAtZ.call(F.S,F.f,F.fam,a,k,sigma*F.fam.Hcol[col],lev,br);
}
for(const nJ of [1,2,3,4,7,12])for(const graded of [false,true])for(const sign of [-1,1]){
  const F=fixture(nJ,graded,sign),{S,fam,f}=F;
  for(let degree=0;degree<=Math.min(3,nJ-1);degree++){
    for(let col=0;col<24;col++)for(let j=0;j<nJ;j++)f[col*fam.stride+j]=fam.sn[j]**degree+col/13;
    for(const a of [-2,-1,0,1,2])for(const k of [-1,0,7,8])for(const x of [-0.2,0,0.031,0.43,0.78,1,1.2]){
      const col=(a<0?-1-a:a)*8+S.kw(a<0?k+4:k),sgn=a<0?sign:1;
      const expected=sgn*(x**degree+col/13);
      near(value(F,a,k,x),expected,Math.abs(expected),`polynomial n=${nJ} degree=${degree} x=${x}`);
    }
  }
  for(let col=0;col<24;col++)for(let j=0;j<nJ;j++)f[col*fam.stride+j]=Math.sin((j+1)*(col+2))+fam.sn[j]**4;
  for(const a of [-1,0,2])for(const k of [-1,0,8])for(const lev of [0,1,nJ-1])for(const x of [-0.2,0.07,0.41,0.93,1.2]){
    const col=(a<0?-1-a:a)*8+S.kw(a<0?k+4:k),z=x*fam.Hcol[col];
    check(Object.is(value(F,a,k,x,false,lev),oldValue(S,f,fam,a,k,z,lev,false)),'anchored arithmetic changed');
  }
  if(nJ<3)continue;
  for(let knot=1;knot<nJ-1;knot++)for(const a of [-1,0,2])for(const k of [-1,0,8]){
    const x=fam.sn[knot],d=Math.min(x-fam.sn[knot-1],fam.sn[knot+1]-x)/4;
    const v0=value(F,a,k,x),vp=[1,2,3].map(j=>value(F,a,k,x+j*d)),vm=[1,2,3].map(j=>value(F,a,k,x-j*d));
    const right=(-11*v0+18*vp[0]-9*vp[1]+2*vp[2])/(6*d);
    const left=(11*v0-18*vm[0]+9*vm[1]-2*vm[2])/(6*d);
    const scale=Math.max(Math.abs(v0),...vp.map(Math.abs),...vm.map(Math.abs))/d;
    near(right,left,scale,`C1 n=${nJ} graded=${graded} knot=${knot}`);
  }
}
{
  const F=fixture(9,false,1),{S,fam,f}=F,h=1/8,knot=4,x=fam.sn[knot];
  for(let col=0;col<24;col++)for(let j=0;j<9;j++)f[col*fam.stride+j]=(fam.sn[j]-x)**4;
  const d=h/4,v=t=>oldValue(S,f,fam,0,0,t*fam.Hcol[0],undefined,true);
  const right=(-11*v(x)+18*v(x+d)-9*v(x+2*d)+2*v(x+3*d))/(6*d);
  const left=(11*v(x)-18*v(x-d)+9*v(x-2*d)-2*v(x-3*d))/(6*d);
  near(right-left,-4*h**3,1,'independent old-stencil quartic witness');
  check(Math.abs(right-left)>1e-3,'old-stencil mutation must be distinguishable from roundoff');
}
console.log(JSON.stringify({suite:'bracket-reconstruction',checks,passed:checks,source}));
