export const REFERENCE_RADIUS_M=0.02425/2;
const read=(env,key,fallback,zero=false)=>{
 const value=Number(env[key]??fallback);
 if(!Number.isFinite(value)||(zero?value<0:value<=0))throw Error(key+' must be '+(zero?'nonnegative':'positive')+' and finite');
 return value;
};
export function patternCase(env={}){
 if(env.PATTERN_DEPTH_MM!==undefined&&env.PATTERN_VOLUME_ML!==undefined)throw Error('Specify depth OR volume, not both');
 const R=REFERENCE_RADIUS_M;
 const depthInput=env.PATTERN_DEPTH_MM!==undefined;
 const volume=depthInput?null:read(env,'PATTERN_VOLUME_ML',2.5)*1e-6;
 const h=depthInput?read(env,'PATTERN_DEPTH_MM',1)*1e-3:volume/(Math.PI*R*R);
 const contact=env.PATTERN_CONTACT||'free';if(!['free','pinned'].includes(contact))throw Error('Unsupported contact');
 const options={nr:read(env,'PATTERN_NR',16),nth:read(env,'PATTERN_NTH',48),nz:read(env,'PATTERN_NZ',16),R,h,
  rho:998.2,nu:1.004e-6,gamma:0.07274,g:9.80665,accel:read(env,'PATTERN_ACCEL_G',0.5,true)*9.80665,
  omegaD:2*Math.PI*read(env,'PATTERN_HZ',56),contact,rStretch:2.2,zStretch:2.2};
 return {options,depthInput:{mode:depthInput?'specified-depth':'volume-over-area',depthM:h,volumeM3:depthInput?Math.PI*R*R*h:volume,meniscus:'not specified; flat reference geometry'},
  initialCondition:'common-double-zero-rim-broadband-v1'};
}
export function commonRadial(x,m,b){return x**m*(1-x*x)**2*(1+b*(1-x*x)**2);}
export function seedCommonSurface(S,seed=1729,targetRms=1e-7){
 if(!Number.isInteger(seed)||seed<0||!Number.isFinite(targetRms)||targetRms<=0)throw Error('Invalid initial seed or RMS');
 let rng=seed>>>0;const random=()=>{rng=(Math.imul(rng,1664525)+1013904223)>>>0;return rng/4294967296;};
 const weights=Array.from(S.rc,(r,i)=>r*S.drc[i]*S.dth),area=weights.reduce((a,b)=>a+b,0)*S.nth;
 S.eta.fill(0);S.u.fill(0);S.v.fill(0);S.w.fill(0);
 for(let m=1;m<=Math.min(12,Math.floor(S.nth/3));m++){
  const phase=2*Math.PI*random(),b=2*random()-1;
  const radial=Array.from(S.rc,r=>commonRadial(r/S.R,m,b));
  const norm=Math.sqrt(radial.reduce((sum,x,i)=>sum+x*x*weights[i]*S.nth/2,0)/area);
  for(let i=0;i<S.nr;i++)for(let k=0;k<S.nth;k++)S.eta[S.ie(i,k)]+=radial[i]/norm*Math.cos(m*k*S.dth+phase);
 }
 const rms=Math.sqrt(S.eta.reduce((a,x,e)=>a+x*x*weights[Math.floor(e/S.nth)],0)/area);
 for(let e=0;e<S.eta.length;e++)S.eta[e]*=targetRms/rms;
 return {seed,targetRms,profile:'common-double-zero-rim-broadband-v1',weights,area};
}
