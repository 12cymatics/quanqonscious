#!/usr/bin/env node
import {createRequire} from 'node:module';
import {mkdirSync,writeFileSync,appendFileSync,readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {patternCase,seedCommonSurface} from './pattern_case.mjs';
const require=createRequire(import.meta.url);
const out=resolve(process.env.PATTERN_OUT||'ns-pattern-output'); mkdirSync(out,{recursive:true});
const num=(name,d)=>{const v=Number(process.env[name]??d);if(!Number.isFinite(v)||v<=0)throw Error(name+' must be positive finite');return v;};
const frequency=num('PATTERN_HZ',56), cycles=num('PATTERN_CYCLES',256), wallSeconds=num('PATTERN_WALL_SECONDS',2700);
const safety=num('PATTERN_SAFETY',0.4), seed=num('PATTERN_SEED',1729), seedRms=num('PATTERN_SEED_RMS_M',1e-7);
const {options,depthInput,initialCondition}=patternCase(process.env);
const root=resolve(process.env.PATTERN_REPO||'.');
const paths=['dns/faraday-cell3d.js','dns/faraday-cell3d-wasm.js','dns/faraday_cell3d.cpp','dns/faraday_cell3d.wasm'];
const hashes=Object.fromEntries(paths.map(p=>[p,createHash('sha256').update(readFileSync(resolve(root,p))).digest('hex')]));
const {FaradayCell3DWasm}=require(resolve(root,'dns/faraday-cell3d-wasm.js'));
const provenance={depthInput,initialCondition,engine:'full nonlinear 3D free-surface Navier-Stokes, freshly built C++/WASM',options,frequency,cycles,safety,seed,seedRms,wallSeconds,hashes,
 commit:process.env.GITHUB_SHA||null,run:process.env.GITHUB_RUN_ID||null,
 assumptions:['Rigid whole-cup vertical translation: cup-frame g+a cos(omega t), fixed no-slip floor and wall.',
 'Acceleration is an uncalibrated sensitivity input, not inferred from paper mV.',
 'Nominal water properties near 20 C; actual experiment temperature unknown.',
 'Free contact means solver 90-degree condition; no measured meniscus/contact-angle calibration.',
 'Finite pilot discretization; no continuum or experimental validation claim.',
 'Common double-zero-rim broadband m=1..12 equal area-energy initial perturbations; identical across contact and depth; none singled out as target.',
 'Signed surface Fourier mode is not directly equivalent to photographed phase-combined folds; no optical forward model is supplied.'],
 paper:{doi:'10.14294/WATER.2017.6',diameter_m:0.02425,referenceVolumesMl:[1.5,2.5],selectedReferenceVolumeM3:depthInput.volumeM3,observed56HzFolds:[6,6,6]},
 diagnostics:'Angular DFT power area-weighted across radius. Settling and period errors are prespecified diagnostics, not proof of asymptotic attractor.'};
writeFileSync(out+'/config.json',JSON.stringify(provenance,null,2));
const W=new FaradayCell3DWasm(options), S=W.js;
const seeded=seedCommonSurface(S,seed,seedRms);
const area=seeded.weights,areaSum=seeded.area;
provenance.initialSurfaceSha256=createHash('sha256').update(Buffer.from(S.eta.buffer,S.eta.byteOffset,S.eta.byteLength)).digest('hex');
writeFileSync(out+'/config.json',JSON.stringify(provenance,null,2));
S.refreshMetric();W.pushState();W.refreshMetric();
const start=Date.now(), samples=[], recent=[];
const sampleCount=Math.ceil(cycles*8),endTime=cycles/frequency;
let steps=0,dtMin=Infinity,dtMax=0,status='running',error=null,lastProgress=start;
let lastValidatedTime=0,minStepDepth=Infinity,maxStepCgResidual=0;
function validateState(){
 for(const name of ['u','v','w','eta','H'])for(const x of W.views[name])if(!Number.isFinite(x))throw Error('Nonfinite '+name+' at '+W.t);
 for(const x of W.views.H){if(!(x>0))throw Error('Nonpositive depth at '+W.t);minStepDepth=Math.min(minStepDepth,x);}
 const residual=W.cgResidual;if(!Number.isFinite(residual)||residual>=1e-11)throw Error('Invalid pressure residual '+residual+' at '+W.t);
 maxStepCgResidual=Math.max(maxStepCgResidual,residual);lastValidatedTime=W.t;
}
function snapshot(){
 const eta=Array.from(W.views.eta);let sq=0,mean=0,peak=0;
 for(let i=0;i<S.nr;i++)for(let k=0;k<S.nth;k++){const x=eta[S.ie(i,k)];sq+=area[i]*x*x;mean+=area[i]*x;peak=Math.max(peak,Math.abs(x));}
 const power=Array(S.nth/2+1).fill(0);
 for(let i=0;i<S.nr;i++)for(let m=0;m<power.length;m++){
 let re=0,im=0;for(let k=0;k<S.nth;k++){const x=eta[S.ie(i,k)];re+=x*Math.cos(m*k*S.dth)/S.nth;im-=x*Math.sin(m*k*S.dth)/S.nth;}
 power[m]+=area[i]*S.nth*(re*re+im*im)*(m===0||m===S.nth/2?1:2)/areaSum;
 }
 const row={time:W.t,driveCycles:W.t*frequency,rms:Math.sqrt(sq/areaSum),peak,mean:mean/areaSum,power,cgResidual:W.cgResidual,maxDivergence:W.maxDivergence(),steps};
 samples.push(row);recent.push({time:W.t,eta});if(recent.length>1024)recent.shift();
 appendFileSync(out+'/metrics.ndjson',JSON.stringify(row)+'\n');
 appendFileSync(out+'/surface.ndjson',JSON.stringify({time:W.t,eta})+'\n');
 return row;
}
writeFileSync(out+'/grid.json',JSON.stringify({rf:Array.from(S.rf),rc:Array.from(S.rc),nth:S.nth,dtheta:S.dth,etaIndex:'i*nth+k',etaUnits:'m'}));
try{
 validateState();snapshot();let sample=1;
 while(sample<=sampleCount){
 if((Date.now()-start)/1000>=wallSeconds){status='partial-wall-budget';break;}
 const target=sample===sampleCount?endTime:sample/(8*frequency);
 W.pullState(); const dt=Math.min(W.js.stableStep(safety),target-W.t);
 if(!(dt>0&&Number.isFinite(dt)))throw Error('Invalid stable dt '+dt);
 W.step(dt);validateState();steps++;dtMin=Math.min(dtMin,dt);dtMax=Math.max(dtMax,dt);
 if(steps===100||Date.now()-lastProgress>=60000){const elapsed=(Date.now()-start)/1000;console.log(JSON.stringify({type:'progress',cycles:W.t*frequency,steps,elapsedSeconds:elapsed,projectedTotalSeconds:elapsed*cycles/(W.t*frequency)}));lastProgress=Date.now();}
 if(W.t>=target-4*Number.EPSILON*Math.max(1,target)){const row=snapshot();sample++;
 if(sample%64===1){const elapsed=(Date.now()-start)/1000;console.log(JSON.stringify({cycles:row.driveCycles,rms:row.rms,steps,elapsedSeconds:elapsed,projectedTotalSeconds:elapsed*cycles/row.driveCycles}));}
 }
 }
 if(status==='running')status='completed-horizon';
}catch(e){status='solver-error';error=e.stack||String(e);}
W.pullState();writeFileSync(out+'/final-state.json',JSON.stringify({time:W.t,lastValidatedTime,stateValidity:error?'raw failed or invalid step; only prior validated snapshots may be interpreted':'accepted finite positive-depth state',eta:Array.from(S.eta),u:Array.from(S.u),v:Array.from(S.v),w:Array.from(S.w)}));
const tail=samples.slice(-256), p=Array(S.nth/2+1).fill(0);
for(const row of tail)row.power.forEach((x,m)=>p[m]+=x/Math.max(1,tail.length));
let dominant=1;for(let m=2;m<p.length;m++)if(p[m]>p[dominant])dominant=m;
const total=p.reduce((a,b)=>a+b,0),share=total>0?p[dominant]/total:0;
function lagError(lag,sign){let a=0,b=0;for(let t=Math.max(lag,recent.length-256);t<recent.length;t++)for(let e=0;e<S.NE;e++){const x=recent[t].eta[e],y=recent[t-lag].eta[e],w=area[Math.floor(e/S.nth)];a+=w*(x-sign*y)**2;b+=w*x*x;}return b?Math.sqrt(a/b):null;}
const blockRms=offset=>{const rows=samples.slice(-512+offset,-256+offset||undefined);return rows.length?Math.sqrt(rows.reduce((a,x)=>a+x.rms*x.rms,0)/rows.length):null;};
const previousRms=blockRms(0),lastRms=blockRms(256);
const envelopeChange=previousRms?Math.abs(lastRms-previousRms)/previousRms:null;
const antiPeriodError=lagError(8,-1),twoPeriodError=lagError(16,1);
const established=status!=='solver-error'&&W.t*frequency>=128&&samples.length>=513&&lastRms>10*seedRms&&envelopeChange<0.05&&twoPeriodError<0.1;
const subharmonicEstablished=established&&antiPeriodError!==null&&antiPeriodError<0.1;
const summary={status,error,lastValidatedTime,minStepDepth,maxStepCgResidual,simulatedDriveCycles:W.t*frequency,wallSeconds:(Date.now()-start)/1000,steps,dtMin,dtMax,
 dominantSignedMode:dominant,dominantPowerShare:share,phaseCombinedFoldCandidate:2*dominant,angularPower:p,
 lastRms,previousRms,envelopeChange,antiPeriodError,twoPeriodError,establishedDiagnostic:established,subharmonicEstablished,
 comparison:!established?'inconclusive: no established saturated pattern at this finite horizon':'conditional numerical surface pattern available for phase-aware inspection; signed Fourier mode alone does not establish photographed fold agreement',
 convergence:'NOT ESTABLISHED. Repeat same physical horizon at half safety and independently refined nr,nth,nz before claiming resolution-independent selection.'};
writeFileSync(out+'/summary.json',JSON.stringify(summary,null,2));console.log(JSON.stringify(summary,null,2));
const frames=recent.slice(-16),scale=Math.max(seedRms,...frames.flatMap(f=>f.eta.map(Math.abs)));
for(let f=0;f<frames.length;f++){
 const frame=frames[f];let svg=`<svg xmlns="http://www.w3.org/2000/svg" width="640" height="690" viewBox="-320 -320 640 690"><rect x="-320" y="-320" width="640" height="690" fill="white"/>`;
 for(let i=0;i<S.nr;i++)for(let k=0;k<S.nth;k++){
 const r0=290*S.rf[i]/S.R,r1=290*S.rf[i+1]/S.R,a=(k-.5)*S.dth,b=(k+.5)*S.dth;
 const xy=(r,t)=>`${r*Math.cos(t)},${r*Math.sin(t)}`;
 const v=Math.max(-1,Math.min(1,frame.eta[S.ie(i,k)]/scale)),fade=Math.round(255*(1-Math.abs(v))),color=v>0?`rgb(255,${fade},${fade})`:`rgb(${fade},${fade},255)`;
 svg+=`<path d="M${xy(r0,a)} L${xy(r1,a)} A${r1},${r1} 0 0 1 ${xy(r1,b)} L${xy(r0,b)} ${r0?`A${r0},${r0} 0 0 0 ${xy(r0,a)}`:''} Z" fill="${color}"/>`;
 }
 svg+=`<text x="-300" y="320" font-size="16">NS surface height; cycle ${(frame.time*frequency).toFixed(3)}</text><text x="-300" y="345" font-size="14">Blue/red = -/+ ${(scale*1e6).toPrecision(4)} micrometres. Not a photograph.</text></svg>`;
 writeFileSync(out+`/surface-phase-${String(f).padStart(2,'0')}.svg`,svg);
}
W.release();if(error)process.exitCode=1;
