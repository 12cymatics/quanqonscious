#!/usr/bin/env node
import {createRequire} from 'node:module';
import {mkdirSync,writeFileSync,appendFileSync,readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
const require=createRequire(import.meta.url);
const out=resolve(process.env.PATTERN_OUT||'ns-pattern-output'); mkdirSync(out,{recursive:true});
const num=(name,d)=>{const v=Number(process.env[name]??d);if(!Number.isFinite(v)||v<=0)throw Error(name+' must be positive finite');return v;};
const frequency=num('PATTERN_HZ',56), cycles=num('PATTERN_CYCLES',256), wallSeconds=num('PATTERN_WALL_SECONDS',2700);
const safety=num('PATTERN_SAFETY',0.4), seed=num('PATTERN_SEED',1729), seedRms=num('PATTERN_SEED_RMS_M',1e-7);
const options={nr:num('PATTERN_NR',16),nth:num('PATTERN_NTH',48),nz:num('PATTERN_NZ',16),R:0.02425/2,
 h:2.5e-6/(Math.PI*(0.02425/2)**2),rho:998.2,nu:1.004e-6,gamma:0.07274,g:9.80665,
 accel:num('PATTERN_ACCEL_G',0.5)*9.80665,omegaD:2*Math.PI*frequency,contact:process.env.PATTERN_CONTACT||'free',rStretch:2.2,zStretch:2.2};
const root=resolve(process.env.PATTERN_REPO||'.');
const paths=['dns/faraday-cell3d.js','dns/faraday-cell3d-wasm.js','dns/faraday_cell3d.cpp','dns/faraday_cell3d.wasm'];
const hashes=Object.fromEntries(paths.map(p=>[p,createHash('sha256').update(readFileSync(resolve(root,p))).digest('hex')]));
const {FaradayCell3DWasm}=require(resolve(root,'dns/faraday-cell3d-wasm.js'));
const provenance={engine:'full nonlinear 3D free-surface Navier-Stokes, freshly built C++/WASM',options,frequency,cycles,safety,seed,seedRms,wallSeconds,hashes,
 commit:process.env.GITHUB_SHA||null,run:process.env.GITHUB_RUN_ID||null,
 assumptions:['Rigid whole-cup vertical translation: cup-frame g+a cos(omega t), fixed no-slip floor and wall.',
 'Acceleration is an uncalibrated sensitivity input, not inferred from paper mV.',
 'Nominal water properties near 20 C; actual experiment temperature unknown.',
 'Free contact means solver 90-degree condition; no measured meniscus/contact-angle calibration.',
 'Finite pilot discretization; no continuum or experimental validation claim.',
 'Smooth broadband m=1..12 equal area-energy initial perturbations; none singled out as target.',
 'Signed surface Fourier mode is not directly equivalent to photographed phase-combined folds; no optical forward model is supplied.'],
 paper:{doi:'10.14294/WATER.2017.6',diameter_m:0.02425,volume_m3:2.5e-6,observed56HzFolds:[6,6,6]},
 diagnostics:'Angular DFT power area-weighted across radius. Settling and period errors are prespecified diagnostics, not proof of asymptotic attractor.'};
writeFileSync(out+'/config.json',JSON.stringify(provenance,null,2));
const W=new FaradayCell3DWasm(options), S=W.js;
let rng=seed>>>0;const random=()=>{rng=(Math.imul(rng,1664525)+1013904223)>>>0;return rng/4294967296;};
const area=Array.from(S.rc,(r,i)=>r*S.drc[i]*S.dth), areaSum=area.reduce((a,b)=>a+b,0)*S.nth;
const maxMode=Math.min(12,Math.floor(S.nth/3));
for(let m=1;m<=maxMode;m++){
 const phase=2*Math.PI*random(), b=2*random()-1;
 const radial=Array.from(S.rc,r=>{const x=r/S.R;return x**m*(S.contact==='free'?1-m/(m+2)*x*x:1-x*x)*(1+b*(1-x*x)**2);});
 const norm=Math.sqrt(radial.reduce((sum,x,i)=>sum+x*x*area[i]*S.nth/2,0)/areaSum);
 for(let i=0;i<S.nr;i++)for(let k=0;k<S.nth;k++)S.eta[S.ie(i,k)]+=radial[i]/norm*Math.cos(m*k*S.dth+phase);
}
let rms=Math.sqrt(S.eta.reduce((a,x,e)=>a+x*x*area[Math.floor(e/S.nth)],0)/areaSum);
for(let e=0;e<S.eta.length;e++)S.eta[e]*=seedRms/rms;
S.refreshMetric();W.pushState();W.refreshMetric();
const start=Date.now(), interval=1/(8*frequency), samples=[], recent=[];
let steps=0,dtMin=Infinity,dtMax=0,status='running',error=null;
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
 snapshot();let sample=1;
 while(W.t<cycles/frequency){
 if((Date.now()-start)/1000>=wallSeconds){status='partial-wall-budget';break;}
 const target=Math.min(sample*interval,cycles/frequency);
 W.pullState(); const dt=Math.min(W.js.stableStep(safety),target-W.t);
 if(!(dt>0&&Number.isFinite(dt)))throw Error('Invalid stable dt '+dt);
 W.step(dt);steps++;dtMin=Math.min(dtMin,dt);dtMax=Math.max(dtMax,dt);
 if(W.t>=target-1e-13){const row=snapshot();sample++;
 if(sample%64===1){const elapsed=(Date.now()-start)/1000;console.log(JSON.stringify({cycles:row.driveCycles,rms:row.rms,steps,elapsedSeconds:elapsed,projectedTotalSeconds:elapsed*cycles/row.driveCycles}));}
 }
 }
 if(status==='running')status='completed-horizon';
}catch(e){status='solver-error';error=e.stack||String(e);}
W.pullState();writeFileSync(out+'/final-state.json',JSON.stringify({time:W.t,eta:Array.from(S.eta),u:Array.from(S.u),v:Array.from(S.v),w:Array.from(S.w)}));
const tail=samples.slice(-256), p=Array(S.nth/2+1).fill(0);
for(const row of tail)row.power.forEach((x,m)=>p[m]+=x/Math.max(1,tail.length));
const dominant=p.indexOf(Math.max(...p.slice(1))),total=p.reduce((a,b)=>a+b,0),share=p[dominant]/Math.max(1e-300,total);
function lagError(lag,sign){let a=0,b=0;for(let t=Math.max(lag,recent.length-256);t<recent.length;t++)for(let e=0;e<S.NE;e++){const x=recent[t].eta[e],y=recent[t-lag].eta[e],w=area[Math.floor(e/S.nth)];a+=w*(x-sign*y)**2;b+=w*x*x;}return b?Math.sqrt(a/b):null;}
const blockRms=offset=>{const rows=samples.slice(-512+offset,-256+offset||undefined);return rows.length?Math.sqrt(rows.reduce((a,x)=>a+x.rms*x.rms,0)/rows.length):null;};
const previousRms=blockRms(0),lastRms=blockRms(256);
const envelopeChange=previousRms?Math.abs(lastRms-previousRms)/previousRms:null;
const antiPeriodError=lagError(8,-1),twoPeriodError=lagError(16,1);
const established=W.t*frequency>=128&&samples.length>=513&&lastRms>10*seedRms&&envelopeChange<0.05&&twoPeriodError<0.1;
const summary={status,error,simulatedDriveCycles:W.t*frequency,wallSeconds:(Date.now()-start)/1000,steps,dtMin,dtMax,
 dominantSignedMode:dominant,dominantPowerShare:share,phaseCombinedFoldCandidate:2*dominant,angularPower:p,
 lastRms,previousRms,envelopeChange,antiPeriodError,twoPeriodError,establishedDiagnostic:established,
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
