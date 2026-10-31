import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import assert from 'node:assert/strict';
import {uniformPatternSamples} from './pattern_samples.mjs';
const read=(dir,name)=>JSON.parse(readFileSync(dir+'/'+name,'utf8'));
const dirs=[process.env.PATTERN_BASELINE_DIR||'ns-pattern-baseline',process.env.PATTERN_CURRENT_DIR||'ns-pattern-output'],configs=dirs.map(d=>read(d,'config.json'));
assert.equal(configs[0].run,'41147436690');
assert.equal(configs[0].options.accel,0.5*9.80665);assert.equal(configs[1].options.accel,9.80665);
for(const key of Object.keys(configs[0].options))if(key!=='accel')assert.equal(configs[0].options[key],configs[1].options[key],key+' unchanged');
for(const key of ['frequency','safety','seed','seedRms'])assert.equal(configs[0][key],configs[1][key],key+' unchanged');
for(const key of ['initialCondition','initialSurfaceSha256'])assert.equal(configs[0][key],configs[1][key],key+' unchanged; legacy and common-rim seeds cannot be compared as drive-only changes');
for(const key of ['dns/faraday-cell3d.js','dns/faraday-cell3d-wasm.js','dns/faraday_cell3d.cpp'])assert.equal(configs[0].hashes[key],configs[1].hashes[key],key+' unchanged');
const results=[];
for(let c=0;c<dirs.length;c++){
 const dir=dirs[c],config=configs[c],summary=read(dir,'summary.json'),grid=read(dir,'grid.json');
 assert.ok(!summary.error&&summary.status!=='solver-error','no solver failure');
 const raw=readFileSync(dir+'/surface.ndjson','utf8').trim().split('\n').map(JSON.parse);
 const normalized=uniformPatternSamples(config,summary,raw);
 const frames=normalized.frames.filter(f=>f.time<=72/config.frequency+1e-12);
 assert.ok(frames.length>=577,'common 72-cycle horizon reached');
 assert.ok(Math.abs(frames.at(-1).time*config.frequency-72)<1e-9,'same physical endpoint');
 const {nr,nth,h}=config.options,weights=grid.rc.map((r,i)=>r*(grid.rf[i+1]-grid.rf[i])*2*Math.PI/nth),area=weights.reduce((a,b)=>a+b,0)*nth;
 frames.forEach((f,j)=>{assert.ok(Math.abs(f.time-j/(8*config.frequency))<1e-10);assert.equal(f.eta.length,nr*nth);f.eta.forEach(x=>assert.ok(typeof x==='number'&&Number.isFinite(x)&&h+x>0));});
 const tail=frames.slice(-256),power=Array(nth/2+1).fill(0);
 const rms=f=>Math.sqrt(f.eta.reduce((sum,x,e)=>sum+weights[Math.floor(e/nth)]*x*x,0)/area);
 for(const frame of tail)for(let i=0;i<nr;i++)for(let m=0;m<power.length;m++){
  let re=0,im=0;for(let k=0;k<nth;k++){const x=frame.eta[i*nth+k];re+=x*Math.cos(m*k*2*Math.PI/nth)/nth;im-=x*Math.sin(m*k*2*Math.PI/nth)/nth;}
  power[m]+=weights[i]*nth*(re*re+im*im)*(m===0||m===nth/2?1:2)/(area*tail.length);
 }
 let dominant=1;for(let m=2;m<power.length;m++)if(power[m]>power[dominant])dominant=m;
 const total=power.reduce((a,b)=>a+b,0);
 function recurrence(lag,sign){let a=0,b=0;for(let j=frames.length-256;j<frames.length;j++)for(let e=0;e<nr*nth;e++){const x=frames[j].eta[e],y=frames[j-lag].eta[e],w=weights[Math.floor(e/nth)];a+=w*(x-sign*y)**2;b+=w*x*x;}return b?Math.sqrt(a/b):null;}
 results.push({extraTerminalObservation:normalized.extraTerminalObservation,run:config.run,commit:config.commit,peakAccelerationMps2:config.options.accel,physicalEndpointS:frames.at(-1).time,
  initialRmsM:rms(frames[0]),endpointRmsM:rms(frames.at(-1)),windowRmsM:Math.sqrt(tail.reduce((s,f)=>s+rms(f)**2,0)/tail.length),
  dominantSignedMode:total>0?dominant:null,dominantPowerShare:total>0?power[dominant]/total:0,angularPower:power,
  antiPeriodError:recurrence(8,-1),twoPeriodError:recurrence(16,1)});
}
const report={comparison:'prespecified drive-amplitude sensitivity; not calibration, mode fitting, or experimental validation',
 cycles:72,window:'last 256 samples through cycle72, sampled every eighth drive period',results,
 windowRmsRatio1gToHalfG:results[0].windowRmsM>0?results[1].windowRmsM/results[0].windowRmsM:null,
 limits:'Neither 72-cycle case establishes asymptotic saturation or continuum-converged pattern selection. Reflected-light images are not surface-height targets.'};
mkdirSync('ns-drive-comparison',{recursive:true});writeFileSync('ns-drive-comparison/comparison.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
