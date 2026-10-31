import assert from 'node:assert/strict';
export function uniformPatternSamples(config,summary,raw){
 assert.ok(raw.length>0,'saved samples exist');
 const endpoint=config.cycles/config.frequency,nominal=config.cycles*8;
 let frames=raw,extraTerminalObservation=null;
 if(Number.isInteger(nominal)&&summary.status==='completed-horizon'&&raw.length===nominal+2){
  const scheduled=raw[nominal],extra=raw[nominal+1],timeTolerance=4*Number.EPSILON*Math.max(1,endpoint);
  assert.ok(Math.abs(scheduled.time-endpoint)<=timeTolerance,'last scheduled sample is at horizon');
  assert.ok(extra.time>scheduled.time&&extra.time-scheduled.time<=timeTolerance,'extra terminal observation differs only by floating-point endpoint time');
  assert.ok(Math.abs(extra.time-endpoint)<=timeTolerance,'extra terminal observation is at horizon');
  assert.equal(extra.eta.length,scheduled.eta.length);
  let peak=config.seedRms,maxDelta=0;
  for(let i=0;i<extra.eta.length;i++){
   assert.ok(typeof extra.eta[i]==='number'&&Number.isFinite(extra.eta[i])&&typeof scheduled.eta[i]==='number'&&Number.isFinite(scheduled.eta[i]),'terminal surface is finite');
   peak=Math.max(peak,Math.abs(extra.eta[i]),Math.abs(scheduled.eta[i]));maxDelta=Math.max(maxDelta,Math.abs(extra.eta[i]-scheduled.eta[i]));
  }
  const surfaceTolerance=1024*Number.EPSILON*peak;
  assert.ok(maxDelta<=surfaceTolerance,'extra endpoint state change is at roundoff scale, not another physical observation');
  extraTerminalObservation={rawIndex:nominal+1,time:extra.time,scheduledTime:scheduled.time,maxSurfaceDeltaM:maxDelta,surfaceRoundoffBoundM:surfaceTolerance,
   handling:'raw record retained and separately validated; excluded only from uniform-cadence statistics'};
  frames=raw.slice(0,-1);
 }
 if(summary.status==='completed-horizon'&&Number.isInteger(nominal))assert.equal(frames.length,nominal+1,'completed horizon contains every scheduled sample');
 frames.forEach((f,j)=>assert.ok(typeof f.time==='number'&&Number.isFinite(f.time)&&Math.abs(f.time-j/(8*config.frequency))<=1e-10*Math.max(1,f.time),'snapshot cadence is one eighth drive period'));
 return {frames,extraTerminalObservation};
}
