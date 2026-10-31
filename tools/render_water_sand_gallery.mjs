import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {resolve,basename} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {uniformPatternSamples} from './pattern_samples.mjs';

const finite=(x,label)=>assert.ok(typeof x==='number'&&Number.isFinite(x),label+' must be finite');
const esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const rng=seed=>{let s=seed>>>0;return()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s/4294967296;};};
export function selectPhasePair(config,summary,raw,requestedCycle){
 const {nr,nth,h,R}=config.options;
 assert.ok(Number.isInteger(nr)&&nr>0&&Number.isInteger(nth)&&nth>0,'valid grid dimensions');
 [h,R,config.frequency].forEach(x=>{finite(x,'physical dimension');assert.ok(x>0);});
 raw.forEach(f=>{finite(f.time,'sample time');assert.equal(f.eta.length,nr*nth);f.eta.forEach(x=>{finite(x,'surface elevation');assert.ok(h+x>0,'positive liquid depth');});});
 const normalized=uniformPatternSamples(config,summary,raw),frames=normalized.frames;
 const completed=Math.floor(frames.at(-1).time*config.frequency+1e-9);
 const endpoint=requestedCycle===undefined?completed:requestedCycle;
 assert.ok(Number.isInteger(endpoint)&&endpoint>=1,'at least one completed drive cycle is needed for a phase pair');
 assert.ok(endpoint<=completed,'requested phase pair is not yet available');
 const selected=[endpoint-1,endpoint].map(cycle=>{
  const f=frames[cycle*8];assert.ok(f,'scheduled phase sample exists');
  assert.ok(Math.abs(f.time*config.frequency-cycle)<=1e-8,'actual phase sample time matches requested cycle');
  return {cycle,time:f.time,eta:f.eta.slice()};
 });
 return {selected,extraTerminalObservation:normalized.extraTerminalObservation,endpointCycle:endpoint};
}
export function pointField(grid,options,eta,{count=16000,seed=2691}={}){
 assert.ok(Number.isInteger(count)&&count>0&&count<=100000,'bounded point count');
 assert.equal(grid.rf.length,options.nr+1);assert.equal(eta.length,options.nr*options.nth);
 const random=rng(seed),out=[];
 for(let n=0;n<count;n++){
  const radius=options.R*Math.sqrt(random()),theta=2*Math.PI*random();
  let lo=0,hi=options.nr;
  while(lo+1<hi){const m=(lo+hi)>>1;if(grid.rf[m]<=radius)lo=m;else hi=m;}
  const k=Math.floor(theta/(2*Math.PI/options.nth)+0.5)%options.nth;
  const value=eta[lo*options.nth+k];finite(value,'rendered elevation');
  out.push({x:radius/options.R*Math.cos(theta),y:radius/options.R*Math.sin(theta),eta:value,cell:lo*options.nth+k});
 }
 return out;
}
function rms(grid,options,eta){
 let sum=0,area=0;
 for(let i=0;i<options.nr;i++){
  const w=(grid.rf[i+1]**2-grid.rf[i]**2)/2;
  for(let k=0;k<options.nth;k++){sum+=w*eta[i*options.nth+k]**2;area+=w;}
 }return Math.sqrt(sum/area);
}
export function renderPanel(record,frame,index,scale,{count=16000,seed=2691}={}){
 const points=pointField(record.grid,record.config.options,frame.eta,{count,seed});
 const o=record.config.options,f=record.config.frequency;
 let svg='<rect width="640" height="770" fill="#10171c"/>';
 svg+='<circle cx="320" cy="332" r="273" fill="#172129" stroke="#586772" stroke-width="1"/>';
 for(const p of points){
  const relative=scale>0?Math.min(1,Math.abs(p.eta)/scale):0;
  const color=p.eta>=0?'#eacb8b':'#82c8db',opacity=0.13+0.82*Math.sqrt(relative);
  svg+=`<circle cx="${(320+270*p.x).toFixed(3)}" cy="${(332+270*p.y).toFixed(3)}" r="1.05" fill="${color}" opacity="${opacity.toFixed(4)}"/>`;
 }
 svg+=`<text x="26" y="32" fill="#f1eee6" font-family="sans-serif" font-size="20">${esc(record.label)} · phase sample ${index===0?'I':'II'}</text>`;
 svg+=`<text x="26" y="57" fill="#b8c6ce" font-family="sans-serif" font-size="14">${f} Hz · ${(o.h*1000).toFixed(5)} mm · ${(o.accel/o.g).toFixed(3)} g peak · ${esc(o.contact)}</text>`;
 svg+=`<text x="26" y="638" fill="#f1eee6" font-family="sans-serif" font-size="16">Actual cycle ${frame.cycle} · t=${frame.time.toFixed(9)} s</text>`;
 svg+=`<text x="26" y="665" fill="#b8c6ce" font-family="sans-serif" font-size="14">Gold: +eta; blue: -eta · pair scale ±${(scale*1e6).toPrecision(5)} µm</text>`;
 svg+=`<text x="26" y="689" fill="#b8c6ce" font-family="sans-serif" font-size="14">RMS ${(rms(record.grid,o,frame.eta)*1e6).toPrecision(5)} µm · ${esc(record.summary.status)}</text>`;
 svg+='<text x="26" y="714" fill="#f0c58c" font-family="sans-serif" font-size="14">Sand-like WATER illustration, not grains or a photograph.</text>';
 svg+='<text x="26" y="738" fill="#b8c6ce" font-family="sans-serif" font-size="13">Snapshots are 1 drive period apart; phase opposition is unverified.</text>';
 svg+=`<text x="26" y="758" fill="#8a9ba5" font-family="monospace" font-size="10">Source ${esc(record.config.commit||'unknown')} / run ${esc(record.config.run||'unknown')}</text>`;
 return svg;
}
export function readCase(dir,requestedCycle){
 const read=name=>JSON.parse(readFileSync(resolve(dir,name),'utf8'));
 const config=read('config.json'),summary=read('summary.json'),grid=read('grid.json');
 const text=readFileSync(resolve(dir,'surface.ndjson'),'utf8');
 const raw=text.trim().split('\n').filter(Boolean).map(JSON.parse);
 assert.match(config.engine,/full nonlinear 3D free-surface Navier-Stokes/,'actual NS provenance required');
 assert.ok(!summary.error&&summary.status!=='solver-error','failed solver output is not illustrated as an accepted trajectory');
 assert.equal(grid.nth,config.options.nth);assert.equal(grid.rf.length,config.options.nr+1);
 grid.rf.forEach((x,i)=>{finite(x,'radial face');if(i)assert.ok(x>grid.rf[i-1]);});
 assert.ok(Math.abs(grid.rf[0])<=Number.EPSILON&&Math.abs(grid.rf.at(-1)-config.options.R)<=8*Number.EPSILON*config.options.R,'grid matches physical radius');
 return {dir,label:basename(resolve(dir)),config,summary,grid,...selectPhasePair(config,summary,raw,requestedCycle),
  surfaceSha256:createHash('sha256').update(text).digest('hex')};
}
export function generateGallery(inputs,out,{requestedCycle,count=16000,seed=2691}={}){
 mkdirSync(out,{recursive:true});
 const records=inputs.map((input,index)=>{
  try{return {...readCase(input.dir,requestedCycle),label:input.label||basename(input.dir),index};}
  catch(error){return {label:input.label||basename(input.dir),dir:input.dir,index,error:String(error.message)};}
 });
 const height=110+770*records.length;
 let gallery=`<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="${height}" viewBox="0 0 1280 ${height}"><rect width="1280" height="${height}" fill="#0b1116"/><text x="28" y="38" fill="#f1eee6" font-family="sans-serif" font-size="26">Water geometry · sand-like illustration</text><text x="28" y="69" fill="#b8c6ce" font-family="sans-serif" font-size="15">Actual Navier–Stokes snapshots. Independent amplitude scale per case; do not compare texture brightness as amplitude.</text><text x="28" y="94" fill="#b8c6ce" font-family="sans-serif" font-size="14">Equal-area deterministic stippling of cell values; no particle dynamics, mode substitution or sign-flipped snapshots.</text>`;
 const manifest={schema:'water-ns-stipple-gallery/1',display:{method:'equal-area deterministic stipple, piecewise-constant cell-centred signed eta',pointCountPerPanel:count,seed,
  color:'gold nonnegative eta, blue negative eta',opacity:'0.13+0.82*sqrt(abs(eta)/pairPeak)',scale:'same peak across each actual temporal pair; independent across cases',
  limits:'not physical sand transport; not a reflected-light forward model; texture is not an amplitude measurement'},cases:[]};
 for(const record of records){
  if(record.error){
   gallery+=`<g transform="translate(0 ${110+record.index*770})"><rect width="1280" height="770" fill="#1b1719"/><text x="28" y="50" fill="#ffb4a6" font-family="sans-serif" font-size="22">${esc(record.label)}: no accepted phase pair</text><text x="28" y="90" fill="#d9c4c0" font-family="sans-serif" font-size="14">${esc(record.error)}</text></g>`;
   manifest.cases.push({label:record.label,input:record.dir,status:'not-rendered',reason:record.error});continue;
  }
  const scale=Math.max(...record.selected.flatMap(f=>f.eta.map(Math.abs)));
  const pair=record.selected.map((frame,index)=>{
   const panel=renderPanel(record,frame,index,scale,{count,seed});
   const name=`case-${String(record.index).padStart(2,'0')}-phase-${index+1}.svg`;
   writeFileSync(resolve(out,name),`<svg xmlns="http://www.w3.org/2000/svg" width="640" height="770" viewBox="0 0 640 770">${panel}</svg>`);
   gallery+=`<g transform="translate(${index*640} ${110+record.index*770})">${panel}</g>`;
   return {file:name,timeS:frame.time,driveCycle:frame.cycle,rmsM:rms(record.grid,record.config.options,frame.eta)};
  });
  writeFileSync(resolve(out,`case-${String(record.index).padStart(2,'0')}-selected-raw.json`),JSON.stringify({sourceCommit:record.config.commit,sourceRun:record.config.run,frames:record.selected}));
  manifest.cases.push({label:record.label,input:record.dir,status:record.summary.status,sourceCommit:record.config.commit,sourceRun:record.config.run,
   surfaceSha256:record.surfaceSha256,options:record.config.options,frequencyHz:record.config.frequency,initialSeed:record.config.seed,initialRmsM:record.config.seedRms,
   pairScaleM:scale,extraTerminalObservation:record.extraTerminalObservation,phaseSamples:pair});
 }
 gallery+='</svg>';writeFileSync(resolve(out,'gallery.svg'),gallery);writeFileSync(resolve(out,'manifest.json'),JSON.stringify(manifest,null,2));
 return manifest;
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const args=process.argv.slice(2),historical=args.length===0;
 const inputs=historical?[{dir:process.env.OLD_HALF_G||'old-half-g',label:'0.5 g historical pilot'},{dir:process.env.OLD_ONE_G||'old-one-g/ns-pattern-output',label:'1.0 g historical pilot'}]:args.map(dir=>({dir}));
 const requestedCycle=process.env.GALLERY_CYCLE===undefined?(historical?72:undefined):Number(process.env.GALLERY_CYCLE);
 const manifest=generateGallery(inputs,resolve(process.env.GALLERY_OUT||'water-sand-gallery'),{requestedCycle});
 console.log(JSON.stringify(manifest,null,2));if(manifest.cases.some(c=>c.status==='not-rendered'))process.exitCode=1;
}
