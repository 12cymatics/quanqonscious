import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, stat } from 'node:fs/promises';
import { resolve, extname, sep, join } from 'node:path';
import { tmpdir } from 'node:os';
import { chromium, webkit } from 'playwright';

// This gate exercises the delivered Sand visualization and 3D scene, each with a real default 4K download.
// It never substitutes a modal field for the solver, a mock solver, a tiny export, or a synthetic click.
const root=resolve(process.argv[2]||'faraday-cell');
const entry=process.env.FARADAY_QA_ENTRY||'index.html';
const engineName=process.env.FARADAY_BROWSER||'chromium',engine={chromium,webkit}[engineName];
assert(engine,`Unknown browser: ${engineName}`);
const out=resolve(process.env.FARADAY_QA_ARTIFACTS||join(tmpdir(),`faraday-unified-${engineName}`));
await mkdir(out,{recursive:true});
const results=[],failures=[];
const server=createServer(async(req,res)=>{
  try{
    const u=new URL(req.url,'http://localhost');
    if(u.pathname==='/favicon.ico'){res.writeHead(204);res.end();return;}
    const path=resolve(root,'.'+decodeURIComponent(u.pathname));
    if(path!==root&&!path.startsWith(root+sep)){res.writeHead(403);res.end();return;}
    const target=(await stat(path)).isDirectory()?resolve(path,'index.html'):path;
    res.writeHead(200,{'Content-Type':{'.html':'text/html; charset=utf-8','.js':'text/javascript','.wasm':'application/wasm'}[extname(target)]||'application/octet-stream'});
    res.end(await readFile(target));
  }catch(error){res.writeHead(404);res.end(String(error));}
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const origin=`http://127.0.0.1:${server.address().port}`;
let browser,page,scenario,pageErrors;
const record=(name,evidence={})=>{const r={scenario:scenario.name,name,passed:true,...evidence};results.push(r);console.log(JSON.stringify(r));};
async function pixels(){return page.locator('#scene3d').evaluate(canvas=>{
  const gl=canvas.getContext('webgl2');if(!gl)throw new Error('Missing WebGL2 scene context');
  const rgba=new Uint8Array(canvas.width*canvas.height*4);gl.readPixels(0,0,canvas.width,canvas.height,gl.RGBA,gl.UNSIGNED_BYTE,rgba);
  let hash=2166136261,lo=255,hi=0,lit=0;
  for(let i=0;i<rgba.length;i+=4){hash=Math.imul(hash^rgba[i],16777619);lo=Math.min(lo,rgba[i]);hi=Math.max(hi,rgba[i]);if(rgba[i]+rgba[i+1]+rgba[i+2]>100)lit++;}
  return{width:canvas.width,height:canvas.height,hash:hash>>>0,spread:hi-lo,lit};
});}
async function sandPixels(){return page.locator('#c').evaluate(canvas=>{
  const a=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;let lo=255,hi=0,hash=2166136261;
  for(let i=0;i<a.length;i+=4){lo=Math.min(lo,a[i]);hi=Math.max(hi,a[i]);hash=Math.imul(hash^a[i],16777619);}
  return{width:canvas.width,height:canvas.height,spread:hi-lo,hash:hash>>>0};
});}
async function sandSettings(){return page.evaluate(()=>({experimentMode,cellKey,freq,amp,temp,depth,rim,view,
  generalInput:JSON.parse(JSON.stringify(generalInput))}));}
async function physics(){return page.evaluate(()=>{
  function hash(array){const bytes=new Uint8Array(array.buffer,array.byteOffset,array.byteLength);let h=2166136261;for(const b of bytes)h=Math.imul(h^b,16777619);return h>>>0;}
  const f=UNIFIED3D.currentFrame(),c=CELL3D.clocks();
  return{time:f.time,steps:f.steps,generation:f.generation,initial:f.initial,
    positionsHash:hash(f.positions),particlesHash:hash(f.particles),flowHash:hash(f.flowVectors),
    particleCount:f.particles.length/3,clockTime:c.physT,clockSteps:c.stepsDone,
    input:JSON.parse(JSON.stringify(generalInput)),frequency:freq};
});}
async function pause(){
  const c=await page.evaluate(()=>CELL3D.clocks());
  if(c.live||c.starting)await page.locator('#pauseSolver').click();
  await page.waitForFunction(()=>{const c=CELL3D.clocks();return !c.live&&!c.pending&&!c.starting&&c.paused;},undefined,{timeout:30000});
  const a=await physics();await page.waitForTimeout(600);assert.deepEqual(await physics(),a,'Pause did not freeze the shared fluid and particle state');return a;
}
async function visibleControls(selectors){
  await page.evaluate(()=>scrollTo(0,0));
  const controls=await page.evaluate(selectors=>selectors.map(selector=>{
    const e=document.querySelector(selector),r=e?.getBoundingClientRect();
    if(!e||!r)return{selector,missing:true};
    const at=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
    return{selector,visible:!!r.width&&!!r.height,inViewport:r.x>=0&&r.y>=0&&r.right<=innerWidth&&r.bottom<=innerHeight,
      unobscured:at===e||e.contains(at),rect:{x:r.x,y:r.y,width:r.width,height:r.height}};
  }),selectors);
  for(const c of controls)assert(c.visible&&c.inViewport&&c.unobscured,`Unreachable first-viewport control: ${JSON.stringify(c)}`);
  return controls;
}
try{
  browser=await engine.launch({headless:true,...(process.env.FARADAY_BROWSER_EXECUTABLE?{executablePath:process.env.FARADAY_BROWSER_EXECUTABLE}:{}),
    ...(engineName==='chromium'?{args:['--enable-unsafe-swiftshader']}:{} )});
  for(scenario of [{name:'desktop',width:1440,height:1000},{name:'mobile',width:390,height:844}]){
    const context=await browser.newContext({viewport:{width:scenario.width,height:scenario.height},acceptDownloads:true});
    page=await context.newPage();page.setDefaultTimeout(20000);pageErrors=[];
    page.on('pageerror',e=>pageErrors.push(String(e.stack||e)));
    page.on('console',m=>{if(m.type()==='error')pageErrors.push(m.text());});
    await page.addInitScript(()=>{
      globalThis.__qaExportFrames=[];globalThis.__qaExportLongTasks=[];globalThis.__qaExportHeartbeat={count:0,last:null,maxGap:0};globalThis.__qaExportMeasure=false;
      if(globalThis.PerformanceObserver?.supportedEntryTypes?.includes('longtask'))new PerformanceObserver(list=>{
        for(const e of list.getEntries())__qaExportLongTasks.push({start:e.startTime,duration:e.duration});
      }).observe({type:'longtask',buffered:true});
      function sample(){const now=performance.now(),h=__qaExportHeartbeat;if(__qaExportMeasure){
        if(h.last!==null)h.maxGap=Math.max(h.maxGap,now-h.last);h.count++;
        const text=document.getElementById('expBtn')?.textContent;if(text!==__qaExportFrames.at(-1)?.text)__qaExportFrames.push({time:now,text});}h.last=now;
        requestAnimationFrame(sample);}
      requestAnimationFrame(sample);
    });
    console.log(JSON.stringify({scenario:scenario.name,stage:'load',url:`${origin}/${entry}`}));
    await page.goto(`${origin}/${entry}`,{waitUntil:'load',timeout:30000});
    assert.match(await page.title(),/Faraday|cymatic/i);assert.equal(new URL(page.url()).pathname,'/'+entry);
    await page.waitForFunction(()=>typeof state!=='undefined'&&state&&!dirty&&!useSolver&&view==='sand'&&ETA.some(v=>v!==0));
    for(const selector of ['#viewSeg [data-v="sand"]','#play','#sprinkleBtn'])assert.equal(await page.locator(selector).isEnabled(),true,`${selector} is disabled on entry`);
    assert.equal(await page.locator('#solverOverlay').isVisible(),false);assert.equal(await page.locator('#scene3d').isVisible(),false);
    const sandInitial=await sandPixels();assert(sandInitial.spread>10,'The initial sand picture is blank');
    const sandControls=await visibleControls(['#workspaceSeg','#fnum','#play','#sprinkleBtn']);
    await page.screenshot({path:join(out,scenario.name+'-initial-sand.png')});
    record('visible-sand-on-entry-with-reachable-controls',{initial:sandInitial,controls:sandControls});
    const grains=await page.evaluate(()=>Array.from(PX.slice(0,1024)));
    await page.waitForFunction(before=>PX.slice(0,before.length).some((x,i)=>x!==before[i]),grains);
    await page.waitForFunction(before=>{const c=document.getElementById('c'),a=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
      let h=2166136261;for(let i=0;i<a.length;i+=4)h=Math.imul(h^a[i],16777619);return(h>>>0)!==before;},sandInitial.hash);
    record('real-grains-and-visible-picture-move');

    if(scenario.name==='desktop'){
      if(process.env.FARADAY_QA_EXPORT_SIZE)await page.locator(`#expSeg [data-s="${process.env.FARADAY_QA_EXPORT_SIZE}"]`).click();
      const size=Number(await page.locator('#expSeg [aria-pressed="true"]').getAttribute('data-s'));
      if(!process.env.FARADAY_QA_EXPORT_SIZE)assert.equal(size,3840,'The default 4K export changed');
      const downloadPromise=page.waitForEvent('download',{timeout:600000}).then(download=>({download}),error=>({error}));
      const start=Date.now(),perfStart=await page.evaluate(()=>{__qaExportHeartbeat={count:0,last:null,maxGap:0};__qaExportMeasure=true;return performance.now();});
      await page.locator('#expBtn').click({timeout:600000});const result=await downloadPromise;if(result.error)throw result.error;
      const download=result.download;assert.equal(await download.failure(),null);
      await page.waitForFunction(()=>!exporting,undefined,{timeout:600000});
      const perfEnd=await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>{
        __qaExportMeasure=false;resolve(performance.now());
      }))));
      const feedback=await page.evaluate(({start,end})=>({frames:__qaExportFrames.filter(x=>x.time>=start&&x.time<=end),heartbeat:{...__qaExportHeartbeat},
        longestTaskMs:Math.max(0,...__qaExportLongTasks.filter(x=>x.start>=start&&x.start<=end).map(x=>x.duration))}),{start:perfStart,end:perfEnd});
      await page.evaluate(()=>{__qaExportHeartbeat={count:0,last:null,maxGap:0};});
      assert(feedback.frames.length>0,'The Sand export never displayed progress on an animation frame');
      assert(feedback.frames[0].time-perfStart<2000,'The Sand export did not paint progress within two seconds');
      assert(feedback.heartbeat.count>10&&feedback.heartbeat.maxGap<1000,`The Sand export blocked animation-frame feedback: ${JSON.stringify(feedback.heartbeat)}`);
      const path=join(out,`sand-${size}.png`);await download.saveAs(path);const png=await readFile(path);
      assert.deepEqual(Array.from(png.subarray(0,8)),[137,80,78,71,13,10,26,10]);
      const decoded=await page.evaluate(async encoded=>{
        const image=new Image();image.src='data:image/png;base64,'+encoded;await image.decode();
        const c=document.createElement('canvas');c.width=image.naturalWidth;c.height=image.naturalHeight;const ctx=c.getContext('2d');ctx.drawImage(image,0,0);
        const p=ctx.getImageData(0,0,c.width,c.height).data;let lo=255,hi=0,lit=0;
        for(let i=0;i<p.length;i+=4){lo=Math.min(lo,p[i]);hi=Math.max(hi,p[i]);if(p[i]>40)lit++;}
        return{width:c.width,height:c.height,spread:hi-lo,lit};
      },png.toString('base64'));
      assert.equal(decoded.width,size);assert.equal(decoded.height,size);
      assert(decoded.spread>10&&decoded.lit>10000,'Downloaded PNG decodes but contains no meaningful sand picture');
      assert.match(download.suggestedFilename(),new RegExp(`sand.*${size}.*\\.png$`,'i'));
      record('real-sand-export-downloads-and-decodes',{size,bytes:png.length,decoded,durationMs:Date.now()-start,filename:download.suggestedFilename(),feedback});
    }

    await page.locator('#play').click();
    await page.waitForFunction(()=>running===false&&document.getElementById('play').getAttribute('aria-pressed')==='false'
      &&document.getElementById('play').textContent==='paused',undefined,{timeout:5000});
    const paused=await page.evaluate(()=>({simT,positions:Array.from(PX.slice(0,1024))}));
    await page.waitForTimeout(500);
    assert.deepEqual(await page.evaluate(()=>({simT,positions:Array.from(PX.slice(0,1024))})),paused,'Pause did not freeze the sand');
    await page.locator('#viewSeg [data-v="height"]').click();
    await page.waitForFunction(()=>view==='height'&&!dirty);
    const heightBefore=await sandPixels(),fieldBefore=await page.evaluate(()=>Array.from(ETA));
    await page.locator('#fnum').fill('56');await page.locator('#fnum').press('Tab');
    await page.waitForFunction(()=>freq===56&&!dirty&&!renderError);
    assert.equal(await page.locator('#f').inputValue(),'56');
    const changed=await page.evaluate(before=>ETA.reduce((n,v,i)=>n+(v!==before[i]),0),fieldBefore);
    assert(changed>100,'Frequency changed the label but not the actual field');
    const heightAfterFrequency=await sandPixels();
    assert.notEqual(heightAfterFrequency.hash,heightBefore.hash,'Frequency did not redraw the paused height field');
    await page.locator('#phaseSeg [data-p="-1"]').click();
    await page.waitForFunction(()=>phaseSign===-1&&!dirty);
    assert.notEqual((await sandPixels()).hash,heightAfterFrequency.hash,'Phase did not redraw the paused height field');
    assert.deepEqual(await page.evaluate(()=>({simT,positions:Array.from(PX.slice(0,1024))})),paused,'Drawing controls advanced paused grains or time');
    await page.locator('#phaseSeg [data-p="1"]').click();
    await page.locator('#viewSeg [data-v="sand"]').click();
    await page.waitForFunction(()=>view==='sand'&&!dirty);
    const beforeSprinkle=await sandPixels();
    await page.locator('#sprinkleBtn').click();
    await page.waitForFunction(before=>PX.slice(0,before.length).reduce((n,v,i)=>n+(v!==before[i]),0)>100,paused.positions,{timeout:5000});
    const sprinkled=await page.evaluate(()=>Array.from(PX.slice(0,1024)));
    await page.waitForTimeout(200);
    assert.notEqual((await sandPixels()).hash,beforeSprinkle.hash,'Re-sprinkle did not redraw while paused');
    assert.deepEqual(await page.evaluate(()=>({simT,positions:Array.from(PX.slice(0,1024))})),{simT:paused.simT,positions:sprinkled});
    const resumeWall=Date.now();await page.locator('#play').click();
    await page.waitForFunction(t=>running&&simT>t,paused.simT);
    assert((await page.evaluate(()=>simT))-paused.simT<=(Date.now()-resumeWall)/1000+.15,'Sand resume includes elapsed paused time');
    record('paused-frequency-phase-and-resprinkle-redraw-without-advancing-grains',{changed});
    const views=[];
    for(const name of ['height','optical','nodal','sand']){
      await page.locator(`#viewSeg [data-v="${name}"]`).click();
      await page.waitForFunction(name=>view===name&&!dirty&&!renderError,name);
      const image=await sandPixels();assert(image.spread>10,`${name} rendering is blank`);views.push({name,...image});
    }
    assert(new Set(views.map(v=>v.hash)).size===views.length,'Render controls produced the same image');
    record('frequency-changes-field-and-all-render-views-work',{views});
    const sandBefore=await sandSettings();

    await page.locator('#workspaceSeg [data-workspace="ns"]').click();
    await page.waitForFunction(()=>useSolver&&typeof UNIFIED3D!=='undefined'&&UNIFIED3D.scene?.stats().ready&&UNIFIED3D.scene.stats().renderCount>0,undefined,{timeout:30000});
    assert.equal(await page.locator('#scene3d').isVisible(),true);assert.equal(await page.locator('#c').isVisible(),false);
    const initial=await page.evaluate(()=>({useSolver,experimentMode,rim,model:state.model,stats:UNIFIED3D.scene.stats()}));
    assert.equal(initial.useSolver,true);assert.equal(initial.experimentMode,'general');assert.equal(initial.rim,'pinned');
    assert(initial.stats.triangles>100&&initial.stats.surfaceVertices>100,'The scene contains no actual 3D surface mesh');
    const picture=await pixels();assert(picture.spread>10&&picture.lit>100,'The scene is blank');
    const captions=await page.evaluate(()=>['.stage .ov.bl','.stage .ov.br'].map(selector=>{
      const r=document.querySelector(selector).getBoundingClientRect();return{x:r.x,y:r.y,right:r.right,bottom:r.bottom};
    }));
    assert(captions[0].bottom<=captions[1].y||captions[1].bottom<=captions[0].y||captions[0].right<=captions[1].x||captions[1].right<=captions[0].x,'Scene captions overlap');
    const controls=await visibleControls(['#workspaceSeg','#runSolver','#pauseSolver','#resetSolver']);
    await page.screenshot({path:join(out,scenario.name+'-unified-entry.png')});
    record('single-nonlinear-scene-with-reachable-shared-controls',{initial,picture,controls,captions});

    await page.waitForFunction(()=>{const c=CELL3D.clocks(),f=UNIFIED3D.scene.stats();if(c.err)throw new Error(c.err);return c.have&&c.physT>0&&f.time===c.physT;},undefined,{timeout:90000});
    const first=await physics(),firstPicture=await pixels();
    await page.waitForFunction(t=>{const c=CELL3D.clocks(),f=UNIFIED3D.scene.stats();if(c.err)throw new Error(c.err);return c.physT>t&&f.time===c.physT&&f.steps===c.stepsDone;},first.time,{timeout:90000});
    const evolved=await physics();assert(evolved.time>first.time&&evolved.steps>first.steps);
    assert.equal(evolved.time,evolved.clockTime);assert.equal(evolved.steps,evolved.clockSteps);
    assert(evolved.particleCount>0,'There are no flow particles');
    await page.waitForFunction(before=>{
      const f=UNIFIED3D.currentFrame(),a=new Uint8Array(f.particles.buffer,f.particles.byteOffset,f.particles.byteLength);let h=2166136261;
      for(const b of a)h=Math.imul(h^b,16777619);return(h>>>0)!==before;
    },first.particlesHash,{timeout:90000,polling:200});
    const moving=await physics();assert.notEqual(moving.particlesHash,first.particlesHash,'Particles do not move with solved flow');
    assert(moving.clockTime>first.clockTime,'Particles advanced without physical time');
    await page.screenshot({path:join(out,scenario.name+'-evolving-3d.png')});
    record('actual-fluid-and-particles-advance-on-one-solved-timeline',{first,evolved:moving,firstPicture,picture:await pixels()});

    const frozen=await pause();
    const camera=await page.evaluate(()=>UNIFIED3D.scene.camera()),beforeOrbit=await pixels();
    const rect=await page.locator('#scene3d').boundingBox();assert(rect);
    const x=rect.x+rect.width*.45,y=rect.y+rect.height*.5;
    await page.mouse.move(x,y);await page.mouse.down();await page.mouse.move(x+rect.width*.17,y+rect.height*.08,{steps:8});await page.mouse.up();
    await page.waitForFunction(c=>UNIFIED3D.scene.camera().yaw!==c.yaw,camera);
    const afterOrbit=await pixels();assert.notEqual(afterOrbit.hash,beforeOrbit.hash,'Camera orbit did not change rendered perspective');
    assert.deepEqual(await physics(),frozen,'Camera orbit mutated the physical state');
    const beforeZoom=await page.evaluate(()=>UNIFIED3D.scene.camera());
    await page.locator('#scene3d').hover();await page.mouse.wheel(0,-120);
    await page.waitForFunction(c=>UNIFIED3D.scene.camera().distance!==c.distance,beforeZoom);
    assert.deepEqual(await physics(),frozen,'Camera zoom mutated the physical state');
    await page.screenshot({path:join(out,scenario.name+'-orbited-paused.png')});
    record('real-orbit-and-zoom-preserve-frozen-physics',{beforeOrbit,afterOrbit,camera:await page.evaluate(()=>UNIFIED3D.scene.camera())});

    const oldScale=await page.locator('#heightScale').inputValue();
    await page.locator('#heightScale').fill('1');await page.locator('#heightScale').press('Tab');
    await page.waitForFunction(()=>UNIFIED3D.scene.stats().heightScale===1);
    const metric=await pixels();assert.deepEqual(await physics(),frozen,'Display scale changed SI geometry or fluid state');
    await page.locator('#heightScale').fill(oldScale);await page.locator('#heightScale').press('Tab');
    await page.waitForFunction(scale=>UNIFIED3D.scene.stats().heightScale===Number(scale),oldScale);
    assert.notEqual((await pixels()).hash,metric.hash,'Explicit height magnification did not change drawing');
    assert.deepEqual(await physics(),frozen);
    await page.locator('#topView').click();
    await page.waitForFunction(()=>UNIFIED3D.scene.camera().elevation>1.5);
    const top=await pixels();await page.locator('#resetCamera').click();
    await page.waitForFunction(()=>UNIFIED3D.scene.camera().elevation<1.5);
    assert.notEqual((await pixels()).hash,top.hash,'Top view and perspective have the same projection');
    assert.deepEqual(await physics(),frozen);
    record('metric-scale-and-camera-presets-change-only-display',{metric,top});

    // Early computed velocities are physical and can be subpixel on mobile.
    // Use the public display-only arrow scale to make the layer observable,
    // while requiring the SI fluid state to remain exactly unchanged.
    const oldVectorScale=await page.locator('#vectorScale').inputValue();
    await page.locator('#vectorScale').fill('10000');await page.locator('#vectorScale').press('Tab');
    assert.equal(await page.evaluate(()=>UNIFIED3D.captureExport().options.vectorScaleSeconds),10000);
    assert.deepEqual(await physics(),frozen,'Velocity arrow display scale changed physical velocity');
    const layerResults=[];
    for(const selector of ['#showParticles','#showSurface','#showVectors']){
      const checked=await page.locator(selector).isChecked(),before=await pixels();
      await page.locator(selector).setChecked(!checked);await page.waitForTimeout(150);
      const after=await pixels();assert.notEqual(after.hash,before.hash,`${selector} did not change the visible layer`);
      assert.deepEqual(await physics(),frozen,`${selector} switched or reset the physical model`);
      await page.locator(selector).setChecked(checked);await page.waitForTimeout(100);
      layerResults.push({selector,before:before.hash,after:after.hash});
    }
    await page.locator('#vectorScale').fill(oldVectorScale);await page.locator('#vectorScale').press('Tab');
    assert.deepEqual(await physics(),frozen);
    record('surface-particles-and-velocity-are-layers-of-identical-state',{layers:layerResults,visibleArrowDisplayScaleSeconds:10000});

    if(scenario.name==='desktop'){
      if(process.env.FARADAY_QA_EXPORT_SIZE)await page.locator(`#expSeg [data-s="${process.env.FARADAY_QA_EXPORT_SIZE}"]`).click();
      const size=Number(await page.locator('#expSeg [aria-pressed="true"]').getAttribute('data-s'));
      if(!process.env.FARADAY_QA_EXPORT_SIZE)assert.equal(size,3840);
      const captured=await page.evaluate(()=>{const s=UNIFIED3D.captureExport();return{time:s.frame.time,generation:s.frame.generation,steps:s.frame.steps,camera:s.camera};});
      const downloadPromise=page.waitForEvent('download',{timeout:120000}).then(download=>({download}),error=>({error}));
      const start=Date.now(),perfStart=await page.evaluate(()=>{__qaExportMeasure=true;return performance.now();});
      await page.locator('#expBtn').click();const result=await downloadPromise;if(result.error)throw result.error;
      const download=result.download;assert.equal(await download.failure(),null);
      await page.waitForFunction(()=>!exporting);
      const perfEnd=await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>{
        __qaExportMeasure=false;resolve(performance.now());
      }))));
      const path=join(out,`unified-${size}.png`);await download.saveAs(path);const png=await readFile(path);
      assert.deepEqual(Array.from(png.subarray(0,8)),[137,80,78,71,13,10,26,10]);
      const decoded=await page.evaluate(async encoded=>{
        const image=new Image();image.src='data:image/png;base64,'+encoded;await image.decode();
        const c=document.createElement('canvas');c.width=image.naturalWidth;c.height=image.naturalHeight;const ctx=c.getContext('2d');ctx.drawImage(image,0,0);
        const p=ctx.getImageData(0,0,c.width,c.height).data;let lo=255,hi=0,lit=0;
        for(let i=0;i<p.length;i+=4){lo=Math.min(lo,p[i]);hi=Math.max(hi,p[i]);if(p[i]+p[i+1]+p[i+2]>100)lit++;}
        return{width:c.width,height:c.height,spread:hi-lo,lit};
      },png.toString('base64'));
      assert.equal(decoded.width,size);assert.equal(decoded.height,size);assert(decoded.spread>10&&decoded.lit>10000,'Export is blank');
      assert.deepEqual(await physics(),frozen,'Saving regenerated or advanced the paused fluid/particles');
      assert.deepEqual(await page.evaluate(()=>UNIFIED3D.scene.camera()),captured.camera,'Export changed the viewport camera');
      const saved=await page.evaluate(()=>UNIFIED3D.lastExport());
      assert.equal(saved.physicalTime,captured.time);assert.equal(saved.generation,captured.generation);assert.equal(saved.steps,captured.steps);
      assert.deepEqual(saved.camera,captured.camera);assert.equal(saved.particles,frozen.particleCount);
      const feedback=await page.evaluate(({start,end})=>({frames:__qaExportFrames.filter(x=>x.time>=start&&x.time<=end),heartbeat:__qaExportHeartbeat,
        longestTaskMs:Math.max(0,...__qaExportLongTasks.filter(x=>x.start>=start&&x.start<=end).map(x=>x.duration))}),{start:perfStart,end:perfEnd});
      assert(feedback.heartbeat.count>1,'Export responsiveness was not observed across animation frames');
      assert(feedback.heartbeat.maxGap<1000,`Export blocked animation-frame feedback for a second: ${JSON.stringify({durationMs:Date.now()-start,...feedback})}`);
      assert(feedback.longestTaskMs<1000,'Export blocked the main thread for a second');
      record('real-default-4k-download-is-a-snapshot-of-the-shared-scene',{size,bytes:png.length,decoded,durationMs:Date.now()-start,captured,filename:download.suggestedFilename(),feedback});
    }

    await page.locator('#seedParticles').click();
    await page.waitForFunction(before=>{
      const f=UNIFIED3D.currentFrame(),a=new Uint8Array(f.particles.buffer,f.particles.byteOffset,f.particles.byteLength);let h=2166136261;
      for(const b of a)h=Math.imul(h^b,16777619);return(h>>>0)!==before;
    },frozen.particlesHash,{timeout:30000,polling:200});
    const reseeded=await physics();assert.equal(reseeded.time,frozen.time);assert.equal(reseeded.steps,frozen.steps);
    assert.equal(reseeded.positionsHash,frozen.positionsHash);assert.equal(reseeded.flowHash,frozen.flowHash);
    assert.notEqual(reseeded.particlesHash,frozen.particlesHash);
    await page.waitForTimeout(400);assert.deepEqual(await physics(),reseeded,'Reseeding unpaused the simulation');
    record('reseed-changes-only-marker-initial-positions-in-the-same-fluid-state',{frozen,reseeded});

    await page.locator('#pauseSolver').click();
    await page.waitForFunction(t=>{const c=CELL3D.clocks();if(c.err)throw new Error(c.err);return c.physT>t;},frozen.time,{timeout:90000});
    const resumed=await pause();assert(resumed.time>frozen.time&&resumed.generation===frozen.generation,'Resume restarted the model');
    record('pause-resume-retains-the-same-fluid-and-particle-run',{frozen,resumed});

    const history=await page.evaluate(()=>UNIFIED3D.historyState());
    assert(history.frames.length>2,'Computed frames were not recorded');
    assert(history.frames.every(f=>f.time<=resumed.time),'History contains future unevolved frames');
    await page.locator('#timeline').focus();await page.locator('#timeline').press('Home');
    await page.waitForFunction(()=>!UNIFIED3D.historyState().following);
    const oldest=await page.evaluate(()=>UNIFIED3D.scene.stats());assert.equal(oldest.time,history.frames[0].time);
    await page.locator('#replayToggle').click();
    await page.waitForFunction(t=>UNIFIED3D.scene.stats().time>t,oldest.time,{timeout:12000});
    const playback=await page.evaluate(()=>({display:UNIFIED3D.scene.stats(),clock:CELL3D.clocks(),history:UNIFIED3D.historyState()}));
    assert(playback.display.time<=resumed.time,'Replay extrapolated beyond computed time');
    assert.equal(playback.clock.physT,resumed.time);assert.equal(playback.clock.stepsDone,resumed.steps);
    assert(history.frames.some(f=>f.time===playback.display.time&&f.steps===playback.display.steps),'Replay displayed an uncomputed frame');
    await page.screenshot({path:join(out,scenario.name+'-recorded-playback.png')});
    await page.locator('#liveView').click();
    await page.waitForFunction(()=>UNIFIED3D.historyState().following);
    assert.deepEqual(await physics(),resumed,'Returning to live view changed the paused simulation');
    record('recorded-motion-replays-only-computed-shared-frames',{storedFrames:history.frames.length,oldestTime:oldest.time,replayTime:playback.display.time,latestTime:resumed.time});

    await page.locator('#resetSolver').click();
    await page.waitForFunction(()=>{const c=CELL3D.clocks(),f=UNIFIED3D.scene.stats();return c.physT===0&&c.stepsDone===0&&f.initial&&f.time===0;});
    const reset=await physics();assert.equal(reset.time,0);assert.equal(reset.steps,0);assert.equal(reset.initial,true);
    await page.waitForTimeout(400);assert.deepEqual(await physics(),reset,'Reset did not hold the inspectable initial state');
    await page.locator('#runSolver').click();await page.waitForFunction(()=>{const c=CELL3D.clocks();if(c.err)throw new Error(c.err);return c.have&&c.physT>0;},undefined,{timeout:90000});
    await pause();record('reset-and-run-apply-to-the-whole-simulation',{reset});
    const fluid=async()=>{const{input,frequency,...state}=await physics();return state;};
    const solverLeft=await fluid();
    await page.locator('#workspaceSeg [data-workspace="sand"]').click();
    await page.waitForFunction(()=>!useSolver&&!dirty&&view==='sand');
    const sandAfter=await sandSettings();
    for(const key of Object.keys(sandBefore))assert.deepEqual(sandAfter[key],sandBefore[key],`Switching workspace changed Sand ${key}`);
    assert.equal(await page.locator('#c').isVisible(),true);assert.equal(await page.locator('#scene3d').isVisible(),false);
    const back=await page.evaluate(()=>Array.from(PX.slice(0,1024)));
    await page.waitForFunction(before=>PX.slice(0,before.length).some((x,i)=>x!==before[i]),back);
    assert.deepEqual(await fluid(),solverLeft,'The Sand visualization advanced or changed the fluid state');
    await visibleControls(['#workspaceSeg','#fnum','#play','#sprinkleBtn']);
    await page.screenshot({path:join(out,scenario.name+'-returned-to-sand.png')});
    record('workspaces-keep-independent-settings-and-return-to-moving-sand',{sandBefore});
    assert.deepEqual(pageErrors,[],`${scenario.name} runtime errors`);record('console-and-page-health',{errors:pageErrors});
    await context.close();page=null;
  }
}catch(error){
  failures.push(String(error.stack||error));process.exitCode=1;
  if(page)try{await page.screenshot({path:join(out,`${scenario?.name||'page'}-failure.png`),fullPage:true});
    await writeFile(join(out,`${scenario?.name||'page'}-failure.html`),await page.content());
    failures.push({scenario:scenario?.name,url:page.url(),browserErrors:pageErrors,clocks:await page.evaluate(()=>CELL3D.clocks()),startupStatus:await page.locator('#startupStatus').innerText()});
  }catch(e){failures.push('Failure capture: '+String(e));}
}finally{
  if(browser)await browser.close();await new Promise(r=>server.close(r));
  const scope='HTTP desktop/mobile: moving Sand visualization on entry with real controls and 4K export; one nonlinear 3D fluid/particle state, true camera and visual layers, actual solved motion, pause/resume/reset, real same-state 4K export; not sediment-contact validation';
  await writeFile(join(out,'results.json'),JSON.stringify({engine:engineName,scope,results,failures},null,2));
  console.log(JSON.stringify({engine:engineName,scope,results,failures},null,2));
}
