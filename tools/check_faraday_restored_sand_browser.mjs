import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, stat } from 'node:fs/promises';
import { resolve, extname, sep, join } from 'node:path';
import { tmpdir } from 'node:os';
import { pathToFileURL } from 'node:url';
import { chromium, webkit } from 'playwright';

// Recovery gate for the retained sand visualization, not a validation of
// resolved sediment physics. The old export computes its own deterministic
// high-resolution deposit; it is not a screenshot of the preview particles.
const root=resolve(process.argv[2]||'faraday-cell');
const entry=process.env.FARADAY_QA_ENTRY||'index.html';
const transport=process.env.FARADAY_QA_TRANSPORT||'http';
assert(['http','file'].includes(transport));
const engineName=process.env.FARADAY_BROWSER||'chromium',engine={chromium,webkit}[engineName];
assert(engine,`Unknown browser: ${engineName}`);
const out=resolve(process.env.FARADAY_QA_ARTIFACTS||join(tmpdir(),`faraday-restored-sand-${engineName}`));
await mkdir(out,{recursive:true});
const nativePath=process.env.FARADAY_QA_NATIVE_RECORDING;
const nativeText=nativePath?await readFile(nativePath,'utf8'):null;
const nativeFrames=nativeText?.trim().split('\n').map(JSON.parse).filter(r=>r.type==='frame');
if(nativePath)assert(nativeFrames.length>1,'Provide an actual native multi-frame CLI recording');

// Digests of the retained implementations at recovery baseline 2017485.
// Defaults, navigation and import plumbing may change; these algorithms may not
// silently change under an "original sand restored" claim.
const originalFunctions={
  sprinkle:'9870d1e5517ccd1eb42a287a3bf2a49910ad29f777ffcf72e3ff76458b6a574c',
  stepGrains:'b24bd5990053592355a279a9a2b82da817222b977a4011a068eddec02daaba0b',
  binGrains:'dfd0097d181d0b0408f5fbea7dc72491ea88792e611ada25e355a27452c79ca4',
  mulberry32:'24a545cb0d02e5c898eda19c2e830cf785ba4c0a0526626010eb05d90c5c4300',
  finishSurface:'7fc6bb63a9efc560e9bc80049996e0a9ad568af6178b3a1dbbef2315237c1ad0',
  renderSurfaceCpu:'c2e8e9174e44033469c8004824f355f24a2abc0cc1f12ae872e798613cb5d4b8',
  exportDiagnosticImage:'0ac8bd2dd4a8fd299caa3f487ddb07e105b15ca33cd78241bacf93793e4d6be6'
};
const results=[],failures=[],skipped=[];
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
const url=transport==='file'?pathToFileURL(resolve(root,entry)).href:`http://127.0.0.1:${server.address().port}/${entry}`;
let browser,page,scenario,pageErrors;
const record=(name,evidence={})=>{const r={scenario:scenario.name,name,passed:true,...evidence};results.push(r);console.log(JSON.stringify(r));};
async function openControl(selector){
  const ancestors=page.locator(selector).locator('xpath=ancestor::details');
  for(let i=0;i<await ancestors.count();i++){
    const details=ancestors.nth(i);
    if(!await details.evaluate(e=>e.open))await details.locator(':scope > summary').click();
  }
}
async function snapshot(name){return page.evaluate(name=>{
  __sandQA.saved[name]=__sandQA.snapshot();return __sandQA.summary(__sandQA.saved[name]);
},name);}
async function equalSnapshot(name,label,ignoreEngine=false){
  const differences=await page.evaluate(({name,ignoreEngine})=>{
    const before=__sandQA.saved[name],after=__sandQA.snapshot();
    return __sandQA.diff(ignoreEngine?{...before,settings:{...before.settings,renderEngine:after.settings.renderEngine}}:before,after);
  },{name,ignoreEngine});
  assert.deepEqual(differences,[],label);
}
async function pause(){
  if(await page.evaluate(()=>running))await page.locator('#play').click();
  await page.waitForFunction(()=>!running&&!dirty);
  const frozen=await snapshot('paused');await page.waitForTimeout(250);
  await equalSnapshot('paused','Pause changed particle arrays, bins, settings or time');return frozen;
}
async function picture(){return page.evaluate(()=>{
  // The GPU intentionally discards its framebuffer after compositing. Read in
  // the drawing task, as the existing renderer parity gate does.
  if(renderEngine==='gpu')GPU_SURFACE.draw(state);
  const p=renderEngine==='gpu'?GPU_SURFACE.readRGBA():ctx.getImageData(0,0,GR,GR).data;
  let lo=255,hi=0,lit=0,hash=2166136261;
  for(let i=0;i<p.length;i+=4){lo=Math.min(lo,p[i]);hi=Math.max(hi,p[i]);if(p[i]+p[i+1]+p[i+2]>150)lit++;hash=Math.imul(hash^p[i],16777619);}
  return{width:GR,height:GR,spread:hi-lo,lit,hash:hash>>>0};
});}
async function nativeRoundTrip(resume){
  if((await page.evaluate(()=>running))!==resume)await page.locator('#play').click();
  await openControl('#importNative');
  const chooserPromise=page.waitForEvent('filechooser');await page.locator('#importNative').click();
  const chooser=await chooserPromise;
  // The capture listener runs in the same event task as the import handler, so
  // a running preview cannot take a hidden extra step between capture and enter.
  await page.evaluate(()=>{
    document.querySelector('#nativeFile').addEventListener('change',()=>{
      __sandQA.beforeNative=__sandQA.snapshot();__sandQA.nativeEntered=performance.now();
    },{once:true,capture:true});
    document.querySelector('#returnFluid').addEventListener('click',()=>{
      __sandQA.afterNative=__sandQA.snapshot();__sandQA.nativeReturned=performance.now();
    },{once:true});
  });
  await chooser.setFiles(resolve(nativePath));
  await page.waitForFunction(()=>NATIVE_PLAYBACK.status().active&&!NATIVE_PLAYBACK.status().loading);
  assert.equal(await page.evaluate(()=>NATIVE_PLAYBACK.status().failure),null);
  assert.equal(await page.evaluate(()=>NATIVE_PLAYBACK.status().complete),true);
  const imported=await page.evaluate(()=>({engine:UNIFIED3D.currentFrame().source.engine,geometry:UNIFIED3D.scene.stats().particleGeometry,time:UNIFIED3D.currentFrame().time}));
  assert.equal(imported.engine,'native-cpp-recording');assert.equal(imported.geometry,'physical-radius spheres');
  await page.waitForTimeout(1100);
  assert.deepEqual(await page.evaluate(()=>__sandQA.diffArrays(__sandQA.beforeNative,__sandQA.snapshot())),[],'Native playback mutated the retained sand arrays');
  await openControl('#returnFluid');await page.locator('#returnFluid').click();
  const evidence=await page.evaluate(()=>({differences:__sandQA.diff(__sandQA.beforeNative,__sandQA.afterNative),
    before:__sandQA.summary(__sandQA.beforeNative),after:__sandQA.summary(__sandQA.afterNative),
    nativeElapsedMs:__sandQA.nativeReturned-__sandQA.nativeEntered,native:NATIVE_PLAYBACK.status().active,useSolver}));
  assert.deepEqual(evidence.differences,[],'Native round trip changed saved sand particles, clock or controls');
  assert.equal(evidence.native,false);assert.equal(evidence.useSolver,false);
  if(resume){
    await page.waitForFunction(()=>simT>__sandQA.afterNative.settings.simT);
    const clock=await page.evaluate(()=>({advance:simT-__sandQA.afterNative.settings.simT,elapsed:(performance.now()-__sandQA.nativeReturned)/1000}));
    assert(clock.advance>0&&clock.advance<=clock.elapsed+.05,'Sand clock included time spent in native playback');
    await pause();evidence.resumedClock=clock;
  }else{await page.waitForTimeout(250);assert.deepEqual(await page.evaluate(()=>__sandQA.diff(__sandQA.afterNative,__sandQA.snapshot())),[],'Paused sand resumed after native playback');}
  record(`native-round-trip-preserves-${resume?'running':'paused'}-sand-exactly`,evidence);
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
      globalThis.__sandExportPerf={enabled:false,count:0,last:null,maxGap:0,labels:[],longTasks:[]};
      if(globalThis.PerformanceObserver?.supportedEntryTypes?.includes('longtask'))new PerformanceObserver(list=>{
        if(__sandExportPerf.enabled)for(const entry of list.getEntries())__sandExportPerf.longTasks.push(entry.duration);
      }).observe({type:'longtask',buffered:true});
      function tick(){const p=__sandExportPerf,now=performance.now();if(p.enabled){if(p.last!==null)p.maxGap=Math.max(p.maxGap,now-p.last);p.count++;
        const label=document.querySelector('#expBtn')?.textContent;if(label!==p.labels.at(-1))p.labels.push(label);}p.last=now;requestAnimationFrame(tick);}
      requestAnimationFrame(tick);
    });
    await page.goto(url,{waitUntil:'load',timeout:30000});
    await page.waitForFunction(()=>typeof state!=='undefined'&&state&&FARADAY_STARTUP.drawn&&!dirty);
    assert.equal(await page.locator('#workspaceSeg').isVisible(),false,'Separate workspaces are exposed');
    const initial=await page.evaluate(()=>({useSolver,view,rim,running,NP,renderEngine,failed:FARADAY_STARTUP.failed}));
    assert.deepEqual(initial,{useSolver:false,view:'sand',rim:'pinned',running:true,NP:60000,renderEngine:'cpu',failed:false});
    assert(await page.locator('#c').isVisible());assert.equal(await page.locator('#scene3d').isVisible(),false);
    assert.equal(await page.locator('#runSolver').isVisible(),false);
    for(const selector of ['#play','#sprinkleBtn','#reset','#fnum'])assert(await page.locator(selector).isVisible(),`Missing restored control ${selector}`);
    const source=await page.evaluate(()=>Object.fromEntries([sprinkle,stepGrains,binGrains,mulberry32,finishSurface,renderSurfaceCpu,exportDiagnosticImage].map(fn=>[fn.name,fn.toString()])));
    for(const [name,digest] of Object.entries(originalFunctions))assert.equal(createHash('sha256').update(source[name]).digest('hex'),digest,`Retained ${name} implementation changed`);
    assert.deepEqual(await page.evaluate(()=>({DRIFT,DAMP,AGIT,PHI_RCP_PREVIEW,TOPPLE_BUDGET})),{DRIFT:2.2,DAMP:.62,AGIT:.85,PHI_RCP_PREVIEW:.84,TOPPLE_BUDGET:60000});
    await page.evaluate(()=>{
      const arrays=()=>({PX,PY,VX,VY,BIN,OCC});
      function hash(a){let h=2166136261;for(const b of new Uint8Array(a.buffer,a.byteOffset,a.byteLength))h=Math.imul(h^b,16777619);return h>>>0;}
      function snapshot(){return{arrays:Object.fromEntries(Object.entries(arrays()).map(([key,value])=>[key,value.slice()])),
        settings:{...captureWorkspace(),useSolver,renderEngine,running,simT,settle,swapAcc},
        controls:Object.fromEntries(['f','fnum','a','t','h','ag','configurationMode','comparePaper','referenceCell'].map(id=>{const e=document.getElementById(id);return[id,{value:e.value,checked:e.checked}];}))};}
      function diffArrays(a,b){const diff=[];for(const key of Object.keys(a.arrays)){const x=a.arrays[key],y=b.arrays[key];
        const p=new Uint8Array(x.buffer,x.byteOffset,x.byteLength),q=new Uint8Array(y.buffer,y.byteOffset,y.byteLength);
        if(p.length!==q.length||p.some((v,i)=>v!==q[i]))diff.push(key);}return diff;}
      function diff(a,b){const result=diffArrays(a,b);for(const key of ['settings','controls'])if(JSON.stringify(a[key])!==JSON.stringify(b[key]))result.push(key);return result;}
      function summary(a){return{hashes:Object.fromEntries(Object.entries(a.arrays).map(([key,value])=>[key,hash(value)])),settings:a.settings,
        particles:a.arrays.PX.length,binTotal:a.arrays.BIN.reduce((a,b)=>a+b,0),finite:['PX','PY','VX','VY'].every(k=>a.arrays[k].every(Number.isFinite))};}
      globalThis.__sandQA={snapshot,diff,diffArrays,summary,saved:{}};
    });
    const start=await snapshot('start');await page.waitForTimeout(350);const moved=await snapshot('moved');
    assert.notEqual(start.hashes.PX,moved.hashes.PX,'Default sand is static');assert(moved.settings.simT>start.settings.simT);
    assert(moved.finite);assert.equal(moved.binTotal,60000);
    const image=await picture();assert(image.spread>10&&image.lit>1000,'Restored sand canvas is blank');
    await page.screenshot({path:join(out,scenario.name+'-restored-sand.png'),fullPage:true});
    record('default-original-60000-grain-sand-moves',{initial,start,moved,image,sourceDigests:originalFunctions});
    await pause();await snapshot('renderer');
    await page.evaluate(()=>{__sandQA.cpuPixels=ctx.getImageData(0,0,GR,GR).data.slice();});
    await openControl('#engSeg [data-e="gpu"]');await page.locator('#engSeg [data-e="gpu"]').click();
    await page.waitForFunction(()=>renderEngine==='gpu'&&GPU_SURFACE.ready&&!dirty&&!renderError);
    assert(await page.locator('#cgl').isVisible());await equalSnapshot('renderer','CPU/GPU selection changed paused sand state',true);
    const parity=await page.evaluate(()=>{GPU_SURFACE.draw(state);const a=__sandQA.cpuPixels,b=GPU_SURFACE.readRGBA();let max=0,different=0;for(let i=0;i<a.length;i++){const d=Math.abs(a[i]-b[i]);max=Math.max(max,d);if(d)different++;}return{max,different,channels:a.length,renderer:GPU_SURFACE.renderer};});
    assert(parity.max<=1,`CPU/GPU sand pixels diverged: ${JSON.stringify(parity)}`);
    await page.locator('#play').click();const gpuStart=await snapshot('gpuStart');await page.waitForTimeout(250);const gpuMoved=await snapshot('gpuMoved');
    assert.notEqual(gpuStart.hashes.PX,gpuMoved.hashes.PX,'GPU sand is static');await pause();
    await page.locator('#engSeg [data-e="cpu"]').click();await page.waitForFunction(()=>renderEngine==='cpu'&&!dirty);
    record('cpu-gpu-render-same-paused-grains-and-both-animate',{parity,gpuStart:gpuStart.hashes,gpuMoved:gpuMoved.hashes});

    const beforeSprinkle=await snapshot('beforeSprinkle');await page.locator('#sprinkleBtn').click();await page.waitForFunction(()=>!dirty);
    const sprinkled=await snapshot('sprinkled');assert.notEqual(sprinkled.hashes.PX,beforeSprinkle.hashes.PX);assert.equal(sprinkled.settings.settle,0);
    assert.equal(sprinkled.settings.simT,beforeSprinkle.settings.simT);assert(await page.evaluate(()=>VX.every(x=>x===0)&&VY.every(x=>x===0)));
    await page.locator('#reset').click();await page.waitForFunction(()=>!dirty);assert.equal(await page.evaluate(()=>simT),0);
    assert.deepEqual(await page.evaluate(()=>__sandQA.diffArrays(__sandQA.saved.sprinkled,__sandQA.snapshot())),[],'Re-excite re-sprinkled particles');
    await page.locator('#fnum').fill('115');await page.locator('#fnum').press('Tab');await page.waitForFunction(()=>freq===115&&!dirty);
    await openControl('#phaseSeg [data-p="-1"]');await page.locator('#phaseSeg [data-p="-1"]').click();await page.waitForFunction(()=>phaseSign===-1&&!dirty);
    const sandBeforeView=await picture();
    await page.locator('#viewSeg [data-v="height"]').click();await page.waitForFunction(()=>view==='height'&&!dirty);
    assert.notEqual((await picture()).hash,sandBeforeView.hash);await page.locator('#viewSeg [data-v="sand"]').click();await page.waitForFunction(()=>view==='sand'&&!dirty);
    assert.equal(await page.evaluate(()=>running),false);record('restored-resprinkle-reexcite-frequency-phase-and-view-controls-work');

    if(nativePath){
      // A nonzero auto-phase accumulator catches accidental phase-cadence reset.
      await page.locator('#phaseSeg [data-p="swap"]').click();await page.locator('#play').click();
      await page.waitForFunction(()=>swapAcc>0);await pause();await nativeRoundTrip(false);
      await page.locator('#engSeg [data-e="gpu"]').click();await page.waitForFunction(()=>renderEngine==='gpu'&&!dirty);
      await nativeRoundTrip(true);
      await page.locator('#engSeg [data-e="cpu"]').click();await page.waitForFunction(()=>renderEngine==='cpu'&&!dirty);
    }
    else skipped.push({scenario:scenario.name,check:'native-round-trip',reason:'FARADAY_QA_NATIVE_RECORDING not supplied; no fabricated native data'});

    if(scenario.name==='desktop'&&process.env.FARADAY_QA_SKIP_EXPORT!=='1'){
      // Restore the documented default field before exercising its retained
      // high-resolution sand export; changing frequency can change mode count.
      await page.locator('#fnum').fill('111');await page.locator('#fnum').press('Tab');
      await page.locator('#phaseSeg [data-p="1"]').click();await page.waitForFunction(()=>freq===111&&phaseSign===1&&!dirty);
      const beforeDrive=await page.evaluate(()=>simT);await page.locator('#play').click();
      await page.waitForFunction(t=>simT>t+.7,beforeDrive);
      await pause();const preview=await snapshot('export');
      assert(await page.evaluate(()=>state.expression>0),'Export must exercise nonzero pattern transport');
      if(process.env.FARADAY_QA_EXPORT_SIZE)await page.locator(`#expSeg [data-s="${process.env.FARADAY_QA_EXPORT_SIZE}"]`).click();
      const size=Number(await page.locator('#expSeg [aria-pressed="true"]').getAttribute('data-s'));if(!process.env.FARADAY_QA_EXPORT_SIZE)assert.equal(size,3840);
      // Match the retained browser gate's ten-minute allowance. The full pinned
      // superposition is not reduced to make a download deadline pass.
      const downloadPromise=page.waitForEvent('download',{timeout:600000}).then(download=>({download}),error=>({error}));
      const started=Date.now();await page.evaluate(()=>{__sandExportPerf.enabled=true;__sandExportPerf.last=performance.now();});
      await page.locator('#expBtn').click();
      const refusal=page.waitForFunction(()=>!exporting&&/^export failed:/.test(document.querySelector('#expNote').textContent),undefined,{timeout:600000})
        .then(async()=>({error:new Error(await page.locator('#expNote').textContent())}),error=>({error}));
      const result=await Promise.race([downloadPromise,refusal]);if(result.error)throw result.error;
      assert.equal(await result.download.failure(),null);await page.waitForFunction(()=>!exporting);
      const feedback=await page.evaluate(()=>{__sandExportPerf.enabled=false;return __sandExportPerf;});
      const filename=result.download.suggestedFilename();assert.match(filename,/modal.*sand/);
      const path=join(out,`restored-sand-${size}.png`);await result.download.saveAs(path);const png=await readFile(path);
      assert.deepEqual(Array.from(png.subarray(0,8)),[137,80,78,71,13,10,26,10]);assert.equal(png.readUInt32BE(16),size);assert.equal(png.readUInt32BE(20),size);
      const decoded=await page.evaluate(async encoded=>{const image=new Image();image.src='data:image/png;base64,'+encoded;await image.decode();
        const c=document.createElement('canvas');c.width=512;c.height=512;const ctx=c.getContext('2d');ctx.drawImage(image,0,0,512,512);
        const p=ctx.getImageData(0,0,512,512).data;let lo=255,hi=0,lit=0;for(let i=0;i<p.length;i+=4){lo=Math.min(lo,p[i]);hi=Math.max(hi,p[i]);if(p[i]+p[i+1]+p[i+2]>150)lit++;}
        return{width:image.naturalWidth,height:image.naturalHeight,spread:hi-lo,lit};},png.toString('base64'));
      assert(decoded.spread>10&&decoded.lit>1000,'4K sand export is blank');await equalSnapshot('export','Export mutated the paused preview');
      assert(feedback.count>2&&feedback.labels.length>2,'Export did not show progress across animation frames');
      assert(feedback.maxGap<1000,`Export froze animation feedback for a second (${feedback.maxGap} ms)`);
      assert(Math.max(0,...feedback.longTasks)<1000,'Export blocked the main thread for a second');
      record('real-4k-sand-download-preserves-preview-and-reports-progress',{size,filename,bytes:png.length,decoded,durationMs:Date.now()-started,preview,feedback,
        exportKind:'Retained deterministic high-resolution sand calculation, not a preview screenshot or resolved grain-fluid result'});
    }else if(scenario.name==='desktop')skipped.push({scenario:scenario.name,check:'4k-export',reason:'FARADAY_QA_SKIP_EXPORT=1'});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'Restored UI overflows viewport');
    assert.deepEqual(pageErrors,[]);record('viewport-and-runtime-health',{errors:pageErrors});await context.close();page=null;
  }
  scenario={name:'cpu-without-webgl2'};
  const context=await browser.newContext({viewport:{width:1440,height:1000}});
  page=await context.newPage();pageErrors=[];page.on('pageerror',e=>pageErrors.push(String(e)));
  await page.addInitScript(()=>{const original=HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext=function(type,...args){return type==='webgl2'?null:original.call(this,type,...args);};});
  await page.goto(url,{waitUntil:'load',timeout:30000});
  await page.waitForFunction(()=>typeof state!=='undefined'&&state&&FARADAY_STARTUP.drawn&&simT>0&&!dirty);
  assert.equal(await page.evaluate(()=>FARADAY_STARTUP.failed),false);assert(await page.locator('#c').isVisible());
  const cpuPicture=await picture();assert(cpuPicture.spread>10&&cpuPicture.lit>1000);
  await openControl('#engSeg [data-e="gpu"]');await page.locator('#engSeg [data-e="gpu"]').click();
  await page.waitForFunction(()=>renderEngine==='gpu'&&!!renderError);assert.match(await page.locator('#renderStatus').textContent(),/GPU.*paused/i);
  await page.locator('#engSeg [data-e="cpu"]').click();await page.waitForFunction(()=>renderEngine==='cpu'&&!renderError&&!dirty);
  assert(await page.locator('#c').isVisible());assert.deepEqual(pageErrors,[]);
  record('default-cpu-sand-does-not-require-webgl2-and-explicit-gpu-failure-recovers',{cpuPicture});
  await context.close();page=null;
}catch(error){
  failures.push(String(error.stack||error));process.exitCode=1;
  if(page)try{await page.screenshot({path:join(out,`${scenario?.name||'page'}-failure.png`),fullPage:true});
    failures.push({scenario:scenario?.name,url:page.url(),browserErrors:pageErrors,state:await page.evaluate(()=>({useSolver,nativeMode,view,running,simT,settle,renderEngine,renderError,exporting,
      progress:document.querySelector('#expBtn').textContent,error:document.querySelector('#expNote').textContent,feedback:__sandExportPerf}))});
  }catch(e){failures.push('Failure capture: '+String(e));}
}finally{
  if(browser)await browser.close();await new Promise(r=>server.close(r));
  const scope=`${transport} desktop/mobile retained sand recovery: original algorithms, motion, CPU/GPU parity, controls, optional real native round trips and actual export; not sediment-physics validation`;
  const report={engine:engineName,transport,scope,nativeRecordingPath:nativePath||null,results,skipped,failures};
  await writeFile(join(out,'results.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
}
