import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdir, writeFile, stat } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium, webkit } from 'playwright';

const engineName=process.env.FARADAY_BROWSER||'chromium', engine={chromium,webkit}[engineName];
assert(engine,`Unknown browser: ${engineName}`);
const transport=process.env.FARADAY_QA_TRANSPORT||'file';
assert(['file','http'].includes(transport),`Unknown transport: ${transport}`);
const root=resolve(process.argv[2]||'faraday-cell');
const out=resolve(process.env.FARADAY_QA_ARTIFACTS||'faraday-browser-evidence');
await mkdir(out,{recursive:true});
const results=[],errors=[];
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
async function scenePixels(){
  return page.locator('#scene3d').evaluate(canvas=>{
    const gl=canvas.getContext('webgl2'); if(!gl)throw new Error('The 3D canvas has no WebGL2 context');
    const p=new Uint8Array(canvas.width*canvas.height*4);gl.readPixels(0,0,canvas.width,canvas.height,gl.RGBA,gl.UNSIGNED_BYTE,p);
    let lo=255,hi=0,lit=0,hash=2166136261;
    for(let i=0;i<p.length;i+=4){lo=Math.min(lo,p[i]);hi=Math.max(hi,p[i]);if(p[i]+p[i+1]+p[i+2]>100)lit++;hash=Math.imul(hash^p[i],16777619);}
    return {width:canvas.width,height:canvas.height,spread:hi-lo,lit,hash:hash>>>0,renderer:gl.getParameter(gl.RENDERER)};
  });
}
async function pause(){
  await page.locator('#pauseSolver').click();
  await page.waitForFunction(()=>{const c=CELL3D.clocks();return !c.live&&!c.pending&&!c.starting&&c.paused;},undefined,{timeout:30000});
  const before=await page.evaluate(()=>({clock:CELL3D.clocks(),camera:UNIFIED3D.scene.camera()}));
  await page.waitForTimeout(600);
  assert.deepEqual(await page.evaluate(()=>({time:CELL3D.clocks().physT,steps:CELL3D.clocks().stepsDone})),
    {time:before.clock.physT,steps:before.clock.stepsDone},'Pause did not freeze physical time and solver steps');
  return before;
}
try{
  browser=await engine.launch({headless:true,...(process.env.FARADAY_BROWSER_EXECUTABLE?{executablePath:process.env.FARADAY_BROWSER_EXECUTABLE}:{}),
    ...(engineName==='chromium'?{args:['--enable-unsafe-swiftshader']}:{} )});
  const local=transport==='file'?[
    {name:'folder-file',url:pathToFileURL(resolve(root,'index.html')).href},
    {name:'standalone-file',url:pathToFileURL(resolve(root,'faraday-cell-standalone.html')).href}
  ]:[{name:'standalone-http',url:origin+'/faraday-cell-standalone.html'}];
  for(scenario of [
    {name:'folder-http',url:origin+'/index.html'},
    {name:'source-http',url:origin+'/cymatic.html'},...local,
    {name:'folder-mobile',url:origin+'/index.html',viewport:{width:390,height:844}}
  ]){
    // The recovered default is the retained sand visualization. This existing
    // gate explicitly exercises the fluid-only diagnostic.
    scenario={...scenario,url:scenario.url+'?solver=1'};
    const context=await browser.newContext({viewport:scenario.viewport||{width:1440,height:1000}});
    page=await context.newPage();pageErrors=[];
    page.on('pageerror',e=>pageErrors.push(String(e.stack||e)));
    page.on('console',m=>{if(m.type()==='error')pageErrors.push(m.text());});
    console.log(JSON.stringify({scenario:scenario.name,stage:'load'}));
    await page.goto(scenario.url,{waitUntil:'load',timeout:30000});
    assert.match(await page.title(),/Faraday|cymatic/i);
    assert.equal(page.url(),scenario.url);
    await page.waitForFunction(()=>typeof UNIFIED3D!=='undefined'&&UNIFIED3D.scene?.stats().ready&&UNIFIED3D.scene.stats().renderCount>0,undefined,{timeout:30000});
    assert.equal(await page.locator('#scene3d').isVisible(),true);
    assert.equal(await page.locator('#workspaceSeg').isVisible(),false,'Separate Sand/NS workspaces are still exposed');
    const initial=await page.evaluate(()=>({experimentMode,rim,useSolver,model:state.model,
      stats:UNIFIED3D.scene.stats(),drawn:FARADAY_STARTUP.drawn,failed:FARADAY_STARTUP.failed}));
    assert.equal(initial.experimentMode,'general');assert.equal(initial.rim,'pinned');assert.equal(initial.useSolver,true);
    assert.equal(initial.failed,false);assert(initial.stats.triangles>100,'No actual triangulated surface');
    const pixels=await scenePixels();assert(pixels.spread>10&&pixels.lit>100,'The 3D viewport is blank');
    results.push({name:scenario.name+'-unified-3d-on-entry',initial,pixels,passed:true});
    await page.waitForFunction(()=>{const c=CELL3D.clocks(),f=UNIFIED3D.scene.stats();if(c.err)throw new Error(c.err);return c.have&&c.physT>0&&f.time===c.physT;},undefined,{timeout:90000});
    const first=await page.evaluate(()=>({clock:CELL3D.clocks(),stats:UNIFIED3D.scene.stats()}));
    assert.equal(first.stats.initial,false);assert.equal(first.stats.time,first.clock.physT);
    assert.equal(first.stats.steps,first.clock.stepsDone);
    assert(first.stats.particles>0,'The shared state has no particles');
    await page.screenshot({path:resolve(out,scenario.name+'-running.png'),fullPage:true});
    results.push({name:scenario.name+'-actual-shared-evolution',first,passed:true});
    const frozen=await pause();
    const camera=await page.evaluate(()=>UNIFIED3D.scene.camera());
    const before=await scenePixels();
    await page.locator('#scene3d').focus();await page.keyboard.press('ArrowRight');
    await page.waitForFunction(c=>UNIFIED3D.scene.camera().yaw!==c.yaw,camera);
    const after=await scenePixels();assert.notEqual(after.hash,before.hash,'Orbit did not change the rendered perspective');
    assert.deepEqual(await page.evaluate(()=>({time:CELL3D.clocks().physT,steps:CELL3D.clocks().stepsDone})),
      {time:frozen.clock.physT,steps:frozen.clock.stepsDone},'Camera motion changed physical time');
    await page.screenshot({path:resolve(out,scenario.name+'-rotated-paused.png'),fullPage:true});
    results.push({name:scenario.name+'-pause-and-real-camera',before,after,passed:true});
    await page.locator('#pauseSolver').click();
    await page.waitForFunction(t=>{const c=CELL3D.clocks();if(c.err)throw new Error(c.err);return c.physT>t;},frozen.clock.physT,{timeout:90000});
    await pause();
    await page.locator('#resetSolver').click();
    await page.waitForFunction(()=>{const c=CELL3D.clocks(),f=UNIFIED3D.scene.stats();return c.physT===0&&c.stepsDone===0&&f.initial&&f.time===0;});
    assert.match(await page.locator('#solverStatus').innerText(),/initial|not evolved/i);
    assert.deepEqual(pageErrors,[],`${scenario.name} runtime errors`);
    results.push({name:scenario.name+'-resume-reset-and-console',passed:true,errors:pageErrors});
    await context.close();page=null;
  }
  const context=await browser.newContext({javaScriptEnabled:false});page=await context.newPage();
  await page.goto(origin+'/index.html?solver=1');assert.match(await page.locator('#startupStatus').innerText(),/Not started/);
  await page.screenshot({path:resolve(out,'javascript-disabled-notice.png'),fullPage:true});
  results.push({name:'javascript-disabled-preview',passed:true});await context.close();page=null;
}catch(error){
  errors.push(String(error.stack||error));process.exitCode=1;
  if(page)try{await page.screenshot({path:resolve(out,`${scenario?.name||'startup'}-failure.png`),fullPage:true});
    await writeFile(resolve(out,`${scenario?.name||'startup'}-failure.html`),await page.content());
    errors.push({scenario:scenario?.name,url:page.url(),browserErrors:pageErrors,clocks:await page.evaluate(()=>CELL3D.clocks()),startupStatus:await page.locator('#startupStatus').innerText()});
  }catch(e){errors.push('Failure capture: '+String(e));}
}finally{
  if(browser)await browser.close();await new Promise(r=>server.close(r));
  const scope='One evolving pinned nonlinear state in an interactive depth-tested 3D viewport; shared particle/solver timestamps; actual pause/resume/reset; '
    +(transport==='file'?'includes file:// opening':'explicit HTTP scope; file:// not validated');
  await writeFile(resolve(out,'results.json'),JSON.stringify({engine:engineName,transport,scope,results,errors},null,2));
  console.log(JSON.stringify({engine:engineName,transport,scope,results,errors},null,2));
}
