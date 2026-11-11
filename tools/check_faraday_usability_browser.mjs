import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, stat } from 'node:fs/promises';
import { resolve, extname, sep, join } from 'node:path';
import { tmpdir } from 'node:os';
import { chromium, webkit } from 'playwright';

// Run against the extracted delivery folder; use FARADAY_QA_ENTRY=cymatic.html
// to exercise the checkout before packaging. This gate uses real controls and
// an actual PNG download. It does not substitute a tiny export or a mock solver.
const root = resolve(process.argv[2] || 'faraday-cell');
const entry = process.env.FARADAY_QA_ENTRY || 'index.html';
const engineName = process.env.FARADAY_BROWSER || 'chromium';
const engine = { chromium, webkit }[engineName];
assert(engine, `Unknown browser: ${engineName}`);
const out = resolve(process.env.FARADAY_QA_ARTIFACTS || join(tmpdir(), `faraday-usability-${engineName}`));
await mkdir(out, { recursive: true });
const results = [], failures = [];
const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname === '/favicon.ico') { res.writeHead(204); res.end(); return; }
    const path = resolve(root, '.' + decodeURIComponent(url.pathname));
    if (path !== root && !path.startsWith(root + sep)) { res.writeHead(403); res.end(); return; }
    const file = (await stat(path)).isDirectory() ? resolve(path, 'index.html') : path;
    res.writeHead(200, { 'Content-Type': { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json' }[extname(file)] || 'application/octet-stream' });
    res.end(await readFile(file));
  } catch (error) { res.writeHead(404); res.end(String(error)); }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const origin = `http://127.0.0.1:${server.address().port}`;
let browser, page, activeScenario, pageErrors;
const record = (name, evidence = {}) => {
  const result = { scenario: activeScenario, name, passed: true, ...evidence };
  results.push(result); console.log(JSON.stringify(result));
};
async function firstViewport(selectors) {
  await page.evaluate(() => scrollTo(0, 0));
  const controls = await page.evaluate(selectors => selectors.map(selector => {
    const e = document.querySelector(selector); if (!e) return { selector, missing: true };
    const r = e.getBoundingClientRect(), visible = getComputedStyle(e).display !== 'none' && r.width > 0 && r.height > 0;
    const center = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
    return { selector, visible, inViewport: r.left >= 0 && r.top >= 0 && r.right <= innerWidth && r.bottom <= innerHeight,
      unobscured: !!center && (e === center || e.contains(center)), rect: { x:r.x, y:r.y, width:r.width, height:r.height } };
  }), selectors);
  for (const control of controls) assert(control.visible && control.inViewport && control.unobscured,
    `Primary control is not reachable in the first viewport: ${JSON.stringify(control)}`);
  return controls;
}
async function pixels() {
  return page.locator('#c').evaluate(canvas => {
    const a = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
    let lo = 255, hi = 0, hash = 2166136261;
    for (let i = 0; i < a.length; i += 4) { lo = Math.min(lo, a[i]); hi = Math.max(hi, a[i]); hash = Math.imul(hash ^ a[i], 16777619); }
    return { width:canvas.width, height:canvas.height, spread:hi-lo, hash:hash >>> 0 };
  });
}
async function settings() {
  return page.evaluate(() => ({ experimentMode, cellKey, freq, amp, temp, depth, rim, view,
    generalInput: JSON.parse(JSON.stringify(generalInput)) }));
}
async function clock() { return page.evaluate(() => CELL3D.clocks()); }
async function settlePausedClock() {
  let previous, same = 0;
  const start = Date.now();
  while (Date.now() - start < 15000) {
    const next = await clock();
    assert(!next.err, next.err);
    if (next.paused && !next.live && previous && next.stepsDone === previous.stepsDone && next.physT === previous.physT) same++;
    else same = 0;
    if (same >= 3) return next;
    previous = next;
    await page.waitForTimeout(250);
  }
  throw new Error('The nonlinear physical clock did not settle after Pause');
}
try {
  const executablePath = process.env.FARADAY_BROWSER_EXECUTABLE;
  browser = await engine.launch({ headless: true, ...(executablePath ? { executablePath } : {}) });
  for (const viewport of [{ name:'desktop', width:1440, height:1000 }, { name:'mobile', width:390, height:844 }]) {
    activeScenario = viewport.name;
    const context = await browser.newContext({ viewport: { width:viewport.width, height:viewport.height }, acceptDownloads:true });
    page = await context.newPage(); page.setDefaultTimeout(20000); pageErrors = [];
    page.on('pageerror', e => pageErrors.push(String(e.stack || e)));
    page.on('console', m => { if (m.type() === 'error') pageErrors.push(m.text()); });
    await page.addInitScript(() => {
      globalThis.__faradayExportFrames=[]; globalThis.__faradayLongTasks=[];
      globalThis.__faradayExportHeartbeat={count:0,last:null,maxGap:0};
      if (globalThis.PerformanceObserver?.supportedEntryTypes?.includes('longtask'))
        new PerformanceObserver(list => { for (const e of list.getEntries()) __faradayLongTasks.push({start:e.startTime,duration:e.duration}); })
          .observe({type:'longtask',buffered:true});
      function sample(){
        const text=document.getElementById('expBtn')?.textContent;
        if(typeof exporting!=='undefined' && exporting && text){
          const now=performance.now(), h=__faradayExportHeartbeat;
          if(h.last!==null) h.maxGap=Math.max(h.maxGap,now-h.last); h.last=now; h.count++;
          if(__faradayExportFrames.at(-1)?.text!==text) __faradayExportFrames.push({time:now,text});
        }
        requestAnimationFrame(sample);
      }
      requestAnimationFrame(sample);
    });
    console.log(JSON.stringify({scenario:activeScenario,stage:'loading',url:`${origin}/${entry}`}));
    await page.goto(`${origin}/${entry}`, { waitUntil:'load', timeout:30000 });
    assert.match(await page.title(), /Faraday|cymatic/i);
    assert.equal(new URL(page.url()).pathname, '/' + entry);
    await page.waitForFunction(() => typeof state !== 'undefined' && state && !dirty && !useSolver && view === 'sand' && ETA.some(v => v !== 0));
    assert.equal(await page.locator('#viewSeg [data-v="sand"]').isEnabled(), true);
    assert.equal(await page.locator('#play').isEnabled(), true);
    assert.equal(await page.locator('#sprinkleBtn').isEnabled(), true);
    assert.equal(await page.locator('#solverOverlay').isVisible(), false);
    const initial = await pixels(); assert(initial.spread > 10, 'The initial sand picture is blank');
    const controls = await firstViewport(['#workspaceSeg', '#fnum', '#play', '#sprinkleBtn']);
    await page.screenshot({ path:join(out, `${viewport.name}-initial-sand.png`) });
    record('visible-sand-on-entry-with-reachable-controls', { initial, controls });

    const grains = await page.evaluate(() => Array.from(PX.slice(0,1024)));
    await page.waitForFunction(before => PX.slice(0,before.length).some((x,i) => x !== before[i]), grains);
    await page.waitForFunction(before => {
      const canvas=document.getElementById('c'), a=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
      let h=2166136261; for(let i=0;i<a.length;i+=4)h=Math.imul(h^a[i],16777619);
      return (h>>>0)!==before;
    }, initial.hash);
    record('real-grains-and-visible-picture-move');

    if (viewport.name === 'desktop') {
      if (process.env.FARADAY_QA_EXPORT_SIZE)
        await page.locator(`#expSeg [data-s="${process.env.FARADAY_QA_EXPORT_SIZE}"]`).click();
      const requestedSize = Number(await page.locator('#expSeg [aria-pressed="true"]').getAttribute('data-s'));
      assert([2048,3840].includes(requestedSize), `Unsupported export control: ${requestedSize}`);
      if (!process.env.FARADAY_QA_EXPORT_SIZE) assert.equal(requestedSize,3840,'The default 4K export changed');
      const downloadPromise = page.waitForEvent('download', { timeout:600000 })
        .then(download=>({download}),error=>({error}));
      const exportStartedAt=await page.evaluate(()=>performance.now()), exportStartedWall=Date.now();
      console.log(JSON.stringify({scenario:activeScenario,stage:'export-start',requestedSize}));
      let reporting=false;
      const reporter=setInterval(async()=>{
        if(reporting)return; reporting=true; const started=Date.now();
        try {
          const status=await page.evaluate(()=>({progress:document.getElementById('expBtn').textContent,
            heartbeat:__faradayExportHeartbeat}));
          console.log(JSON.stringify({scenario:activeScenario,stage:'export-progress',elapsedMs:Date.now()-exportStartedWall,
            responseMs:Date.now()-started,...status}));
        }catch(error){ console.log(JSON.stringify({stage:'export-progress-unavailable',error:String(error)})); }
        finally{reporting=false;}
      },15000);
      let outcome,clickMs;
      try {
        await page.locator('#expBtn').click({timeout:600000}); clickMs=Date.now()-exportStartedWall;
        outcome=await downloadPromise;
      } finally {clearInterval(reporter);}
      if(outcome.error) throw outcome.error;
      const download=outcome.download, durationMs=Date.now()-exportStartedWall;
      const progress=await page.evaluate(start=>({frames:__faradayExportFrames.filter(e=>e.time>=start),
        longTasks:__faradayLongTasks.filter(e=>e.start>=start),heartbeat:__faradayExportHeartbeat}),exportStartedAt);
      assert(progress.frames.length>0,'The export never displayed progress on an animation frame');
      assert(progress.frames[0].time-exportStartedAt<2000,'The export did not paint initial progress within two seconds');
      assert(progress.heartbeat.count>10 && progress.heartbeat.maxGap<1000,
        `Export blocked visible animation-frame feedback: ${JSON.stringify(progress.heartbeat)}`);
      assert.equal(await download.failure(), null);
      const pngPath = join(out, `sand-${requestedSize}.png`); await download.saveAs(pngPath);
      const png = await readFile(pngPath);
      assert.deepEqual(Array.from(png.subarray(0,8)), [137,80,78,71,13,10,26,10]);
      const decoded = await page.evaluate(async encoded => {
        const img=new Image(); img.src='data:image/png;base64,'+encoded; await img.decode();
        const canvas=document.createElement('canvas'); canvas.width=img.naturalWidth; canvas.height=img.naturalHeight;
        const ctx=canvas.getContext('2d'); ctx.drawImage(img,0,0); const rgba=ctx.getImageData(0,0,canvas.width,canvas.height).data;
        let lo=255,hi=0,lit=0; for(let i=0;i<rgba.length;i+=4){lo=Math.min(lo,rgba[i]);hi=Math.max(hi,rgba[i]);if(rgba[i]>40)lit++;}
        return {width:img.naturalWidth,height:img.naturalHeight,spread:hi-lo,lit};
      }, png.toString('base64'));
      assert.equal(decoded.width,requestedSize); assert.equal(decoded.height,requestedSize);
      assert(decoded.spread>10 && decoded.lit>10000,'Downloaded PNG decodes but contains no meaningful sand picture');
      assert.match(download.suggestedFilename(), new RegExp(`sand.*${requestedSize}.*\\.png$`,'i'));
      record('real-sand-export-downloads-and-decodes', { requestedSize, defaultSize:!process.env.FARADAY_QA_EXPORT_SIZE,
        filename:download.suggestedFilename(), bytes:png.length, decoded, durationMs,clickMs,
        firstProgressMs:progress.frames[0].time-exportStartedAt,
        longestTaskMs:Math.max(0,...progress.longTasks.map(t=>t.duration)),heartbeat:progress.heartbeat,progress:progress.frames });
    }

    await page.locator('#play').click();
    await page.waitForFunction(() => running === false
      && document.getElementById('play').getAttribute('aria-pressed') === 'false'
      && document.getElementById('play').textContent === 'paused', undefined, { timeout:5000 });
    const paused = await page.evaluate(() => ({ running, simT, positions:Array.from(PX.slice(0,1024)) }));
    assert.equal(paused.running, false);
    await page.waitForTimeout(500);
    assert.deepEqual(await page.evaluate(() => ({ simT, positions:Array.from(PX.slice(0,1024)) })),
      { simT:paused.simT, positions:paused.positions });
    await page.locator('#viewSeg [data-v="height"]').click();
    await page.waitForFunction(() => view==='height'&&!dirty);
    const heightBefore=await pixels(), fieldBefore=await page.evaluate(() => Array.from(ETA));
    await page.locator('#fnum').fill('56'); await page.locator('#fnum').press('Tab');
    await page.waitForFunction(() => freq === 56 && !dirty && !renderError);
    assert.equal(await page.locator('#f').inputValue(), '56');
    const changed=await page.evaluate(before => ETA.reduce((n,v,i) => n+(v!==before[i]),0),fieldBefore);
    assert(changed>100,'Frequency changed the label but not the actual field');
    const heightAfterFrequency=await pixels();
    assert.notEqual(heightAfterFrequency.hash,heightBefore.hash,'Frequency did not redraw the paused height field');
    await page.locator('#phaseSeg [data-p="-1"]').click();
    await page.waitForFunction(() => phaseSign===-1&&!dirty);
    assert.notEqual((await pixels()).hash,heightAfterFrequency.hash,'Phase did not redraw the paused height field');
    assert.deepEqual(await page.evaluate(() => ({simT,positions:Array.from(PX.slice(0,1024))})),
      {simT:paused.simT,positions:paused.positions},'Drawing controls advanced paused grains or time');
    await page.locator('#phaseSeg [data-p="1"]').click();
    await page.locator('#viewSeg [data-v="sand"]').click();
    await page.waitForFunction(() => view==='sand'&&!dirty);
    const beforeSprinkle=await pixels();
    await page.locator('#sprinkleBtn').click();
    await page.waitForFunction(before => PX.slice(0,before.length)
      .reduce((count,value,index) => count + (value !== before[index]), 0) > 100,
      paused.positions, { timeout:5000 });
    const sprinkled = await page.evaluate(() => Array.from(PX.slice(0,1024)));
    assert(sprinkled.filter((v,i) => v !== paused.positions[i]).length > 100, 'Re-sprinkle did not redistribute grains');
    await page.waitForTimeout(200);
    assert.notEqual((await pixels()).hash,beforeSprinkle.hash,'Re-sprinkle did not redraw while paused');
    assert.deepEqual(await page.evaluate(() => ({simT,positions:Array.from(PX.slice(0,1024))})),
      {simT:paused.simT,positions:sprinkled});
    const resumeWall = Date.now(); await page.locator('#play').click();
    await page.waitForFunction(t => running && simT > t, paused.simT);
    assert((await page.evaluate(() => simT)) - paused.simT <= (Date.now()-resumeWall)/1000 + .15,
      'Sand resume includes elapsed paused time');
    record('paused-frequency-phase-and-resprinkle-redraw-without-advancing-grains', {changed});
    record('sand-pause-and-resume-operate');

    const views = [];
    for (const name of ['height','optical','nodal','sand']) {
      await page.locator(`#viewSeg [data-v="${name}"]`).click();
      await page.waitForFunction(name => view===name && !dirty && !renderError, name);
      const image = await pixels(); assert(image.spread > 10, `${name} rendering is blank`);
      views.push({ name, ...image });
    }
    assert(new Set(views.map(v => v.hash)).size === views.length, 'Render controls produced the same image');
    record('frequency-changes-field-and-all-render-views-work', { changed, views });

    const sandSettings = await settings();
    await page.locator('#workspaceSeg [data-workspace="ns"]').click();
    await page.waitForFunction(() => useSolver && !dirty && state && CELL3D.clocks().initial);
    const initialNs = await page.evaluate(() => ({ clocks:CELL3D.clocks(), experimentMode, rim, engine:cell3dEngine,
      model:state.model, modalModes:state.states.length, signature:state.visualSignature, source:CELL3D.snapshot().provenance }));
    assert.equal(initialNs.experimentMode,'general'); assert.equal(initialNs.rim,'pinned'); assert.equal(initialNs.engine,'cpp');
    assert.equal(initialNs.clocks.physT,0); assert.equal(initialNs.clocks.stepsDone,0); assert.equal(initialNs.clocks.have,false);
    assert.equal(initialNs.source.fieldKind,'initial-condition'); assert.equal(initialNs.source.steps,0);
    assert.equal(initialNs.modalModes,0); assert.equal(initialNs.signature,null);
    assert.equal(await page.locator('#solverOverlay').isVisible(),false);
    assert.match(await page.locator('#solverStatus').innerText(),/initial|not evolved/i);
    const nsControls = await firstViewport(['#workspaceSeg','#runSolver','#pauseSolver','#resetSolver']);
    assert((await pixels()).spread>10,'The specified initial condition preview is blank');
    await page.screenshot({path:join(out,`${viewport.name}-ns-initial-condition.png`)});
    record('explicit-ns-workspace-shows-labeled-initial-condition', { initialNs, controls:nsControls });

    const runStart=Date.now(); await page.locator('#runSolver').click();
    await page.waitForFunction(() => CELL3D.clocks().have || CELL3D.clocks().err,undefined,{timeout:90000});
    const firstFrameMs=Date.now()-runStart;
    const evolved=await clock(); assert(!evolved.err,evolved.err); assert(evolved.physT>0&&evolved.stepsDone>0);
    assert.equal(evolved.engine,'cpp');
    assert.equal(await page.evaluate(() => CELL3D.snapshot().provenance.fieldKind),'evolved-solver-frame');
    await page.evaluate(() => { globalThis.__usabilityRun=CELL3D.lastRun(); });
    await page.locator('#pauseSolver').click(); const frozen=await settlePausedClock();
    await page.waitForTimeout(500); const still=await clock();
    assert.equal(still.physT,frozen.physT); assert.equal(still.stepsDone,frozen.stepsDone);
    assert.match(await page.locator('#solverStatus').innerText(),/paused/i);
    await page.locator('#pauseSolver').click();
    const immediately=await clock(); assert(immediately.stepsDone>=frozen.stepsDone&&immediately.physT>=frozen.physT,'Resume restarted the physical state');
    await page.waitForFunction(steps => CELL3D.clocks().stepsDone>steps || CELL3D.clocks().err,frozen.stepsDone,{timeout:60000});
    assert.equal(await page.evaluate(() => CELL3D.lastRun()===globalThis.__usabilityRun),true,'Resume created a replacement run');
    const resumed=await clock(); assert(!resumed.err,resumed.err);
    await page.screenshot({path:join(out,`${viewport.name}-ns-running.png`)});
    await page.locator('#resetSolver').click();
    await page.waitForFunction(() => CELL3D.clocks().initial && !CELL3D.clocks().have && CELL3D.clocks().physT===0);
    await page.waitForTimeout(500); const reset=await clock();
    assert.equal(reset.stepsDone,0); assert.equal(reset.physT,0); assert.equal(reset.live,false);
    record('real-ns-run-pause-resume-and-reset-preserve-physical-state', { firstFrameMs, evolved, frozen, resumed, reset });

    await page.locator('#fnum').fill('84.5'); await page.locator('#fnum').press('Tab');
    if (await page.locator('#parameterDetails').getAttribute('open') === null)
      await page.locator('#parameterDetails > summary').click();
    await page.locator('#expDiameter').fill('35'); await page.locator('#expDiameter').press('Tab');
    await page.waitForFunction(() => !dirty && freq===84.5 && generalInput.diameterMm===35);
    const nsSettings=await settings();
    await page.locator('#workspaceSeg [data-workspace="sand"]').click();
    await page.waitForFunction(() => !useSolver && !dirty && view==='sand');
    const sandBack=await settings();
    for(const key of ['experimentMode','cellKey','freq','amp','temp','depth','rim','view'])
      assert.deepEqual(sandBack[key],sandSettings[key],`Switching workspace changed Sand ${key}`);
    assert.equal(await page.locator('#viewSeg [data-v="sand"]').isEnabled(),true);
    await page.locator('#workspaceSeg [data-workspace="ns"]').click();
    await page.waitForFunction(() => useSolver && !dirty && CELL3D.clocks().initial);
    const nsBack=await settings();
    assert.equal(nsBack.freq,nsSettings.freq); assert.deepEqual(nsBack.generalInput,nsSettings.generalInput);
    await page.locator('#workspaceSeg [data-workspace="sand"]').click();
    await page.waitForFunction(() => !useSolver&&!dirty&&view==='sand');
    if (viewport.name === 'desktop') {
      if (await page.locator('#parameterDetails').getAttribute('open') === null)
        await page.locator('#parameterDetails > summary').click();
      await page.locator('#configurationMode').selectOption('general');
      await page.locator('#viewSeg [data-v="height"]').click();
      await page.waitForFunction(() => !dirty && experimentMode==='general');
      for (const [id,bad,good] of [['expRho','-1','1000'],['fnum','-1','56']]) {
        await page.locator('#'+id).fill(bad); await page.locator('#'+id).press('Tab');
        await page.waitForFunction(() => document.getElementById('experimentError').textContent.length>0
          && document.getElementById('expBtn').disabled);
        const refusal=await page.evaluate(async()=>{
          const error=document.getElementById('experimentError').textContent;
          try { await exportImage(129,()=>{}); return {error,reason:null}; }
          catch(e){return {error,reason:e.message};}
        });
        assert.equal(refusal.reason,refusal.error,'Invalid custom Sand inputs exported a stale valid image');
        assert.equal(await page.locator('#'+id).inputValue(),bad,'Invalid input was silently replaced');
        await page.locator('#'+id).fill(good); await page.locator('#'+id).press('Tab');
        await page.waitForFunction(() => !dirty && !document.getElementById('experimentError').textContent
          && !document.getElementById('expBtn').disabled);
      }
      await page.locator('#configurationMode').selectOption('reference');
      await page.locator('#viewSeg [data-v="sand"]').click();
      await page.waitForFunction(() => !dirty && view==='sand');
      record('invalid-custom-sand-density-and-frequency-refuse-export-until-corrected');
    }
    assert.deepEqual(pageErrors,[],`Runtime errors in ${viewport.name}`);
    await firstViewport(['#workspaceSeg','#fnum','#play','#sprinkleBtn']);
    await page.screenshot({path:join(out,`${viewport.name}-returned-to-sand.png`)});
    record('workspaces-keep-independent-settings-and-return-to-working-sand', { sandSettings, nsSettings });
    await context.close(); page=null;
  }
} catch(error) {
  failures.push(String(error.stack||error));
  console.error(String(error.stack||error));
  if(page) try {
    const exportState=await page.evaluate(()=>({progress:document.getElementById('expBtn')?.textContent,
      heartbeat:globalThis.__faradayExportHeartbeat,frames:globalThis.__faradayExportFrames,
      longTasks:globalThis.__faradayLongTasks}));
    await page.screenshot({path:join(out,`${activeScenario}-failure.png`),fullPage:true});
    failures.push({scenario:activeScenario,url:page.url(),pageErrors,exportState});
  } catch(captureError){ failures.push(String(captureError)); }
  process.exitCode=1;
} finally {
  if(browser)await browser.close(); await new Promise(r=>server.close(r));
  const report={engine:engineName,scope:'HTTP usability on desktop/mobile; actual moving sand, real PNG export, explicit NS initial condition and physical run controls; not a file:// or high-resolution convergence test',results,failures};
  await writeFile(join(out,'results.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
}
