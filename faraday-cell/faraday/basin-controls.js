(function(){
'use strict';
const byId=id=>document.getElementById(id);
const setText=(id,value)=>{const element=byId(id);if(element.textContent!==value)element.textContent=value};
const presets={pure:{salinity:0,densityScale:1,viscosityScale:1,surfaceScale:1},saline09:{salinity:9,densityScale:1,viscosityScale:1,surfaceScale:1},seawater:{salinity:35,densityScale:1,viscosityScale:1,surfaceScale:1},glycerol20:{salinity:0,densityScale:1.046,viscosityScale:1.93,surfaceScale:0.925}};
const gridPresets={original:[12,24,8,2.2,2.2],medium:[24,48,12,2.2,2.2],fine:[32,64,16,2.2,2.2]};
const gridIds=['basin-nr','basin-nth','basin-nz','basin-r-stretch','basin-z-stretch'];
const fields=[['expDiameter','diameterMm'],['expDepth','depthMm'],['expRho','rho'],['expMu','mu'],['expSigma','sigma'],['expGravity','gravity'],['expAccel','acceleration'],['expSeedM','seedMode'],['expSeedFraction','seedFraction']];
let timer=null,pending=false,runningIntent=initialAutostart||CELL3D.clocks().live,view='top',lastPhysicalKey='',inputError='',manualSurface=false,manualViscosity=false,lastTime=0,disposed=false;
function number(id,min=0,max=Infinity,integer=false){const element=byId(id);const value=Number(element.value);if(!element.value.trim()||!Number.isFinite(value)||value<min||value>max||integer&&!Number.isSafeInteger(value))throw new RangeError(element.closest('.control')?.querySelector('label')?.textContent.trim()+' requires '+(integer?'an integer':'a number')+' between '+min+' and '+(Number.isFinite(max)?max:'the finite input limit')+'.');return value}
function waterProperties(t,s,preset){
const densityPure=999.842594+.06793952*t-.009095290*t*t+.0001001685*t**3-.000001120083*t**4+.000000006536332*t**5;
const a=.824493-.0040899*t+.000076438*t*t-.00000082467*t**3+.0000000053875*t**4;
const b=-.00572466+.00010227*t-.0000016546*t*t;
const density=(densityPure+a*s+b*s**1.5+.00048314*s*s)*preset.densityScale;
const T=(t+273.15)/647.096,rho=densityPure/322;
const h0=[1.67752,2.20462,.6366564,-.241605];
const terms=[[0,0,.520094],[1,0,.0850895],[2,0,-1.08374],[3,0,-.289555],[0,1,.222531],[1,1,.999115],[2,1,1.88797],[3,1,1.26613],[5,1,.120573],[0,2,-.281378],[1,2,-.906851],[2,2,-.772479],[3,2,-.489837],[4,2,-.257040],[0,3,.161913],[1,3,.257399],[0,4,-.0325372],[3,4,.0698452],[4,5,.00872102],[3,6,-.00435673],[5,6,-.000593264]];
const mu0=100*Math.sqrt(T)/h0.reduce((sum,h,i)=>sum+h/T**i,0);
const pureMu=1e-6*mu0*Math.exp(rho*terms.reduce((sum,[i,j,h])=>sum+h*(1/T-1)**i*(rho-1)**j,0));
const mass=s/1000,ma=1.541+.01998*t-.0000952*t*t,mb=7.974-.07561*t+.0004724*t*t;
const mu=pureMu*(1+ma*mass+mb*mass*mass)*preset.viscosityScale;
const tau=1-T,sigma=.2358*tau**1.256*(1-.625*tau)*(1+.0003766*s+.000002347*s*t)*preset.surfaceScale;
if(![density,mu,sigma].every(v=>Number.isFinite(v)&&v>0))throw new RangeError('The entered liquid properties are not finite and positive.');
return{rho:density,mu,sigma};
}
function thermalUpdate(){const t=number('temperature-number',1,92),s=number('salinity',0,40),preset=presets[byId('medium').value];if(!preset)throw new RangeError('Unknown liquid medium.');const properties=waterProperties(t,s,preset);if(!manualSurface)byId('surface-tension').value=String(properties.sigma*1000);if(!manualViscosity)byId('viscosity').value=String(properties.mu*1000);return properties}
function readConfiguration(){
const frequencyHz=number('frequency-number',5,300000),strokeUm=number('amplitude-number',.01,10000),depthMm=number('water-depth-number',.000001,1000000),diameterMm=number('plate-width-number',.000001,1000000);
const properties=thermalUpdate(),sigma=manualSurface?number('surface-tension',20,90)/1000:properties.sigma,mu=manualViscosity?number('viscosity',.2,20)/1000:properties.mu;
const seedMode=number('basin-seed-mode',0,1000000,true),amplitudeUm=number('basin-seed-amplitude',.000001),gravity=number('basin-gravity',0),acceleration=strokeUm*1e-6*(2*Math.PI*frequencyHz)**2;
const experiment={diameterMm,depthMm,rho:properties.rho,mu,sigma,gravity,acceleration,seedMode,seedFraction:amplitudeUm/(depthMm*1000)};
FARADAY_KERNEL.experimentParameters({...experiment,frequencyHz,contact:'pinned'});
const mode=byId('basin-grid-preset').value==='auto'?'auto':'manual';
const grid=mode==='auto'?{mode}:{mode,nr:number('basin-nr',4,250000,true),nth:number('basin-nth',12,250000,true),nz:number('basin-nz',4,250000,true),rStretch:number('basin-r-stretch',0),zStretch:number('basin-z-stretch',0)};
if(mode==='manual'&&(grid.nth%2||grid.nth<=2*seedMode))throw new RangeError('Angular cells must be even and exceed twice the initial azimuthal order.');
if(mode==='manual'&&grid.nr*grid.nth*grid.nz>250000)throw new RangeError('The explicit grid exceeds the 250,000-cell solver arena.');
const engine=byId('basin-engine').value,threads=number('basin-threads',1,CELL3D_CORES,true);
if(mode==='manual'&&threads>grid.nr)throw new RangeError('Solver threads cannot exceed the number of radial rows.');
if(!['cpp','js'].includes(engine))throw new RangeError('Choose the C++ or JavaScript engine.');
return{experiment,frequencyHz,strokeUm,grid,engine,threads};
}
function format(value,digits=3){return Number(value.toPrecision(digits)).toLocaleString('en-US',{maximumFractionDigits:9})}
function updateReadouts(configuration){
const c=configuration||readConfiguration(),e=c.experiment;
setText('hero-frequency',format(c.frequencyHz,8));setText('frequency-output',format(c.frequencyHz,8)+' Hz');setText('amplitude-output',format(c.strokeUm,6)+' µm');
setText('hero-dimensions',format(e.diameterMm,6)+' mm × '+format(e.depthMm,6)+' mm water');setText('hero-material',byId('material').value.toUpperCase());
setText('water-depth-output',format(e.depthMm,6)+' mm');setText('temperature-output',format(number('temperature-number',1,92),5)+' °C');
setText('surface-tension-output',format(e.sigma*1000,6)+' mN/m');setText('viscosity-output',format(e.mu*1000,6)+' mPa·s');setText('salinity-output',format(number('salinity',0,40),4)+' PSU');
setText('width-output',format(e.diameterMm,6)+' mm');setText('height-output',format(number('plate-height',1,15),4)+' mm');setText('thickness-output',format(number('thickness',.2,2),4)+' mm');
setText('micro-scale-output',format(number('micro-scale',100,5000),6)+' µm');setText('centre-drive-status','Whole cup · '+format(e.acceleration,8)+' m/s² peak acceleration · '+format(c.strokeUm,6)+' µm peak displacement');
setText('thermal-response','Density '+format(e.rho,7)+' kg/m³ · liquid properties follow the entered medium and temperature.');
updateResolution(c);
}
function updateResolution(configuration){
const c=configuration,e=c.experiment,actual=CELL3D.clocks().grid;
const grid=c.grid.mode==='auto'?actual:c.grid;
if(!grid){setText('basin-grid-status','Automatic physical grid · awaiting sizing');setText('basin-resolution-warning','');return}
const k=FARADAY_KERNEL.kOfOmega(Math.PI*c.frequencyHz,e.sigma,e.rho,e.depthMm/1000,e.gravity),lambda=2*Math.PI/k,R=e.diameterMm/2000;
const rStretch=grid.rStretch,nr=grid.nr,t=Math.tanh(rStretch);
let minWidth=Infinity,maxWidth=0,last=0;
for(let i=1;i<=nr;i++){const next=i===nr?R:rStretch===0?R*i/nr:R*Math.tanh(rStretch*i/nr)/t;minWidth=Math.min(minWidth,next-last);maxWidth=Math.max(maxWidth,next-last);last=next}
const radialMin=lambda/maxWidth,radialMax=lambda/minWidth,angular=lambda/(R*2*Math.PI/grid.nth);
setText('basin-grid-status',grid.nr+' × '+grid.nth+' × '+grid.nz+' = '+(grid.nr*grid.nth*grid.nz).toLocaleString()+' numerical cells · '+(c.grid.mode==='auto'?'automatic selection':'entered explicitly'));
const inadequate=radialMin<2||angular<2;
setText('basin-resolution-warning',(inadequate?'Under-resolved at the drive wavelength. ':'Drive-wavelength sampling: ')+format(lambda*1000,5)+' mm wavelength; '+format(radialMin,4)+'–'+format(radialMax,4)+' radial cells and '+format(angular,4)+' outer angular cells per wavelength.'+(inadequate?' Select Automatic physical grid to request resolution.':' Sampling alone does not establish convergence.'));
byId('basin-resolution-warning').dataset.underresolved=String(inadequate);
}
function synchronizeLegacy(c){freq=c.frequencyHz;generalInput={...c.experiment};cell3dGrid={...c.grid};cell3dEngine=c.engine;cell3dThreads=c.threads;rim='pinned';experimentMode='general';useSolver=true;workspace='ns';byId('fnum').value=String(freq);byId('f').value=String(freq);for(const[id,key]of fields)byId(id).value=String(generalInput[key]);}
function clearError(){inputError='';byId('experimentError').textContent='';byId('sceneError').textContent='';renderError=null;setText('basin-error','')}
function reportInput(error){inputError=String(error.message||error);setText('basin-error',inputError);setText('basin-state-status','CHECK INPUT');CELL3D.pause();clearTimeout(timer);timer=null;pending=false;initialAutostart=false;return false}
function applyConfiguration(force=false){
clearTimeout(timer);timer=null;pending=false;
try{
const c=readConfiguration(),key=JSON.stringify(c);updateReadouts(c);clearError();
if(!force&&key===lastPhysicalKey){if(runningIntent&&!CELL3D.clocks().live){if(!CELL3D.resume()){solverRequested=true;CELL3D.start(state)}}return true}
initialAutostart=false;CELL3D.reset();synchronizeLegacy(c);resolutionKey=null;problemKey=null;solverRequested=false;lastPhysicalKey=key;UNIFIED3D.live();dirty=true;recompute();
if(runningIntent){solverRequested=true;CELL3D.start(state)}
return true;
}catch(error){return reportInput(error)}
}
function schedulePhysical(){
try{const c=readConfiguration();updateReadouts(c);inputError='';setText('basin-error','');if(!pending)CELL3D.pause();pending=true;initialAutostart=false;setText('basin-state-status','APPLYING');clearTimeout(timer);timer=setTimeout(()=>applyConfiguration(),240)}catch(error){reportInput(error)}
}
function displayChange(){
try{
const options={stoneMaterial:byId('material').value,wallThicknessMm:number('plate-height',1,15),baseThicknessMm:number('thickness',.2,2),renderDensity:number('quality',1,2),heightScale:number('basin-height-gain',1,10000),colorScaleM:number('basin-colour-scale',.000001)*1e-6,colorMode:'height',showSurface:byId('basin-surface').checked,showParticles:byId('basin-particles').checked,showVectors:byId('basin-vectors').checked};
UNIFIED3D.scene.setOptions(options);if(view==='micro')UNIFIED3D.scene.microView(number('micro-scale',100,5000));UNIFIED3D.scene.render();updateReadouts();
}catch(error){setText('basin-error',String(error.message||error))}
}
function selectView(next){view=next;for(const[name,id]of[['top','view-top'],['orbit','view-orbit'],['micro','view-micro']]){const selected=name===next;byId(id).classList.toggle('active',selected);byId(id).setAttribute('aria-pressed',String(selected))}byId('basin-stage').classList.toggle('top-view',next==='top');byId('basin-stage').classList.toggle('micro-mode',next==='micro');if(next==='top')UNIFIED3D.scene.topView();else if(next==='orbit')UNIFIED3D.scene.resetCamera();else UNIFIED3D.scene.microView(number('micro-scale',100,5000));UNIFIED3D.scene.render();setText('basin-micro-note',next==='micro'?'Showing '+format(number('micro-scale',100,5000),6)+' µm of the same computed water surface.':'MICRO enlarges the same computed water surface.')}
function bindPair(sliderId,numberId,readSlider=v=>v,writeSlider=v=>v,after=()=>{}){const slider=byId(sliderId),input=byId(numberId);slider.addEventListener('input',()=>{input.value=String(readSlider(Number(slider.value)));after();schedulePhysical()});const update=()=>{if(input.value.trim()&&Number.isFinite(Number(input.value)))slider.value=String(writeSlider(Number(input.value)));after();schedulePhysical()};input.addEventListener('input',update);input.addEventListener('change',update)}
bindPair('frequency-slider','frequency-number',v=>5*(60000)**(v/1000),v=>1000*Math.log(v/5)/Math.log(60000));
bindPair('amplitude','amplitude-number',v=>10**v,v=>Math.log10(v));
bindPair('water-depth','water-depth-number');bindPair('plate-width','plate-width-number');bindPair('temperature','temperature-number',v=>v,v=>v,()=>{manualSurface=false;manualViscosity=false});
byId('medium').addEventListener('change',()=>{const preset=presets[byId('medium').value];if(!preset)return reportInput(new Error('Unknown liquid medium.'));byId('salinity').value=String(preset.salinity);manualSurface=false;manualViscosity=false;schedulePhysical()});
byId('salinity').addEventListener('input',()=>{manualSurface=false;manualViscosity=false;schedulePhysical()});
byId('surface-tension').addEventListener('input',()=>{manualSurface=true;schedulePhysical()});byId('viscosity').addEventListener('input',()=>{manualViscosity=true;schedulePhysical()});
for(const id of['basin-seed-mode','basin-seed-amplitude','basin-gravity']){byId(id).addEventListener('input',schedulePhysical);byId(id).addEventListener('change',schedulePhysical)}
for(const id of['basin-engine','basin-threads'])byId(id).addEventListener('change',schedulePhysical);
for(const id of gridIds){byId(id).addEventListener('input',()=>{byId('basin-grid-preset').value='manual';schedulePhysical()});byId(id).addEventListener('change',schedulePhysical)}
byId('basin-grid-preset').addEventListener('change',()=>{const selected=byId('basin-grid-preset').value,values=gridPresets[selected];if(values)gridIds.forEach((id,i)=>{byId(id).value=String(values[i])});for(const id of gridIds)byId(id).disabled=selected==='auto';schedulePhysical()});
for(const id of['material','quality','basin-surface','basin-particles','basin-vectors'])byId(id).addEventListener('change',displayChange);
for(const id of['plate-height','thickness','micro-scale','basin-height-gain','basin-colour-scale']){byId(id).addEventListener('input',displayChange);byId(id).addEventListener('change',displayChange)}
for(const[name,id]of[['top','view-top'],['orbit','view-orbit'],['micro','view-micro']])byId(id).addEventListener('click',()=>selectView(name));
byId('basin-run').addEventListener('click',()=>{runningIntent=true;applyConfiguration(true)});
byId('basin-pause').addEventListener('click',()=>{const wasRunning=pending?runningIntent:CELL3D.clocks().live||initialAutostart;initialAutostart=false;runningIntent=!wasRunning;if(!runningIntent){clearTimeout(timer);if(pending)applyConfiguration();CELL3D.pause()}else if(pending||!CELL3D.clocks().ready)applyConfiguration();else{UNIFIED3D.live();CELL3D.resume()}});
byId('basin-refresh').addEventListener('click',()=>applyConfiguration(true));
byId('basin-export').addEventListener('click',()=>{if(pending&&!applyConfiguration())return;byId('expBtn').click()});
byId('basin-replay').addEventListener('click',()=>{runningIntent=false;initialAutostart=false;UNIFIED3D.play()});byId('basin-live').addEventListener('click',()=>UNIFIED3D.live());
byId('basin-timeline').addEventListener('input',()=>{runningIntent=false;initialAutostart=false;UNIFIED3D.seek(Number(byId('basin-timeline').value))});
for(const count of [...new Set([1,2,4,8,CELL3D_CORES])].filter(n=>n<=CELL3D_CORES&&n>=1).sort((a,b)=>a-b)){if(count===1)continue;const option=document.createElement('option');option.value=String(count);option.textContent=String(count);byId('basin-threads').append(option)}
for(const id of['ro','polynomialStatus','exportPolynomials','polynomialExportStatus'])byId('basin-provenance').append(byId(id));
function refresh(now){
if(disposed)return;
if(now-lastTime>120){lastTime=now;const c=CELL3D.clocks(),scene=UNIFIED3D.scene.stats(),history=UNIFIED3D.historyState();
const status=inputError?'CHECK INPUT':pending?'APPLYING':c.err?'STOPPED':c.starting?'STARTING':c.live?'RUNNING':c.have||c.ready?'PAUSED':'READY';
setText('basin-live-status',status);setText('basin-state-status',status);setText('basin-engine-status',cell3dEngine==='cpp'?'WASM':'JAVASCRIPT');setText('basin-pause',(pending?runningIntent:c.live||initialAutostart)?'PAUSE':'RESUME');setText('basin-run',c.have||c.ready||c.starting?'RESTART':'START');
const current=history.frames.findLast(frame=>frame.time===history.currentTime);const shown=current?.time||0;setText('basin-physical-time',(history.following?'Computed time':'Recorded time')+' · '+(shown*1000).toFixed(6)+' ms · '+(current?.steps||0).toLocaleString()+' steps');
setText('basin-motion-status',inputError?'Correct the entered values.':pending?'Applying the new physical inputs…':c.err?'Solver stopped · '+c.err:c.starting?c.phase:c.have?(c.live?'Evolving':'Paused')+' · '+format(c.physT*freq,5)+' drive cycles · '+format(c.perStep,4)+' ms per CPU step':'Specified initial surface · zero initial velocity');
setText('basin-renderer',scene.renderer+' · height ×'+byId('basin-height-gain').value+' · fixed colour scale '+byId('basin-colour-scale').value+' µm');
if(!inputError)setText('basin-error',c.err||byId('sceneError').textContent||'');
byId('startupStatus').hidden=FARADAY_STARTUP.drawn&&!FARADAY_STARTUP.failed;
byId('basin-export').disabled=!!inputError||!!c.err||byId('expBtn').disabled;setText('basin-export',exporting?'RENDERING 4K PNG…':'EXPORT VIEW · 4K PNG');setText('basin-export-note',byId('expNote').textContent+(exporting?' · '+((performance.now()-exportStartedAt)/1000).toFixed(1)+' s elapsed':''));
const count=history.frames.length;byId('basin-replay').disabled=count<2;setText('basin-replay',history.playing?'PAUSE RECORDING':'PLAY RECORDING');byId('basin-timeline').max=String(Math.max(0,count-1));
if(document.activeElement!==byId('basin-timeline'))byId('basin-timeline').value=String(Math.max(0,history.frames.findIndex(frame=>frame.time===history.currentTime)));
setText('basin-recorded-note',count?count+' actual frames · '+(history.frames[0].time*1000).toFixed(6)+'–'+(history.frames[count-1].time*1000).toFixed(6)+' ms'+(history.playing?' · recorded interval played over 8 seconds':''):'Waiting for computed frames');
if(c.grid&&byId('basin-grid-preset').value==='auto')gridIds.forEach((id,i)=>{byId(id).value=String([c.grid.nr,c.grid.nth,c.grid.nz,c.grid.rStretch,c.grid.zStretch][i])});
if(!inputError&&!pending){try{updateResolution(readConfiguration())}catch(error){reportInput(error)}}
}
requestAnimationFrame(refresh);
}
window.addEventListener('pagehide',()=>{disposed=true;clearTimeout(timer)},{once:true});
try{thermalUpdate();const configuration=readConfiguration();synchronizeLegacy(configuration);lastPhysicalKey=JSON.stringify(configuration);resolutionKey=null;problemKey=null;recompute();updateReadouts(configuration);displayChange();selectView('top');globalThis.FARADAY_BASIN_CONTROLS_READY=true;requestAnimationFrame(refresh)}catch(error){reportInput(error);FARADAY_STARTUP.fail(error)}
globalThis.FARADAY_BASIN=Object.freeze({readConfiguration,waterProperties,applyConfiguration,selectView,status:()=>({runningIntent,pending,inputError,view,physical:CELL3D.clocks()}),provenance:{siteCommit:'dca62dc0e6890d4e0da8d80046dad5535bd4dd76',appearanceSource:'Cymatica stone basin',solver:'current 29-method nonlinear liquid solver'}});
})();
