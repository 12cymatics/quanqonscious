/* Read actual native JSONL output into the existing 3D observation format.
   This adapter never advances a solver or supplies missing physical fields.
   feed()/finish()/iterate() are generators: consume every yielded event so a
   later malformed record cannot conceal the preceding coherent frames. */
(function(root,factory){
  const api=factory(typeof module==='object'&&module.exports
    ?require('../dns/cell3d-observation.js'):root.FARADAY_CELL3D_OBSERVATION);
  if(typeof module==='object'&&module.exports)module.exports=api;
  root.FARADAY_NATIVE_RECORDING=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(OBS){
  'use strict';
  const FORMAT='faraday-native-recording',VERSION=1;
  const SCOPE='Experimental resolved rigid-sphere volume constraints; imported native results, not independently validated measurements of south-east Queensland beach sand.';
  const DEFAULT_LIMITS=Object.freeze({maxLineChars:32*1024*1024,maxCaseChars:16*1024*1024,
    maxCells:1000000,maxGrains:100000,maxArrayElements:4000000,maxFrameBytes:64*1024*1024});
  const copy=value=>JSON.parse(JSON.stringify(value));
  const finite=(x,name)=>{if(typeof x!=='number'||!Number.isFinite(x))throw new TypeError(name+' must be finite.');return x;};
  const positive=(x,name)=>{finite(x,name);if(!(x>0))throw new RangeError(name+' must be positive.');return x;};
  const nonnegative=(x,name)=>{finite(x,name);if(x<0)throw new RangeError(name+' must be nonnegative.');return x;};
  const integer=(x,name,min=0)=>{if(!Number.isSafeInteger(x)||x<min)throw new RangeError(name+' must be a safe integer >= '+min+'.');return x;};
  const text=(x,name)=>{if(typeof x!=='string'||!x.trim())throw new TypeError(name+' must be a nonempty string.');return x;};
  function keys(value,expected,name){
    if(!value||typeof value!=='object'||Array.isArray(value))throw new TypeError(name+' must be an object.');
    for(const key of expected)if(!Object.hasOwn(value,key))throw new TypeError(name+'.'+key+' is required.');
    for(const key of Object.keys(value))if(!expected.includes(key))throw new TypeError(name+'.'+key+' is not recognised.');
  }
  function vector(value,name){
    if(!Array.isArray(value)||value.length!==3)throw new RangeError(name+' must contain three components.');
    for(let i=0;i<3;i++)finite(value[i],name+'['+i+']');
    return value;
  }
  function grain(value,name,fluid){
    keys(value,['id','centre','radius','density','velocity','omega'],name);
    integer(value.id,name+'.id');vector(value.centre,name+'.centre');
    positive(value.radius,name+'.radius');positive(value.density,name+'.density');
    if(!(value.density>fluid.density))throw new RangeError(name+'.density must exceed the liquid density for the excess-mass model.');
    vector(value.velocity,name+'.velocity');vector(value.omega,name+'.omega');
    const mass=(4/3)*Math.PI*value.radius**3*value.density;
    positive(mass,name+' mass');positive(.4*mass*value.radius**2,name+' inertia');
    return value;
  }
  function shape(grid,limits){
    integer(grid.nr,'nr',4);integer(grid.nth,'nth',8);integer(grid.nz,'nz',4);
    if(grid.nth%2)throw new RangeError('nth must be even.');
    const cells=grid.nr*grid.nth*grid.nz;
    if(!Number.isSafeInteger(cells)||cells>limits.maxCells)throw new RangeError('Fluid cell count exceeds the recording reader limit; no smaller grid was substituted.');
    const lengths={rf:grid.nr+1,rc:grid.nr,sf:grid.nz+1,sc:grid.nz,eta:grid.nr*grid.nth,Ht:grid.nr*grid.nth,
      u:(grid.nr+1)*grid.nth*grid.nz,v:cells,w:grid.nr*grid.nth*(grid.nz+1)};
    for(const [name,n] of Object.entries(lengths))if(!Number.isSafeInteger(n)||n>limits.maxArrayElements)
      throw new RangeError(name+' exceeds the recording reader array limit.');
    return lengths;
  }
  function observationShape(grid,grainCount,glyphLimit,limits){
    // This is the array layout emitted by cell3d-observation.frame(), plus the
    // physical grain fields attached below. Mirror its deterministic glyph
    // strides so the byte bound includes every emitted array before allocation.
    const {nr,nth,nz}=grid;
    let sr=1,st=1,sz=1;
    const count=()=>Math.ceil(nr/sr)*Math.ceil(nth/st)*Math.ceil(nz/sz);
    while(glyphLimit&&count()>glyphLimit){
      const ar=nr/sr,at=nth/st,az=nz/sz;
      if(ar>=at&&ar>=az)sr++;else if(at>=az)st++;else sz++;
    }
    const vertices=1+(nr+1)*nth,indices=nth*(1+2*nr)*3,glyphs=glyphLimit?count():0;
    for(const [name,n] of Object.entries({surfacePositions:3*vertices,surfaceIndices:indices,
      surfaceDisplacement:vertices,flowPositions:3*glyphs,flowVectors:3*glyphs,
      flowDisplacement:glyphs,grainPositions:3*grainCount,grainVelocities:3*grainCount,
      grainAngularVelocities:3*grainCount,grainScalars:grainCount}))
      if(!Number.isSafeInteger(n)||n>limits.maxArrayElements)
        throw new RangeError('Observed '+name+' exceeds the recording reader array limit.');
    return 16*vertices+4*indices+28*glyphs+100*grainCount;
  }
  function validateDeclaredGrid(record,c){
    // Native prepare-case.mjs uses FaradayCell3D, whose radial and sigma maps
    // are gradeToEnd/gradeBothEnds in dns/faraday-disc.js. Centres are arithmetic
    // face midpoints. Allow only 16 machine epsilons of the domain length for
    // cross-runtime tanh/rounding differences; retain the supplied coordinates.
    const radial=i=>{
      const n=c.grid.nr,L=c.fluid.radius,s=c.grid.rStretch;
      if(i===0)return 0;if(i===n)return L;
      return s===0?L*i/n:L*Math.tanh(s*i/n)/Math.tanh(s);
    };
    const vertical=j=>{
      const n=c.grid.nz,s=c.grid.zStretch;
      if(j===0)return 0;if(j===n)return 1;
      return s===0?j/n:.5*(1+Math.tanh(s*(2*j/n-1))/Math.tanh(s));
    };
    for(const [name,at,length] of [['rf',radial,c.fluid.radius],['rc',i=>.5*(radial(i)+radial(i+1)),c.fluid.radius],
      ['sf',vertical,1],['sc',j=>.5*(vertical(j)+vertical(j+1)),1]])
      for(let i=0;i<record[name].length;i++){
        const expected=at(i);
        if(!Number.isFinite(expected)||Math.abs(record[name][i]-expected)>16*Number.EPSILON*length)
          throw new RangeError('Native '+name+'['+i+'] differs from the declared grid mapping.');
      }
  }
  function readCase(json,limits){
    if(typeof json!=='string'||!json.length||json.length>limits.maxCaseChars)throw new RangeError('Native case provenance is missing or exceeds its reader limit.');
    let c;try{c=JSON.parse(json);}catch(e){throw new SyntaxError('Native case provenance is not complete JSON.');}
    keys(c,['format','version','fluid','drive','grid','initial','grains','materialProvenance','contact','resourceBudgetBytes','duration','dt','outputInterval'],'case');
    if(c.format!=='faraday-native-case'||c.version!==1)throw new TypeError('Unsupported native case format/version.');
    keys(c.fluid,['radius','depth','density','dynamicViscosity','surfaceTension','gravity','contactLine'],'fluid');
    for(const name of ['radius','depth','density','dynamicViscosity','surfaceTension'])positive(c.fluid[name],'fluid.'+name);
    nonnegative(c.fluid.gravity,'fluid.gravity');if(c.fluid.contactLine!=='pinned')throw new TypeError('Native recording requires the pinned contact line.');
    keys(c.grid,['nr','nth','nz','rStretch','zStretch'],'grid');shape(c.grid,limits);
    nonnegative(c.grid.rStretch,'grid.rStretch');nonnegative(c.grid.zStretch,'grid.zStretch');
    keys(c.drive,['frequency','acceleration'],'drive');nonnegative(c.drive.frequency,'drive.frequency');finite(c.drive.acceleration,'drive.acceleration');
    if(c.initial?.type==='flat')keys(c.initial,['type'],'initial');
    else if(c.initial?.type==='seed'){
      keys(c.initial,['type','azimuthalOrder','amplitude'],'initial');integer(c.initial.azimuthalOrder,'initial.azimuthalOrder');nonnegative(c.initial.amplitude,'initial.amplitude');
      if(2*c.initial.azimuthalOrder>=c.grid.nth)throw new RangeError('Initial mode is outside the recorded grid.');
    }else throw new TypeError('Unknown native initial-condition type.');
    keys(c.contact,['friction','restitution'],'contact');nonnegative(c.contact.friction,'contact.friction');nonnegative(c.contact.restitution,'contact.restitution');
    if(c.contact.restitution>1)throw new RangeError('Restitution must not exceed one.');
    keys(c.materialProvenance,['kind','description','source'],'materialProvenance');
    if(!['illustrative','measured'].includes(c.materialProvenance.kind))throw new TypeError('Unknown material-provenance kind.');
    text(c.materialProvenance.description,'materialProvenance.description');text(c.materialProvenance.source,'materialProvenance.source');
    integer(c.resourceBudgetBytes,'resourceBudgetBytes',1);
    for(const name of ['duration','dt','outputInterval'])positive(c[name],name);
    if(c.dt>c.duration)throw new RangeError('Recorded requested dt exceeds duration.');
    if(!Array.isArray(c.grains)||c.grains.length>limits.maxGrains)throw new RangeError('Native grain count is missing or exceeds its reader limit.');
    const ids=new Set();
    for(let i=0;i<c.grains.length;i++){
      const g=grain(c.grains[i],'case.grains['+i+']',c.fluid);if(ids.has(g.id))throw new RangeError('Duplicate native grain ID.');ids.add(g.id);
      if(Math.hypot(g.centre[0],g.centre[1])+g.radius>c.fluid.radius||g.centre[2]-g.radius<0)
        throw new RangeError('Declared initial grain penetrates the cylinder or floor.');
      if(g.centre[2]+g.radius>c.fluid.depth+(c.initial.type==='seed'?c.initial.amplitude:0))
        throw new RangeError('Declared initial grain exceeds the initial surface upper bound.');
    }
    return c;
  }
  function cloneFrame(frame){
    const out={};
    for(const [key,value] of Object.entries(frame))out[key]=ArrayBuffer.isView(value)?new value.constructor(value):value&&typeof value==='object'?copy(value):value;
    return Object.freeze(out);
  }
  function createSession(options={}){
    if(!OBS||typeof OBS.createFrame!=='function')throw new Error('The native recording reader requires cell3d-observation.js.');
    const known=['generation','maxFlowGlyphs',...Object.keys(DEFAULT_LIMITS)];
    for(const key of Object.keys(options))if(!known.includes(key))throw new TypeError('Unknown native reader option '+key+'.');
    const limits={...DEFAULT_LIMITS};
    for(const key of Object.keys(limits))if(options[key]!==undefined)limits[key]=integer(options[key],key,1);
    const generation=integer(options.generation??0,'generation'),maxFlowGlyphs=integer(options.maxFlowGlyphs??384,'maxFlowGlyphs');
    if(maxFlowGlyphs>limits.maxArrayElements/3)throw new RangeError('Flow glyph count exceeds the reader array limit.');
    let pending='',lineNumber=0,header=null,caseValue=null,grainIdentities=null,grid=null,frames=0,lastTime=null,lastSteps=null,latest=null;
    let terminal=null,closed=false,failed=false,failure=null,feeding=false;
    const status=()=>Object.freeze({headerSeen:!!header,frames,lastTime,lastSteps,terminal,closed,failed,error:failure});
    function refuse(error){failed=true;failure=error.message;closed=true;throw error;}
    function decode(line){
      lineNumber++;
      if(lineNumber===1&&line.charCodeAt(0)===0xfeff)line=line.slice(1);
      if(!line.trim())return null;
      if(terminal)throw new Error('Unexpected record after the native '+terminal+' terminator.');
      let record;try{record=JSON.parse(line);}catch(e){throw new SyntaxError('Native recording line '+lineNumber+' is not complete JSON.');}
      if(!record||typeof record!=='object'||Array.isArray(record))throw new TypeError('Native records must be JSON objects.');
      if(record.type==='error'){
        keys(record,['type','message'],'error');text(record.message,'error.message');terminal='error';failed=true;failure=record.message;
        return Object.freeze({type:'error',message:record.message,framesRetained:frames});
      }
      if(!header){
        keys(record,['type','format','version','caseJson','nativeFluidBytes'],'header');
        if(record.type!=='native-case'||record.format!==FORMAT||record.version!==VERSION)throw new TypeError('The first record must be a supported native-case header.');
        const c=readCase(record.caseJson,limits);integer(record.nativeFluidBytes,'nativeFluidBytes',1);
        if(record.nativeFluidBytes>c.resourceBudgetBytes)throw new RangeError('Recorded native allocation exceeds the declared case budget.');
        caseValue=c;grainIdentities=new Map(c.grains.map(g=>[g.id,g]));
        header=Object.freeze({...record,case:copy(c),scope:SCOPE,pressureAvailable:false});
        return Object.freeze({type:'header',header:copy(header)});
      }
      if(record.type==='frame'){
        keys(record,['type','time','steps','initial','nr','nth','nz','R','h','contact','rf','rc','sf','sc','eta','Ht','u','v','w','grains','diagnostics'],'frame');
        finite(record.time,'frame.time');integer(record.steps,'frame.steps');
        if(typeof record.initial!=='boolean')throw new TypeError('frame.initial must be boolean.');
        if(frames===0){if(record.time!==0||record.steps!==0||!record.initial)throw new RangeError('A native recording must begin with its actual initial frame at t=0, step=0.');}
        else if(!(record.time>lastTime&&record.steps>lastSteps&&!record.initial))throw new RangeError('Native frame clocks must advance together; duplicate or backwards frames are refused.');
        if(record.time>caseValue.duration+8*Number.EPSILON*Math.max(caseValue.duration,record.time))throw new RangeError('Native frame time exceeds the declared duration.');
        for(const name of ['nr','nth','nz'])if(record[name]!==caseValue.grid[name])throw new RangeError('Native frame grid differs from its case.');
        if(record.R!==caseValue.fluid.radius||record.h!==caseValue.fluid.depth||record.contact!=='pinned')throw new RangeError('Native frame geometry differs from its case.');
        const lengths=shape(record,limits);
        if(!Array.isArray(record.grains)||record.grains.length!==grainIdentities.size||record.grains.length>limits.maxGrains)throw new RangeError('Native frame grain count differs from its case.');
        // maxFrameBytes bounds each physical/observation array representation,
        // not total process memory, JSON objects, or temporary cloning overhead.
        const inputBytes=8*(Object.values(lengths).reduce((a,b)=>a+b,0)+12*record.grains.length);
        const observedBytes=observationShape(record,record.grains.length,maxFlowGlyphs,limits);
        const bytes=Math.max(inputBytes,observedBytes);
        if(!Number.isSafeInteger(bytes)||bytes>limits.maxFrameBytes)throw new RangeError('Native frame exceeds the reader byte budget; no fields or grains were dropped.');
        // Validate all counts and all values before allocating typed-array copies.
        for(const [name,length] of Object.entries(lengths)){
          if(!Array.isArray(record[name])||record[name].length!==length)throw new RangeError('Native '+name+' length does not match its staggered grid.');
          for(let i=0;i<length;i++)finite(record[name][i],name+'['+i+']');
          if(grid&&['rf','rc','sf','sc'].includes(name))for(let i=0;i<length;i++)if(record[name][i]!==grid[name][i])throw new RangeError('Native coordinate arrays changed within one recording.');
        }
        if(!grid)validateDeclaredGrid(record,caseValue);
        const ids=new Set();
        for(let i=0;i<record.grains.length;i++){
          const g=grain(record.grains[i],'frame.grains['+i+']',caseValue.fluid),original=grainIdentities.get(g.id);
          if(ids.has(g.id)||!original)throw new RangeError('Native frame contains duplicate or unknown grain IDs.');ids.add(g.id);
          if(g.radius!==original.radius||g.density!==original.density)throw new RangeError('Native grain size or density changed within a rigid-grain recording.');
          if(frames===0)for(const name of ['centre','velocity','omega'])for(let k=0;k<3;k++)if(g[name][k]!==original[name][k])throw new RangeError('Initial native grain state differs from the declared case.');
        }
        keys(record.diagnostics,['dt','slipResidual','divergence','constraintIterations','markers','eventRefinements','contacts'],'diagnostics');
        for(const name of ['dt','slipResidual','divergence'])nonnegative(record.diagnostics[name],'diagnostics.'+name);
        for(const name of ['constraintIterations','markers','eventRefinements','contacts'])integer(record.diagnostics[name],'diagnostics.'+name);
        if(record.initial?record.diagnostics.dt!==0:!(record.diagnostics.dt>0))throw new RangeError('Native step diagnostics do not match the initial/evolved frame clock.');
        const physical={...record,generation};
        for(const name of Object.keys(lengths))physical[name]=Float64Array.from(record[name]);
        const count=record.grains.length,particles=new Float64Array(count*3),radii=new Float64Array(count),density=new Float64Array(count),grainIds=new Float64Array(count),velocity=new Float64Array(count*3),omega=new Float64Array(count*3);
        record.grains.forEach((g,i)=>{particles.set(g.centre,i*3);radii[i]=g.radius;density[i]=g.density;grainIds[i]=g.id;velocity.set(g.velocity,i*3);omega.set(g.omega,i*3);});
        const observed=OBS.createFrame(physical,{particles,generation,steps:record.steps,initial:record.initial,maxFlowGlyphs});
        for(const [name,value] of Object.entries(observed))if(ArrayBuffer.isView(value))for(let i=0;i<value.length;i++)finite(value[i],'observed '+name+'['+i+']');
        finite(observed.maxSpeed,'observed maximum speed');
        const frame=Object.freeze({...observed,particleRadii:radii,grainIds,grainDensity:density,grainVelocities:velocity,grainAngularVelocities:omega,
          grainProvenance:{model:'experimental-resolved-rigid-sphere-volume-constraints',geometry:'spheres',motion:'recorded-native-state',physicalTime:record.time,steps:record.steps,
            material:copy(caseValue.materialProvenance),contact:copy(caseValue.contact),scope:SCOPE},
          source:{engine:'native-cpp-recording',format:FORMAT,version:VERSION,caseJson:header.caseJson,nativeFluidBytes:header.nativeFluidBytes,
            parameters:{R:record.R,h:record.h,rho:caseValue.fluid.density,nu:caseValue.fluid.dynamicViscosity/caseValue.fluid.density,gamma:caseValue.fluid.surfaceTension,g:caseValue.fluid.gravity,omegaD:2*Math.PI*caseValue.drive.frequency,accel:caseValue.drive.acceleration,contact:'pinned'},
            diagnostics:copy(record.diagnostics),pressureAvailable:false},
          provenance:{...observed.provenance,source:'native-recording',grainModel:'experimental-resolved-rigid-sphere-volume-constraints'},
          units:{...observed.units,radius:'m',density:'kg/m^3',angularVelocity:'rad/s'}});
        // Publish only after the complete observation and particle geometry succeed.
        if(!grid)grid=Object.fromEntries(['rf','rc','sf','sc'].map(name=>[name,physical[name]]));
        frames++;lastTime=record.time;lastSteps=record.steps;latest=frame;
        return Object.freeze({type:'frame',frame:cloneFrame(frame)});
      }
      if(record.type==='complete'){
        keys(record,['type','time','steps'],'complete');finite(record.time,'complete.time');integer(record.steps,'complete.steps');
        if(!frames||record.time!==lastTime||record.steps!==lastSteps)throw new RangeError('Completion does not match the last emitted native frame.');
        if(Math.abs(record.time-caseValue.duration)>8*Number.EPSILON*Math.max(record.time,caseValue.duration))throw new RangeError('Completion precedes the requested native duration.');
        terminal='complete';return Object.freeze({...record,frames});
      }
      throw new TypeError('Unknown or misplaced native record type.');
    }
    function* feed(chunk){
      if(closed)throw new Error('This native reader is closed.');
      if(feeding)throw new Error('Consume the previous native-reader generator before feeding another chunk.');
      if(typeof chunk!=='string')throw new TypeError('feed requires decoded text chunks; iterateStream accepts UTF-8 byte chunks.');
      feeding=true;let consumed=false;
      try{
        let start=0;
        while(start<chunk.length){
          const newline=chunk.indexOf('\n',start),end=newline<0?chunk.length:newline;
          if(pending.length+end-start>limits.maxLineChars)throw new RangeError('Native JSONL record exceeds the line-length budget before JSON parsing.');
          pending+=chunk.slice(start,end);start=end+(newline<0?0:1);
          if(newline<0)break;
          const line=pending;pending='';const event=decode(line);if(event)yield event;
        }
        consumed=true;
      }catch(error){refuse(error);}finally{
        feeding=false;
        if(!consumed){closed=true;if(!failed){failed=true;failure='Native reader events were not fully consumed; remaining input was not accepted.';}}
      }
    }
    function* finish(){
      if(closed)throw new Error('This native reader is closed.');
      if(feeding)throw new Error('Consume pending native-reader events before finishing.');
      feeding=true;let consumed=false;
      try{
        if(pending.length){const line=pending;pending='';const event=decode(line);if(event)yield event;}
        if(!terminal)throw new Error('Native recording ended without a complete/error terminator; the retained frames are incomplete results.');
        closed=true;consumed=true;
      }catch(error){refuse(error);}finally{
        feeding=false;
        if(!consumed){closed=true;if(!failed){failed=true;failure='Native reader final events were not fully consumed; EOF was not accepted.';}}
      }
    }
    return Object.freeze({feed,finish,status,header:()=>header?copy(header):null,lastFrame:()=>latest?cloneFrame(latest):null,limits:Object.freeze({...limits})});
  }
  function* iterate(textValue,options={}){const session=createSession(options);yield*session.feed(textValue);yield*session.finish();}
  async function* iterateStream(chunks,options={}){
    const session=createSession(options),decoder=new TextDecoder('utf-8',{fatal:true});
    for await(const chunk of chunks){
      if(typeof chunk==='string'){const flush=decoder.decode();if(flush)yield*session.feed(flush);yield*session.feed(chunk);}
      else if(chunk instanceof Uint8Array)yield*session.feed(decoder.decode(chunk,{stream:true}));
      else throw new TypeError('Native stream chunks must be strings or UTF-8 Uint8Arrays.');
    }
    const flush=decoder.decode();if(flush)yield*session.feed(flush);yield*session.finish();
  }
  return Object.freeze({FORMAT,VERSION,SCOPE,DEFAULT_LIMITS,createSession,iterate,iterateStream});
});
