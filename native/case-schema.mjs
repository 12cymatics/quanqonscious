/* Strict, versioned SI inputs for a resolved native grain/fluid case. No grain
   counts, physical properties, grid sizes or contact laws are filled in here.
   The caller declares them, including an empty grains array for a zero-grain
   benchmark. A sphere is the geometry of this model, not a measured beach shape. */
export const CASE_FORMAT = 'faraday-native-case';
export const CASE_VERSION = 1;
export const BINARY_MAGIC = 'FNSCASE1';
export const BINARY_VERSION = 1;
export const ARRAY_FIELDS = Object.freeze([
  [0,'rf'],[1,'sf'],[2,'rc'],[3,'sc'],[4,'drc'],[5,'dsc'],[6,'drf'],[7,'dsf'],[8,'rx'],
  [9,'u'],[10,'v'],[11,'w'],[12,'om'],[13,'p'],[14,'eta'],[20,'Ht'],
  [62,'twc'],[63,'tws'],[64,'_pcBand']
].map(Object.freeze));
export const MATERIAL_NOTICE = 'Illustrative silica-rich spherical grains are declared case geometry and density, not measured shapes, grading or silica content of a south-east Queensland beach sample.';
const MAX_INDEX = 2147483647n;
const keys = (value, required, where) => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError(where+' must be an object');
  for (const key of required) if (!Object.hasOwn(value,key)) throw new TypeError(where+'.'+key+' is required; no physical default is substituted');
  for (const key of Object.keys(value)) if (!required.includes(key)) throw new TypeError(where+'.'+key+' is not a recognised input');
};
const number = (value,where) => {if(typeof value!=='number'||!Number.isFinite(value))throw new TypeError(where+' must be a finite number');return value;};
const positive = (value,where) => {number(value,where);if(!(value>0))throw new RangeError(where+' must be positive');return value;};
const nonnegative = (value,where) => {number(value,where);if(value<0)throw new RangeError(where+' must be nonnegative');return value;};
const integer = (value,where,min=0) => {number(value,where);if(!Number.isSafeInteger(value)||value<min)throw new RangeError(where+' must be a safe integer >= '+min);return value;};
const string = (value,where) => {if(typeof value!=='string'||!value.trim())throw new TypeError(where+' must be a nonempty string');return value;};
const vector = (value,where) => {if(!Array.isArray(value)||value.length!==3)throw new TypeError(where+' must contain exactly three SI components');return value.map((x,i)=>number(x,where+'['+i+']'));};
const finiteDerived = (value,where) => {if(!Number.isFinite(value)||!(value>0))throw new RangeError(where+' is outside finite positive floating-point range');return value;};

export function validateCase(input) {
  keys(input,['format','version','fluid','drive','grid','initial','grains','materialProvenance','contact','resourceBudgetBytes','duration','dt','outputInterval'],'case');
  if(input.format!==CASE_FORMAT||input.version!==CASE_VERSION)throw new TypeError('case requires format '+CASE_FORMAT+' and version '+CASE_VERSION);
  const f=input.fluid,d=input.drive,g=input.grid,c=input.contact,p=input.materialProvenance;
  keys(f,['radius','depth','density','dynamicViscosity','surfaceTension','gravity','contactLine'],'fluid');
  const fluid={radius:positive(f.radius,'fluid.radius'),depth:positive(f.depth,'fluid.depth'),density:positive(f.density,'fluid.density'),dynamicViscosity:positive(f.dynamicViscosity,'fluid.dynamicViscosity'),surfaceTension:positive(f.surfaceTension,'fluid.surfaceTension'),gravity:nonnegative(f.gravity,'fluid.gravity'),contactLine:f.contactLine};
  if(f.contactLine!=='pinned')throw new RangeError('fluid.contactLine must be pinned; a moving contact-line law is not substituted');
  finiteDerived(fluid.dynamicViscosity/fluid.density,'kinematic viscosity');
  finiteDerived(Math.PI*fluid.radius*fluid.radius*fluid.depth,'vessel volume');
  keys(d,['frequency','acceleration'],'drive');
  const drive={frequency:nonnegative(d.frequency,'drive.frequency'),acceleration:number(d.acceleration,'drive.acceleration')};
  if(!Number.isFinite(2*Math.PI*drive.frequency)||!Number.isFinite(fluid.gravity+Math.abs(drive.acceleration)))throw new RangeError('drive frequency or effective gravity exceeds finite floating-point range');
  keys(g,['nr','nth','nz','rStretch','zStretch'],'grid');
  const grid={nr:integer(g.nr,'grid.nr',4),nth:integer(g.nth,'grid.nth',8),nz:integer(g.nz,'grid.nz',4),rStretch:nonnegative(g.rStretch,'grid.rStretch'),zStretch:nonnegative(g.zStretch,'grid.zStretch')};
  if(grid.nth%2)throw new RangeError('grid.nth must be even');
  checkNativeIndexRange(grid);
  let initial;
  if(input.initial?.type==='flat'){keys(input.initial,['type'],'initial');initial={type:'flat'};}
  else if(input.initial?.type==='seed'){
    keys(input.initial,['type','azimuthalOrder','amplitude'],'initial');
    initial={type:'seed',azimuthalOrder:integer(input.initial.azimuthalOrder,'initial.azimuthalOrder'),amplitude:nonnegative(input.initial.amplitude,'initial.amplitude')};
    if(2*initial.azimuthalOrder>=grid.nth)throw new RangeError('initial.azimuthalOrder must be below the azimuthal Nyquist limit');
  }else throw new TypeError('initial.type must be flat or seed');
  keys(c,['friction','restitution'],'contact');
  const contact={friction:nonnegative(c.friction,'contact.friction'),restitution:nonnegative(c.restitution,'contact.restitution')};
  if(contact.restitution>1)throw new RangeError('contact.restitution must be <= 1 for this dissipative contact model');
  keys(p,['kind','description','source'],'materialProvenance');
  if(!['illustrative','measured'].includes(p.kind))throw new TypeError('materialProvenance.kind must be illustrative or measured');
  const materialProvenance={kind:p.kind,description:string(p.description,'materialProvenance.description'),source:string(p.source,'materialProvenance.source')};
  if(!Array.isArray(input.grains))throw new TypeError('grains must be an explicit array; no particle count is assumed');
  if(input.grains.length>0xffffffff)throw new RangeError('grain count exceeds the portable case format');
  const ids=new Set();
  const grains=input.grains.map((x,i)=>{
    const name='grains['+i+']';keys(x,['id','centre','radius','density','velocity','omega'],name);
    const grain={id:integer(x.id,name+'.id'),centre:vector(x.centre,name+'.centre'),radius:positive(x.radius,name+'.radius'),density:positive(x.density,name+'.density'),velocity:vector(x.velocity,name+'.velocity'),omega:vector(x.omega,name+'.omega')};
    if(ids.has(grain.id))throw new RangeError('grain IDs must be unique');ids.add(grain.id);
    if(!(grain.density>fluid.density))throw new RangeError(name+'.density must exceed fluid density for the positive excess-mass formulation');
    finiteDerived((4/3)*Math.PI*grain.radius**3*grain.density,name+' mass');
    finiteDerived((4/3)*Math.PI*grain.radius**5*grain.density*0.4,name+' inertia');
    if(Math.hypot(grain.centre[0],grain.centre[1])+grain.radius>fluid.radius)throw new RangeError(name+' penetrates the cylindrical wall');
    if(grain.centre[2]-grain.radius<0)throw new RangeError(name+' penetrates the floor');
    const upper=fluid.depth+(initial.type==='seed'?initial.amplitude:0);
    if(grain.centre[2]+grain.radius>upper)throw new RangeError(name+' is not submerged below the initial free-surface upper bound');
    return grain;
  });
  // The native coupler additionally checks actual curved-surface clearance,
  // local resolution and lubrication gaps. Bounding the vessel is not that proof.
  const bins=new Map(),binSize=2*grains.reduce((r,g)=>Math.max(r,g.radius),0);
  for(const a of grains){
    const at=a.centre.map(x=>Math.floor(x/binSize));
    for(let x=-1;x<=1;x++)for(let y=-1;y<=1;y++)for(let z=-1;z<=1;z++){
      const key=[at[0]+x,at[1]+y,at[2]+z].join(',');
      for(const b of bins.get(key)||[])if(Math.hypot(...a.centre.map((v,k)=>v-b.centre[k]))<a.radius+b.radius)throw new RangeError('initial grains '+a.id+' and '+b.id+' overlap; positions are not repaired');
    }
    const key=at.join(',');if(!bins.has(key))bins.set(key,[]);bins.get(key).push(a);
  }
  const resourceBudgetBytes=integer(input.resourceBudgetBytes,'resourceBudgetBytes',1),duration=positive(input.duration,'duration'),dt=positive(input.dt,'dt'),outputInterval=positive(input.outputInterval,'outputInterval');
  if(dt>duration)throw new RangeError('dt exceeds duration; the requested step is not reduced automatically');
  if(!Number.isSafeInteger(Math.ceil(duration/dt)))throw new RangeError('duration/dt exceeds safe step-count range');
  if(!Number.isSafeInteger(Math.ceil(duration/outputInterval)))throw new RangeError('duration/outputInterval exceeds safe output-count range');
  return {format:CASE_FORMAT,version:CASE_VERSION,fluid,drive,grid,initial,grains,materialProvenance,contact,resourceBudgetBytes,duration,dt,outputInterval};
}

export function checkNativeIndexRange(grid) {
  const r=BigInt(grid.nr),t=BigInt(grid.nth),z=BigInt(grid.nz),m=t/2n+1n;
  const counts=[r+2n,z+2n,(r+2n)*t,(r+1n)*t*z,r*t*(z+1n),r*z,m*r*z,m*r*z*(z+1n),400n*(r+t+z)];
  if(counts.some(n=>n>MAX_INDEX))throw new RangeError('grid exceeds the canonical native signed-index or band-preconditioner range; no coarser grid is substituted');
}

/* Counts are BigInt until final serialization. pcBand scales as
   (nth/2+1)*nr*nz*(nz+1), not merely with the fluid cell count. These exact
   typed-array counts are checked against the canonical constructor in tests. */
export function estimateResources(grid) {
  integer(grid.nr,'grid.nr',4);integer(grid.nth,'grid.nth',8);integer(grid.nz,'grid.nz',4);
  if(grid.nth%2)throw new RangeError('grid.nth must be even');
  const r=BigInt(grid.nr),t=BigInt(grid.nth),z=BigInt(grid.nz),m=t/2n+1n;
  const nu=(r+1n)*t*z,nv=r*t*z,nw=r*t*(z+1n),np=nv,ne=r*t,hr=(r+1n)*t,nx=(r+2n)*t,band=m*r*z*(z+1n),hat=m*r*z;
  const jsDoubles=4n*nu+4n*nv+6n*nw+6n*np+20n*ne+4n*hr+9n*nx+band+2n*hat+2n*t+3n*t*(z+1n)+2n*t*z+6n*r+8n*z+66n;
  const intBytes=4n*(4n*r+1n),jsCellArrayBytes=8n*jsDoubles+intBytes;
  const temporaryBandBytes=8n*(6n*r*z*(z+1n)+2n*np);
  const preparationPeakArrayBytes=2n*jsCellArrayBytes+temporaryBandBytes+65536n;
  const nativeFluidArrayBytes=8n*(jsDoubles-54n+t+nw+2n*np)+intBytes;
  return {cells:np,pcBandElements:band,pcBandBytes:8n*band,jsCellArrayBytes,temporaryBandBytes,preparationPeakArrayBytes,nativeFluidArrayBytes};
}
export function resourceReport(grid){return Object.fromEntries(Object.entries(estimateResources(grid)).map(([key,value])=>[key,value<=BigInt(Number.MAX_SAFE_INTEGER)?Number(value):value.toString()]));}
export function assertPreparationBudget(c,availableBytes=Infinity){
  const e=estimateResources(c.grid),budget=BigInt(c.resourceBudgetBytes);
  if(e.preparationPeakArrayBytes>budget||e.nativeFluidArrayBytes>budget)throw new RangeError('requested grid needs '+e.preparationPeakArrayBytes+' preparation-array bytes and '+e.nativeFluidArrayBytes+' native-fluid-array bytes, above resourceBudgetBytes='+budget+'; includes '+e.pcBandBytes+' pressure-band bytes. No allocation or coarsening performed.');
  if(Number.isFinite(availableBytes)&&e.preparationPeakArrayBytes>BigInt(Math.floor(availableBytes)))throw new RangeError('preparing this grid needs '+e.preparationPeakArrayBytes+' array bytes, above this machine\'s available memory '+Math.floor(availableBytes)+'; no allocation or coarsening performed.');
  return e;
}
