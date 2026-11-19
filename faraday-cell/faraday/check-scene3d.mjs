import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import vm from 'node:vm';
const require = createRequire(import.meta.url);
const S = require('./scene3d.js');
let checks = 0;
function check(name, fn) { fn(); checks++; console.log('  ok  ' + name); }
const fixture = () => ({ R: 0.02, h: 0.004, time: 0.01, generation: 7, steps: 120, initial: false,
  positions: new Float32Array([0, 0, 0.005, 0.02, 0, 0.004, 0, 0.02, 0.004]),
  indices: new Uint32Array([0, 1, 2]), particles: new Float64Array([0, 0, 0.0025]),
  particleDisplacement: new Float32Array([0.0005]),
  flowPositions: new Float32Array([0, 0, 0.0025]), flowVectors: new Float32Array([0.1, 0.2, -0.3]),
  flowDisplacement: new Float32Array([0.0005]), source: { physical: { contact: 'pinned' } },
  provenance: { fieldKind: 'evolved-solver-frame', physicalTime: 0.01 },
  units: {position:'m',velocity:'m/s',time:'s'}, grid:{nr:8,nth:16,nz:8},
  flowSampling:{count:1,totalCells:1024,displayOnly:true} });
const close = (a, b, tol = 2e-7) => assert(Math.abs(a - b) <= tol, `${a} differs from ${b}`);
check('the offline export worker boots without DOM or fetched dependencies and reports missing capabilities', () => {
  let reply;
  const worker = { postMessage: value => { reply = value; } };
  const context = vm.createContext({ self: worker });
  new vm.Script(S.exportWorkerSource()).runInContext(context);
  assert.equal(typeof context.FARADAY_SCENE3D.create, 'function');
  assert.equal(typeof worker.onmessage, 'function');
  worker.onmessage({ data: { size: 3840, snapshot: {} } });
  assert.equal(reply.unsupported, true);
  assert.match(reply.reason, /OffscreenCanvas/);
});
check('a captured export is independent of later surface, particle and provenance edits', () => {
  const f = fixture(), copy = S.captureFrame(f);
  f.positions.fill(9); f.particles.fill(9); f.source.physical.contact = 'changed';
  f.provenance.physicalTime = 9; f.units.position = 'px'; f.grid.nr = 99; f.flowSampling.count = 99;
  close(copy.positions[2], 0.005); close(copy.particles[2], 0.0025);
  assert.equal(copy.source.physical.contact, 'pinned');
  assert.equal(copy.provenance.physicalTime, 0.01); assert.equal(copy.units.position,'m');
  assert.equal(copy.grid.nr,8); assert.equal(copy.flowSampling.count,1);
  assert.equal(copy.time, 0.01); assert.equal(copy.generation, 7); assert.equal(copy.steps, 120);
});
check('metric geometry preserves cylindrical radius and physical floor-to-surface depth', () => {
  const f = fixture(), p = S.displayedPositions(f.positions, null, f, 1, true);
  close(p[3], 1); close(p[4], 0.1); close(p[8], -1);
  const floor = S.displayedPositions(new Float64Array([0, 0, 0]), null, f, 1, false);
  close(floor[1], -0.1); close(p[4] - floor[1], f.h / f.R);
});
check('fixed gain magnifies displacement only, keeping physical arrays and floor unchanged', () => {
  const f = fixture(), original = Array.from(f.positions);
  const p = S.displayedPositions(f.positions, null, f, 4, true);
  close(p[1], (0.004 + 4 * 0.001 - 0.002) / 0.02);
  close(p[4], 0.1); close(p[7], 0.1);
  assert.deepEqual(Array.from(f.positions), original);
  const floor = S.displayedPositions(new Float64Array([0, 0, 0]), new Float64Array([0]), f, 4, false);
  close(floor[1], -0.1);
});
check('interior marker magnification follows sigma times surface displacement', () => {
  const f = fixture();
  const p = S.displayedPositions(f.particles, f.particleDisplacement, f, 4, false);
  close(p[1], (0.0025 + 3 * 0.0005 - 0.002) / 0.02);
  close(f.particles[2], 0.0025);
});
check('display gain stays fixed when consecutive physical amplitudes differ', () => {
  const f = fixture(), next = fixture(); next.positions[2] = 0.006;
  const a = S.displayedPositions(f.positions, null, f, 4, true);
  const b = S.displayedPositions(next.positions, null, next, 4, true);
  close(b[1] - a[1], 4 * 0.001 / 0.02);
});
check('invalid geometry is refused rather than clipped or replaced', () => {
  const f = fixture(); f.positions[0] = NaN; assert.throws(() => S.captureFrame(f), /finite/);
  const g = fixture(); g.indices = [0, 1, 100]; assert.throws(() => S.captureFrame(g), /outside/);
  const h = fixture(); h.indices = [0, 0.5, 2]; assert.throws(() => S.captureFrame(h), /outside/);
  const j = fixture(); j.flowVectors = new Float32Array(0); assert.throws(() => S.captureFrame(j), /counts differ/);
  const k = fixture(); k.particleDisplacement = [0, 1]; assert.throws(() => S.captureFrame(k), /count/);
});
check('triangle normals represent the supplied tilted three-dimensional surface', () => {
  const normals = S.normalsFor(new Float32Array([0, 0, 0, 1, 1, 0, 0, 0, -1]), new Uint32Array([0, 1, 2]));
  const d = Math.sqrt(0.5); close(normals[0], -d); close(normals[1], d); close(normals[2], 0);
});
check('perspective projection makes nearer geometry larger with correct depth ordering', () => {
  const camera = { yaw: 0, elevation: 0, distance: 3, targetX: 0, targetY: 0 };
  const near = S.projectPoint([1, 0, 1], camera, 1), far = S.projectPoint([1, 0, -1], camera, 1);
  assert(near.x > far.x && near.depth < far.depth); close(near.x / far.x, 2);
  close(S.projectPoint([0, 0, 0], camera, 1).x, 0);
});
check('orbit changes the projected view without modifying physical coordinates', () => {
  const p = [1, 0.25, 0], before = [...p];
  const a = S.projectPoint(p, { yaw: 0, elevation: 0.5, distance: 3 }, 1);
  const b = S.projectPoint(p, { yaw: Math.PI / 2, elevation: 0.5, distance: 3 }, 1);
  assert(Math.abs(a.x - b.x) > 0.1); assert.deepEqual(p, before);
});
check('camera zoom changes perspective while recorded physical time stays unchanged', () => {
  const f = S.captureFrame(fixture());
  const a = S.projectPoint([0.5, 0.1, 0], { yaw: 0, elevation: 0, distance: 4 }, 1);
  const b = S.projectPoint([0.5, 0.1, 0], { yaw: 0, elevation: 0, distance: 2 }, 1);
  close(b.x / a.x, 2); assert.equal(f.time, 0.01);
});
check('optional physical grain radii and their provenance survive an independent capture', () => {
  const f = fixture(); f.particleRadii = new Float64Array([0.000125]);
  f.grainProvenance = { geometry: 'sphere', units: { radius: 'm' } };
  const saved = S.captureFrame(f);
  f.particleRadii[0] = 10; f.grainProvenance.units.radius = 'pixels';
  assert(saved.particleRadii instanceof Float64Array);
  assert.equal(saved.particleRadii[0], 0.000125);
  assert.equal(saved.grainProvenance.units.radius, 'm');
  assert.equal(S.captureFrame(fixture()).particleRadii, null);
  assert.equal(S.DEFAULT_OPTIONS.surfaceOpacity, null);
});
check('grain radius validation refuses malformed or nonphysical size data', () => {
  for (const radii of [[], [0], [-0.1], [NaN], [Infinity], [0.0001, 0.0002]]) {
    const f = fixture(); f.particleRadii = radii;
    assert.throws(() => S.captureFrame(f), /radi|finite/);
  }
});
check('native grain identities and rigid-body state remain independent and validated in snapshots', () => {
  const f=fixture();f.particleRadii=new Float64Array([.000125]);f.grainIds=new Float64Array([7]);f.grainDensity=new Float64Array([2650]);
  f.grainVelocities=new Float64Array([1,2,3]);f.grainAngularVelocities=new Float64Array([4,5,6]);
  const saved=S.captureFrame(f);f.grainVelocities.fill(0);f.grainIds[0]=8;
  assert.deepEqual(Array.from(saved.grainVelocities),[1,2,3]);assert.equal(saved.grainIds[0],7);
  for(const [key,value] of [['grainIds',[-1]],['grainIds',[.5]],['grainDensity',[0]],['grainVelocities',[1,2]],['grainAngularVelocities',[0,Infinity,0]]])
    assert.throws(()=>S.captureFrame({...f,[key]:value}),/grain|finite/);
});
check('finite spheres use metre-valued radii independent of surface display gain', () => {
  const f = fixture(); f.particleRadii = new Float64Array([0.000125]);
  const saved = S.captureFrame(f), r = S.displayedParticleRadii(saved)[0];
  close(r * saved.R, 0.000125, 1e-11);
  const a = S.displayedPositions(saved.particles, saved.particleDisplacement, saved, 1, false);
  const b = S.displayedPositions(saved.particles, saved.particleDisplacement, saved, 5000, false);
  assert.notEqual(a[1], b[1]);
  assert.equal(S.displayedParticleRadii(saved)[0], r);
  assert.equal(saved.particleRadii[0], 0.000125);
});
check('sphere billboard bounds are exact perspective tangents, including off-axis centres', () => {
  const camera={yaw:0,elevation:0,distance:3,targetX:0,targetY:0}, f=1/Math.tan(42*Math.PI/360);
  const radius=.1, bounds=S.projectedSphereBounds([0,0,0],radius,camera,1);
  close(bounds.max[0],f*radius/Math.sqrt(9-radius*radius),1e-12);
  close(bounds.min[0],-bounds.max[0],1e-12);
  const centre=[.8,.4,.6], d=3-centre[2], off=S.projectedSphereBounds(centre,radius,camera,1.5);
  for(let axis=0;axis<2;axis++)for(const edge of [off.min[axis],off.max[axis]]){
    const m=edge/(axis?f:f/1.5), residual=(centre[axis]-m*d)**2-radius**2*(1+m*m);
    close(residual,0,1e-12);
  }
  for(let j=0;j<=24;j++)for(let k=0;k<48;k++){
    const phi=j*Math.PI/24,theta=k*2*Math.PI/48;
    const p=[centre[0]+radius*Math.sin(phi)*Math.cos(theta),centre[1]+radius*Math.cos(phi),centre[2]+radius*Math.sin(phi)*Math.sin(theta)];
    const q=S.projectPoint(p,camera,1.5);
    assert(q.x>=off.min[0]-1e-7&&q.x<=off.max[0]+1e-7&&q.y>=off.min[1]-1e-7&&q.y<=off.max[1]+1e-7);
  }
  assert.deepEqual(S.projectedSphereBounds([0,0,2.95],.1,camera,1),{min:[-1,-1],max:[1,1]});
});
check('finite sphere silhouettes change with physical radius and perspective depth', () => {
  const camera={yaw:0,elevation:0,distance:3,targetX:0,targetY:0};
  const width=(r,z)=>S.projectPoint([r,0,z],camera,1).x-S.projectPoint([-r,0,z],camera,1).x;
  close(width(0.1,0)/width(0.05,0),2);
  close(width(0.05,1)/width(0.05,-1),2);
  assert(S.projectPoint([0,0,0.1],camera,1).depth<S.projectPoint([0,0,0],camera,1).depth);
});
console.log(`${checks} scene geometry checks passed; WebGL interactions are checked in the browser suite.`);
