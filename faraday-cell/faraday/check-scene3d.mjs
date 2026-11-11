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
console.log(`${checks} scene geometry checks passed; WebGL interactions are checked in the browser suite.`);
