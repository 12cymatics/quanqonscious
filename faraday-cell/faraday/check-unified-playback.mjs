// Exercise the page's actual recorded-frame controller without a graphics device.
// Drawing is a stub; frame validation/copying uses the real scene module. Real
// browser checks cover GPU output and the worker-driven UI transitions.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const sceneModule = require('./scene3d.js');
const html = readFileSync(new URL('../cymatic.html', import.meta.url), 'utf8');
const marker = 'const UNIFIED3D = (() => {';
const start = html.indexOf(marker), end = html.indexOf('\n})();', start);
assert(start >= 0 && end > start, 'the actual unified controller exists');
assert.equal(html.indexOf(marker, start + marker.length), -1, 'the controller is unambiguous');
const controller = html.slice(start, end + '\n})();'.length);

function frame(time, generation = 1, particleCount = 2) {
  const particles = new Float64Array(3 * particleCount);
  if (particles.length) particles.set([0.1, 0.2, 0.4]);
  return { R: 1, h: 0.5, time, generation, steps: Math.round(time * 1000), initial: time === 0,
    positions: new Float32Array([0, 0, 0.51, 1, 0, 0.5, 0, 1, 0.5]),
    indices: new Uint32Array([0, 1, 2]), particles,
    flowPositions: new Float32Array([0, 0, 0.25]), flowVectors: new Float32Array([0, 0, 0]),
    etaRange: [0, 0.01], maxSpeed: 0,
    provenance: { parameters: { omegaD: 2 * Math.PI * 111 }, specimen: { name: 'specified liquid' } } };
}

function harness() {
  const elements = new Map(), queuedFrames = [];
  let drawn = null, camera = { yaw: 0.4, elevation: 0.6, distance: 3 }, options = { heightScale: 5000 };
  let pauses = 0, renders = 0, saved = null;
  const scene = {
    setFrame(f) { drawn = sceneModule.captureFrame(f); return scene; },
    setCamera(value) { camera = { ...camera, ...value }; return scene; },
    setOptions(value) { options = { ...options, ...value }; return scene; },
    render() { renders++; },
    stats: () => ({ ready: !!drawn }),
    snapshot() {
      if (!drawn) throw new Error('No frame has been drawn.');
      return { frame: sceneModule.captureFrame(drawn), camera: { ...camera }, options: { ...options } };
    },
    exportCanvas(size, snapshot) { saved = { size, snapshot }; return { width: size, height: size }; }
  };
  const context = vm.createContext({
    $: id => {
      if (!elements.has(id)) elements.set(id, { value: '', textContent: '', disabled: false,
        attributes: {}, setAttribute(key, value) { this.attributes[key] = value; } });
      return elements.get(id);
    },
    FARADAY_SCENE3D: { ...sceneModule, create: () => scene },
    CELL3D: { pause() { pauses++; } },
    generalInput: { seedFraction: 0.0001, depthMm: 4 }, freq: 111,
    requestAnimationFrame: callback => { queuedFrames.push(callback); return queuedFrames.length; },
    structuredClone, ArrayBuffer,
  });
  vm.runInContext(controller, context);
  const api = vm.runInContext('UNIFIED3D', context);
  return { api, scene, elements, pauses: () => pauses, renders: () => renders,
    saved: () => saved, paint: () => { const callback = queuedFrames.shift(); assert(callback); callback(1); } };
}

let passed = 0;
async function check(name, run) {
  await run(); passed++; console.log(`  ok   ${name}`);
}

await check('recording retains original physical timestamps and SI geometry', () => {
  const h = harness(), f = frame(0.125);
  h.api.acceptFrame(f); h.api.tick(500000);
  assert.equal(h.api.currentFrame().time, 0.125);
  assert.deepEqual(Array.from(h.api.currentFrame().positions), Array.from(f.positions));
  assert.equal(h.api.currentFrame().source.physicalTime, 0.125);
  assert.equal(h.api.currentFrame().source.model, 'nonlinear-incompressible-free-surface-navier-stokes');
  assert.equal(h.api.currentFrame().source.particleModel, 'passive-fluid-tracers');
  assert.equal(h.pauses(), 0);
  assert.equal(h.renders(), 1);
});

await check('accepted history is independent of the caller and returned snapshots', () => {
  const h = harness(), first = frame(0.1), expected = Array.from(first.particles);
  h.api.acceptFrame(first);
  first.particles[0] = 0.9; first.positions[2] = 9;
  first.provenance.specimen.name = 'mutated outside controller';
  h.api.acceptFrame(frame(0.2)); h.api.seek(0);
  const captured = h.api.currentFrame();
  assert.deepEqual(Array.from(captured.particles), expected);
  assert.equal(captured.positions[2], Math.fround(0.51));
  assert.equal(captured.source.specimen.name, 'specified liquid');
  captured.particles[0] = 8;
  assert.deepEqual(Array.from(h.api.currentFrame().particles), expected);
});

await check('stale generations and backwards time cannot replace recorded state', () => {
  const h = harness(); h.api.acceptFrame(frame(0.1, 4));
  assert.equal(h.api.acceptFrame(frame(0.5, 3)), false);
  assert.equal(h.api.currentFrame().generation, 4);
  assert.equal(h.api.currentFrame().time, 0.1);
  assert.throws(() => h.api.acceptFrame(frame(0.05, 4)), /backwards/i);
  assert.equal(h.api.historyState().frames.length, 1);
  h.api.acceptFrame(frame(0, 5));
  assert.equal(h.api.historyState().frames.length, 1);
  assert.equal(h.api.currentFrame().generation, 5);
  assert.equal(h.api.currentFrame().time, 0);
});

await check('invalid metadata is refused before replacing a valid frame', () => {
  const h = harness(); h.api.acceptFrame(frame(0.1));
  for (const invalid of [frame(-1), frame(NaN), frame(0.2, 1.5)])
    assert.throws(() => h.api.acceptFrame(invalid), /metadata/i);
  assert.equal(h.api.historyState().frames.length, 1);
  assert.equal(h.api.currentFrame().time, 0.1);
});

await check('history retains at most 180 frames and discloses its available interval', () => {
  const h = harness();
  for (let i = 0; i < 200; i++) h.api.acceptFrame(frame(i / 1000));
  const state = h.api.historyState();
  assert.equal(state.frames.length, 180);
  assert.equal(state.frames[0].time, 0.02);
  assert.equal(state.frames.at(-1).time, 0.199);
  assert.match(h.elements.get('historyNote').textContent, /180 frames/);
  assert.match(h.elements.get('historyNote').textContent, /20\.000000–199\.000000 ms/);
});

await check('the 32 MiB history bound is enforced independently of frame count', () => {
  const h = harness();
  for (let i = 0; i < 7; i++) h.api.acceptFrame(frame(i / 1000, 1, 250000));
  const state = h.api.historyState();
  assert(state.bytes <= 32 * 1024 * 1024);
  assert(state.frames.length > 1 && state.frames.length < 7);
  assert.equal(state.frames.at(-1).time, 0.006);
});

await check('an oversized single observation cannot silently exceed the history bound', () => {
  const h = harness(); h.api.acceptFrame(frame(0.1));
  assert.throws(() => h.api.acceptFrame(frame(0.2, 1, 1500000)), /history|MiB|memory|large|limit/i);
  assert.equal(h.api.historyState().frames.length, 1);
  assert.equal(h.api.currentFrame().time, 0.1);
});

await check('seeking pauses the solver and keeps new incoming frames out of the selected view', () => {
  const h = harness(); h.api.acceptFrame(frame(0.1)); h.api.acceptFrame(frame(0.2));
  h.api.seek(0);
  assert.equal(h.pauses(), 1);
  assert.equal(h.api.historyState().following, false);
  assert.equal(h.api.currentFrame().time, 0.1);
  h.api.acceptFrame(frame(0.3)); h.api.tick(100000);
  assert.equal(h.api.currentFrame().time, 0.1);
  assert.throws(() => h.api.seek(100), /outside/i);
  assert.equal(h.pauses(), 1);
  h.api.live();
  assert.equal(h.api.currentFrame().time, 0.3);
  assert.equal(h.api.historyState().following, true);
  assert.equal(h.pauses(), 1, 'returning to the latest view does not restart physics');
});

await check('replay displays only already-recorded frames and ends without extrapolation', () => {
  const h = harness();
  for (const time of [0.01, 0.02, 0.06, 0.11]) h.api.acceptFrame(frame(time));
  h.api.play(); assert.equal(h.pauses(), 1); h.api.tick(1000);
  assert.equal(h.api.currentFrame().time, 0.01);
  h.api.tick(5000); assert.equal(h.api.currentFrame().time, 0.06);
  h.api.acceptFrame(frame(0.15));
  h.api.tick(20000);
  assert.equal(h.api.currentFrame().time, 0.11);
  assert.equal(h.api.historyState().playing, false);
  h.api.tick(1000000); assert.equal(h.api.currentFrame().time, 0.11);
  h.api.live(); assert.equal(h.api.currentFrame().time, 0.15);
});

await check('pausing replay and changing the camera never advances physical time', () => {
  const h = harness(); h.api.acceptFrame(frame(0.1)); h.api.acceptFrame(frame(0.2));
  h.api.play(); h.api.tick(0); h.api.tick(4000); h.api.play();
  const frozen = h.api.currentFrame();
  h.scene.setCamera({ yaw: 2 }); h.scene.setOptions({ heightScale: 1 }); h.api.tick(100000);
  assert.equal(h.api.currentFrame().time, frozen.time);
  assert.deepEqual(Array.from(h.api.currentFrame().particles), Array.from(frozen.particles));
  assert.deepEqual(Array.from(h.api.currentFrame().positions), Array.from(frozen.positions));
});

await check('reset removes old history and prevents exporting a previous generation', () => {
  const h = harness(); h.api.acceptFrame(frame(0.1)); h.api.reset();
  assert.equal(h.api.historyState().frames.length, 0);
  assert.equal(h.api.currentFrame(), null);
  assert.throws(() => h.api.captureExport(), /frame|state|initial/i);
  h.api.reset(frame(0, 2));
  assert.equal(h.api.captureExport().frame.generation, 2);
  assert.equal(h.api.captureExport().provenance.fieldKind, 'initial-condition');
});

await check('image export freezes the displayed frame, particles and camera before yielding', async () => {
  const h = harness(); h.api.acceptFrame(frame(0.1));
  const original = h.api.currentFrame(), messages = [];
  const exported = h.api.exportImage(3840, message => messages.push(message));
  h.api.acceptFrame(frame(0.2)); h.scene.setCamera({ yaw: 2 }); h.scene.setOptions({ heightScale: 1 });
  h.paint(); const result = await exported;
  const saved = h.saved();
  assert.equal(saved.size, 3840);
  assert.equal(saved.snapshot.frame.time, 0.1);
  assert.deepEqual(Array.from(saved.snapshot.frame.particles), Array.from(original.particles));
  assert.equal(saved.snapshot.camera.yaw, 0.4);
  assert.equal(saved.snapshot.options.heightScale, 5000);
  assert.equal(result.source.physicalTime, 0.1);
  assert.equal(h.api.lastExport().physicalTime, 0.1);
  assert.equal(result.canvas.width, 3840);
  assert(messages.length > 0);
});

console.log(`${passed} unified playback checks passed, 0 failed`);
