import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const K = require('./kernel.js');
const input = {
  diameterMm: 37.5, depthMm: 6.2, rho: 920, mu: 0.004,
  sigma: 0.031, gravity: 1.62, acceleration: 12.5,
  frequencyHz: 22.75, contact: 'pinned', seedMode: 3, seedFraction: 0.0002
};
const state = K.resolveExperimentState(input);
assert.deepEqual(state.experiment, input);
assert.equal(state.water.nu, input.mu/input.rho);
assert.equal(state.cellDiameterMm, input.diameterMm);
assert.equal(state.depthMm, input.depthMm);
assert.equal(state.gravity, input.gravity);
assert.equal(state.accelAssumed, input.acceleration);
assert.equal(state.frequencyHz, input.frequencyHz);
assert.equal(state.expression, 1);
assert.equal(state.visualSignature, null);
assert.equal(state.replicates, null);
assert.equal(state.transitionStudy, null);
assert.equal(state.responseHz, null);
assert.equal(state.transport, null);
assert.deepEqual(state.states, []);
assert.deepEqual(state.nearestTheoryModes, []);
assert.ok(!Object.hasOwn(state, 'damping'));
assert.ok(!Object.hasOwn(state, 'competition'));
assert.ok(!Object.hasOwn(state, 'onset'));

const medium = Object.getOwnPropertyDescriptor(K.ATLAS, 'medium');
const transition = Object.getOwnPropertyDescriptor(K.TRANSITION, '115');
try {
  Object.defineProperty(K.ATLAS, 'medium', { configurable: true, get(){ throw new Error('Atlas accessed by general physics.'); } });
  Object.defineProperty(K.TRANSITION, '115', { configurable: true, get(){ throw new Error('Transition prior accessed by general physics.'); } });
  assert.deepEqual(K.resolveExperimentState(input), state);
  const overlap = K.resolveExperimentState({ ...input, diameterMm: 24.25, frequencyHz: 115 });
  assert.equal(overlap.visualSignature, null);
  assert.deepEqual(overlap.states, []);
} finally {
  Object.defineProperty(K.ATLAS, 'medium', medium);
  Object.defineProperty(K.TRANSITION, '115', transition);
}

for (const frequencyHz of [0.125, 49.125, 199.5, 5000.25, 20000])
  assert.equal(K.resolveExperimentState({ ...input, frequencyHz }).frequencyHz, frequencyHz);
const unforced = K.resolveExperimentState({ ...input, acceleration: 0, gravity: 0 });
assert.equal(unforced.accelAssumed, 0);
assert.equal(unforced.gravity, 0);
assert.equal(unforced.expression, 1);
for (const key of ['diameterMm', 'depthMm', 'rho', 'mu', 'sigma', 'frequencyHz']){
  for (const value of [0, -1, NaN, Infinity, '1'])
    assert.throws(() => K.resolveExperimentState({ ...input, [key]: value }), /finite positive/);
}
for (const key of ['gravity', 'acceleration']){
  for (const value of [-1, NaN, Infinity, '1'])
    assert.throws(() => K.resolveExperimentState({ ...input, [key]: value }), /finite non-negative/);
}
for (const value of [-1, 1.5, NaN, Infinity])
  assert.throws(() => K.resolveExperimentState({ ...input, seedMode: value }), /safe integer/);
for (const value of [0, 1, 2.5])
  assert.equal(K.resolveExperimentState({ ...input, seedFraction: value }).experiment.seedFraction, value);
for (const value of [-1, NaN, Infinity, '1'])
  assert.throws(() => K.resolveExperimentState({ ...input, seedFraction: value }), /finite non-negative/);
assert.throws(() => K.resolveExperimentState({ ...input, contact: 'invented' }), /contact/);

const k = 173, depth = 0.007;
const expected = Math.sqrt((input.gravity*k + input.sigma*k**3/input.rho)*Math.tanh(k*depth));
assert.equal(K.omegaOf(k, input.sigma, input.rho, depth, input.gravity), expected);
const recovered = K.kOfOmega(expected, input.sigma, input.rho, depth, input.gravity);
assert.ok(Math.abs(recovered-k)/k < 1e-12);
const low = K.pinnedEdgeSpectrum(2, 0.02, 0.04, 950, 0.006, 8, 2, 1.62);
const earth = K.pinnedEdgeSpectrum(2, 0.02, 0.04, 950, 0.006, 8, 2, 9.80665);
assert.ok(low.pinned.every((mode, i) => mode.omega < earth.pinned[i].omega));
console.log('PASS: general SI input fidelity, no atlas/prior access, no modal substitution, explicit invalid-input rejection, and optional modal gravity propagation.');
