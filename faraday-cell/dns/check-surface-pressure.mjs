import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { FaradayCell3D, suggestCell3DGrid } = require('./faraday-cell3d.js');
const { FaradayCell3DWasm } = require('./faraday-cell3d-wasm.js');
const { besselJ } = require('../faraday/kernel.js');
const options = { nr: 8, nth: 12, nz: 6, R: 0.017, h: 0.004,
  rho: 875, nu: 4e-6, gamma: 0.028, g: 0, accel: 0, omegaD: 0 };

function cell(amplitude, contact = 'free') {
  const s = new FaradayCell3D({ ...options, contact });
  for (let i = 0; i < s.nr; i++)
    for (let k = 0; k < s.nth; k++) {
      const r = s.rc[i] / s.R;
      const radial = contact === 'free' ? r * r * (2 - r * r) : r * r * (1 - r * r);
      s.eta[s.ie(i, k)] = amplitude * s.h * radial * Math.cos(2 * (k + 0.5) * s.dth);
    }
  s.refreshMetric();
  return s;
}

function gradient(s, p, surfaceP) {
  const out = [new Float64Array(s.NU), new Float64Array(s.NV), new Float64Array(s.NW)];
  s.gradient(p, ...out, surfaceP);
  return out;
}

function maxAbs(a) {
  let value = 0;
  for (const x of a) value = Math.max(value, Math.abs(x));
  return value;
}

for (const contact of ['free', 'pinned'])
  for (const amplitude of [0, 0.1, 0.4])
    test(`matching constant surface pressure has zero gradient: ${contact}, eta/h=${amplitude}`, () => {
      const s = cell(amplitude, contact);
      for (const value of [-1, 1, 7]) {
        const result = gradient(s, new Float64Array(s.NP).fill(value),
          new Float64Array(s.NE).fill(value));
        for (const component of result) assert.equal(maxAbs(component), 0);
      }
    });

test('a matching affine vertical pressure gives its physical gradient through the flat surface', () => {
  const s = cell(0);
  const slope = 137, offset = 3;
  const p = new Float64Array(s.NP), ps = new Float64Array(s.NE).fill(slope * s.h + offset);
  for (let i = 0; i < s.nr; i++)
    for (let k = 0; k < s.nth; k++)
      for (let j = 0; j < s.nz; j++) p[s.ip(i, k, j)] = slope * s.sc[j] * s.h + offset;
  const [u, v, w] = gradient(s, p, ps);
  assert.equal(maxAbs(u), 0);
  assert.equal(maxAbs(v), 0);
  for (let i = 0; i < s.nr; i++)
    for (let k = 0; k < s.nth; k++) {
      assert.equal(w[s.iw(i, k, 0)], 0);
      for (let j = 1; j <= s.nz; j++) assert.ok(Math.abs(w[s.iw(i, k, j)] - slope) < 1e-9);
    }
});

for (const contact of ['free', 'pinned'])
  test(`constant prescribed pressure cannot accelerate liquid initially at rest: ${contact}`, () => {
    const s = cell(0.4, contact);
    const before = s.eta.slice();
    s.surfacePressure = out => out.fill(1);
    s.step(1e-7);
    for (const a of [s.u, s.v, s.w]) assert.ok(maxAbs(a) < 1e-14);
    for (let i = 0; i < s.eta.length; i++) assert.ok(Math.abs(s.eta[i] - before[i]) < 1e-20);
    assert.ok(s.maxDivergence() < 1e-9);
  });

test('adding the same pressure offset leaves pinned nonlinear evolution unchanged', () => {
  const a = cell(0.3, 'pinned'), b = cell(0.3, 'pinned');
  const surfacePressure = b.surfacePressure.bind(b);
  b.surfacePressure = out => {
    surfacePressure(out);
    for (let i = 0; i < out.length; i++) out[i] += 7;
    return out;
  };
  a.step(1e-7);
  b.step(1e-7);
  for (const name of ['u', 'v', 'w', 'eta'])
    for (let i = 0; i < a[name].length; i++)
      assert.ok(Math.abs(a[name][i] - b[name][i]) < 1e-14, `${name}[${i}] changed under pressure gauge shift`);
  for (let i = 0; i < a.p.length; i++) assert.ok(Math.abs(b.p[i] - a.p[i] - 7) < 1e-8);
});

test('zero gravity is a valid physical input; negative and nonfinite values fail', () => {
  assert.equal(cell(0).gravity(), 0);
  for (const g of [-1, NaN, Infinity, -Infinity])
    assert.throws(() => new FaradayCell3D({ ...options, g }));
});

test('grid sizing accepts capillary waves at zero gravity and rejects invalid gravity', () => {
  const input = { ...options, m: 0, k: 2.404825557695773 / options.R, contact: 'pinned',
    omegaResponse: 400, omegaDrive: 800, besselJ, maxCells: 20000 };
  const grid = suggestCell3DGrid(input);
  assert.ok(grid.tried.length > 0);
  for (const g of [-1, NaN, Infinity, -Infinity])
    assert.throws(() => suggestCell3DGrid({ ...input, g }));
});

test('surface pressure arrays must match the physical surface grid', () => {
  const s = cell(0.1);
  assert.throws(() => gradient(s, new Float64Array(s.NP), new Float64Array(s.NE - 1)), /surfaceP/);
});

test('the pinned nonlinear surface retains a fixed rim and no-slip wall kinematics', () => {
  const s = cell(0.4, 'pinned');
  s.step(1e-7);
  assert.ok(maxAbs(s.Ht) > 0, 'the interior surface must evolve');
  for (let k = 0; k < s.nth; k++) {
    const th = (k + 0.5) * s.dth;
    assert.equal(s.etaAt(s.R, th), 0);
    assert.equal(s.etaDotAt(s.R, th), 0);
    assert.equal(s._bcU('rim', s.R, th, 1), 0);
    assert.equal(s._bcV('rim', s.R, th, 1), 0);
    assert.equal(s._bcW('rim', s.R, th, 1), 0);
  }
});

test('JavaScript and WebAssembly agree on the affine pressure operator', () => {
  const wasm = new FaradayCell3DWasm({ ...options, contact: 'pinned' });
  try {
    const s = wasm.js, shape = cell(0.4, 'pinned');
    s.eta.set(shape.eta);
    s.refreshMetric();
    wasm.pushState().refreshMetric();
    for (const offset of [0, 7]) {
      for (let i = 0; i < s.NP; i++) s.p[i] = offset + (i % 17) / 13;
      for (let i = 0; i < s.NE; i++) s._ps[i] = offset + (i % 11) / 7;
      const expected = gradient(s, s.p, s._ps);
      wasm.views.p.set(s.p);
      wasm.views.psurf.set(s._ps);
      const actual = wasm.gradient('p', 'psurf');
      for (let c = 0; c < 3; c++) assert.deepEqual(actual[c], expected[c]);
    }
    wasm.views.p.fill(1);
    wasm.views.psurf.fill(1);
    for (const a of wasm.gradient('p', 'psurf')) assert.equal(maxAbs(a), 0);
  } finally {
    wasm.release();
  }
});

test('WebAssembly rejects mismatched pressure buffers before reading raw memory', () => {
  const wasm = new FaradayCell3DWasm({ ...options, contact: 'pinned' });
  try {
    assert.throws(() => wasm.gradient('p', 'rf'), /surface|pressure|length|grid/i);
    assert.throws(() => wasm.gradient('rf', 'psurf'), /pressure|length|grid/i);
  } finally {
    wasm.release();
  }
});

if (process.argv.includes('--audit-free'))
  test('UNSUPPORTED free contact line must satisfy no-slip wall kinematics', () => {
    const s = cell(0.4, 'free');
    s.step(1e-7);
    for (let k = 0; k < s.nth; k++) {
      const th = (k + 0.5) * s.dth;
      assert.equal(s.etaDotAt(s.R, th), s._bcW('rim', s.R, th, 1),
        'the moving free rim lacks a compatible slip/contact-line closure');
    }
  });
