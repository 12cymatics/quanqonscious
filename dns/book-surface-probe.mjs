import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
const require = createRequire(import.meta.url);
const { FaradayCell3D } = require('./faraday-cell3d.js');
const FLUID = { rho: 998.2041322005837, nu: 1.0035510043427237e-6,
                gamma: 0.07273614042160757, g: 9.80665, nz: 6 };
const request = JSON.parse(readFileSync(0, 'utf8'));
if (request.mode !== 'grid' && request.mode !== 'measure')
  throw new TypeError(`mode = ${JSON.stringify(request.mode)}: either "grid" or "measure"`);
const out = request.cases.map(c => {
  const S = new FaradayCell3D({ ...FLUID, nr: c.nr, nth: c.nth, R: c.R, h: c.h, contact: c.contact });
  const order = [];
  for (let i = 0; i < S.nr; i++) for (let k = 0; k < S.nth; k++) order.push(S.ie(i, k));
  if (request.mode === 'grid')
    return { nr: S.nr, nth: S.nth, cells: S.NE, rc: Array.from(S.rc), order };
  if (!Array.isArray(c.eta) || c.eta.length !== S.NE)
    throw new RangeError(`eta has ${c.eta && c.eta.length} values for ${S.NE} surface cells`);
  for (let e = 0; e < S.NE; e++){
    if (typeof c.eta[e] !== 'number' || !Number.isFinite(c.eta[e]))
      throw new TypeError(`eta[${e}] = ${c.eta[e]} is not a finite double`);
    S.eta[e] = c.eta[e];
  }
  S.refreshMetric();
  const kappa = Array.from(S.curvature(new Float64Array(S.NE)));
  const slope2 = new Array(S.NE), sr = new Array(S.NE), st = new Array(S.NE);
  for (let i = 0; i < S.nr; i++)
    for (let k = 0; k < S.nth; k++){
      const e = S.ie(i, k);
      slope2[e] = S.surfaceSlopeSquared(i, k);
      const n = S.surfaceNormal(S.rc[i], (k + 0.5)*S.dth);
      sr[e] = n.sr; st[e] = n.st;
    }
  return { nr: S.nr, nth: S.nth, kappa, slope2, sr, st };
});
process.stdout.write(JSON.stringify(out));
