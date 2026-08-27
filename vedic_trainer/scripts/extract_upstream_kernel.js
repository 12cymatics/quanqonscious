// Runs the UPSTREAM STRICT_SUTRA_KERNEL out of the tracked simulator HTML.
//
// Why this exists
// ---------------
// Falsification criterion 3 asked whether this package's exact-Q kernel agrees
// with the upstream definition, and every document answered that it could not
// be checked because `vedic_v18.24_full_kernel.html` "is external to this
// repository". It is not: it is tracked at the work-tree root. The comparison
// was always possible and was never made.
//
// This lifts the strict kernel and its dependencies out of the HTML *verbatim*
// -- by source-slicing, not by reimplementation -- and evaluates them. A
// reimplementation would compare this package against itself, which is the
// defect `verify_bit_exact.py` already documents about its own fixtures.
'use strict';
const fs = require('fs');
const path = require('path');

const HTML = path.join(__dirname, '..', '..', 'vedic_v18.24_full_kernel.html');

function sliceConst(src, decl) {
  // From `decl` to the line that closes it at column 0 (`};` or `}`).
  const i = src.indexOf(decl);
  if (i < 0) throw new Error(`not found in upstream HTML: ${decl}`);
  const m = /\n\};/.exec(src.slice(i));
  if (!m) throw new Error(`no terminator for: ${decl}`);
  return src.slice(i, i + m.index + m[0].length);
}
function sliceLine(src, needle) {
  const i = src.indexOf(needle);
  if (i < 0) throw new Error(`not found in upstream HTML: ${needle}`);
  const e = src.indexOf('\n', i);
  return src.slice(i, e);
}

const src = fs.readFileSync(HTML, 'utf8');

const parts = [
  sliceLine(src, 'const gcd = (a, b) =>'),
  sliceConst(src, 'const strictBi = v => {').replace(/\n\};$/, '\n};'),
  sliceConst(src, 'const Q = {'),
  'const Bi = strictBi;',
  'Q.ZERO = Q.mk(0n); Q.ONE = Q.mk(1n); Q.TWO = Q.mk(2n);',
  sliceLine(src, 'const SUTRA_SUM = 435n;'),
  src.slice(src.indexOf('const SUTRA_KIND = ['),
            src.indexOf('];', src.indexOf('const SUTRA_KIND = [')) + 2),
  sliceConst(src, 'const ALPHA = {'),
  sliceConst(src, 'const VTX = {'),
  // MSTVQ is touched only for id 29's display side-effect; the field update
  // above it is already complete. Stubbed rather than lifted so no display
  // code enters the comparison path.
  'const MSTVQ = { setB(){}, compute(){}, computeExact(){} };',
  'let lambda = [Q.mk(0n), Q.mk(0n), Q.mk(0n), Q.mk(0n)];',
  sliceConst(src, 'const STRICT_SUTRA_KERNEL = {'),
];

// eslint-disable-next-line no-eval
const load = new Function(parts.join('\n\n') + `
  return { Q, VTX, ALPHA, SUTRA_KIND, STRICT_SUTRA_KERNEL };
`);
const U = load();

function applyOne(id, psiPairs, strength) {
  for (let i = 0; i < 16; i++) U.VTX.psi[i] = U.Q.mk(BigInt(psiPairs[i][0]), BigInt(psiPairs[i][1]));
  U.STRICT_SUTRA_KERNEL.applyOne({ id, strength });
  return U.VTX.psi.map(p => [p.n.toString(), p.d.toString()]);
}

const req = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = req.map(({ id, psi, strength }) => applyOne(id, psi, strength));
process.stdout.write(JSON.stringify(out));
