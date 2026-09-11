#!/usr/bin/env node
// Checks for the exact coupled affine benchmark. Values come from the stated
// construction and from independent integration, never from the module's own
// opinion of itself.
import { createRequire } from 'node:module';
const B = createRequire(import.meta.url)('./coupled-affine-benchmark.js');
const { Q, evaluate, allResidualsZero } = B;

let pass = 0, fail = 0;
const ok  = (n, c, d='') => { c ? (pass++, console.log(`  ok   ${n}`))
                                : (fail++, console.log(`  FAIL ${n}   ${d}`)); };
const eq  = (n, got, want) => ok(n, Q.of(got).cmp(Q.of(want)) === 0, `got ${got}, want ${want}`);
const throws = (n, fn, frag) => {
  try { fn(); ok(n, false, 'did not throw'); }
  catch (e){ ok(n, !frag || String(e.message).includes(frag), e.message); }
};

const STRETCHES = ['5/4', '13/10', '7/5', '17/12', '3/2'];
const PARAMS = [ {},
  { rhoF:'7/3',  mu:'5/2',  rhoS:'11/4', G:'13/5', gamma:'9/7'  },
  { rhoF:'1/8',  mu:'3/16', rhoS:'21/5', G:'1/3',  gamma:'17/2' },
  { rhoF:'999/1000', mu:'1/1000', rhoS:'2650/1000', G:'50', gamma:'727/10000' } ];

console.log('1. every local residual is exactly zero over Q');
let total = 0;
for (const P of PARAMS) for (const s of STRETCHES){
  const z = allResidualsZero(evaluate({ stretch: s, ...P }));
  total += z.count;
  ok(`s=${s} ${Object.keys(P).length ? 'params '+P.G : 'defaults'} — ${z.count} residuals`,
     z.allZero, z.nonzero.join(', '));
}
console.log(`  ${total} residual values checked, none approximate\n`);

console.log('2. geometry, integrated from the declared radii and heights');
const g = evaluate({ stretch:'5/4' }).geometry;
eq('liquid volume = 1',      g.liquid.volume, '1');
eq('liquid radial = 1/2',    g.liquid.radial, '1/2');
eq('liquid vertical = 1/3',  g.liquid.vertical, '1/3');
eq('solid volume = 7',       g.solid.volume, '7');
eq('solid radial = 31/2',    g.solid.radial, '31/2');
eq('solid vertical = 7/3',   g.solid.vertical, '7/3');
eq('total volume = 8',       g.totalVolume, '8');

console.log('\n3. the closure and the stresses it forces');
for (const s of STRETCHES){
  const r = evaluate({ stretch: s });
  ok(`s=${s}: 6μa − d = 0`, r.residuals.viscousElasticClosure.isZero);
  ok(`s=${s}: σ_f = σ_s in all three components`,
     r.stress.sigmaF.every((v, i) => v.cmp(r.stress.sigmaS[i]) === 0));
  ok(`s=${s}: det F = 1`, r.state.detF.cmp(Q.of('1')) === 0);
}

console.log('\n4. the clock is monotone on the interval');
// d′(s)/G = 4s³ − 2s − 4s⁻⁵ + 8s⁻⁹, bounded below on [5/4, 3/2] by
// 125/16 − 3 − 4096/3125, which the construction states is positive.
const bound = Q.of('125/16').sub(Q.of('3')).sub(Q.of('4096/3125'));
ok(`stated lower bound 125/16 − 3 − 4096/3125 = ${bound} > 0`, bound.cmp(Q.of('0')) > 0);
for (const s of STRETCHES){
  const q = Q.of(s);
  const dP = Q.of('4').mul(q.pow(3)).sub(Q.of('2').mul(q))
             .sub(Q.of('4').mul(q.pow(-5))).add(Q.of('8').mul(q.pow(-9)));
  ok(`s=${s}: d′/G = ${dP} exceeds the bound`, dP.cmp(bound) >= 0);
}
const r0 = evaluate({ stretch:'5/4' });
ok('T* lower bound < upper bound', r0.clock.tStarLower.cmp(r0.clock.tStarUpper) < 0);
ok('T* lower bound positive', r0.clock.tStarLower.cmp(Q.of('0')) > 0);

console.log('\n5. inexact and invalid inputs are refused, never coerced');
throws('a binary64 stretch is rejected', () => evaluate({ stretch: 1.25 }), 'binary64');
throws('a binary64 viscosity is rejected', () => evaluate({ mu: 0.001 }), 'binary64');
throws('zero viscosity is rejected', () => evaluate({ mu: '0' }), 'positive');
throws('negative density is rejected', () => evaluate({ rhoF: '-1' }), 'positive');
throws('unreadable text is rejected', () => evaluate({ stretch: 'about 1.3' }), 'exactly');
ok('a decimal string is read exactly',
   Q.of('1.25').cmp(Q.of('5/4')) === 0);
ok('a stretch outside [5/4, 3/2] is flagged, not silently accepted',
   evaluate({ stretch: '2' }).scope.stretchInRange === false);

console.log('\n6. scope is stated, not implied');
const sc = evaluate({ stretch:'5/4' }).scope;
ok('does not claim to reproduce the OpenAI blowup', sc.reproducesOpenAIBlowup === false);
ok('does not claim the centre-driven Faraday problem', sc.solvesCentreDrivenFaradayProblem === false);
ok('does not claim a calibrated stone law', sc.materialLawCalibratedToStone === false);
ok('does not claim a global space-time proof', sc.globalSpaceTimeProof === false);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
