import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';

const HERE = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const D = require(join(HERE, 'dyadic128.js'));
const WASMFLOAT = require(join(HERE, 'wasm-float-scan.js'));

let passed = 0, failed = 0;
function ok(cond, name, detail = ''){
  if (cond){ passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '\n       ' + detail : ''}`); }
}

const work = mkdtempSync(join(tmpdir(), 'dyadic128-wasm-'));
function build(src, out){
  execFileSync('clang++', ['--target=wasm32', '-O3', '-mbulk-memory', '-fno-exceptions', '-fno-rtti', '-nostdlib',
    '-Wl,--no-entry', '-Wl,--export-all', '-I', HERE, '-o', out, src], { encoding: 'utf8' });
  return readFileSync(out);
}

console.log('1. the module is built from dns/dyadic128_ops.cpp and holds no floating point');
const bytes = build(join(HERE, 'dyadic128_ops.cpp'), join(work, 'ops.wasm'));
const module = new WebAssembly.Module(bytes);
ok(WebAssembly.Module.imports(module).length === 0, 'the module imports nothing');
const scan = WASMFLOAT.scanWasm(bytes);
ok(scan.functions > 0 && scan.instructions > 1000, `decoded ${scan.functions} function bodies, ${scan.instructions} instructions`);
ok(scan.floatInstructions === 0 && scan.floatTypes === 0,
  `no f32 or f64 instruction, local, global or signature (${scan.floatInstructions} instructions, ${scan.floatTypes} types)`,
  JSON.stringify(scan.floatOpcodes));
{
  const ctrl = join(work, 'control.cpp');
  writeFileSync(ctrl, [
    'extern "C" {',
    'double Dm[8]; float Fm[8];',
    'double f64ops(double a, double b){ return (a + b)*(a - b)/b; }',
    'float f32ops(float a, float b){ return (a + b)*(a - b)/b; }',
    'int cmp64(double a, double b){ return a < b; }',
    'int cmp32(float a, float b){ return a < b; }',
    'long long trunc64(double a){ return (long long)a; }',
    'double conv64(long long x){ return (double)x; }',
    'double load64(int i){ return Dm[i]; }',
    'void store64(int i, double x){ Dm[i] = x; }',
    'float load32(int i){ return Fm[i]; }',
    'void store32(int i, float x){ Fm[i] = x; }',
    'double konst(){ return 0.375; }',
    '}', ''].join('\n'));
  const cs = WASMFLOAT.scanWasm(build(ctrl, join(work, 'control.wasm')));
  const want = { 'f32.load': 0x2A, 'f64.load': 0x2B, 'f32.store': 0x38, 'f64.store': 0x39, 'f64.const': 0x44,
                 'f32.lt': 0x5D, 'f64.lt': 0x63, 'f32.add': 0x92, 'f32.sub': 0x93, 'f32.mul': 0x94, 'f32.div': 0x95,
                 'f64.add': 0xA0, 'f64.sub': 0xA1, 'f64.mul': 0xA2, 'f64.div': 0xA3, 'i64.trunc_f64_s': 0xB0,
                 'f64.convert_i64_s': 0xB9 };
  const missed = Object.entries(want).filter(([, op]) => !cs.floatOpcodes[op]).map(([k]) => k);
  ok(missed.length === 0 && cs.floatTypes > 0,
    `the same scan finds every class of float instruction in a control module built to contain them: loads, stores, constants, comparisons, `
    + `f32 and f64 arithmetic, truncation and conversion (${cs.floatInstructions} instructions, ${cs.floatTypes} types)`,
    missed.length ? `missed: ${missed.join(', ')}` : '');
}

const inst = new WebAssembly.Instance(module, {});
const X = inst.exports;
const CAP = X.capacity();
const M64 = (1n << 64n) - 1n;
const view = () => new DataView(X.memory.buffer);
function put(ptr, k, a){
  const dv = view(), base = ptr + 24*k, mag = a.m < 0n ? -a.m : a.m;
  dv.setBigUint64(base, mag >> 64n, true); dv.setBigUint64(base + 8, mag & M64, true);
  dv.setInt32(base + 16, a.m === 0n ? 0 : a.e, true); dv.setInt32(base + 20, a.m < 0n ? 1 : 0, true);
}
function get(ptr, k){
  const dv = view(), base = ptr + 24*k;
  const mag = (dv.getBigUint64(base, true) << 64n) | dv.getBigUint64(base + 8, true);
  const e = dv.getInt32(base + 16, true), s = dv.getInt32(base + 20, true);
  return { m: s ? -mag : mag, e: mag === 0n ? 0 : e };
}
const same = (x, y) => x.m === y.m && x.e === y.e;

let seed = 0x2545F491;
const rnd = () => { seed ^= seed << 13; seed >>>= 0; seed ^= seed >>> 17; seed ^= seed << 5; seed >>>= 0; return BigInt(seed); };
const big = w => { let x = 0n; for (let i = 0; i < w; i++) x = (x << 32n) | rnd(); return x; };
const P127 = 1n << 127n, P128 = 1n << 128n;
const dy = (m, e) => ({ m, e });
function randD(){
  const kind = Number(rnd() % 8n);
  const sgn = rnd() & 1n ? -1n : 1n, e = Number(rnd() % 300n) - 150;
  if (kind === 0) return D.fromBigInt((rnd() % 7n) - 3n);
  if (kind === 1) return dy(sgn*(P127 + (rnd() % 5n)), e);
  if (kind === 2) return dy(sgn*(P128 - 1n - (rnd() % 5n)), e);
  const n = big(1 + Number(rnd() % 6n))*sgn, d = big(1 + Number(rnd() % 5n)) + 1n;
  const x = D.fromRational(n, d);
  return x.m === 0n ? x : dy(x.m, x.e + e);
}

const constructed = [
  ['an addition that lands exactly on a tie, rounded to even', dy(P127, 1), dy(P127 + 1n, 0)],
  ['an addition tie that rounds up to even', dy(P127 + 1n, 1), dy(P127 + 1n, 0)],
  ['a multiplication exactly on a tie', dy(P127 + (1n << 63n), 0), dy(P127 + (1n << 63n), 0)],
  ['a multiplication a hair above a tie, decided by the lowest word', dy(P127 + (1n << 62n), 0), dy(P127 + (1n << 64n) + 2n, 0)],
  ['a subtraction that cancels to the last bit', dy(P127 + 5n, 3), dy(P127 + 5n, 3)],
  ['a subtraction that cancels all but one bit', dy(P127 + 5n, 3), dy(P127 + 4n, 3)],
  ['an addition 129 binades apart, where the power of two drops a binade', dy(P127, 129), dy(-(P128 - 1n), 0)],
  ['an addition 130 binades apart', dy(P127, 130), dy(-(P128 - 1n), 0)],
  ['a product that carries into a new binade on rounding', dy(P128 - 1n, 0), dy(P128 - 1n, 0)],
  ['a division whose trial quotient digit is one too large, so the long division adds back', dy(P127, 0), dy(0xffffffffffffffff0000000000000001n, 0)],
  ['a second add-back division', dy(0xfffffffffffffffe0000000000000000n, 0), dy(0xffffffffffffffff00000000ffffffffn, 0)],
  ['a third add-back division', dy(0xfffffffffffffffe0000000000000000n, 0), dy(0xfffffffeffffffff0000000000000001n, 0)]
];

console.log('2. add, sub, mul and div agree with dns/dyadic128.js bit for bit');
{
  const N = Math.min(CAP, 6000);
  const As = [], Bs = [];
  for (const [, a, b] of constructed){ As.push(a); Bs.push(b); }
  while (As.length < N){
    let a = randD(), b = randD();
    if (As.length % 7 === 0 && a.m !== 0n) b = dy((rnd() & 1n ? a.m : -a.m) + ((rnd() % 3n) - 1n), a.e + Number(rnd() % 3n) - 1);
    if (b.m !== 0n && ((b.m < 0n ? -b.m : b.m) >> 127n) !== 1n) b = D.fromBigInt(b.m);
    As.push(a); Bs.push(b);
  }
  for (let i = 0; i < N; i++){ put(X.bufA(), i, As[i]); put(X.bufB(), i, Bs[i]); }
  for (const [name, jsOp, wOp] of [['mul', D.mul, X.opMul], ['add', D.add, X.opAdd], ['sub', D.sub, X.opSub]]){
    wOp(N);
    let bad = 0, firstBad = '';
    for (let i = 0; i < N; i++){
      const want = jsOp(As[i], Bs[i]), got = get(X.bufO(), i);
      if (!same(want, got)){ bad++; if (!firstBad) firstBad = `case ${i}: want ${want.m} * 2^${want.e}, got ${got.m} * 2^${got.e}`; }
    }
    ok(bad === 0, `${name}: ${N} operand pairs, ${constructed.length} of them constructed edge cases (${bad} differ)`, firstBad);
  }
  for (let i = 0; i < N; i++) if (Bs[i].m === 0n) put(X.bufB(), i, D.ONE);
  X.opDiv(N);
  let bad = 0, firstBad = '';
  for (let i = 0; i < N; i++){
    const want = D.div(As[i], Bs[i].m === 0n ? D.ONE : Bs[i]), got = get(X.bufO(), i);
    if (!same(want, got)){ bad++; if (!firstBad) firstBad = `case ${i}: want ${want.m} * 2^${want.e}, got ${got.m} * 2^${got.e}`; }
  }
  ok(bad === 0, `div: ${N} operand pairs (${bad} differ)`, firstBad);
  for (let i = 0; i < N; i++) put(X.bufB(), i, Bs[i]);
  X.opCmp(N);
  const C = new Int32Array(X.memory.buffer, X.bufC(), N);
  let badC = 0;
  for (let i = 0; i < N; i++) if (C[i] !== D.cmp(As[i], Bs[i])) badC++;
  ok(badC === 0, `cmp: ${N} operand pairs (${badC} differ)`);
}

console.log('3. square root agrees bit for bit, including the cases that decide its rounding');
{
  const xs = [dy(P127 + 1n, 1), dy(P128 - 1n, 0), dy(P128 - 1n, 1), dy(P127, 0), dy(P127, 1), D.fromBigInt(4n), D.fromBigInt(2n)];
  for (let k = 0n; k < 40n; k++) xs.push(D.fromBigInt((k*k + 3n*k + 1n)*(k*k + 3n*k + 1n)));
  while (xs.length < 4000){ const a = randD(); xs.push(a.m < 0n ? D.neg(a) : a); }
  for (let i = 0; i < xs.length; i++) put(X.bufA(), i, xs[i]);
  X.opSqrt(xs.length);
  let bad = 0, firstBad = '';
  for (let i = 0; i < xs.length; i++){
    const want = D.sqrt(xs[i]), got = get(X.bufO(), i);
    if (!same(want, got)){ bad++; if (!firstBad) firstBad = `case ${i}: want ${want.m} * 2^${want.e}, got ${got.m} * 2^${got.e}`; }
  }
  ok(bad === 0, `sqrt: ${xs.length} arguments, among them r(r+1), where the remainder equals the root (${bad} differ)`, firstBad);
}

console.log('4. conversions');
{
  X.opFromI64(2000, -1000n, 1n);
  let bad = 0;
  for (let i = 0; i < 2000; i++) if (!same(get(X.bufO(), i), D.fromBigInt(BigInt(i - 1000)))) bad++;
  X.opFromI64(3, -(1n << 63n), (1n << 62n));
  for (let i = 0; i < 3; i++) if (!same(get(X.bufO(), i), D.fromBigInt(-(1n << 63n) + BigInt(i)*(1n << 62n)))) bad++;
  ok(bad === 0, `integers from -1000 to 999 and around -2^63 (${bad} differ)`);
  const xs = [];
  for (let i = 0; i < 500; i++) xs.push(randD());
  for (let i = 0; i < xs.length; i++) put(X.bufA(), i, xs[i]);
  X.opScale2(xs.length, -37);
  let badS = 0;
  for (let i = 0; i < xs.length; i++) if (!same(get(X.bufO(), i), D.scale2(xs[i], -37))) badS++;
  ok(badS === 0, `scaling by 2^-37 (${badS} differ)`);
}

console.log('5. time per operation (reported, not gated)');
{
  const n = 4096;
  for (let i = 0; i < n; i++){ put(X.bufA(), i, D.fromRational(BigInt(1000 + 7*i), BigInt(3 + i % 11))); put(X.bufB(), i, D.fromRational(-BigInt(17 + i), BigInt(9 + i % 5))); }
  const per = {};
  for (const [k, f] of [['mul', X.opMul], ['add', X.opAdd], ['div', X.opDiv], ['sqrt', X.opSqrt]]){
    for (let r = 0; r < 20; r++) f(n);
    const t = process.hrtime.bigint();
    for (let r = 0; r < 200; r++) f(n);
    per[k] = (Number(process.hrtime.bigint() - t)/(200*n)).toFixed(1) + ' ns';
  }
  console.log('       ' + JSON.stringify(per));
}

rmSync(work, { recursive: true, force: true });
console.log(`\n${passed} passed, ${failed} failed`);
process.exitCode = failed ? 1 : 0;
