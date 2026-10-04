const K=require('./boundarykernel.js');
const PI=Math.PI;
let pass=0, fail=0;
const check=(ok,what)=>{ ok?pass++:fail++; console.log((ok?'  ok   ':'  FAIL ')+what); };

console.log('1. E(m) accuracy vs published reference values');
// references from mpmath at 25 digits. The bound is 2 ulp for m <= 0.99; near
// m = 1 the AGM degrades because b0 = sqrt(1-m) -> 0, so 8 ulp is allowed there
// and the measured worst case is 1.11e-15 at m = 0.999.
const refs=[[0,1.5707963267948966],[0.05,1.5509733517804725],[0.1,1.5307576368977631],
            [0.25,1.4674622093394272],[0.5,1.3506438810476755],[0.75,1.2110560275684601],
            [0.9,1.1047747327040733],[0.99,1.0159935450252239],
            [0.999,1.0021707908344453],[0.99999,1.0000332138990837]];
const ULP=Number.EPSILON;   // 2.22e-16
for(const [m,ref] of refs){
  const got=K.completeEllipticE(m);
  const bound = m>0.99 ? 8*ULP : 2*ULP;
  check(Math.abs(got-ref)<=bound,
    `E(${m}) = ${got.toFixed(16)}  |err| ${Math.abs(got-ref).toExponential(2)} <= ${(bound).toExponential(1)}`);
}

console.log('2. E(m) domain is enforced, not silently NaN');
for(const bad of [1, 1.5, -0.1, NaN, Infinity, '0.5']){
  let threw=false; try{ K.completeEllipticE(bad); }catch(e){ threw=true; }
  check(threw, `E(${String(bad)}) throws`);
}

console.log('3. AGM cost vs the 2048-interval Simpson it replaced');
function simpsonE(m){const n=2048,h=(PI/2)/n;let s=0;
 for(let i=0;i<=n;i++){const t=i*h,v=Math.sqrt(1-m*Math.sin(t)**2);
  s+=(i===0||i===n?1:i%2===0?2:4)*v;} return s*h/3;}
let t0=Date.now(); for(let i=0;i<200000;i++) K.completeEllipticE(0.5); const tAgm=Date.now()-t0;
t0=Date.now(); for(let i=0;i<200000;i++) simpsonE(0.5); const tSim=Date.now()-t0;
console.log(`  AGM ${tAgm} ms, Simpson ${tSim} ms over 200k calls -> ${(tSim/tAgm).toFixed(0)}x faster`);
check(tAgm<tSim, 'AGM is cheaper');
check(Math.abs(K.completeEllipticE(5/9)-simpsonE(5/9))<1e-13, 'AGM and Simpson agree to 1e-13');

console.log('4. Order independence (the index*0.0007 term is gone)');
const oldCoupling=(b,i)=>0.016+(1-b.minorRatio)*0.024+i*0.0007;
const oldProduct=(arr,arc)=>{let p=1;arr.forEach((b,i)=>{
  p*=1-oldCoupling(b,i)*Math.sin(2*PI*arc/b.perimeterRatio)**2;});return p;};
const newProduct=(arr,arc)=>{let p=1;for(const b of arr){
  p*=1-K.couplingFor(b)*Math.sin(2*PI*arc/b.perimeterRatio)**2;}return p;};
const fwd=K.BOUNDARY_EQUATIONS.slice(), rev=fwd.slice().reverse();
const dOld=Math.abs(oldProduct(fwd,1.7)-oldProduct(rev,1.7));
const dNew=Math.abs(newProduct(fwd,1.7)-newProduct(rev,1.7));
console.log(`  old formula changes by ${dOld.toExponential(3)} when reversed`);
console.log(`  new formula changes by ${dNew.toExponential(3)} when reversed`);
check(dOld>1e-9,'the old coupling really was order dependent');
// Exact bit-equality under reordering is NOT achievable: floating-point
// multiplication is not associative. Demonstrate that directly, so the bound
// below is a consequence of IEEE-754 and not a widened tolerance.
const assocA=(0.1*0.2)*0.3, assocB=0.1*(0.2*0.3);
check(assocA!==assocB,
  `float multiply is non-associative: (0.1*0.2)*0.3 = ${assocA} != 0.1*(0.2*0.3) = ${assocB}`);
check(dNew<=4*Number.EPSILON,
  `the new coupling is order independent to ${(dNew/Number.EPSILON).toFixed(1)} ulp `
  +`(old was ${(dOld/Number.EPSILON).toExponential(1)} ulp)`);

console.log('5. reinforcedR4 guards');
check(K.reinforcedR4(0)===1,'R4 at radius 0 is exactly 1');
for(const bad of [[1,[0,1]],[1,[-2]],[1,[]],[NaN,undefined]]){
  let threw=false; try{ K.reinforcedR4(bad[0],bad[1]); }catch(e){ threw=true; }
  check(threw,`reinforcedR4(${bad[0]}, ${JSON.stringify(bad[1])}) throws`);
}

console.log('6. boundaryTerms / product reject bad input');
for(const args of [[NaN,1],[1,NaN],['a',1],[1,Infinity]]){
  let threw=false; try{ K.factoredBoundaryProduct(args[0],args[1]); }catch(e){ threw=true; }
  check(threw,`factoredBoundaryProduct(${String(args[0])}, ${String(args[1])}) throws`);
}
const p=K.factoredBoundaryProduct(1.7,0.9);
check(p>0&&p<=1,`product in (0,1]: ${p.toFixed(9)}`);
check(Math.abs(K.factoredBoundaryImpedance(1.7,0.9)-(1-p))<1e-15,'impedance = 1 - product');

console.log('7. the verifier can actually fail (regeneration)');
const rec=K.BOUNDARY_EQUATIONS.find(b=>b.key==='thom-a');
const corrupted={...rec, perimeterRatio: rec.perimeterRatio*1.0001};
// re-run the verifier's own comparison against the corrupted value
const got=corrupted.evaluate();
check(Math.abs(got-corrupted.perimeterRatio)>1e-12,
  `a 0.01% drift in thom-a is detected: formula ${got.toFixed(9)} vs stored ${corrupted.perimeterRatio.toFixed(9)}`);
check(K.verify().filter(r=>!r.ok).map(r=>r.key).join(',')==='egg-i,egg-ii',
  'exactly egg-i and egg-ii fail, nothing else');

console.log('8. published values');
for(const b of K.BOUNDARY_EQUATIONS)
  console.log(`  ${b.key.padEnd(15)} P/D = ${b.perimeterRatio.toFixed(9)}  minor = ${b.minorRatio.toFixed(9)}  ${b.derived?'derived':'UNVERIFIED'}`);

console.log('9. the page embeds the real kernel, not a copy of it that has drifted');
// boundaries.html inlines boundarykernel.js verbatim so every figure on the page
// is computed by the shipped code. An inlined copy is exactly the kind of thing
// that rots silently, so require it byte-identical rather than merely disclose it.
{
  const fs = require('fs'), path = require('path');
  const pagePath = path.join(__dirname, 'boundaries.html');
  const src = fs.readFileSync(path.join(__dirname, 'boundarykernel.js'), 'utf8');
  if (!fs.existsSync(pagePath)) {
    check(false, 'boundaries.html is missing — the page cannot embed the kernel');
  } else {
    const page = fs.readFileSync(pagePath, 'utf8');
    const open = page.indexOf('\n<script>\n');
    const close = page.indexOf('\n</script>\n', open);
    check(open !== -1 && close !== -1, 'boundaries.html carries an inlined kernel block');
    const embedded = open === -1 ? '' : page.slice(open + 10, close + 1);
    check(embedded === src,
      embedded === src
        ? `embedded kernel is byte-identical to boundarykernel.js (${src.length} bytes)`
        : `embedded kernel has DRIFTED: ${embedded.length} bytes embedded vs ${src.length} on disk — rebuild the page`);
  }
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail?1:0);
