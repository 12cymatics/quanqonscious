// Render-path smoke test: drive the animation loop and the UI handlers,
// failing on any console.error the render try/catch would otherwise swallow.
import fs from 'fs';
const html = fs.readFileSync(new URL('../R4_tesseract_cymatic_v4.html', import.meta.url),'utf8');
let src = html.match(/<script>\n([\s\S]*?)<\/script>/)[1].replace(/^'use strict';/,'');

const errors = [];
const origErr = console.error;
console.error = (...a) => errors.push(a.join(' '));

const els = new Map();
const mkEl = id => { const e = {
  id, innerHTML:'', textContent:'', className:'', style:{}, children:[], disabled:false,
  appendChild(c){this.children.push(c);}, removeChild(){this.children.shift();},
  addEventListener(t,f){ (this._h ||= {})[t]=f; }, classList:{add(){},remove(){}},
  scrollTop:0, scrollHeight:0, getBoundingClientRect:()=>({left:0,top:0}),
  getContext:()=>ctxStub, width:0, height:0 }; els.set(id,e); return e; };
const ctxStub = new Proxy({}, { get:(t,p)=>{
  if (p==='createRadialGradient'||p==='createLinearGradient') return ()=>({addColorStop(){}});
  if (p==='measureText') return ()=>({width:0});
  return typeof t[p]==='undefined' ? ()=>{} : t[p]; }, set:()=>true });
globalThis.document = { getElementById:id=>els.get(id)||mkEl(id), querySelectorAll:()=>[], createElement:()=>mkEl('tmp') };
globalThis.window = { addEventListener(){} };
globalThis.innerWidth=1600; globalThis.innerHeight=900;
let rafFn=null; globalThis.requestAnimationFrame = f => { rafFn=f; return 1; };
globalThis.setTimeout = () => 0;
globalThis.performance = performance;

src += `\nglobalThis.__R = { render, QUEUE, Compose, St, Q, CY, TS, TO, cubes, injectCymatic, verifyInverses };`;
new Function(src)();
const X = globalThis.__R;

let pass=0, fail=0;
const ok=(c,m,x='')=>{ c?(pass++,console.log(`  ✓ ${m}`)):(fail++,console.log(`  ✗ ${m} ${x}`)); };

console.log('\n── RENDER LOOP ─────────────────────────────────────────────');
ok(errors.length===0, 'init() completed with no console.error', errors.join(' | '));
ok(typeof rafFn==='function', 'requestAnimationFrame handed a callback');

for (let f=1; f<=180; f++) X.render(f*16.7);
ok(errors.length===0, `180 frames rendered clean`, errors.slice(0,2).join(' | '));

console.log('\n── FRAME WORK ACTUALLY HAPPENS ─────────────────────────────');
ok(X.TS.edges.length===32, `tesseract has 32 edges (${X.TS.edges.length})`);
ok(X.TS.proj.every(p=>p.every(Number.isFinite)), 'all 16 projected vertices finite');
ok(X.TO.pts.length===(X.TO.rings+1)*(X.TO.seg+1), `torus table built (${X.TO.pts.length} pts)`);
ok(X.TO.pts.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)&&Number.isFinite(p.z)), 'torus points finite');
ok(X.cubes.length===29, `29 sūtra cubes (${X.cubes.length})`);
ok(X.CY.field && X.CY.field.some(v=>v!==0), 'cymatic field is non-trivial');
ok(X.CY.field.every(Number.isFinite), 'cymatic field all finite (no NaN/Inf)');

console.log('\n── TESSERACT RIGIDITY (no drift over long runs) ────────────');
{ // edge lengths in the 4-D base must be identical at frame 1 and frame 100000
  const len = () => { X.TS.update(1); const a=X.TS.proj.map(p=>p.slice());
    X.TS.update(100000); return a; };
  const before = X.TS.proj.map(p=>p.slice());
  X.TS.update(1); const f1 = X.TS.proj.map(p=>p.slice());
  for (let f=2; f<50000; f+=997) X.TS.update(f);
  X.TS.update(1); const f1b = X.TS.proj.map(p=>p.slice());
  const same = f1.every((p,i)=>p.every((v,k)=>Math.abs(v-f1b[i][k])<1e-12));
  ok(same, 'frame 1 reproduces bit-identically after 50k intervening frames');
}
{ // the base is never mutated
  const b = X.TS.base;
  ok(b.every(v=>v.every(c=>c===1||c===-1)), 'base vertices remain exactly ±1 (never mutated)');
}

console.log('\n── UI PATH ─────────────────────────────────────────────────');
X.cubes[0].tap(); X.cubes[6].tap(); X.cubes[12].tap();
ok(X.QUEUE.items.length===3, `tapping cubes enqueues (${X.QUEUE.items})`);
for (let m=0;m<6;m++){ X.QUEUE.mode=m; X.QUEUE.ui(); }
ok(errors.length===0, 'queue UI renders for all 6 modes', errors.join(' | '));
X.QUEUE.clear();
ok(X.QUEUE.items.length===0, 'clear empties the queue');
X.verifyInverses();
ok(errors.length===0, 'verify gate runs clean', errors.join(' | '));

console.log('\n── PERFORMANCE BUDGET ──────────────────────────────────────');
{ const t0=performance.now(); for(let f=1;f<=300;f++) X.render(f*16.7);
  const per=(performance.now()-t0)/300;
  ok(per < 16.7, `mean frame ${per.toFixed(2)}ms < 16.7ms (60fps budget)`); }
{ const t0=performance.now(); X.CY.compute(500);
  ok(performance.now()-t0 < 8, `cymatic field recompute ${(performance.now()-t0).toFixed(2)}ms`); }

console.error = origErr;
console.log(`\n${'═'.repeat(64)}\n  ${pass} passed, ${fail} failed\n${'═'.repeat(64)}`);
process.exit(fail?1:0);
