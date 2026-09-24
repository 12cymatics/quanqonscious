#!/usr/bin/env node
/* Every theorem name the page cites must exist in the environment `lake build`
   produced. Run after ProvenanceExport.lean has written lean_provenance.json.
   Exits non-zero naming any citation with no theorem behind it. */
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const ROOT = process.cwd();
const HTML = join(ROOT, 'vedic_v18.51.1_exact_phi.html');
const JSONF = join(ROOT, 'lean_provenance.json');
if (!existsSync(JSONF)) { console.error('lean_provenance.json missing — run the Lean export first'); process.exit(1); }

const env = JSON.parse(readFileSync(JSONF, 'utf8'));
const html = readFileSync(HTML, 'utf8');

/* The page cites the final name component. That is only sound while the
   component is unique: if two namespaces both end in `foo`, deleting the one
   the page means leaves the citation satisfied by the other. Ambiguity in a
   cited name is therefore a failure, not a note. */
const bySuffix = new Map();
for (const full of env.kernelChecked.concat(env.compilerTrusted)) {
  const s = full.split('.').pop();
  if (!bySuffix.has(s)) bySuffix.set(s, []);
  bySuffix.get(s).push(full);
}

const strings = src => [...(src.match(/'[^']*'/g) || [])].map(s => s.slice(1, -1));
const cites = new Map();
const add = (name, where) => {
  if (!cites.has(name)) cites.set(name, new Set());
  cites.get(name).add(where);
};

for (const key of ['kernelChecked', 'wheelerChecked', 'goldenChecked', 'modeChecked', 'compilerTrusted']) {
  const m = html.match(new RegExp(key + ': \\[([^\\]]*)\\]'));
  if (!m) { console.error('LEAN_PROVED.' + key + ' not found'); process.exit(1); }
  for (const n of strings(m[1])) add(n, 'LEAN_PROVED.' + key);
}

/* channel gating: literal arrays only; `theorems: LEAN_PROVED.x` is covered above */
const channelRe = /(\w+): \{\s*\n\s*draws: '[^']*',\s*\n\s*theorems: \[([^\]]*)\]/g;
let m;
while ((m = channelRe.exec(html)) !== null)
  for (const n of strings(m[2])) add(n, 'VISUAL_CERTIFICATE.' + m[1]);

const missing = [...cites].filter(([n]) => !bySuffix.has(n));
const ambiguous = [...cites].filter(([n]) => (bySuffix.get(n) || []).length > 1);
console.log(`citations: ${cites.size} distinct names, ${bySuffix.size} theorem suffixes in the built environment`);

if (missing.length) {
  console.error(`\nFAIL — ${missing.length} cited name(s) with no theorem behind them:`);
  for (const [n, where] of missing) console.error(`  ${n}  <- ${[...where].join(', ')}`);
}
if (ambiguous.length) {
  console.error(`\nFAIL — ${ambiguous.length} cited name(s) resolve to more than one theorem;`);
  console.error(`the citation cannot say which, so deleting the intended one would go unnoticed:`);
  for (const [n, where] of ambiguous)
    console.error(`  ${n}  <- ${[...where].join(', ')}\n      ${bySuffix.get(n).join('\n      ')}`);
}
if (missing.length || ambiguous.length) process.exit(1);
console.log('OK — every cited theorem exists, and each name resolves to exactly one');
