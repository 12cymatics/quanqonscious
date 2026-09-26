#!/usr/bin/env node
/* Rewrites the provenance fields of vedic_v18.51.1_exact_phi.html from what
   `lake build` actually produced.
   Inputs:  lean_provenance.json  (written by ProvenanceExport.lean)
            lean-toolchain, SutraWS/*.lean, .lake/build/lib/SutraWS/*.olean
   The page may then claim only what the compiler checked. */
import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join } from 'node:path';

const ROOT = process.cwd();
const HTML = join(ROOT, 'vedic_v18.51.1_exact_phi.html');
const JSONF = join(ROOT, 'lean_provenance.json');

for (const f of [HTML, JSONF, join(ROOT, 'lean-toolchain')])
  if (!existsSync(f)) { console.error('missing input: ' + f); process.exit(1); }

const env = JSON.parse(readFileSync(JSONF, 'utf8'));
const toolchain = readFileSync(join(ROOT, 'lean-toolchain'), 'utf8').trim();

/* corpus = the Lean sources that were compiled, hashed in a fixed order */
const leanDir = join(ROOT, 'SutraWS');
const sources = ['SutraWS.lean', ...readdirSync(leanDir).filter(f => f.endsWith('.lean')).sort().map(f => 'SutraWS/' + f)];
const corpusHash = createHash('sha256');
for (const rel of sources) corpusHash.update(rel + '\0' + readFileSync(join(ROOT, rel)));
const leanCorpusSha256 = corpusHash.digest('hex');

/* strip block comments and docstrings first: several of them discuss `sorry` */
let sorryCount = 0;
for (const rel of sources) {
  const body = readFileSync(join(ROOT, rel), 'utf8')
    .replace(/\/-[\s\S]*?-\//g, '')
    .split('\n').filter(l => !/^\s*--/.test(l)).join('\n');
  sorryCount += (body.match(/\bsorry\b/g) || []).length;
}

const oleanDir = join(ROOT, '.lake', 'build', 'lib', 'SutraWS');
const oleans = existsSync(oleanDir)
  ? readdirSync(oleanDir).filter(f => f.endsWith('.olean')).sort().map(f => 'SutraWS/' + f)
  : [];
if (existsSync(join(ROOT, '.lake', 'build', 'lib', 'SutraWS.olean'))) oleans.unshift('SutraWS.olean');

/* the page cites the final name component; it must stay unambiguous */
const shortOf = n => n.split('.').pop();
const shorts = new Map();
for (const n of env.kernelChecked) {
  const s = shortOf(n);
  if (!shorts.has(s)) shorts.set(s, []);
  shorts.get(s).push(n);
}
const ambiguous = [...shorts].filter(([, v]) => v.length > 1).map(([k]) => k);

let html = readFileSync(HTML, 'utf8');
const before = html;
const subst = [];
function put(re, next, label) {
  const hit = html.match(re);
  if (!hit) { console.error('pattern not found: ' + label); process.exit(1); }
  if (hit[0] !== next) subst.push(label);
  html = html.replace(re, () => next);
}

put(/leanCorpusSha256: '[^']*'/, `leanCorpusSha256: '${leanCorpusSha256}'`, 'leanCorpusSha256');
put(/leanToolchain: '[^']*'/, `leanToolchain: '${toolchain}'`, 'leanToolchain');
put(/leanTheorems: \d+/, `leanTheorems: ${env.kernelCheckedCount + env.compilerTrustedCount}`, 'leanTheorems');
put(/leanSorryCount: \d+/, `leanSorryCount: ${sorryCount}`, 'leanSorryCount');
put(/leanOleans: \[[^\]]*\]/, `leanOleans: [${oleans.map(o => `'${o}'`).join(',')}]`, 'leanOleans');
put(/leanAxiomsKernelChecked: \d+/, `leanAxiomsKernelChecked: ${env.kernelCheckedCount}`, 'leanAxiomsKernelChecked');
put(/leanAxiomsCompilerTrusted: \d+/, `leanAxiomsCompilerTrusted: ${env.compilerTrustedCount}`, 'leanAxiomsCompilerTrusted');
put(/leanAxiomsUsed: \[[^\]]*\]/,
    `leanAxiomsUsed: [${env.axiomsUsed.map(a => `'${a}'`).join(',')}]`, 'leanAxiomsUsed');

writeFileSync(HTML, html);

const cited = [...(before.match(/'[a-z][a-z0-9_]{5,}'/g) || [])].map(s => s.slice(1, -1));
const known = new Set(env.kernelChecked.concat(env.compilerTrusted).map(shortOf));

console.log(JSON.stringify({
  toolchain, leanCorpusSha256, sorryCount,
  oleans: oleans.length,
  kernelChecked: env.kernelCheckedCount,
  compilerTrusted: env.compilerTrustedCount,
  axiomsUsed: env.axiomsUsed,
  ambiguousShortNames: ambiguous,
  fieldsChanged: subst
}, null, 2));
