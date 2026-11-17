#!/usr/bin/env node
import { readFileSync, existsSync } from 'node:fs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const NAME = 'QuanQonscious-Simulation-Suite';
const ZIP = join(REPO, 'product', 'dist', `${NAME}-1.0.0.zip`);
const FILES = [
  ['product/START-HERE.html', 'START-HERE.html'],
  ['product/GUIDE.html', 'GUIDE.html'],
  ['product/LICENSE.txt', 'LICENSE.txt'],
  ['product/THIRD-PARTY-NOTICES.txt', 'THIRD-PARTY-NOTICES.txt'],
  ['vedic_v18.51.1_exact_phi.html', 'vedic-sutra-simulator.html'],
  ['faraday-cell/faraday-cell-standalone.html', 'faraday-cell.html'],
];
let pass = 0, fail = 0;
const ok = (c, label, detail = '') => { console.log(`  ${c ? 'ok  ' : 'FAIL'} ${label}${c || !detail ? '' : '  -- ' + detail}`); c ? pass++ : fail++; };

ok(existsSync(ZIP), 'the product ZIP is built');
const listed = execFileSync('unzip', ['-Z1', ZIP], { encoding: 'utf8' }).split('\n').filter(x => x && !x.endsWith('/')).sort();
const want = [...FILES.map(f => `${NAME}/${f[1]}`), `${NAME}/SHA256SUMS.txt`].sort();
ok(JSON.stringify(listed) === JSON.stringify(want), 'the ZIP holds exactly the seven product files', listed.join(' '));
const inZip = name => execFileSync('unzip', ['-p', ZIP, `${NAME}/${name}`], { maxBuffer: 1 << 28 });
for (const [from, to] of FILES) ok(inZip(to).equals(readFileSync(join(REPO, from))), `${to} is byte for byte ${from}`);
for (const to of FILES.map(f => f[1]).filter(t => t.endsWith('.html'))) {
  const text = inZip(to).toString('utf8');
  const outside = text.match(/(?:src|href)\s*=\s*["']https?:\/\/[^"']+|url\(\s*["']?https?:\/\/[^)]+|@import\s+["']https?:\/\/[^"']+/g) || [];
  ok(outside.length === 0, `${to} loads nothing from the network`, outside.slice(0, 3).join(' '));
}
const start = inZip('START-HERE.html').toString('utf8');
const links = [...start.matchAll(/href="([^"]+)"/g)].map(m => m[1]);
ok(links.length === 5 && links.every(l => want.includes(`${NAME}/${l}`)), 'every link on the start page is a file in the ZIP', links.join(' '));
const guide = inZip('GUIDE.html').toString('utf8');
ok([...guide.matchAll(/href="([^"]+)"/g)].every(m => want.includes(`${NAME}/${m[1]}`)), 'every link in the guide is a file in the ZIP');
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
