#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync, rmSync, copyFileSync, utimesSync, readdirSync, statSync } from 'node:fs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const NAME = 'QuanQonscious-Simulation-Suite';
const VERSION = '1.0.0';
const OUT = join(REPO, 'product', 'dist');
const DIR = join(OUT, NAME);
const ZIP = join(OUT, `${NAME}-${VERSION}.zip`);
const EPOCH = new Date(Date.UTC(2026, 10, 16));

const FILES = [
  ['product/START-HERE.html', 'START-HERE.html'],
  ['product/GUIDE.html', 'GUIDE.html'],
  ['product/LICENSE.txt', 'LICENSE.txt'],
  ['product/THIRD-PARTY-NOTICES.txt', 'THIRD-PARTY-NOTICES.txt'],
  ['vedic_v18.51.1_exact_phi.html', 'vedic-sutra-simulator.html'],
  ['faraday-cell/faraday-cell-standalone.html', 'faraday-cell.html'],
];

const sha = f => createHash('sha256').update(readFileSync(f)).digest('hex');

rmSync(OUT, { recursive: true, force: true });
mkdirSync(DIR, { recursive: true });
const lines = [];
for (const [from, to] of FILES) {
  copyFileSync(join(REPO, from), join(DIR, to));
  lines.push(`${sha(join(DIR, to))}  ${to}`);
}
writeFileSync(join(DIR, 'SHA256SUMS.txt'), lines.join('\n') + '\n');
for (const f of readdirSync(DIR)) utimesSync(join(DIR, f), EPOCH, EPOCH);
utimesSync(DIR, EPOCH, EPOCH);
execFileSync('zip', ['-X', '-q', '-r', '-9', ZIP, NAME], { cwd: OUT, env: { ...process.env, TZ: 'UTC' } });
const total = readdirSync(DIR).reduce((n, f) => n + statSync(join(DIR, f)).size, 0);
console.log(`wrote ${NAME}/  ${readdirSync(DIR).length} files, ${(total / 1024).toFixed(1)} KB`);
console.log(`wrote ${NAME}-${VERSION}.zip  ${(statSync(ZIP).size / 1024).toFixed(1)} KB  sha256 ${sha(ZIP)}`);
