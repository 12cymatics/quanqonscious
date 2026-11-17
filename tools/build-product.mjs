#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync, rmSync, utimesSync, readdirSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { NAME, VERSION, ZIP, FILES, productFile } from './product-files.mjs';

const OUT = dirname(ZIP);
const DIR = join(OUT, NAME);
const EPOCH = new Date(Date.UTC(2026, 10, 16));
const sha = b => createHash('sha256').update(b).digest('hex');

rmSync(OUT, { recursive: true, force: true });
mkdirSync(DIR, { recursive: true });
const lines = [];
for (const entry of FILES) {
  const bytes = productFile(entry);
  writeFileSync(join(DIR, entry[1]), bytes);
  lines.push(`${sha(bytes)}  ${entry[1]}`);
}
writeFileSync(join(DIR, 'SHA256SUMS.txt'), lines.join('\n') + '\n');
for (const f of readdirSync(DIR)) utimesSync(join(DIR, f), EPOCH, EPOCH);
utimesSync(DIR, EPOCH, EPOCH);
execFileSync('zip', ['-X', '-q', '-r', '-9', ZIP, NAME], { cwd: OUT, env: { ...process.env, TZ: 'UTC' } });
const total = readdirSync(DIR).reduce((n, f) => n + statSync(join(DIR, f)).size, 0);
console.log(`wrote ${NAME}/  ${readdirSync(DIR).length} files, ${(total / 1024).toFixed(1)} KB`);
console.log(`wrote ${NAME}-${VERSION}.zip  ${(statSync(ZIP).size / 1024).toFixed(1)} KB  sha256 ${sha(readFileSync(ZIP))}`);
