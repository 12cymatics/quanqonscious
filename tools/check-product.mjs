#!/usr/bin/env node
import { existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { NAME, ZIP, FILES, productFile } from './product-files.mjs';

let pass = 0, fail = 0;
const ok = (c, label, detail = '') => { console.log(`  ${c ? 'ok  ' : 'FAIL'} ${label}${c || !detail ? '' : '  -- ' + detail}`); c ? pass++ : fail++; };

ok(existsSync(ZIP), 'the product ZIP is built');
const listed = execFileSync('unzip', ['-Z1', ZIP], { encoding: 'utf8' }).split('\n').filter(x => x && !x.endsWith('/')).sort();
const want = [...FILES.map(f => `${NAME}/${f[1]}`), `${NAME}/SHA256SUMS.txt`].sort();
ok(JSON.stringify(listed) === JSON.stringify(want), 'the ZIP holds exactly the seven product files', listed.join(' '));
const inZip = name => execFileSync('unzip', ['-p', ZIP, `${NAME}/${name}`], { maxBuffer: 1 << 28 });
for (const entry of FILES)
  ok(inZip(entry[1]).equals(productFile(entry)), `${entry[1]} is ${entry[0]}${entry[2] ? ' with the atlas taken out and nothing else changed' : ', byte for byte'}`);
const faraday = inZip('faraday-cell.html').toString('utf8');
const records = faraday.match(/"l":\[[^\]]*\],"h":\[/g) || [];
ok(faraday.includes('const ATLAS={};\n') && records.length === 0, 'the Faraday page carries no atlas record', `${records.length} records`);
ok(!/atlas: '(small|medium)'/.test(faraday), 'no cell of the Faraday page names an atlas');
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
const sums = inZip('SHA256SUMS.txt').toString('utf8').trim().split('\n');
ok(sums.length === FILES.length, 'SHA256SUMS.txt lists every other file');
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
