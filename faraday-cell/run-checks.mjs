#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const SUITES = [
  {"name":"faraday/check-startup.mjs","file":"faraday/check-startup.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"faraday/check-playback.mjs","file":"faraday/check-playback.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"faraday/check-unified-playback.mjs","file":"faraday/check-unified-playback.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"faraday/check-scene3d.mjs","file":"faraday/check-scene3d.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"faraday/check-kernel.mjs","file":"faraday/check-kernel.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"faraday/check-general.mjs","file":"faraday/check-general.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"dns/check-surface-pressure.mjs","file":"dns/check-surface-pressure.mjs","slow":false,"browser":false,"interpreter":null,"args":["--audit-free"]},
  {"name":"dns/check-cell3d-observation.mjs","file":"dns/check-cell3d-observation.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"dns/check-cell3d-tracers.mjs","file":"dns/check-cell3d-tracers.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"dns/check-book-pressure.py","file":"dns/check-book-pressure.py","slow":false,"browser":false,"interpreter":"python3","args":[]},
  {"name":"faraday/check-standalone.mjs","file":"faraday/check-standalone.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"faraday/check-clean.mjs","file":"faraday/check-clean.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"faraday/check-runner.mjs","file":"faraday/check-runner.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"dns/check-dns.mjs","file":"dns/check-dns.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"dns/check-disc.mjs","file":"dns/check-disc.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"dns/check-floquet-mode.mjs","file":"dns/check-floquet-mode.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"dns/check-grains.mjs","file":"dns/check-grains.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"dns/check-dyadic128.mjs","file":"dns/check-dyadic128.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"dns/check-dyadic128-wasm.mjs","file":"dns/check-dyadic128-wasm.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"dns/check-disc-d128.mjs","file":"dns/check-disc-d128.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"dns/check-floquet-d128.mjs","file":"dns/check-floquet-d128.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"dns/check-grains-d128.mjs","file":"dns/check-grains-d128.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"dns/check-wasm-build.mjs","file":"dns/check-wasm-build.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"dns/check-cell3d.mjs","file":"dns/check-cell3d.mjs","slow":true,"browser":false,"interpreter":null,"args":[]},
  {"name":"dns/check-cell3d-wasm.mjs","file":"dns/check-cell3d-wasm.mjs","slow":true,"browser":false,"interpreter":null,"args":[]},
  {"name":"dns/check-cell3d-pool.mjs","file":"dns/check-cell3d-pool.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"dns/check-cell3d-pool-integrity.mjs","file":"dns/check-cell3d-pool-integrity.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"dns/check-cell3d-diagnostics.mjs","file":"dns/check-cell3d-diagnostics.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"boundary/boundarykernel.test.js","file":"boundary/boundarykernel.test.js","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"fsi/check-coupled-affine.mjs","file":"fsi/check-coupled-affine.mjs","slow":false,"browser":false,"interpreter":null,"args":[]},
  {"name":"dns/check-page.mjs","file":"dns/check-page.mjs","slow":false,"browser":true,"interpreter":null,"args":[]}
];

const argv = process.argv.slice(2);
for (const a of argv)
  if (a !== '--fast' && a !== '--list'){
    console.error(`unknown option ${a}. Known: --fast, --list.`);
    process.exit(2);
  }
const fast = argv.includes('--fast');
const list = argv.includes('--list');
const run = SUITES.filter(s => !(fast && s.slow));
const omitted = SUITES.filter(s => !run.includes(s));

if (list){
  console.error('Suite inventory only; no checks executed.');
  for (const s of run)
    console.log(`${s.name}${s.slow ? '   (slow)' : ''}${s.browser ? '   (needs Chrome)' : ''}`);
  if (omitted.length) console.error('Not selected: ' + omitted.map(s => s.name).join(', '));
  process.exit(0);
}
if (omitted.length)
  console.error('INCOMPLETE selection: ' + omitted.map(s => s.name).join(', ')
    + '. A partial execution cannot pass full verification.');

const browserSuites = run.filter(s => s.browser).map(s => s.name);
if (browserSuites.length)
  console.log(`note: ${browserSuites.join(', ')} needs a Chrome or Chromium binary. `
    + `It refuses rather than skipping; set CHROME=/path/to/chrome if it is not found.\n`);

const results = [];
for (const s of run){
  process.stdout.write(`=== ${s.name}${s.slow ? '   (slow -- minutes)' : ''}\n`);
  const t0 = Date.now();
  const r = spawnSync(s.interpreter || process.execPath, [join(HERE, s.file), ...s.args],
                      { cwd: HERE, stdio: ['ignore', 'pipe', 'pipe'], encoding: 'utf8',
                        maxBuffer: 256*1024*1024 });
  const secs = (Date.now() - t0)/1000;
  const out = (r.stdout || '') + (r.stderr || '');
  const okRun = r.status === 0 && !r.error && !r.signal;
  if (okRun) console.log(out.trimEnd().split('\n').slice(-3).join('\n'));
  else console.log(out.trimEnd() + '\nProcess result: '
    + (r.error ? r.error.message : r.signal ? 'signal ' + r.signal : 'exit ' + r.status));
  console.log(`--- ${s.name}: ${okRun ? 'PASS' : 'FAIL'} in ${secs.toFixed(1)} s\n`);
  results.push({ name: s.name, ok: okRun, secs });
}

console.log('summary');
for (const r of results)
  console.log(`  ${r.ok ? 'PASS' : 'FAIL'}  ${r.name}  ${r.secs.toFixed(1)} s`);
const bad = results.filter(r => !r.ok);
if (bad.length){
  console.log(`\n${bad.length} of ${results.length} suites FAILED.`);
  process.exit(1);
}
if (omitted.length){
  console.error(`\n${results.length} selected suites passed; ${omitted.length} required suites NOT RUN.\n`
    + 'INCOMPLETE: run without --fast for full verification.');
  process.exit(2);
}
console.log(`\nAll ${results.length} listed suites executed and passed.`);
