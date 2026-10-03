// Build faraday-cell.zip -- everything needed to run the simulation and its
// suites on a machine that has node and nothing else installed.
//
//     node faraday/build-zip.mjs [outfile]
//
// The zip is GENERATED and never committed. A committed zip is a second copy of
// every file in it, and the first time one side is edited the copy drifts; a
// drifted copy that still *runs* is the worst failure mode this repository has,
// because it returns an answer. It is rebuilt whenever something in MANIFEST
// changes, and faraday/check-zip.mjs builds one, unpacks it somewhere else and
// runs the suites from there.
//
// It carries two things that are not in the checkout:
//
//   faraday-cell-standalone.html   one file, opened by double-click, no server
//                                  and no siblings -- built by build-standalone.mjs
//   run-checks.mjs                 runs every suite in the unpack, in one command
//
// `zip` is used rather than a JavaScript implementation because the output has to
// be a zip any operating system's own unarchiver opens, and that is a format
// question, not an algorithm question. If `zip` is missing this REFUSES and says
// so, rather than writing a tar under a .zip name.

import { readFileSync, writeFileSync, mkdirSync, rmSync, existsSync, copyFileSync,
         chmodSync, statSync } from 'node:fs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { buildStandalone } from './build-standalone.mjs';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..');

/* The directory the zip unpacks into. Named, rather than loose at the top level,
   so an unpack into a folder that already holds something does not scatter 30
   files over it. */
export const TOP = 'faraday-cell';

/* Every file copied from the checkout, by repo-relative path. Explicit, not a
   glob: a glob over dns/ would have swept in the 94 KB plan and the scratch
   JSON of whatever was being measured that week, and -- worse -- a file DELETED
   from the checkout would leave the zip silently smaller with nothing to say so.
   check-zip.mjs asserts every path here resolves and that the suite list below
   covers every suite the checkout has. */
export const MANIFEST = [
  'cymatic.html',

  'faraday/kernel.js',
  'faraday/benchmark.js',
  'faraday/reference.json',
  'faraday/build-standalone.mjs',
  'faraday/build-zip.mjs',
  'faraday/check-kernel.mjs',
  'faraday/check-standalone.mjs',
  'faraday/check-zip.mjs',
  'faraday/zip-README.md',

  'dns/faraday-dns.js',
  'dns/faraday-disc.js',
  'dns/faraday-disc-wasm.js',
  'dns/faraday-floquet.js',
  'dns/faraday-cell3d.js',
  'dns/faraday-cell3d-wasm.js',
  'dns/faraday_disc.cpp',
  'dns/faraday_disc.wasm',
  'dns/faraday_cell3d.cpp',
  'dns/faraday_cell3d.wasm',
  'dns/build-wasm.sh',
  'dns/build-wasm-cell3d.sh',
  'dns/run-cell3d.mjs',
  'dns/check-dns.mjs',
  'dns/check-disc.mjs',
  'dns/check-cell3d.mjs',
  'dns/check-wasm-build.mjs',
  'dns/check-cell3d-wasm.mjs',
  'dns/check-page.mjs',
  'dns/hessenberg-defective-24.json',
  'dns/PLAN-cell3d.md',

  'boundary/boundarykernel.js',
  'boundary/boundarykernel.test.js',
  'boundary/boundaries.html',

  'fsi/coupled-affine-benchmark.js',
  'fsi/check-coupled-affine.mjs'
];

/* The suites run-checks.mjs runs, in order, with the command that runs each.
   `browser` marks the one that needs Chrome: it REFUSES without one rather than
   skipping, which is correct, so run-checks says up front which suite that is
   instead of leaving the reader to read a stack trace. */
export const SUITES = [
  { name: 'faraday/check-kernel.mjs',       file: 'faraday/check-kernel.mjs' },
  { name: 'faraday/check-standalone.mjs',   file: 'faraday/check-standalone.mjs' },
  { name: 'dns/check-dns.mjs',              file: 'dns/check-dns.mjs' },
  { name: 'dns/check-disc.mjs',             file: 'dns/check-disc.mjs' },
  { name: 'dns/check-wasm-build.mjs',       file: 'dns/check-wasm-build.mjs' },
  { name: 'dns/check-cell3d.mjs',           file: 'dns/check-cell3d.mjs', slow: true },
  { name: 'dns/check-cell3d-wasm.mjs',      file: 'dns/check-cell3d-wasm.mjs', slow: true },
  { name: 'boundary/boundarykernel.test.js', file: 'boundary/boundarykernel.test.js' },
  { name: 'fsi/check-coupled-affine.mjs',   file: 'fsi/check-coupled-affine.mjs' },
  { name: 'dns/check-page.mjs',             file: 'dns/check-page.mjs', browser: true }
];

/* run-checks.mjs is generated from SUITES rather than written out, so a suite
   added to SUITES cannot be left out of the runner. */
export function runChecksSource(){
  const rows = SUITES.map(s => '  ' + JSON.stringify(
    { name: s.name, file: s.file, slow: !!s.slow, browser: !!s.browser })).join(',\n');
  return `#!/usr/bin/env node
/* Run every suite in this unpack, in one command.
 *
 *     node run-checks.mjs            every suite
 *     node run-checks.mjs --fast     all but the slow ones, named below
 *     node run-checks.mjs --list     what would run, and nothing else
 *
 * GENERATED by faraday/build-zip.mjs from its SUITES list. Editing this file in
 * place is pointless: the next build overwrites it. Add the suite there.
 *
 * Nothing here skips. A suite that cannot run fails, and the exit code is
 * non-zero, because a run that quietly omitted a suite and said "all passed"
 * would be worse than no runner at all.
 */
import { spawnSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const SUITES = [
${rows}
];

const argv = process.argv.slice(2);
for (const a of argv)
  if (a !== '--fast' && a !== '--list'){
    console.error(\`unknown option \${a}. Known: --fast, --list.\`);
    process.exit(2);
  }
const fast = argv.includes('--fast');
const list = argv.includes('--list');
const run = SUITES.filter(s => !(fast && s.slow));

if (list){
  for (const s of run)
    console.log(\`\${s.name}\${s.slow ? '   (slow)' : ''}\${s.browser ? '   (needs Chrome)' : ''}\`);
  process.exit(0);
}

const browserSuites = run.filter(s => s.browser).map(s => s.name);
if (browserSuites.length)
  console.log(\`note: \${browserSuites.join(', ')} needs a Chrome or Chromium binary. \`
    + \`It refuses rather than skipping; set CHROME=/path/to/chrome if it is not found.\\n\`);

const results = [];
for (const s of run){
  process.stdout.write(\`=== \${s.name}\${s.slow ? '   (slow -- minutes)' : ''}\\n\`);
  const t0 = Date.now();
  const r = spawnSync(process.execPath, [join(HERE, s.file)],
                      { stdio: ['ignore', 'pipe', 'pipe'], encoding: 'utf8' });
  const secs = (Date.now() - t0)/1000;
  const out = (r.stdout || '') + (r.stderr || '');
  const tail = out.trimEnd().split('\\n').slice(-3).join('\\n');
  const okRun = r.status === 0;
  console.log(tail);
  console.log(\`--- \${s.name}: \${okRun ? 'PASS' : 'FAIL'} in \${secs.toFixed(1)} s\\n\`);
  if (!okRun && !tail) console.log(out);
  results.push({ name: s.name, ok: okRun, secs });
}

console.log('summary');
for (const r of results)
  console.log(\`  \${r.ok ? 'PASS' : 'FAIL'}  \${r.name}  \${r.secs.toFixed(1)} s\`);
const bad = results.filter(r => !r.ok);
if (bad.length){
  console.log(\`\\n\${bad.length} of \${results.length} suites FAILED.\`);
  process.exit(1);
}
console.log(\`\\nall \${results.length} suites passed.\`);
`;
}

export function stage(dir){
  const top = join(dir, TOP);
  for (const p of MANIFEST){
    const src = resolve(REPO, p);
    if (!existsSync(src)) throw new Error(
      `MANIFEST names ${p}, which does not exist in the checkout. Either the file `
      + `was renamed or deleted and MANIFEST was not updated -- do not let this zip `
      + `ship silently smaller than it says it is.`);
    const dst = join(top, p);
    mkdirSync(dirname(dst), { recursive: true });
    copyFileSync(src, dst);
  }

  writeFileSync(join(top, 'README.md'), readFileSync(resolve(REPO, 'faraday/zip-README.md')));
  writeFileSync(join(top, 'faraday-cell-standalone.html'), buildStandalone());
  writeFileSync(join(top, 'run-checks.mjs'), runChecksSource());
  chmodSync(join(top, 'run-checks.mjs'), 0o755);
  chmodSync(join(top, 'dns', 'build-wasm.sh'), 0o755);
  chmodSync(join(top, 'dns', 'run-cell3d.mjs'), 0o755);
  return top;
}

export function buildZip(outfile){
  let zipVersion;
  try { zipVersion = execFileSync('zip', ['-v'], { encoding: 'utf8' }).split('\n')[1] || 'zip'; }
  catch { throw new Error(
    `the \`zip\` command is not available, and this will not write a tar under a `
    + `.zip name. Install zip (apt-get install zip / brew install zip) and run again.`); }

  const work = mkdtempSync(join(tmpdir(), 'faraday-zip-'));
  try {
    stage(work);
    if (existsSync(outfile)) rmSync(outfile);
    /* -X drops the extra attribute blocks, -r recurses, and the zip is built from
       inside the work directory so the paths inside it start at faraday-cell/. */
    execFileSync('zip', ['-rqX', outfile, TOP], { cwd: work });
  } finally { rmSync(work, { recursive: true, force: true }); }
  return { outfile, bytes: statSync(outfile).size, zipVersion: zipVersion.trim() };
}

if (import.meta.url === `file://${process.argv[1]}`){
  const out = resolve(process.argv[2] ?? join(REPO, 'faraday-cell.zip'));
  const r = buildZip(out);
  console.log(`wrote ${r.outfile}  ${(r.bytes/1024).toFixed(1)} KB`);
  console.log(`      ${MANIFEST.length} files from the checkout, plus README.md, `
              + `faraday-cell-standalone.html and run-checks.mjs`);
}
