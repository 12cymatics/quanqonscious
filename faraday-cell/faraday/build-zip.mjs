// Build the faraday-cell bundle -- everything needed to run the simulation and
// its suites on a machine that has node and nothing else installed.
//
//     node faraday/build-zip.mjs            the COMMITTED pair, at the repo root:
//                                           faraday-cell/ and faraday-cell.zip
//     node faraday/build-zip.mjs <outfile>  just a zip, written to <outfile>
//
// THE BUNDLE IS COMMITTED, AND THAT WAS NOT THE ORIGINAL DESIGN. It started out
// generated and gitignored, on the grounds that a committed copy of every file is
// a copy that drifts the first time one side is edited -- and a drifted copy that
// still *runs* is the worst failure mode this repository has, because it returns
// an answer. The owner asked for the files together in the repository so they can
// be downloaded from it, and that is a reasonable thing to want: GitHub offers no
// way to download a folder, and the whole repository is 306 MB of generated
// images around 2 MB of simulation.
//
// So the copy is committed and the drift is made impossible to miss instead,
// which is the arrangement this repository already uses for its committed
// WebAssembly modules. faraday/check-bundle.mjs stages a fresh bundle from the
// checkout and requires the committed folder to be byte for byte identical to it,
// file for file with nothing extra and nothing missing, and the committed zip to
// be byte for byte the zip this file would write now. Edit any source without
// running this and CI goes red, naming the file.
//
// THE ZIP IS REPRODUCIBLE, because a committed zip that changed on every rebuild
// would show as modified whenever anyone ran this, and a gate could only compare
// its contents. Every file is given the same modification time and an explicit
// mode, the entries are written in sorted order from a list rather than in
// whatever order the filesystem walks, directory entries and extra attribute
// fields are left out, and the timestamps are taken in UTC -- zip stores DOS times
// in LOCAL time, so the same file zipped in two time zones differs otherwise.
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
         chmodSync, statSync, readdirSync, utimesSync, renameSync } from 'node:fs';
import { dirname, resolve, join, relative } from 'node:path';
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

/* Where the committed pair lives: the folder and the zip, side by side at the
   repository root, so the zip unpacks to a folder of the same name as the one
   beside it. */
export const BUNDLE_DIR = join(REPO, TOP);
export const BUNDLE_ZIP = join(REPO, `${TOP}.zip`);

/* The files that are run directly and so carry the executable bit. Every other
   file is 0644. Set explicitly rather than inherited from the checkout, because
   an inherited mode is whatever the machine that copied it happened to have, and
   git records the executable bit -- a bundle staged on two machines would
   otherwise differ in the one property nobody reads. */
export const EXECUTABLE = new Set([
  'run-checks.mjs', 'dns/build-wasm.sh', 'dns/build-wasm-cell3d.sh', 'dns/run-cell3d.mjs'
]);

/* The modification time every file in the zip carries: 2026-01-01T00:00:00Z.
   Arbitrary, and fixed, which is the whole of its job. */
export const ZIP_EPOCH = Date.UTC(2026, 0, 1)/1000;

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
  'faraday/check-bundle.mjs',
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
  for (const p of bundleFiles(top))
    chmodSync(join(top, p), EXECUTABLE.has(p) ? 0o755 : 0o644);
  return top;
}

/* Every file under a bundle directory, as paths relative to it, SORTED -- the
   order the zip is written in and the order the gate compares in. */
export function bundleFiles(top){
  const out = [];
  const walk = d => {
    for (const f of readdirSync(d, { withFileTypes: true })){
      const p = join(d, f.name);
      if (f.isDirectory()) walk(p); else out.push(relative(top, p));
    }
  };
  walk(top);
  return out.sort();
}

export function buildZip(outfile){
  let zipVersion;
  try { zipVersion = execFileSync('zip', ['-v'], { encoding: 'utf8' }).split('\n')[1] || 'zip'; }
  catch { throw new Error(
    `the \`zip\` command is not available, and this will not write a tar under a `
    + `.zip name. Install zip (apt-get install zip / brew install zip) and run again.`); }

  const work = mkdtempSync(join(tmpdir(), 'faraday-zip-'));
  try {
    const top = stage(work);
    const files = bundleFiles(top);
    for (const p of files) utimesSync(join(top, p), ZIP_EPOCH, ZIP_EPOCH);
    if (existsSync(outfile)) rmSync(outfile);
    /* -X drops the extra attribute blocks, which carry a second, UTC timestamp and
       the owner's uid; -D writes no directory entries, whose times are whatever
       mkdir left; -@ takes the entries from the sorted list on stdin rather than
       from a directory walk. Built from inside the work directory so the paths in
       the zip start at faraday-cell/. TZ=UTC because zip stores DOS times in local
       time. */
    execFileSync('zip', ['-qXD', outfile, '-@'], {
      cwd: work,
      input: files.map(p => `${TOP}/${p}`).join('\n') + '\n',
      env: { ...process.env, TZ: 'UTC' }
    });
  } finally { rmSync(work, { recursive: true, force: true }); }
  return { outfile, bytes: statSync(outfile).size, zipVersion: zipVersion.trim() };
}

/* The committed pair: faraday-cell/ and faraday-cell.zip at the repository root.
   The folder is staged in a sibling temporary directory and moved into place, so
   a failure halfway leaves the old folder whole rather than half deleted -- and
   it is REPLACED rather than overwritten, so a file dropped from MANIFEST
   disappears from the committed folder too instead of lingering there. */
export function writeBundle(){
  if (TOP !== 'faraday-cell' || dirname(BUNDLE_DIR) !== REPO) throw new Error(
    `refusing to replace ${BUNDLE_DIR}: the bundle directory must be faraday-cell `
    + `directly under the repository root, and this would delete whatever is there.`);
  const work = mkdtempSync(join(REPO, '.faraday-cell-staging-'));
  try {
    const top = stage(work);
    rmSync(BUNDLE_DIR, { recursive: true, force: true });
    renameSync(top, BUNDLE_DIR);
  } finally { rmSync(work, { recursive: true, force: true }); }
  const z = buildZip(BUNDLE_ZIP);
  return { dir: BUNDLE_DIR, files: bundleFiles(BUNDLE_DIR).length, zip: z };
}

if (import.meta.url === `file://${process.argv[1]}`){
  if (process.argv[2]){
    const r = buildZip(resolve(process.argv[2]));
    console.log(`wrote ${r.outfile}  ${(r.bytes/1024).toFixed(1)} KB`);
  } else {
    const r = writeBundle();
    console.log(`wrote ${relative(REPO, r.dir)}/  ${r.files} files`);
    console.log(`wrote ${relative(REPO, r.zip.outfile)}  ${(r.zip.bytes/1024).toFixed(1)} KB`);
  }
  console.log(`      ${MANIFEST.length} files from the checkout, plus README.md, `
              + `faraday-cell-standalone.html and run-checks.mjs`);
}
