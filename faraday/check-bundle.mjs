#!/usr/bin/env node
/* Gate for the COMMITTED bundle: faraday-cell/ and faraday-cell.zip at the
 * repository root.
 *
 *     node faraday/check-bundle.mjs
 *
 * They are committed so that the simulation can be downloaded from the repository
 * as one folder, or one file -- GitHub has no way to download a folder, and the
 * whole repository is 306 MB of generated images around 2 MB of simulation. A
 * committed copy of every source file is a copy that drifts the first time one
 * side is edited, and a drifted copy that still RUNS is the worst failure mode
 * this repository has, because it returns an answer. This gate is what makes that
 * drift impossible to miss rather than unlikely: it stages a fresh bundle from the
 * checkout and requires the committed one to be byte for byte the same.
 *
 * The remedy for every failure here is the same, and it is printed:
 *
 *     node faraday/build-zip.mjs        then commit faraday-cell/ and faraday-cell.zip
 *
 * It is the arrangement this repository already uses for its committed
 * WebAssembly modules, and for the same reason: a committed build output is
 * acceptable exactly when something fails the moment it stops matching its source.
 */

import { readFileSync, existsSync, statSync, mkdtempSync, rmSync } from 'node:fs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { stage, buildZip, bundleFiles, TOP, BUNDLE_DIR, BUNDLE_ZIP, EXECUTABLE,
         MANIFEST } from './build-zip.mjs';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const REMEDY = 'run `node faraday/build-zip.mjs` and commit faraday-cell/ and '
             + 'faraday-cell.zip';

let pass = 0; const failures = [];
function ok(cond, label, detail){
  if (cond){ pass++; console.log('  ok   ' + label); }
  else { failures.push(`${label}${detail ? '  -- ' + detail : ''}`);
         console.log('  FAIL ' + label + (detail ? '  -- ' + detail : '')); }
}
const section = t => console.log('\n' + t);
const git = (...a) => execFileSync('git', ['-C', REPO, ...a], { encoding: 'utf8' });

/* Run from an unpack of the bundle, there is no committed copy to check -- the
   unpack IS a copy. Saying so and failing is right; reporting a missing bundle
   would be wrong, and passing would be a pass that checked nothing. */
if (!existsSync(join(REPO, '.git')) && existsSync(join(REPO, 'run-checks.mjs'))
    && !existsSync(BUNDLE_DIR)){
  console.log('refused: this is an unpack of the faraday-cell bundle, not a checkout '
    + 'of the repository. This gate checks the REPOSITORY\'S committed copy of the '
    + 'bundle against its sources, and an unpack has neither. Run it from a checkout.');
  process.exit(1);
}

const work = mkdtempSync(join(tmpdir(), 'faraday-bundle-'));
try {

/* ---- 1. both halves are committed ------------------------------------- */
section('1. both halves are committed, and nothing in them is ignored');
const tracked = new Set(git('ls-files', '--', TOP, `${TOP}.zip`).split('\n').filter(Boolean));
ok(existsSync(BUNDLE_DIR) && statSync(BUNDLE_DIR).isDirectory(),
   'faraday-cell/ exists at the repository root', REMEDY);
ok(tracked.has(`${TOP}.zip`), 'faraday-cell.zip is tracked by git',
   `it is ${existsSync(BUNDLE_ZIP) ? 'on disk but not tracked' : 'missing'}; ${REMEDY}`);
const onDisk = existsSync(BUNDLE_DIR) ? bundleFiles(BUNDLE_DIR) : [];
const untracked = onDisk.filter(p => !tracked.has(`${TOP}/${p}`));
ok(untracked.length === 0 && onDisk.length > 0,
   `every one of the ${onDisk.length} files in faraday-cell/ is tracked`,
   untracked.length ? `untracked: ${untracked.slice(0, 6).join(', ')}; ${REMEDY}`
                    : 'the folder is empty');
/* An ignore rule written for the generated file at the root -- the single-file
   page used to be ignored by bare name, which matches at any depth -- would
   silently keep the bundle's copy out of every commit. git add skips an ignored
   path without a word, so this is checked rather than trusted.
   .
   WITH --no-index, and that is not optional: plain `git check-ignore` does not
   report a file that is already TRACKED, whatever the rules say, because tracked
   files are not subject to them. So a bad rule would pass this check on every
   existing file and bite only the next time a file was added to the bundle --
   which is exactly when nobody would be looking. --no-index tests the rules
   themselves. */
const ignored = onDisk.length
  ? (() => { try { return git('check-ignore', '--no-index', ...onDisk.map(p => `${TOP}/${p}`))
                           .split('\n').filter(Boolean); }
             catch (e){ return e.status === 1 ? [] : ['git check-ignore failed']; } })()
  : [];
ok(ignored.length === 0, 'no .gitignore rule matches anything in the bundle',
   ignored.join(', '));

/* ---- 2. the folder is a fresh stage, byte for byte -------------------- */
section('2. the committed folder is exactly what the sources build today');
const fresh = stage(work);
const want = bundleFiles(fresh);
const extra = onDisk.filter(p => !want.includes(p));
const missing = want.filter(p => !onDisk.includes(p));
ok(extra.length === 0, 'nothing is in the committed folder that a fresh build lacks',
   `${extra.join(', ')} -- a file dropped from MANIFEST that the folder still carries; ${REMEDY}`);
ok(missing.length === 0, 'nothing a fresh build has is missing from it',
   `${missing.join(', ')}; ${REMEDY}`);
const differing = want.filter(p => onDisk.includes(p)
  && !readFileSync(join(BUNDLE_DIR, p)).equals(readFileSync(join(fresh, p))));
ok(differing.length === 0,
   `all ${want.length} files are byte for byte what the sources build today`,
   `stale: ${differing.join(', ')} -- a source was edited and the bundle was not `
   + `rebuilt; ${REMEDY}`);

/* The executable bit as GIT records it, which is what a clone and a download get
   -- not the bit on this disk, which a checkout on another filesystem may not
   carry. */
const modes = new Map(git('ls-files', '-s', '--', TOP).split('\n').filter(Boolean)
  .map(l => { const [meta, path] = l.split('\t'); return [path.slice(TOP.length + 1),
                                                         meta.split(' ')[0]]; }));
const wrongMode = want.filter(p => modes.has(p)
  && modes.get(p) !== (EXECUTABLE.has(p) ? '100755' : '100644'));
ok(wrongMode.length === 0,
   `git records the four runnable files as executable and the rest as not`,
   wrongMode.map(p => `${p} is ${modes.get(p)}`).join(', '));

/* ---- 3. the zip is the zip this checkout builds ----------------------- */
section('3. the committed zip is byte for byte the zip this checkout builds');
const freshZip = join(work, 'fresh.zip');
const built = buildZip(freshZip);
ok(existsSync(BUNDLE_ZIP), 'faraday-cell.zip exists', REMEDY);
if (existsSync(BUNDLE_ZIP)){
  const a = readFileSync(BUNDLE_ZIP), b = readFileSync(freshZip);
  ok(a.equals(b), `byte for byte, ${(b.length/1024).toFixed(1)} KB`,
     `committed ${a.length} bytes, fresh ${b.length}; ${REMEDY}. If only the bytes `
     + `differ and section 4 passes, the zip tool itself changed: ${built.zipVersion}`);

  /* And by content as well as by bytes, so that a failure above says WHICH files
     moved -- and so that if a different zip build ever wrote different bytes for
     the same files, this section would say the contents were still right. */
  const entries = execFileSync('unzip', ['-Z1', BUNDLE_ZIP], { encoding: 'utf8' })
    .split('\n').filter(Boolean);
  ok(JSON.stringify(entries) === JSON.stringify(want.map(p => `${TOP}/${p}`)),
     `its ${entries.length} entries are the folder's files, in sorted order, with no `
     + `directory entries`, `${entries.length} entries against ${want.length} files`);
  const zipStale = want.filter(p => {
    const got = execFileSync('unzip', ['-p', BUNDLE_ZIP, `${TOP}/${p}`],
                             { maxBuffer: 64*1024*1024 });
    return !got.equals(readFileSync(join(fresh, p)));
  });
  ok(zipStale.length === 0, 'every entry decompresses to what the sources build today',
     `stale in the zip: ${zipStale.join(', ')}; ${REMEDY}`);
}

/* ---- 4. the folder and the zip are one bundle ------------------------- */
section('4. what a download gets');
{
  /* The whole point: one file to download, which unpacks to the folder beside it.
     A zip that unpacked to a differently named directory, or loose into wherever
     it was opened, would make the two halves look like two different things. */
  const top = execFileSync('unzip', ['-Z1', BUNDLE_ZIP], { encoding: 'utf8' })
    .split('\n').filter(Boolean).map(p => p.split('/')[0]);
  ok(top.every(t => t === TOP), `the zip unpacks into ${TOP}/ and nowhere else`,
     [...new Set(top)].join(', '));
  ok(existsSync(join(BUNDLE_DIR, 'faraday-cell-standalone.html')),
     'the folder carries the single-file page, which opens by double-click');
  ok(existsSync(join(BUNDLE_DIR, 'README.md')),
     'and a README, which GitHub renders when the folder is opened');
  ok(MANIFEST.includes('faraday/check-bundle.mjs'),
     'and this gate travels with it, so a downloaded copy can still be checked '
     + 'against a checkout');
}

} finally { rmSync(work, { recursive: true, force: true }); }

console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length){
  for (const f of failures) console.log('  - ' + f);
  console.log(`\nTo fix: ${REMEDY}.`);
  process.exit(1);
}
