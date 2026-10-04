#!/usr/bin/env node
/* Gate for faraday/build-zip.mjs.
 *
 *     node faraday/check-zip.mjs
 *
 * A zip is not verified by building it. It is verified by unpacking it somewhere
 * else and running what is inside from THERE, because every fault a zip can have
 * is a fault of what is missing from it: a script whose sibling was left out, a
 * README naming a path that does not travel, a generated file that was committed
 * once and has drifted since. All of those build cleanly.
 *
 * So this builds one, unpacks it into a temp directory, and from inside that
 * directory: compares every file against the checkout byte for byte, rebuilds the
 * standalone page and compares it, reads the README and requires every path it
 * names to resolve, runs the terminal runner, and runs the fast suites.
 *
 * The slow suite (dns/check-cell3d.mjs, minutes) is NOT run here -- it is run by
 * the project's own CI on the checkout, and running it again from the unpack would
 * make this gate unrunnable by hand. What IS asserted is that it travels, that it
 * is in the unpack's runner, and that the runner's --fast flag is the only thing
 * that leaves it out.
 */

import { readFileSync, writeFileSync, mkdtempSync, rmSync, existsSync, readdirSync,
         statSync } from 'node:fs';
import { dirname, resolve, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync, spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { buildStandalone } from './build-standalone.mjs';
import { buildZip, stage, MANIFEST, SUITES, TOP, runChecksSource } from './build-zip.mjs';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..');

let pass = 0; const failures = [];
function ok(cond, label, detail){
  if (cond){ pass++; console.log('  ok   ' + label); }
  else { failures.push(`${label}${detail ? '  -- ' + detail : ''}`);
         console.log('  FAIL ' + label + (detail ? '  -- ' + detail : '')); }
}
const section = t => console.log('\n' + t);

const work = mkdtempSync(join(tmpdir(), 'faraday-checkzip-'));
try {

/* ---- 1. the manifest describes the checkout ------------------------------ */
section('1. the manifest describes the checkout');
{
  const missing = MANIFEST.filter(p => !existsSync(resolve(REPO, p)));
  ok(missing.length === 0, 'every manifest path exists in the checkout',
     missing.length ? missing.join(', ') : `${MANIFEST.length} files`);

  const dup = MANIFEST.filter((p, i) => MANIFEST.indexOf(p) !== i);
  ok(dup.length === 0, 'no path is listed twice', dup.join(', '));

  /* A suite the checkout has and SUITES does not would travel in the zip and
     never be run by the runner inside it -- a silent omission, which is the one
     thing a runner must not do. check-zip.mjs itself is the single exception and
     the exception is named here, not implied: run from inside an unpack it would
     build a zip of the unpack and unpack that, without end. */
  const dirs = ['faraday', 'dns', 'boundary', 'fsi'];
  const found = [];
  for (const d of dirs)
    for (const f of readdirSync(resolve(REPO, d)))
      if (/^check-.*\.mjs$/.test(f) || /\.test\.js$/.test(f)) found.push(`${d}/${f}`);
  /* Two gates travel in the zip and are deliberately not in its runner, and each
     says why rather than being implied. This one, run from inside an unpack, would
     build a zip of the unpack and unpack that, without end. check-bundle.mjs
     checks the REPOSITORY'S committed copy of the bundle against its sources, and
     an unpack has neither -- it refuses there, correctly, so the runner would
     report a failure that means nothing about the unpack it is run in. */
  const NOT_RUN = new Map([
    ['faraday/check-zip.mjs', 'run from an unpack it would recurse'],
    ['faraday/check-bundle.mjs', 'it checks the repository\'s committed copy, which '
                                 + 'an unpack does not have']
  ]);
  const listed = new Set(SUITES.map(s => s.file));
  const unlisted = found.filter(f => !listed.has(f) && !NOT_RUN.has(f));
  ok(unlisted.length === 0,
     'every suite in the checkout is in the runner, bar the two that cannot run '
     + 'from an unpack',
     unlisted.length ? `not in SUITES: ${unlisted.join(', ')}` : `${found.length} found`);
  const phantom = [...NOT_RUN.keys()].filter(f => !found.includes(f));
  ok(phantom.length === 0, 'and both of those exist, so the exception is not stale',
     phantom.join(', '));

  const notTravelling = SUITES.map(s => s.file).filter(f => !MANIFEST.includes(f));
  ok(notTravelling.length === 0, 'every suite the runner runs travels in the zip',
     notTravelling.join(', '));

  /* The manifest read the other way round. Everything above catches a file
     dropped from MANIFEST only when something else in the unpack needs it -- the
     README names it, or a suite requires it -- and two files answer to neither.
     Measured by dropping each from MANIFEST on a disposable copy: without this
     check, faraday/benchmark.js and faraday/zip-README.md each leave the gate
     entirely GREEN while the zip ships without them. benchmark.js is required by
     faraday/check-kernel.mjs, which is too slow to run from the unpack here, and
     zip-README.md is the source README.md is generated from, so the generated copy
     survives its absence. "A file silently missing from the zip" is the one
     failure this gate exists for.

     (boundary/boundaries.html was the example this comment first gave, and the
     injection disproved it: boundary/boundarykernel.test.js reads that page, and
     dropping it fails 2 of that suite's 36 checks from the unpack. It was covered
     all along.)

     So: every source file in these four directories travels, and the list of
     things deliberately left out is EMPTY. A file added to one of them fails this
     check until someone decides whether it belongs in the zip. That is the
     intended cost: a decision made once, rather than a glob that would have swept
     in whatever scratch file was lying in dns/ that week. */
  const EXCLUDED = new Set();
  const carried = new Set(MANIFEST);
  const should = [];
  for (const d of dirs)
    for (const f of readdirSync(resolve(REPO, d)))
      if (/\.(mjs|js|html|json|md|cpp|wasm|sh)$/.test(f)) should.push(`${d}/${f}`);
  const omitted = should.filter(p => !carried.has(p) && !EXCLUDED.has(p));
  ok(omitted.length === 0,
     `all ${should.length} source files in ${dirs.join('/, ')}/ travel in the zip`,
     omitted.length ? `not in MANIFEST: ${omitted.join(', ')}` : '');
  ok(EXCLUDED.size === 0, 'nothing is deliberately left out', [...EXCLUDED].join(', '));

  /* The slow ones are the two that integrate the three-dimensional solver over
     many grids and many steps: its own gate, and the gate that holds the C++
     transcription to the JavaScript bit for bit over a whole drive period. Named
     here rather than counted, so that a third suite becoming slow is a decision
     and not a drift. */
  const slow = SUITES.filter(s => s.slow).map(s => s.file).sort();
  ok(slow.length === 2 && slow[0] === 'dns/check-cell3d-wasm.mjs'
     && slow[1] === 'dns/check-cell3d.mjs',
     'the slow suites are the two three-dimensional solver gates and only those',
     slow.join(', '));
}

/* ---- 2. the staged tree is exactly what the manifest says --------------- */
section('2. the staged tree is exactly what the manifest says');
const top = stage(join(work, 'staged'));
{
  const walk = (d, out = []) => {
    for (const f of readdirSync(d, { withFileTypes: true })){
      const p = join(d, f.name);
      if (f.isDirectory()) walk(p, out); else out.push(relative(top, p));
    }
    return out;
  };
  const got = walk(top).sort();
  const want = [...MANIFEST, 'README.md', 'faraday-cell-standalone.html',
                'run-checks.mjs'].sort();
  const extra = got.filter(p => !want.includes(p));
  const absent = want.filter(p => !got.includes(p));
  ok(extra.length === 0, 'nothing is staged that the manifest does not name', extra.join(', '));
  ok(absent.length === 0, 'everything the manifest names is staged', absent.join(', '));
  ok(got.length === want.length, `the tree holds ${want.length} files`, `got ${got.length}`);

  let differing = [];
  for (const p of MANIFEST)
    if (!readFileSync(join(top, p)).equals(readFileSync(resolve(REPO, p)))) differing.push(p);
  ok(differing.length === 0, 'every copied file is byte for byte the checkout\'s',
     differing.join(', '));

  const built = buildStandalone();
  ok(readFileSync(join(top, 'faraday-cell-standalone.html'), 'utf8') === built,
     'the standalone page in the tree is a fresh build of the checkout');
  ok(readFileSync(join(top, 'run-checks.mjs'), 'utf8') === runChecksSource(),
     'the runner in the tree is generated from SUITES');
  ok((statSync(join(top, 'run-checks.mjs')).mode & 0o111) !== 0,
     'the runner is executable');
}

/* ---- 3. the zip itself --------------------------------------------------- */
section('3. the zip itself');
const zipPath = join(work, 'faraday-cell.zip');
const built = buildZip(zipPath);
{
  ok(built.bytes > 200*1024, 'the zip is larger than 200 KB',
     `${(built.bytes/1024).toFixed(1)} KB`);
  const list = execFileSync('unzip', ['-Z1', zipPath], { encoding: 'utf8' })
    .split('\n').filter(Boolean).filter(p => !p.endsWith('/'));
  const want = [...MANIFEST, 'README.md', 'faraday-cell-standalone.html', 'run-checks.mjs']
    .map(p => `${TOP}/${p}`).sort();
  ok(JSON.stringify(list.sort()) === JSON.stringify(want),
     `the zip holds exactly those ${want.length} files under ${TOP}/`,
     `got ${list.length}`);
  ok(list.every(p => p.startsWith(`${TOP}/`)),
     'nothing in the zip unpacks outside its own directory');
}

/* ---- 4. a fresh unpack -------------------------------------------------- */
section('4. a fresh unpack, byte for byte');
const unpackRoot = join(work, 'unpack');
execFileSync('unzip', ['-q', zipPath, '-d', unpackRoot]);
const U = join(unpackRoot, TOP);
{
  let differing = [];
  for (const p of MANIFEST)
    if (!readFileSync(join(U, p)).equals(readFileSync(resolve(REPO, p)))) differing.push(p);
  ok(differing.length === 0, 'the unpack matches the checkout file for file',
     differing.join(', '));
  ok(readFileSync(join(U, 'faraday-cell-standalone.html'), 'utf8') === buildStandalone(),
     'the unpacked standalone page is a fresh build');

  /* The whole point of the single file: it must not reference a sibling, because
     a page opened by double-click has no origin a fetch can satisfy. */
  const html = readFileSync(join(U, 'faraday-cell-standalone.html'), 'utf8');
  const srcs = [...html.matchAll(/<script[^>]*\ssrc="([^"]+)"/g)].map(m => m[1]);
  ok(srcs.length === 0, 'the standalone page loads no external script', srcs.join(', '));
  ok(html.includes('FARADAY_DISC_WASM_BASE64 = "'),
     'the standalone page carries the compiled period map inline');
}

/* ---- 5. the README names nothing that does not travel ------------------- */
section('5. the README names nothing that does not travel');
{
  const readme = readFileSync(join(U, 'README.md'), 'utf8');
  /* Anything that looks like a path to a file in this tree: a bare name with a
     known extension, optionally under one of the directories. Prose is not
     searched for words that happen to contain a slash. */
  const cand = new Set();
  for (const m of readme.matchAll(/\b((?:faraday|dns|boundary|fsi)\/)?([A-Za-z0-9_.-]+\.(?:mjs|js|html|json|md|cpp|wasm|sh|zip))\b/g))
    cand.add((m[1] ?? '') + m[2]);
  const generated = new Set(['faraday-cell.zip']);
  const bad = [...cand].filter(p => !generated.has(p) && !existsSync(join(U, p)));
  ok(bad.length === 0, `every one of the ${cand.size} paths the README names resolves `
     + `in the unpack`, bad.join(', '));
  ok(cand.size >= 10, 'the README names at least ten of them', `${cand.size}`);

  for (const want of ['faraday-cell-standalone.html', 'run-checks.mjs',
                      'dns/run-cell3d.mjs', 'cymatic.html'])
    ok(cand.has(want), `the README tells the reader about ${want}`);

  /* The two settled facts a reader has to be told, or they will file the clock
     ratio as a bug and ask for the physics in f32. */
  ok(/slower than real time/i.test(readme),
     'the README says the solver does not run at real time');
  ok(/f32|single precision|double/i.test(readme),
     'the README says where single precision is and is not used');
}

/* ---- 6. the runner inside the unpack ------------------------------------ */
section('6. the runner inside the unpack');
{
  const list = spawnSync(process.execPath, [join(U, 'run-checks.mjs'), '--list'],
                         { encoding: 'utf8' });
  ok(list.status === 0, 'run-checks.mjs --list exits zero', `status ${list.status}`);
  const named = list.stdout.split('\n').map(s => s.trim().split('   ')[0]).filter(Boolean);
  ok(JSON.stringify(named.sort()) === JSON.stringify(SUITES.map(s => s.name).sort()),
     `--list names all ${SUITES.length} suites`, named.join(', '));

  const fast = spawnSync(process.execPath, [join(U, 'run-checks.mjs'), '--list', '--fast'],
                         { encoding: 'utf8' });
  const fastNamed = fast.stdout.split('\n').map(s => s.trim().split('   ')[0]).filter(Boolean);
  ok(fastNamed.length === SUITES.length - 2
     && !fastNamed.includes('dns/check-cell3d.mjs')
     && !fastNamed.includes('dns/check-cell3d-wasm.mjs'),
     '--fast leaves out exactly the two slow suites', fastNamed.join(', '));

  const bad = spawnSync(process.execPath, [join(U, 'run-checks.mjs'), '--nope'],
                        { encoding: 'utf8' });
  ok(bad.status === 2 && /unknown option/.test(bad.stderr),
     'an unknown option is refused rather than ignored',
     `status ${bad.status}, stderr ${JSON.stringify(bad.stderr.slice(0, 60))}`);
}

/* ---- 7. the terminal runner, from the unpack --------------------------- */
section('7. the terminal runner, from the unpack');
{
  const r = spawnSync(process.execPath,
    [join(U, 'dns', 'run-cell3d.mjs'), '--nr', '6', '--nth', '8', '--nz', '4',
     '--m', '2', '--periods', '0.02', '--frames', '1', '--width', '21'],
    { encoding: 'utf8' });
  ok(r.status === 0, 'it runs out of the unpack with no checkout beside it',
     `status ${r.status}: ${(r.stderr || '').slice(0, 200)}`);
  ok(/slower than real time/.test(r.stdout), 'it reports the clock ratio');
  ok(/max \|div u\| =/.test(r.stdout), 'it reports the divergence of the projected field');
  ok(/energy  total/.test(r.stdout), 'it reports the energy split');
  const m = r.stdout.match(/peak \|eta\| = ([0-9.]+) mm/);
  ok(m && Number(m[1]) > 0, 'the surface it drew is not identically flat', m && m[1]);

  /* And the C++ engine, from the unpack: the module and its loader have to travel
     and the loader has to find the module beside itself. It refuses rather than
     falling back, so if the bytes had not travelled this would fail here rather
     than quietly run the JavaScript and report its time under the C++ name. */
  const cpp = spawnSync(process.execPath,
    [join(U, 'dns', 'run-cell3d.mjs'), '--engine', 'cpp', '--nr', '6', '--nth', '8',
     '--nz', '4', '--m', '2', '--periods', '0.02', '--frames', '1', '--width', '21'],
    { encoding: 'utf8' });
  ok(cpp.status === 0, 'the C++ engine runs out of the unpack too',
     `status ${cpp.status}: ${(cpp.stderr || '').slice(0, 200)}`);
  ok(/faraday_cell3d\.wasm/.test(cpp.stdout),
     'and says which module answered', (cpp.stdout || '').slice(0, 120));

  const bad = spawnSync(process.execPath, [join(U, 'dns', 'run-cell3d.mjs'), '--nope', '1'],
                        { encoding: 'utf8' });
  ok(bad.status === 1 && /unknown option/.test(bad.stderr),
     'an unknown option is refused', `status ${bad.status}`);
  const bad2 = spawnSync(process.execPath, [join(U, 'dns', 'run-cell3d.mjs'), '--nr', 'x'],
                         { encoding: 'utf8' });
  ok(bad2.status === 1 && /not a finite number/.test(bad2.stderr),
     'a non-numeric value is refused rather than becoming NaN', `status ${bad2.status}`);
  const bad3 = spawnSync(process.execPath, [join(U, 'dns', 'run-cell3d.mjs'), '--nth', '9'],
                         { encoding: 'utf8' });
  ok(bad3.status === 1 && /even azimuthal count/.test(bad3.stderr),
     'the solver\'s own grid refusal reaches the user', `status ${bad3.status}`);
}

/* ---- 8. the fast suites, run from the unpack --------------------------- */
section('8. the fast suites, run from the unpack');
{
  /* Chosen for crossing directory boundaries, not for being easy: each one
     requires files the zip had to carry from somewhere else, so a missing sibling
     fails here and nowhere else. check-standalone reads cymatic.html and every
     script it inlines; check-wasm-build requires three files from dns/ AND
     faraday/kernel.js, and rebuilds dns/faraday_disc.cpp with clang to compare
     the fresh module against the committed one -- which is the strongest
     did-it-travel check available, and costs two seconds.

     Pass counts are asserted, because a suite that collected nothing exits zero
     too.

     dns/check-dns.mjs is deliberately NOT here. It was, and it made this gate a
     nine-minute run on its own: measured at 506 s on this container, against the
     2.2 s of check-wasm-build, which crosses the same directory boundary. A gate
     nobody can afford to run by hand stops being run. */
  const expect = [
    ['faraday/check-standalone.mjs', 12],
    ['dns/check-wasm-build.mjs', 5],
    ['boundary/boundarykernel.test.js', 36],
    ['fsi/check-coupled-affine.mjs', 102]
  ];
  for (const [file, n] of expect){
    const r = spawnSync(process.execPath, [join(U, file)], { encoding: 'utf8' });
    const out = (r.stdout || '') + (r.stderr || '');
    ok(r.status === 0, `${file} passes from the unpack`,
       `status ${r.status}: ${out.trim().split('\n').slice(-2).join(' | ').slice(0, 200)}`);
    const m = out.match(/(\d+)\s+(?:checks\s+)?passed/);
    ok(m && Number(m[1]) === n, `${file} ran all ${n} of its checks`,
       m ? `reported ${m[1]}` : 'no count in its output');
  }
}

} finally { rmSync(work, { recursive: true, force: true }); }

console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length){ for (const f of failures) console.log('  - ' + f); process.exit(1); }
