// Build a single-file, self-contained copy of cymatic.html.
//
// cymatic.html loads faraday/kernel.js and the two dns/ solver files through
// relative <script src>, so the page only runs from inside a checkout with
// those directories beside it. Handing someone the .html alone gives them a
// page that loads and then throws on the first kernel call.
//
// This inlines those scripts so the result opens by double-click, with no
// server and no siblings. It is GENERATED, never committed: a second copy of
// 66 KB of kernel would drift from the original the first time one side was
// edited, and a drifted copy that still *runs* is the worst failure mode this
// repository has, because it returns an answer.
//
//     node faraday/build-standalone.mjs [outfile]
//
// Gated by faraday/check-standalone.mjs, which builds one and checks it.

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..');

/* The scripts the page loads, in order, each as {tag, path}. The tag is
   matched verbatim and the inlined block KEEPS its id: the Navier-Stokes panel
   reads its own solver source back out of those elements to build its worker,
   so an id dropped here would leave the built page unable to run the check. */
export const SCRIPTS = [
  { tag: '<script src="faraday/kernel.js"></script>',
    path: 'faraday/kernel.js', open: '<script>' },
  { tag: '<script id="dnsSolver" src="dns/faraday-dns.js"></script>',
    path: 'dns/faraday-dns.js', open: '<script id="dnsSolver">' },
  { tag: '<script id="dnsFloquet" src="dns/faraday-floquet.js"></script>',
    path: 'dns/faraday-floquet.js', open: '<script id="dnsFloquet">' }
];

// `overrides` maps a repo-relative path to substitute content. It exists so
// check-standalone.mjs can exercise the </script> guard on a source that
// really contains one, instead of asserting a regex against itself. Nothing
// in the normal build path passes it.
export function buildStandalone(overrides = {}) {
  const read = p => overrides[p] ?? readFileSync(resolve(REPO, p), 'utf8');
  const html = read('cymatic.html');

  for (const { tag, path } of SCRIPTS){
    const n = html.split(tag).length - 1;
    if (n !== 1)
      throw new Error(
        `cymatic.html must contain the tag for ${path} exactly once, found ${n}. `
        + `If it was reordered, reformatted or renamed, update SCRIPTS here -- do `
        + `not let this silently inline nothing.`);
  }

  const sources = SCRIPTS.map(({ path }) => [path, read(path)]);

  // A literal </script> anywhere in the source would close the inlined block
  // early, and the rest of the kernel would render as text on the page.
  for (const [name, src] of sources)
    if (/<\/script/i.test(src))
      throw new Error(
        `${name} contains a literal </script, which would terminate the `
        + `inlined block. Split it (e.g. '<\\/' + 'script') before inlining.`);

  let out = html;
  for (let i = 0; i < SCRIPTS.length; i++){
    const { tag, open } = SCRIPTS[i];
    out = out.replace(tag, `${open}\n${sources[i][1]}\n</script>`);
  }
  return out;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const out = process.argv[2] ?? resolve(REPO, 'faraday-cell-standalone.html');
  writeFileSync(out, buildStandalone());
  console.log(`wrote ${out}`);
}
