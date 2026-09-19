// Build a single-file, self-contained copy of cymatic.html.
//
// cymatic.html loads faraday/kernel.js and faraday/benchmark.js through
// relative <script src>, so the page only runs from inside a checkout with
// that directory beside it. Handing someone the .html alone gives them a
// page that loads and then throws on the first kernel call.
//
// This inlines the two scripts so the result opens by double-click, with no
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

export const PAIR = '<script src="faraday/kernel.js"></script>\n'
                  + '<script src="faraday/benchmark.js"></script>';

// `overrides` maps a repo-relative path to substitute content. It exists so
// check-standalone.mjs can exercise the </script> guard on a source that
// really contains one, instead of asserting a regex against itself. Nothing
// in the normal build path passes it.
export function buildStandalone(overrides = {}) {
  const read = p => overrides[p] ?? readFileSync(resolve(REPO, p), 'utf8');
  const html = read('cymatic.html');

  const n = html.split(PAIR).length - 1;
  if (n !== 1)
    throw new Error(
      `cymatic.html must contain the kernel/benchmark <script src> pair exactly `
      + `once, found ${n}. If the tags were reordered, reformatted or a third `
      + `was added, update PAIR here -- do not let this silently inline nothing.`);

  const sources = [['faraday/kernel.js', read('faraday/kernel.js')],
                   ['faraday/benchmark.js', read('faraday/benchmark.js')]];

  // A literal </script> anywhere in the source would close the inlined block
  // early, and the rest of the kernel would render as text on the page.
  for (const [name, src] of sources)
    if (/<\/script/i.test(src))
      throw new Error(
        `${name} contains a literal </script, which would terminate the `
        + `inlined block. Split it (e.g. '<\\/' + 'script') before inlining.`);

  const inlined = sources
    .map(([, src]) => `<script>\n${src}\n</script>`)
    .join('\n');

  return html.replace(PAIR, inlined);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const out = process.argv[2] ?? resolve(REPO, 'faraday-cell-standalone.html');
  writeFileSync(out, buildStandalone());
  console.log(`wrote ${out}`);
}
