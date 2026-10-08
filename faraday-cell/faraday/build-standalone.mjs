// Build a single-file, self-contained copy of cymatic.html.
//
// cymatic.html loads faraday/kernel.js and the two dns/ solver files through
// relative <script src>, so the page only runs from inside a checkout with
// those directories beside it. Handing someone the .html alone gives them a
// page that loads and then throws on the first kernel call.
//
// This inlines those scripts so the result opens by double-click, with no
// server and no siblings. It is GENERATED. Where this command writes it by
// default -- faraday-cell-standalone.html at the repository root -- it is never
// committed: a second copy of 66 KB of kernel would drift from the original the
// first time one side was edited, and a drifted copy that still *runs* is the
// worst failure mode this repository has, because it returns an answer.
//
// ONE copy IS committed, inside the faraday-cell/ bundle, because the owner asked
// for the simulation to be downloadable from the repository as one folder. That
// copy is what faraday/check-bundle.mjs exists for: it rebuilds the bundle from
// the sources and fails CI the moment the committed page differs from what this
// file writes now.
//
//     node faraday/build-standalone.mjs [outfile]
//
// Gated by faraday/check-standalone.mjs, which builds one and checks it.

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..');

/* The scripts the page loads, IN ORDER, each as {tag, path}. The tag is matched
   verbatim and the inlined block KEEPS its id: the Navier-Stokes panel reads its
   own solver source back out of those elements to build its worker, so an id
   dropped here would leave the built page unable to run the check.

   The order is load order and it is load bearing. faraday-floquet.js resolves
   both solvers at its top level -- the periodic box from faraday-dns.js and the
   disc from faraday-disc.js -- and refuses if either is missing, so it comes
   last. */
export const SCRIPTS = [
  { tag: '<script src="faraday/kernel.js"></script>',
    path: 'faraday/kernel.js', open: '<script>' },
  /* The GPU renderer. Inlined like the rest: a single file that kept this as a
     <script src> would have a GPU button whose code could not be fetched from
     file://, which dns/check-page.mjs's "fetches no script at all" would catch. */
  { tag: '<script id="renderGl" src="faraday/render-gl.js"></script>',
    path: 'faraday/render-gl.js', open: '<script id="renderGl">' },
  { tag: '<script id="dnsSolver" src="dns/faraday-dns.js"></script>',
    path: 'dns/faraday-dns.js', open: '<script id="dnsSolver">' },
  { tag: '<script id="dnsDisc" src="dns/faraday-disc.js"></script>',
    path: 'dns/faraday-disc.js', open: '<script id="dnsDisc">' },
  { tag: '<script id="dnsDiscWasm" src="dns/faraday-disc-wasm.js"></script>',
    path: 'dns/faraday-disc-wasm.js', open: '<script id="dnsDiscWasm">' },
  { tag: '<script id="dnsFloquet" src="dns/faraday-floquet.js"></script>',
    path: 'dns/faraday-floquet.js', open: '<script id="dnsFloquet">' },
  /* The three-dimensional solver. AFTER faraday-disc.js, which it takes its graded
     grid maps from at load time and refuses without. Left out of this list when the
     page first referenced it, and dns/check-page.mjs caught it twice over: the
     standalone build kept the <script src> tag, so a single file opened by
     double-click tried to fetch a sibling that file:// gives it no origin for, and
     both "fetches no script at all" and "no resource failed to load" went red. */
  { tag: '<script id="dnsCell3d" src="dns/faraday-cell3d.js"></script>',
    path: 'dns/faraday-cell3d.js', open: '<script id="dnsCell3d">' },
  /* The loader for the three-dimensional C++ module. AFTER faraday-cell3d.js,
     whose FaradayCell3D it asks globalThis for -- a worker has no `require`, so
     the global is the whole mechanism there. */
  { tag: '<script id="dnsCell3dWasm" src="dns/faraday-cell3d-wasm.js"></script>',
    path: 'dns/faraday-cell3d-wasm.js', open: '<script id="dnsCell3dWasm">' },
  { tag: '<script id="dnsCell3dPool" src="dns/cell3d-pool.js"></script>',
    path: 'dns/cell3d-pool.js', open: '<script id="dnsCell3dPool">' }
];

/* The compiled period map travels as base64 in its own tag. A single file opened
   by double-click cannot fetch a sibling -- file:// gives the fetch no origin to
   satisfy -- so the alternative to carrying the bytes is a page whose C++ engine
   refuses, and the panel would then have nothing to run: it asks for the C++
   engine and does not substitute the JavaScript under that name.

   The tag is empty in the checkout, where the bytes are fetched from
   dns/faraday_disc.wasm over http instead. */
export const WASM_TAG = '<script id="dnsWasmBase64"></script>';
export const WASM_PATH = 'dns/faraday_disc.wasm';

/* And the same for the three-dimensional module, which the page's solver panel
   offers as its C++ engine. Without the bytes that engine refuses -- it does not
   substitute the JavaScript under the C++ name -- so a single file that left them
   out would carry a button that could only fail. */
export const CELL_WASM_TAG = '<script id="dnsCell3dWasmBase64"></script>';
export const CELL_WASM_PATH = 'dns/faraday_cell3d.wasm';

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

  if (html.split(WASM_TAG).length - 1 !== 1)
    throw new Error(
      `cymatic.html must contain ${WASM_TAG} exactly once: it is where the `
      + `compiled period map is inlined. Without it the built page has no C++ `
      + `engine and the Navier-Stokes panel refuses, because it does not run the `
      + `JavaScript solver under the C++ engine's name.`);

  if (html.split(CELL_WASM_TAG).length - 1 !== 1)
    throw new Error(
      `cymatic.html must contain ${CELL_WASM_TAG} exactly once: it is where the `
      + `three-dimensional C++ module is inlined. Without it the single file has a `
      + `C++ engine button that can only fail, because that engine refuses rather `
      + `than falling back to the JavaScript one.`);

  let out = html;
  for (let i = 0; i < SCRIPTS.length; i++){
    const { tag, open } = SCRIPTS[i];
    out = out.replace(tag, `${open}\n${sources[i][1]}\n</script>`);
  }

  const wasm = overrides[WASM_PATH] ?? readFileSync(resolve(REPO, WASM_PATH));
  const b64 = Buffer.from(wasm).toString('base64');
  out = out.replace(WASM_TAG,
    `<script id="dnsWasmBase64">\nglobalThis.FARADAY_DISC_WASM_BASE64 = "${b64}";\n</script>`);

  const cellWasm = overrides[CELL_WASM_PATH]
    ?? readFileSync(resolve(REPO, CELL_WASM_PATH));
  const cellB64 = Buffer.from(cellWasm).toString('base64');
  out = out.replace(CELL_WASM_TAG,
    `<script id="dnsCell3dWasmBase64">\n`
    + `globalThis.FARADAY_CELL3D_WASM_BASE64 = "${cellB64}";\n</script>`);
  return out;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const out = process.argv[2] ?? resolve(REPO, 'faraday-cell-standalone.html');
  writeFileSync(out, buildStandalone());
  console.log(`wrote ${out}`);
}
