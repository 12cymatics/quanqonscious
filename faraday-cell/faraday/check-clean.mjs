// The comment-free single-file build must be the same program as the full one.
//
//     ACORN=/path/to/node_modules/acorn node faraday/check-clean.mjs
//
// faraday/build-clean.mjs removes every comment with a tokenizer of its own. This file
// does not trust that tokenizer to grade itself: every script is read back by acorn, an
// independent JavaScript parser, and the clean build must give the same tokens as the
// page it came from, in the same order with the same values, and no comments. A comment
// stripper that took a `//` inside a string or a regular expression for a comment, or
// that trimmed the inside of a template literal, changes a token, and that fails here.
//
// acorn is not vendored. It is found through ACORN, or through node's own resolution,
// and this REFUSES when it is not there: without it nothing below would be checked.
//
// The browser half -- the clean page loads, solves and draws the same numbers as the full
// one -- is in dns/check-page.mjs, which runs this build beside the other two.

import { createRequire } from 'node:module';
import vm from 'node:vm';
import { buildStandalone } from './build-standalone.mjs';
import { applyProse, buildClean, stripJs, PROSE_EDITS, REMOVED_PROSE } from './build-clean.mjs';

const require = createRequire(import.meta.url);
let acorn;
try { acorn = require(process.env.ACORN || 'acorn'); }
catch (e) {
  console.log('clean single-file build\n\n  FAIL acorn was not found'
    + `${process.env.ACORN ? ` at ACORN=${process.env.ACORN}` : ' (set ACORN=/path/to/acorn)'}\n`
    + '         Every token comparison below needs an independent parser; refusing rather '
    + 'than passing with none of them run.');
  process.exit(1);
}

let passed = 0, failed = 0;
const check = (name, fn) => {
  try { fn(); console.log(`  ok   ${name}`); passed++; }
  catch (e) { console.log(`  FAIL ${name}\n         ${e.message}`); failed++; }
};
const eq = (got, want, what) => {
  if (got !== want) throw new Error(`${what}: got ${got}, want ${want}`);
};

const full = buildStandalone();
const edited = applyProse(full);
let clean;
try { clean = buildClean(full); }
catch (e) {
  console.log(`clean single-file build\n\n  FAIL the build refused to run\n         ${e.message}`);
  process.exit(1);
}
const html = clean.out;

console.log('clean single-file build\n');

/* The scripts of a page, read with a pattern of this file's own rather than the builder's
   walk: each <script ...> to the first </script> after it. */
const scriptsOf = page => [...page.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)]
  .map(m => ({ attrs: m[1], body: m[2] }));
const tokens = (src, comments) => {
  const out = [];
  for (const t of acorn.tokenizer(src, { ecmaVersion: 'latest', onComment: comments }))
    out.push(t);
  return out;
};
/* A worker body is a template literal that closes the [ ... ] its sources are joined from. */
const isWorkerBody = (toks, i) => toks[i].type.label === 'template'
  && toks[i + 1] && toks[i + 1].type.label === '`'
  && toks[i + 2] && toks[i + 2].type.label === ']'
  && toks[i + 3] && toks[i + 3].type.label === '.'
  && toks[i + 4] && toks[i + 4].value === 'join';

const before = scriptsOf(edited), after = scriptsOf(html);
let workerBodies = 0, commentsBefore = 0, shaders = 0, glslBefore = 0;

check('the clean page has the same scripts, in the same order, with the same attributes', () => {
  eq(after.length, before.length, 'script count');
  before.forEach((b, i) => eq(after[i].attrs, b.attrs, `attributes of script ${i}`));
});

check('every script gives acorn the same tokens as the page it came from, and no comment', () => {
  before.forEach((b, s) => {
    const cb = [], ca = [];
    const tb = tokens(b.body, cb), ta = tokens(after[s].body, ca);
    commentsBefore += cb.length;
    eq(ca.length, 0, `comments acorn finds in clean script ${s}`);
    eq(ta.length, tb.length, `token count of script ${s}`);
    for (let i = 0; i < tb.length; i++){
      const x = tb[i], y = ta[i];
      if (x.type.label !== y.type.label) throw new Error(
        `script ${s}, token ${i}: ${x.type.label} became ${y.type.label} near `
        + JSON.stringify(b.body.slice(x.start, x.start + 60)));
      const worker = isWorkerBody(tb, i) && isWorkerBody(ta, i);
      if (x.type.label === 'template' && String(x.value).startsWith('#version ')){
        /* a GPU shader is GLSL the page ships as text. GLSL has no string literals, so
           a pattern is an exact reference here: every block and line comment it finds is
           one. The two must read as the same words, and the clean one carry no comment. */
        const glslComments = /\/\*[\s\S]*?\*\/|\/\/[^\n]*/g;
        glslBefore += (x.value.match(glslComments) || []).length;
        const words = t => t.replace(glslComments, ' ').split(/\s+/).filter(Boolean).join(' ');
        if (/\/\*|\/\//.test(y.value)) throw new Error(`a comment is left in a clean shader`);
        eq(words(y.value), words(x.value), 'a shader, comments aside');
        shaders++;
        continue;
      }
      if (x.value === y.value && !worker) continue;
      if (!worker) throw new Error(
        `script ${s}, token ${i} (${x.type.label}): ${JSON.stringify(String(x.value).slice(0, 60))} `
        + `became ${JSON.stringify(String(y.value).slice(0, 60))}`);
      /* a worker body is code the page ships as text: the same check, one level down */
      const wb = [], wa = [];
      const kb = tokens(x.value, wb), ka = tokens(y.value, wa);
      commentsBefore += wb.length;
      eq(wa.length, 0, `comments acorn finds in a clean worker body`);
      eq(ka.length, kb.length, `token count of a worker body`);
      kb.forEach((k, j) => {
        if (k.type.label !== ka[j].type.label || k.value !== ka[j].value) throw new Error(
          `worker body token ${j}: ${JSON.stringify(k.value)} became ${JSON.stringify(ka[j].value)}`);
      });
      workerBodies++;
    }
  });
});

check('the page builds three worker bodies and two shaders, and no other token differs', () => {
  eq(workerBodies, 3, 'worker bodies read by acorn');
  eq(clean.counts.workerBodies, 3, 'worker bodies the builder found');
  eq(shaders, 2, 'shaders read');
  eq(clean.counts.shaders, 2, 'shaders the builder found');
});

check('the builder removed exactly the comments acorn and the GLSL pattern count', () => {
  eq(clean.counts.js, commentsBefore, 'JavaScript comments removed');
  eq(clean.counts.glsl, glslBefore, 'GLSL comments removed');
  if (glslBefore === 0) throw new Error('the shaders had no comment to remove, so this checked nothing');
});

check('every clean script compiles', () => {
  after.forEach(({ body }, s) => { new vm.Script(body, { filename: `clean script ${s}` }); });
});

check('no HTML or CSS comment is left', () => {
  const markup = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  eq(markup.includes('<!--'), false, 'an HTML comment in the markup');
  const styles = [...html.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/g)].map(m => m[1]);
  eq(styles.some(c => c.includes('/*')), false, 'a CSS comment in a <style>');
});

check('no comment marker is left anywhere in the file, outside the two encoded modules', () => {
  /* The compiled WebAssembly travels as base64, whose alphabet has '/' in it, so a '//'
     there is data. Everywhere else in this page none of the three markers occurs at all:
     not in a string, a URL or a pattern either. */
  const modules = html.match(/_BASE64 = "[A-Za-z0-9+/=]+"/g) || [];
  eq(modules.length, 2, 'encoded modules found');
  let rest = html;
  for (const m of modules) rest = rest.replace(m, '');
  const left = ['/*', '*/', '//', '<!--'].filter(t => rest.includes(t));
  eq(left.length, 0, `markers left (${left.join(' ')})`);
});

check('the markup and the style are the page\'s, less their comments', () => {
  /* Every character outside the scripts is the edited page's, in order: the clean build
     only ever deletes there. */
  const outside = p => p.replace(/<script([^>]*)>[\s\S]*?<\/script>/g, '<script$1></script>');
  const a = outside(edited), b = outside(html);
  let j = 0;
  for (let i = 0; i < b.length; i++){
    while (j < a.length && a[j] !== b[i]) j++;
    if (j >= a.length) throw new Error(
      `character ${i} of the clean markup, ${JSON.stringify(b.slice(i, i + 40))}, is not in the page`);
    j++;
  }
  const strippedMarkup = a.replace(/<!--[\s\S]*?-->/g, '').replace(/\s+/g, '');
  eq(b.replace(/\s+/g, ''), strippedMarkup, 'the markup, comments and whitespace aside');
});

check('every prose edit was made exactly once', () => {
  let page = full;
  for (const [what, find] of PROSE_EDITS){
    const n = page.split(find).length - 1;
    if (n !== 1) throw new Error(`${what}: found ${n} times in the full build`);
    page = page.replace(find, '');
  }
});

check('none of the removed sentences is anywhere in the clean page', () => {
  const left = REMOVED_PROSE.filter(t => html.includes(t));
  eq(left.length, 0, `fragments still present (${left.map(t => JSON.stringify(t)).join(', ')})`);
  /* and every one of them was really in the full build, so this is not vacuous */
  const absent = REMOVED_PROSE.filter(t => !full.includes(t));
  eq(absent.length, 0, `fragments the full build never had (${absent.map(t => JSON.stringify(t)).join(', ')})`);
});

check('the stripper keeps a comment-looking // or /* inside strings, templates and regexes', () => {
  /* Each of these is code acorn reads with no comment in it; stripping must change nothing. */
  const cases = [
    "const u = 'http://x.org/a'; const v = \"/* not */\";",
    'const t = `a // b ${"/* c */"} d`;',
    'const r = /\\/\\/[/*]+/g, s = a / b / c;',
    'const q = x.replace(/\\*\\//g, "") /* gone */ + 1;',
    'return /a\\/\\/b/.test(s)',
    'const w = `line  \n  trailing  \n`;'
  ];
  for (const c of cases){
    const want = [];
    for (const t of acorn.tokenizer(c, { ecmaVersion: 'latest', allowReturnOutsideFunction: true }))
      want.push(`${t.type.label}:${t.value}`);
    const r = stripJs(c), got = [];
    for (const t of acorn.tokenizer(r.out, { ecmaVersion: 'latest', allowReturnOutsideFunction: true }))
      got.push(`${t.type.label}:${t.value}`);
    eq(got.join(' '), want.join(' '), `tokens of ${JSON.stringify(c)}`);
  }
});

console.log(`\n${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
