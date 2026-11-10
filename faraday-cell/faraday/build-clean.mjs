#!/usr/bin/env node
/* The single-file page with every comment and every passage of prose taken out.

       node faraday/build-clean.mjs out.html

   It is built from faraday-cell-standalone.html -- the same page, the same scripts, the same
   compiled modules -- by two transformations and nothing else.

   COMMENTS. Every HTML comment, every CSS comment and every JavaScript comment is removed by
   a tokenizer that knows the three languages' strings: single and double quotes, template
   literals with nested ${...} code, regular-expression literals and CSS strings, so that a //
   inside a URL, a regex or a string is left exactly where it is. A removed comment that
   spanned a line break leaves the line break, so automatic semicolon insertion reads the code
   the way it read it before. The page also ships three blocks of JavaScript to its worker
   threads as template literals -- the Floquet worker, the 3-D solver's worker and its row
   workers -- each recognised by the `].join('\n;\n')` that assembles it, and the comments
   inside those are removed too, as code. So are the comments in the two GLSL programs the
   GPU renderer compiles, which also travel as template literals.

   PROSE. The sentences the page shows are taken out of it, and nothing a control, a reading
   or a unit needs: the subtitle under the title, the startup box's sentences, the paragraph
   for a browser without JavaScript, the Navier-Stokes panel's opening paragraph and its
   notes, the footer, the warnings, the export notes, the source note beside the
   acceleration, the flicker-fusion clause on the overlay, the inspector's sentences, the grid
   placeholder and the anchors' tooltips. A failure the page used to explain in a sentence is
   shown by its name -- "simulation error", "solver failed", "did not run", "export failed",
   "GPU unavailable", "drawing paused" -- with its reason in the browser console. The kernel
   still computes its warnings and the solvers still throw their reasons; the page no longer
   shows them.

   faraday/check-clean.mjs reads every script of the result back with acorn, an independent
   parser, and requires the same tokens as the page it came from with no comment among them.
   dns/check-page.mjs runs the result in a browser beside the other two builds -- the Floquet
   panel against the node solver, the 3-D solver in both engines and on several threads, the
   GPU renderer -- and requires no sentence on its screen. */

import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildStandalone } from './build-standalone.mjs';

/* The keywords after which a slash opens a regular expression rather than dividing. */
const REGEX_AFTER_WORD = new Set(['return', 'typeof', 'instanceof', 'in', 'of', 'new', 'delete',
  'void', 'throw', 'case', 'do', 'else', 'yield', 'await']);

/* Removing a comment from `out` + src[i..j): the spaces before it go, and when the comment
   had a line to itself the line goes too. Text inside a string is never touched -- this is
   only called at the level of code, markup or style. A comment that spanned a line break
   leaves one, so automatic semicolon insertion reads the code the way it read it before.
   Returns [out, next index]. */
function cut(out, src, i, j, spansLines){
  out = out.replace(/[ \t]+$/, '');
  let k = j;
  while (k < src.length && (src[k] === ' ' || src[k] === '\t')) k++;
  const atLineStart = out === '' || out.endsWith('\n');
  if (atLineStart) return [out, src[k] === '\n' ? k + 1 : k];
  if (spansLines) return [out + '\n', k];
  return [out + (k >= src.length || src[k] === '\n' ? '' : ' '), k];
}

/* JavaScript with its comments removed. `src` is code; the result is the same code, token
   for token. Returns { out, comments, workerBodies, shaders, glsl }: the JavaScript comments
   removed, how many worker bodies were found and stripped as code, and how many GPU
   shader programs were found and how many GLSL comments came out of them. */
export function stripJs(src){
  let out = '', i = 0, comments = 0, workerBodies = 0, shaders = 0, glsl = 0;
  const n = src.length;
  /* the last significant character or word emitted, to tell a regex from a division */
  let last = '', lastWord = '';
  const isIdent = c => /[A-Za-z0-9_$\u0080-￿]/.test(c);
  const regexAllowed = () => {
    if (last === '') return true;
    if (lastWord && REGEX_AFTER_WORD.has(lastWord)) return true;
    if (isIdent(last) || last === ')' || last === ']' || last === '}') return false;
    return true;
  };
  const emit = s => {
    out += s;
    const t = s.trimEnd();
    if (t){
      last = t[t.length - 1];
      const w = t.match(/[A-Za-z_$][\w$]*$/);
      lastWord = w ? w[0] : '';
    }
  };
  const readString = (at, q) => {
    let j = at + 1;
    while (j < n){
      const c = src[j];
      if (c === '\\'){ j += 2; continue; }
      if (c === q) return j + 1;
      if (c === '\n') throw new Error(`unterminated string at ${at}`);
      j++;
    }
    throw new Error(`unterminated string at ${at}`);
  };
  /* a template literal from the backtick at `at`; the code in each ${...} is stripped like
     any other code. Returns the stripped text and the index past the closing backtick. */
  const readTemplate = at => {
    let j = at + 1, text = '`';
    while (j < n){
      const c = src[j];
      if (c === '\\'){ text += src.slice(j, j + 2); j += 2; continue; }
      if (c === '`') return { text: text + '`', end: j + 1 };
      if (c === '$' && src[j + 1] === '{'){
        let depth = 1, k = j + 2;
        const startCode = k;
        while (k < n){
          const d = src[k];
          if (d === "'" || d === '"'){ k = readString(k, d); continue; }
          if (d === '`'){ k = readTemplate(k).end; continue; }
          if (d === '/' && (src[k + 1] === '/' || src[k + 1] === '*')) throw new Error(
            `a comment inside a template interpolation at ${k}; this stripper does not take it apart`);
          if (d === '{') depth++;
          else if (d === '}' && --depth === 0) break;
          k++;
        }
        if (k >= n) throw new Error(`unterminated template interpolation at ${j}`);
        const inner = stripJs(src.slice(startCode, k));
        comments += inner.comments;
        text += '${' + inner.out + '}';
        j = k + 1;
        continue;
      }
      text += c; j++;
    }
    throw new Error(`unterminated template literal at ${at}`);
  };
  const readRegex = at => {
    let j = at + 1, inClass = false;
    while (j < n){
      const c = src[j];
      if (c === '\\'){ j += 2; continue; }
      if (c === '\n') throw new Error(`unterminated regular expression at ${at}`);
      if (inClass){ if (c === ']') inClass = false; }
      else if (c === '[') inClass = true;
      else if (c === '/'){ j++; break; }
      j++;
    }
    while (j < n && /[a-z]/i.test(src[j])) j++;
    return j;
  };
  while (i < n){
    const c = src[i], d = src[i + 1];
    if (c === '/' && d === '/'){
      let j = i + 2;
      while (j < n && src[j] !== '\n') j++;
      [out, i] = cut(out, src, i, j, false);
      comments++;
      continue;
    }
    if (c === '/' && d === '*'){
      const e = src.indexOf('*/', i + 2);
      if (e < 0) throw new Error(`unterminated block comment at ${i}`);
      [out, i] = cut(out, src, i, e + 2, src.slice(i, e).includes('\n'));
      comments++;
      continue;
    }
    if (c === "'" || c === '"'){
      const j = readString(i, c);
      emit(src.slice(i, j)); i = j;
      /* a string is a value: a slash after it divides */
      last = 'a'; lastWord = '';
      continue;
    }
    if (c === '`'){
      const t = readTemplate(i);
      let text = t.text;
      /* a worker body: a template literal that closes a [ ... ] of sources joined by ';' */
      if (/^\s*\]\.join\('\\n;\\n'\)/.test(src.slice(t.end, t.end + 40))){
        if (text.includes('${')) throw new Error(
          `the worker body ending at ${t.end} interpolates; its text is not plain code`);
        const inner = stripJs(text.slice(1, -1));
        comments += inner.comments;
        workerBodies++;
        shaders += inner.shaders; glsl += inner.glsl;
        text = '`' + inner.out + '`';
      }
      /* a GPU shader: a template literal that is a GLSL program */
      else if (text.startsWith('`#version ')){
        if (text.includes('${')) throw new Error(
          `the shader ending at ${t.end} interpolates; its text is not plain GLSL`);
        const inner = stripGlsl(text.slice(1, -1));
        glsl += inner.comments;
        shaders++;
        text = '`' + inner.out + '`';
      }
      emit(text); i = t.end;
      last = 'a'; lastWord = '';
      continue;
    }
    if (c === '/' && regexAllowed()){
      const j = readRegex(i);
      emit(src.slice(i, j)); i = j;
      last = 'a'; lastWord = '';
      continue;
    }
    if (/\s/.test(c)) out += c; else emit(c);
    i++;
  }
  return { out, comments, workerBodies, shaders, glsl };
}

/* GLSL with its comments removed. GLSL has no string literals, so every // and every /*
   in a shader begins a comment. */
export function stripGlsl(src){
  let out = '', i = 0, comments = 0;
  while (i < src.length){
    if (src[i] === '/' && src[i + 1] === '/'){
      let j = i + 2;
      while (j < src.length && src[j] !== '\n') j++;
      [out, i] = cut(out, src, i, j, false);
      comments++;
      continue;
    }
    if (src[i] === '/' && src[i + 1] === '*'){
      const e = src.indexOf('*/', i + 2);
      if (e < 0) throw new Error(`unterminated GLSL comment at ${i}`);
      [out, i] = cut(out, src, i, e + 2, src.slice(i, e).includes('\n'));
      comments++;
      continue;
    }
    out += src[i]; i++;
  }
  return { out, comments };
}

/* CSS with its comments removed; strings are left alone. */
export function stripCss(src){
  let out = '', i = 0, comments = 0;
  while (i < src.length){
    const c = src[i];
    if (c === '/' && src[i + 1] === '*'){
      const e = src.indexOf('*/', i + 2);
      if (e < 0) throw new Error(`unterminated CSS comment at ${i}`);
      [out, i] = cut(out, src, i, e + 2, false);
      comments++;
      continue;
    }
    if (c === '"' || c === "'"){
      let j = i + 1;
      while (j < src.length && src[j] !== c){ if (src[j] === '\\') j++; j++; }
      out += src.slice(i, j + 1); i = j + 1;
      continue;
    }
    out += c; i++;
  }
  return { out, comments };
}

/* A page with every comment removed: HTML comments in the markup, CSS comments inside
   <style>, JavaScript comments inside <script>. A <script> whose type is not JavaScript
   would carry data, not code, and is left as it is; this page has none. */
export function stripPage(html){
  let out = '', i = 0;
  const counts = { html: 0, css: 0, js: 0, glsl: 0, workerBodies: 0, shaders: 0, scripts: 0, styles: 0 };
  const lower = html.toLowerCase();
  while (i < html.length){
    if (html.startsWith('<!--', i)){
      const e = html.indexOf('-->', i + 4);
      if (e < 0) throw new Error(`unterminated HTML comment at ${i}`);
      [out, i] = cut(out, html, i, e + 3, false);
      counts.html++;
      continue;
    }
    const isScript = lower.startsWith('<script', i), isStyle = lower.startsWith('<style', i);
    if (isScript || isStyle){
      const tagEnd = html.indexOf('>', i);
      const closeTag = isScript ? '</script>' : '</style>';
      const close = lower.indexOf(closeTag, tagEnd);
      if (close < 0) throw new Error(`unterminated <${isScript ? 'script' : 'style'}> at ${i}`);
      const open = html.slice(i, tagEnd + 1), body = html.slice(tagEnd + 1, close);
      let stripped;
      if (isScript){
        if (/\btype\s*=/i.test(open) && !/\btype\s*=\s*"(text\/javascript|module)"/i.test(open))
          stripped = body;
        else {
          const r = stripJs(body);
          counts.js += r.comments; counts.workerBodies += r.workerBodies; counts.scripts++;
          counts.shaders += r.shaders; counts.glsl += r.glsl;
          stripped = r.out;
        }
      } else {
        const r = stripCss(body);
        counts.css += r.comments; counts.styles++;
        stripped = r.out;
      }
      out += open + stripped + closeTag;
      i = close + closeTag.length;
      continue;
    }
    out += html[i]; i++;
  }
  return { out, counts };
}

/* Each removal of prose: an exact text in the commented page and what takes its place.
   Every one must be found exactly once, so a change to the page that moves one of them
   fails this build instead of leaving its prose on screen. A sentence that reported a
   failure is cut to the failure's name, and the reason goes to the browser console. */
export const PROSE_EDITS = [
  ['the subtitle under the title',
   "\n  <p>General circular-cell experiment · SI physical inputs · optional Sheldrake &amp; Sheldrake 2017 comparison</p>", ''],
  ['the example-input explanation',
   'Initial values are editable examples, not measurements.', 'Editable examples · not measurements'],
  ['the general physical-input explanation',
   'Drive: g + a cos(2π f t). Kinematic viscosity is μ/ρ. Initial displacement: (amplitude/depth) × depth × Jₘ(jₘ,₁ r/R) cos(mθ); initial velocity is zero.',
   'g + a cos(2π f t) · ν = μ/ρ · η₀ = (amplitude/depth) h Jₘ(jₘ,₁ r/R) cos(mθ) · u₀ = 0'],
  ['the model scope explanation',
   'One incompressible Newtonian liquid in a rigid circular cylinder; a single-valued free surface; pinned contact line with no-slip floor and wall. Free contact is available only for modal diagnostics. No air dynamics, overturning, contact-angle hysteresis or deformable vessel. Finite numerical discretization, not an exact continuum solution.',
   'Incompressible Newtonian liquid · rigid cylinder · graph surface · pinned rim · no-slip floor and wall · free contact: modal diagnostics only · no gas or overturning · no contact hysteresis or vessel deformation · finite discretization'],
  ['the initial nonlinear pending explanation',
   '>Nonlinear solver not started. Set the experiment inputs and press Run 3D solver.</div>',
   '>Nonlinear solver pending · press Run</div>'],
  ['the live nonlinear pending explanation',
   "solverRequested ? 'Nonlinear solver starting; no replacement surface is displayed.' : 'Nonlinear solver not started. Set the experiment inputs and press Run 3D solver.'",
   "solverRequested ? 'Nonlinear solver starting' : 'Nonlinear solver pending · press Run'"],
  ['the startup box\'s sentence before the page runs; the box stays, hidden until a failure',
   "<div id=\"startupStatus\" role=\"status\" style=\"padding:12px;border:1px solid var(--edge);margin-bottom:12px;line-height:1.6\">Not started. Open index.html in a full browser, not an attachment or Files preview. If this message stays visible, JavaScript has not run.</div>",
   "<div id=\"startupStatus\" role=\"status\" hidden style=\"padding:12px;border:1px solid var(--edge);margin-bottom:12px;line-height:1.6\"></div>"],
  ['the paragraph for a browser without JavaScript', "\n<noscript><p>This simulator requires JavaScript. A static file preview cannot run it. Open the extracted index.html in Safari, Chrome or Firefox on your computer.</p></noscript>", ''],
  ['the startup failure\'s sentence', "    status.textContent = 'Simulation error: ' + String(error && error.message || error)\n      + '. No substitute simulation has been loaded. Reload after resolving this error.';",
   "    console.error(error);\n    status.textContent = 'simulation error';\n    status.hidden = false;"],
  ['the sentence once the canvas draws', "\n    const status = document.getElementById('startupStatus');\n    status.textContent = 'Canvas rendering started. The active model is identified under surface drawn from. Nonlinear Navier\u2013Stokes physical time can advance much more slowly than wall time.';\n    status.style.color = 'var(--grn)';", ''],
  ['the sentence while the page starts', "\ndocument.getElementById('startupStatus').textContent = 'Starting the simulator\u2026';", ''],
  ['the Navier-Stokes panel\'s opening paragraph',
   "\n  <p>The deck above decides onset from a damped Mathieu form — one amplitude equation, with\n"
   + "  all the viscous damping folded into a single rate γ. This solves the incompressible\n"
   + "  Navier–Stokes equations instead, linearised about the flat layer <b>in this disc</b>: the\n"
   + "  same radius, depth, water and drive, with no slip on the floor and the sidewall, the full\n"
   + "  normal-stress balance at the free surface, and the contact line as the control above sets\n"
   + "  it. It integrates one drive period and takes the eigenvalues of that period map; |μ| &gt; 1\n"
   + "  is growth. Nothing is assumed about the damping — the viscosity enters as the diffusion\n"
   + "  term and the boundary layers are resolved by a graded grid.</p>", ''],
  ['the Navier-Stokes panel\'s note',
   "\n  <p id=\"dnsNote\"></p>", ''],
  ['the writer of that note',
   "\n  const note = t => { $('dnsNote').innerHTML = t; };", ''],
  ['the projected-time note', "note(`Projected ${projected < 90 ? projected.toFixed(0) + ' s'\n             : (projected/60).toFixed(1) + ' min'} for ${krylov} period maps on `\n         + `${g.stateSize} unknowns at ${g.stepsPerPeriod} steps each, `\n         + `from twenty steps timed on this machine just now. Running in a worker; `\n         + `the renderer above is unaffected.`);", ''],
  ['the note under the Floquet result', "note(`Growth is ln|\u03bc| over one drive period, from the eigenvalues of the period `\n         + `map of the linearised Navier-Stokes operator in this disc, integrated by `\n         + `the C++ transcription of the solver \u2014 bit for bit the same numbers as the `\n         + `JavaScript, which is asserted rather than assumed. The width on the growth `\n         + `is real: the viscous cell has a dense cluster of decaying modes, so which `\n         + `one is largest is ill conditioned, and the figure is quoted with the spread `\n         + `across Krylov dimensions rather than to a precision it does not have. Where `\n         + `this and the deck's Mathieu figure differ, the difference is damping the `\n         + `single-rate \u03b3 leaves out \u2014 chiefly the sidewall layer.`);", ''],
  ['the note on a state the panel cannot run', "note(`<b>Not runnable on this state:</b> `\n      + `${e.message}`);", 'console.error(e);'],
  ['the note before the first run', "\n  note('Not run yet. The renderer above is unaffected by this panel.');", ''],
  ['the Floquet panel\'s failure sentence',
   "note(`<b>The Navier-Stokes check did not run:</b> ${err.message} Nothing is `\n"
   + "         + `reported in its place.`);",
   "console.error(err); $('dnsOut').innerHTML = cell('verdict', '<b>did not run</b>');"],
  ['the footer',
   "<footer>\n  General nonlinear experiments use the entered SI properties and acceleration. The paper atlas is optional comparison data and never selects nonlinear patterns. The reference reconstruction and modal diagnostic are separate visual models. Image normalization does not alter the solver field. Free-surface overturning, gas dynamics and moving contact lines are outside this solver's scope.\n"
   + "</footer>\n", ''],
  ['the warnings',
   "  $('warnBox').innerHTML = s.warnings.map(w =>\n"
   + "    `<div class=\"warn${w.hard ? ' hard' : ''}\">${w.text}</div>`).join('');\n", ''],
  ['the source sentence beside the acceleration',
   "`<span style=\"color:var(--dim)\">(${dc.gPerMv} g/mV — ${dc.source})</span>`",
   "`<span style=\"color:var(--dim)\">(${dc.gPerMv} g/mV)</span>`"],
  ['the paused-drawing sentence',
   "`${renderEngine.toUpperCase()} drawing paused: ${renderError} Choose CPU to draw there, "
   + "or choose GPU to retry when its context is available.`",
   "(console.error(renderError), `${renderEngine.toUpperCase()} drawing paused`)"],
  ['the reason the GPU is unavailable',
   "`<b>CPU</b> <span class=\"u\">· GPU unavailable: ${GPU_SURFACE.error}</span>`",
   "(console.error(GPU_SURFACE.error), `<b>CPU</b> <span class=\"u\">· GPU unavailable</span>`)"],
  ['the reason the solver failed',
   "if (c.err) return `<b style=\"color:var(--bad,#e66)\">solver failed</b> `\n"
   + "                       + `<span class=\"u\">${c.err}</span>`;",
   "if (c.err) return (console.error(c.err), `<b style=\"color:var(--bad,#e66)\">solver failed</b>`);"],
  ['the inspector\'s missing-atlas sentence',
   "'<span class=\"val sm\">no atlas for this diameter</span>'", "'—'"],
  ['the inspector\'s transition sentences',
   "\n    <div class=\"val sm\">switched in ${(s.transitionStudy.switched*100).toFixed(1)}% of slow amplitude reductions.\n"
   + "    Higher atmospheric pressure favoured the higher-order state.</div>", ''],
  ['the flicker-fusion clause on the overlay',
   "\n    + '  ·  steady: f/2 = ' + s.responseHz.toFixed(1) + ' Hz is above flicker fusion';", ';'],
  ['the anchors\' tooltips', " title=\"${t}\"", ''],
  ['the export note at load',
   "  if (n) n.textContent = saver ? 'saves through the viewer'\n"
   + "    : (typeof claude !== 'undefined' ? 'saving unavailable in this view' : 'saves to disk');\n", ''],
  ['the export failure\'s reason',
   ": 'export failed: ' + (e && (e.message || e.code) || e);",
   ": (console.error(e), 'export failed');"],
  ['the grid placeholder sentence',
   "cell('grid', `<span class=\"u\">sized to the mode when you press run`\n"
   + "                + `</span>`)", "cell('grid', '—')"]
];

/* The sentences PROSE_EDITS takes off the screen, each as a fragment that must not be left
   anywhere in the built page. Held separately from the edits so that a check of the
   output does not merely restate the edit that produced it. */
export const REMOVED_PROSE = [
  'Sheldrake &amp; Sheldrake 2017', 'Not started. Open index.html', 'This simulator requires JavaScript',
  'No substitute simulation has been loaded', 'Canvas rendering started', 'Starting the simulator', 'The deck above decides onset', 'General nonlinear experiments use the entered SI properties',
  'Initial values are editable examples', 'Kinematic viscosity is μ/ρ',
  'One incompressible Newtonian liquid in a rigid circular cylinder',
  'Set the experiment inputs and press Run 3D solver',
  'Choose CPU to draw there', 'GPU unavailable: ', 'no atlas for this diameter',
  'Higher atmospheric pressure favoured', 'above flicker fusion', 'saves through the viewer',
  'saving unavailable in this view', "'saves to disk'", "'export failed: '",
  'sized to the mode when you press run', 'The Navier-Stokes check did not run',
  'Not run yet. The renderer above', 'twenty steps timed on this machine',
  'Growth is ln|μ| over one drive period', 'Not runnable on this state', 'id="dnsNote"',
  "s.warnings.map(w", ' g/mV — ${dc.source}', 'title="${t}"'
];

export function applyProse(html){
  for (const [what, find, rep] of PROSE_EDITS){
    const found = html.split(find).length - 1;
    if (found !== 1) throw new Error(
      `build-clean: ${what} was found ${found} times where it must be found once; `
      + 'the page changed under this edit, so its prose would be left behind.');
    html = html.replace(find, () => rep);
  }
  return html;
}

export function buildClean(source){
  const r = stripPage(applyProse(source === undefined ? buildStandalone() : source));
  if (r.counts.workerBodies !== 3) throw new Error(
    `build-clean: ${r.counts.workerBodies} worker bodies found where the page builds three.`);
  if (r.counts.shaders !== 2) throw new Error(
    `build-clean: ${r.counts.shaders} GPU shaders found where the page compiles two.`);
  return r;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])){
  if (!process.argv[2]){
    console.error('usage: node faraday/build-clean.mjs out.html');
    process.exit(2);
  }
  const out = resolve(process.argv[2]);
  const r = buildClean();
  writeFileSync(out, r.out);
  console.log(`wrote ${out}  ${(r.out.length/1024).toFixed(1)} KB; removed ${r.counts.js} JavaScript, `
    + `${r.counts.glsl} GLSL, ${r.counts.css} CSS and ${r.counts.html} HTML comments; `
    + `${r.counts.workerBodies} worker bodies and ${r.counts.shaders} shaders stripped as code`);
}
