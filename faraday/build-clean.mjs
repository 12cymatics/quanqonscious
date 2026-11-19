#!/usr/bin/env node
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

export function buildClean(source){
  const r = stripPage(source === undefined ? buildStandalone() : source);
  if (r.counts.workerBodies !== 3) throw new Error(
    `build-clean: ${r.counts.workerBodies} template worker bodies found where the page builds three.`);
  if (r.counts.shaders !== 4) throw new Error(
    `build-clean: ${r.counts.shaders} GPU shaders found where the two renderers compile four.`);
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
