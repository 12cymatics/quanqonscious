import { readFileSync } from 'node:fs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const NAME = 'QuanQonscious-Simulation-Suite';
export const VERSION = '1.0.1';
export const ZIP = join(REPO, 'product', 'dist', `${NAME}-${VERSION}.zip`);

const SWAPS = [
  [/const ATLAS=\{[^\n]*\};\n/g, 'const ATLAS={};\n'],
  ["small:  { d: 10.00, atlas: 'small',  reps: 1, note: 'Figure 16, one sweep' }",
   "small:  { d: 10.00, atlas: null,     reps: 0, note: 'no atlas in this build' }"],
  ["medium: { d: 24.25, atlas: 'medium', reps: 3, note: 'Figure 11, three day-replicates' }",
   "medium: { d: 24.25, atlas: null,     reps: 0, note: 'no atlas in this build' }"],
  ['<label style="display:block;padding:10px"><input type="checkbox" id="comparePaper">',
   '<label style="display:none"><input type="checkbox" id="comparePaper">'],
  ['simT:0,comparePaper:true,', 'simT:0,comparePaper:false,'],
  ['The paper atlas is optional comparison data and never selects nonlinear patterns.',
   'This build carries no paper atlas.'],
];

export function withoutAtlas(page) {
  let out = page;
  for (const [from, to] of SWAPS) {
    const found = typeof from === 'string' ? out.split(from).length - 1 : (out.match(from) || []).length;
    if (found !== 1) throw new Error(`expected one place to change, found ${found}: ${String(from).slice(0, 60)}`);
    out = typeof from === 'string' ? out.replace(from, () => to) : out.replace(from, () => to);
  }
  return out;
}

export const FILES = [
  ['product/START-HERE.html', 'START-HERE.html'],
  ['product/GUIDE.html', 'GUIDE.html'],
  ['product/LICENSE.txt', 'LICENSE.txt'],
  ['product/THIRD-PARTY-NOTICES.txt', 'THIRD-PARTY-NOTICES.txt'],
  ['vedic_v18.51.1_exact_phi.html', 'vedic-sutra-simulator.html'],
  ['faraday-cell/faraday-cell-standalone.html', 'faraday-cell.html', withoutAtlas],
];

export function productFile([from, , change]) {
  const raw = readFileSync(join(REPO, from));
  return change ? Buffer.from(change(raw.toString('utf8')), 'utf8') : raw;
}
