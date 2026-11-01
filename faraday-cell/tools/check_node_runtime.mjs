import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const expected=readFileSync(new URL('../.nvmrc',import.meta.url),'utf8').trim();
console.log(JSON.stringify({node:process.version,expected,platform:process.platform,arch:process.arch,versions:process.versions},null,2));
assert.equal(process.versions.node,expected,'selected Node runtime matches .nvmrc');
assert.equal(Number(process.versions.node.split('.')[0]),26,'Node.js 26 is required');
assert.equal(typeof WebSocket,'function','native WebSocket supports browser checks');
assert.equal(typeof fetch,'function','native fetch is available');
