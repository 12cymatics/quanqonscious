import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const html = readFileSync(new URL('../cymatic.html', import.meta.url), 'utf8');
const block = html.match(/<script id="startup">([\s\S]*?)<\/script>/);
assert(block, 'startup diagnostics script exists');
assert(html.includes('Not started. Open index.html in a full browser'), 'static preview explains why nothing runs');
assert(html.includes('<noscript>'), 'JavaScript-disabled browsers receive an explicit notice');
assert(html.includes('name="viewport"'), 'mobile viewport is declared');
const status = { textContent: '', style: {} }, handlers = new Map();
const context = vm.createContext({
  document: { getElementById: id => { assert.equal(id, 'startupStatus'); return status; } },
  window: { addEventListener: (name, callback) => handlers.set(name, callback) }
});
vm.runInContext(block[1], context);
assert.equal(status.textContent, 'Starting the simulator…');
assert.equal(context.FARADAY_STARTUP.drawn, false);
context.FARADAY_STARTUP.ready();
assert.match(status.textContent, /Canvas rendering started/);
assert.equal(context.FARADAY_STARTUP.drawn, true);
assert.match(status.textContent, /physical time/);
handlers.get('error')({ message: 'required kernel missing' });
assert.match(status.textContent, /required kernel missing/);
assert.match(status.textContent, /No substitute/);
assert.equal(context.FARADAY_STARTUP.failed, true);
context.FARADAY_STARTUP.ready();
assert.match(status.textContent, /required kernel missing/, 'a subsequent frame does not hide the error');
handlers.get('unhandledrejection')({ reason: new Error('worker unavailable') });
assert.match(status.textContent, /worker unavailable/);
assert.equal(context.FARADAY_STARTUP.failed, true);
context.FARADAY_STARTUP.ready();
assert.match(status.textContent, /worker unavailable/, 'the rejected worker remains visible');
const render = html.indexOf('  renderSurface(state);', html.indexOf('function loop(now)'));
const ready = html.indexOf('  FARADAY_STARTUP.ready();', render);
assert(render >= 0 && ready > render, 'ready status follows the draw call');
console.log('Startup diagnostics checks passed, 0 failed');
