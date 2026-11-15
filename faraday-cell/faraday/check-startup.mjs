import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const html = readFileSync(new URL('../cymatic.html', import.meta.url), 'utf8');
const block = html.match(/<script id="startup">([\s\S]*?)<\/script>/);
assert(block, 'startup diagnostics script exists');
assert(html.includes('Not started. Open index.html in a full browser'), 'static preview explains why nothing runs');
assert(html.includes('<noscript>'), 'JavaScript-disabled browsers receive an explicit notice');
assert(html.includes('name="viewport"'), 'mobile viewport is declared');
assert(html.includes('<canvas id="scene3d"'), 'the interactive three-dimensional canvas exists');
assert.match(html, /let useSolver = true;/, 'the visible application starts from nonlinear state');
const readinessStart = html.indexOf('  if(useSolver){\n    if(UNIFIED3D.tick(now)){');
const readinessEnd = html.indexOf('}else FARADAY_STARTUP.ready();', readinessStart);
assert(readinessStart >= 0 && readinessEnd > readinessStart, 'the loop gates startup on the real scene draw');
const readiness = html.slice(readinessStart, readinessEnd + '}else FARADAY_STARTUP.ready();'.length);
const autostart = html.match(/let initialAutostart = new URLSearchParams\(location.search\)[^;]+;/);
assert(autostart, 'the page supplies an explicit initial-state inspection option');
const status = { textContent: '', style: {} }, handlers = new Map();
const calls = [];
const context = vm.createContext({
  document: { getElementById: id => { assert.equal(id, 'startupStatus'); return status; } },
  window: { addEventListener: (name, callback) => handlers.set(name, callback) },
  $: id => { assert.equal(id, 'experimentError'); return { textContent: '' }; },
  UNIFIED3D: { tick: () => { calls.push('draw'); return context.drawReady; } },
  CELL3D: { start: () => calls.push('start') },
  URLSearchParams, location: { search: '' }, drawReady: false,
});
vm.runInContext(block[1], context);
vm.runInContext('let useSolver = true, now = 0, clock = { err:null }, solverRequested = false, state = {};'
  + autostart[0], context);
const ready = context.FARADAY_STARTUP.ready.bind(context.FARADAY_STARTUP);
context.FARADAY_STARTUP.ready = () => { calls.push('ready'); ready(); };
assert.equal(status.textContent, 'Starting the simulator…');
assert.equal(context.FARADAY_STARTUP.drawn, false);
vm.runInContext(readiness, context);
assert.deepEqual(calls, ['draw'], 'an incomplete scene cannot announce ready or start integration');
assert.equal(context.FARADAY_STARTUP.drawn, false);
context.drawReady = true;
vm.runInContext(readiness, context);
assert.deepEqual(calls, ['draw', 'draw', 'ready', 'start'], 'integration begins only after a successful scene draw');
assert.match(status.textContent, /3D cell ready/);
assert.equal(context.FARADAY_STARTUP.drawn, true);
assert.match(status.textContent, /Surface and flow tracers share the nonlinear solver state/);
vm.runInContext(readiness, context);
assert.equal(calls.filter(call => call === 'start').length, 1, 'subsequent frames cannot restart the solver');
context.location.search = '?paused=1';
vm.runInContext(autostart[0].replace('let initialAutostart', 'initialAutostart'), context);
vm.runInContext(readiness, context);
assert.equal(calls.filter(call => call === 'start').length, 1, 'initial-state inspection does not start integration');
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
console.log('Startup diagnostics checks passed, 0 failed');
