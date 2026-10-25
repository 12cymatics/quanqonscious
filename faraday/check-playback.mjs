import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const html = readFileSync(fileURLToPath(new URL('../cymatic.html', import.meta.url)), 'utf8');
const extract = (start, end) => {
  const first = html.indexOf(start), last = html.indexOf(end, first);
  assert(first >= 0 && last > first, `page section not found: ${start}`);
  assert.equal(html.indexOf(start, first + 1), -1, `ambiguous page section: ${start}`);
  return html.slice(first, last);
};
const handlers = extract("$('play').addEventListener", "sc.addEventListener");
const clock = extract('function loop(now){', '  if (swapping && running){');
const listeners = new Map(), attributes = new Map();
const button = { textContent: 'running', setAttribute: (k, v) => attributes.set(k, v) };
const context = vm.createContext({
  performance: { now: () => context.now },
  $: id => ({ addEventListener: (event, handler) => listeners.set(`${id}:${event}`, handler) }),
  now: 1000,
});
vm.runInContext('let running = true, t0 = performance.now(), simT = 0, dirty = false, renderError = null;', context);
vm.runInContext(handlers, context);
vm.runInContext(clock + '} finally {} }', context);
const value = expression => vm.runInContext(expression, context);
const frame = time => { context.now = time; value('loop(now)'); };
const click = id => listeners.get(`${id}:click`)({ target: button });
let checks = 0;
const equal = (actual, expected, label) => {
  assert.equal(actual, expected, label);
  console.log(`  ok   ${label}`); checks++;
};
frame(2250);
equal(value('simT'), 1.25, 'running advances the modal clock');
click('play');
equal(value('running'), false, 'pause stops playback');
equal(button.textContent, 'paused', 'pause is visible');
equal(attributes.get('aria-pressed'), 'false', 'pause state is accessible');
frame(12250);
equal(value('simT'), 1.25, 'paused wall time leaves the clock unchanged');
click('play');
equal(value('running'), true, 'resume starts playback');
equal(button.textContent, 'running', 'resume is visible');
equal(attributes.get('aria-pressed'), 'true', 'resume state is accessible');
frame(12500);
equal(value('simT'), 1.5, 'resume excludes ten seconds spent paused');
for (let cycle = 0; cycle < 3; cycle++) {
  click('play');
  const frozen = value('simT');
  frame(context.now + 30000);
  equal(value('simT'), frozen, `pause ${cycle + 2} preserves the clock`);
  click('play');
  frame(context.now + 250);
  equal(value('simT'), frozen + 0.25, `resume ${cycle + 2} excludes paused time`);
}
click('play');
click('reset');
equal(value('simT'), 0, 're-excite resets a paused clock');
equal(value('running'), false, 're-excite preserves paused state');
equal(value('dirty'), true, 're-excite requests a fresh rendering');
frame(context.now + 60000);
equal(value('simT'), 0, 're-excited clock remains paused');
click('play');
frame(context.now + 125);
equal(value('simT'), 0.125, 'resume after paused re-excite starts at zero');
click('reset');
frame(context.now + 500);
equal(value('simT'), 0.5, 're-excite during playback restarts the clock');
console.log(`${checks} playback checks passed, 0 failed`);
