'use strict';

const FLOAT_TYPES = new Set([0x7D, 0x7C]);
const SINGLE_FLOAT = new Set([0x2A, 0x2B, 0x38, 0x39, 0x43, 0x44]);
const isFloatOpcode = op => SINGLE_FLOAT.has(op) || (op >= 0x5B && op <= 0x66) || (op >= 0x8B && op <= 0xA6)
  || (op >= 0xA8 && op <= 0xAB) || (op >= 0xAE && op <= 0xBF);

function scanWasm(bytes){
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let p = 0;
  const fail = msg => { throw new Error(`wasm-float-scan: ${msg} at byte ${p}.`); };
  const byte = () => { if (p >= u8.length) fail('unexpected end of module'); return u8[p++]; };
  const uleb = () => { let r = 0, sh = 0, b; do { b = byte(); r += (b & 0x7f)*Math.pow(2, sh); sh += 7; } while (b & 0x80); return r; };
  const sleb = () => { let b; do { b = byte(); } while (b & 0x80); };
  const res = { functions: 0, instructions: 0, floatInstructions: 0, floatTypes: 0, floatOpcodes: {} };
  const noteFloat = op => { res.floatInstructions++; res.floatOpcodes[op] = (res.floatOpcodes[op] || 0) + 1; };
  const valtype = () => { const t = byte(); if (FLOAT_TYPES.has(t)) res.floatTypes++; return t; };
  const blocktype = () => {
    const t = u8[p];
    if (t === 0x40){ p++; return; }
    if (t === 0x7F || t === 0x7E || t === 0x7D || t === 0x7C || t === 0x7B || t === 0x70 || t === 0x6F){ valtype(); return; }
    sleb();
  };
  function expr(end){
    let depth = 1;
    while (p < end){
      const op = byte();
      res.instructions++;
      if (isFloatOpcode(op)) noteFloat(op);
      if (op === 0x02 || op === 0x03 || op === 0x04){ blocktype(); depth++; continue; }
      if (op === 0x0B){ depth--; if (depth === 0) return; continue; }
      if (op === 0x00 || op === 0x01 || op === 0x05 || op === 0x0F || op === 0x1A || op === 0x1B || op === 0xD1) continue;
      if (op === 0x0C || op === 0x0D || op === 0x10 || (op >= 0x20 && op <= 0x26) || op === 0xD2){ uleb(); continue; }
      if (op === 0x0E){ const n = uleb(); for (let i = 0; i <= n; i++) uleb(); continue; }
      if (op === 0x11){ uleb(); uleb(); continue; }
      if (op === 0x1C){ const n = uleb(); for (let i = 0; i < n; i++) valtype(); continue; }
      if (op >= 0x28 && op <= 0x3E){ uleb(); uleb(); continue; }
      if (op === 0x3F || op === 0x40){ uleb(); continue; }
      if (op === 0x41 || op === 0x42){ sleb(); continue; }
      if (op === 0x43){ p += 4; continue; }
      if (op === 0x44){ p += 8; continue; }
      if (op >= 0x45 && op <= 0xC4) continue;
      if (op === 0xD0){ byte(); continue; }
      if (op === 0xFC){
        const sub = uleb();
        if (sub <= 7){ noteFloat(0xFC00 + sub); continue; }
        if (sub === 8){ uleb(); uleb(); continue; }
        if (sub === 9 || sub === 11 || sub === 13 || sub === 15 || sub === 16 || sub === 17){ uleb(); continue; }
        if (sub === 10 || sub === 12 || sub === 14){ uleb(); uleb(); continue; }
        fail(`unknown 0xFC sub-opcode ${sub}`);
      }
      fail(`opcode 0x${op.toString(16)} is outside the instruction set this scan decodes`);
    }
    fail('function body ended without its end opcode');
  }
  if (u8[0] !== 0x00 || u8[1] !== 0x61 || u8[2] !== 0x73 || u8[3] !== 0x6D) fail('not a WebAssembly module');
  p = 8;
  while (p < u8.length){
    const id = byte(), size = uleb(), end = p + size;
    if (id === 1){
      const n = uleb();
      for (let i = 0; i < n; i++){
        if (byte() !== 0x60) fail('function type expected');
        const np = uleb(); for (let k = 0; k < np; k++) valtype();
        const nr = uleb(); for (let k = 0; k < nr; k++) valtype();
      }
    } else if (id === 6){
      const n = uleb();
      for (let i = 0; i < n; i++){ valtype(); byte(); expr(end); }
    } else if (id === 10){
      const n = uleb();
      for (let i = 0; i < n; i++){
        const bodySize = uleb(), bodyEnd = p + bodySize;
        const groups = uleb();
        for (let g = 0; g < groups; g++){ uleb(); valtype(); }
        expr(bodyEnd);
        if (p !== bodyEnd) fail('function body did not end where its size says');
        res.functions++;
      }
    }
    p = end;
  }
  return res;
}

const WASM_FLOAT_SCAN = Object.freeze({ scanWasm, isFloatOpcode });
if (typeof module !== 'undefined' && module.exports) module.exports = WASM_FLOAT_SCAN;
if (typeof globalThis !== 'undefined') globalThis.WASM_FLOAT_SCAN = WASM_FLOAT_SCAN;
