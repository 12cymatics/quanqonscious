import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, cpSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const root=fileURLToPath(new URL('../',import.meta.url));
const original=readFileSync(join(root,'dns/faraday-cell3d.js'),'utf8');
const start=original.indexOf('      const ds = sn[hi] - sn[lo], t = (ss - sn[lo])/ds;');
const end=original.indexOf('    } else j0 = (lev === undefined ? 1 : lev) - 1;',start);
assert.ok(start>=0&&end>start,'current bracket implementation not found');
const mutations=[
  ['old-stencil',original.slice(0,start)+'      j0 = lo - 1;\n'+original.slice(end)],
  ['wrong-shared-slope',original.replace('const m1 = this.colKnotSlope(f, sn, nJ, base, hi);','const m1 = this.colKnotSlope(f, sn, nJ, base, lo);')],
  ['missing-sigma-scale',original.replace('(t3 - 2*t2 + t)*ds*m0','(t3 - 2*t2 + t)*m0')],
  ['clamped-extrapolation',original.replace('t = (ss - sn[lo])/ds;','t = Math.max(0, Math.min(1, (ss - sn[lo])/ds));')]
];
for(const [name,source] of mutations){
  assert.notEqual(source,original,`${name} did not mutate source`);
  const temp=mkdtempSync(join(tmpdir(),'cymatica-c1-'));
  try{
    cpSync(join(root,'dns'),join(temp,'dns'),{recursive:true});
    const path=join(temp,'dns/faraday-cell3d.js');
    writeFileSync(path,source);
    const run=spawnSync(process.execPath,[join(root,'tools/check_cell3d_reconstruction.mjs')],{
      env:{...process.env,CYMATICA_SOLVER_PATH:path},encoding:'utf8',maxBuffer:16*1024*1024,timeout:120000});
    assert.equal(run.error,undefined,`${name}: transport error`);
    assert.notEqual(run.status,0,`${name}: mutation escaped`);
    assert.match(run.stderr,/AssertionError/);
    assert.match(run.stderr,/C1 n=|polynomial n=/);
    console.log(JSON.stringify({mutation:name,rejected:true,exit:run.status,stderr:run.stderr}));
  }finally{rmSync(temp,{recursive:true,force:true});}
}
console.log(JSON.stringify({suite:'bracket-mutations',passed:mutations.length}));
