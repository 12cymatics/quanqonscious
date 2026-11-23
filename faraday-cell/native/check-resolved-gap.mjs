import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const directory = dirname(fileURLToPath(import.meta.url));
const scratch = mkdtempSync(join(tmpdir(), 'faraday-resolved-gap-'));
try {
  const executable = join(scratch, 'check-resolved-gap');
  const compiler = spawnSync(process.env.CXX || 'c++', [
    '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic',
    join(directory, 'check-resolved-gap.cpp'), '-o', executable,
  ], { encoding: 'utf8' });
  if (compiler.error || compiler.status !== 0) {
    throw new Error(compiler.error?.message || compiler.stderr || 'Compiler failed');
  }
  const check = spawnSync(executable, [], { encoding: 'utf8' });
  if (check.error || check.status !== 0) {
    throw new Error(check.error?.message || check.stderr || 'Geometry check failed');
  }
  const result = JSON.parse(check.stdout.trim());
  if (!result.ok) throw new Error('Geometry check did not pass');
  console.log(JSON.stringify(result));
} finally {
  rmSync(scratch, { recursive: true, force: true });
}
