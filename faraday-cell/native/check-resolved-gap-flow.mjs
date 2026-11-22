import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const directory = dirname(fileURLToPath(import.meta.url));
const scratch = mkdtempSync(join(tmpdir(), 'faraday-resolved-gap-flow-'));
try {
  const executable = join(scratch, 'check-resolved-gap-flow');
  const compiler = spawnSync(process.env.CXX || 'c++', [
    '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic',
    join(directory, 'resolved-gap-flow.cpp'), join(directory, 'check-resolved-gap-flow.cpp'), '-o', executable,
  ], { encoding: 'utf8' });
  if (compiler.error || compiler.status !== 0) throw new Error(compiler.error?.message || compiler.stderr || 'Compiler failed');
  const check = spawnSync(executable, [], { encoding: 'utf8' });
  if (check.stdout) process.stdout.write(check.stdout);
  if (check.error || check.status !== 0) throw new Error(check.error?.message || check.stderr || 'Mapped full NS check failed');
  const reports = check.stdout.trim().split('\n').map(line => JSON.parse(line));
  if (!reports.at(-1)?.ok) throw new Error('Mapped full NS check did not pass');
} finally {
  rmSync(scratch, { recursive: true, force: true });
}
