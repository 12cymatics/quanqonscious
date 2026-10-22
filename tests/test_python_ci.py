from pathlib import Path
import os
import subprocess

import pytest
import yaml


WORKFLOW = Path(__file__).resolve().parents[1] / '.github/workflows/python-app.yml'


@pytest.mark.parametrize(
    ('reported_version', 'expected_returncode'),
    [
        ('4.19.0', 0),
        ('4.19.0,', 0),
        ('4.19.01,', 1),
        ('4x19x0,', 1),
        ('4.19.0-rc1,', 1),
        ('4.20.0,', 1),
    ],
)
def test_ci_requires_exact_lean_version(tmp_path, reported_version, expected_returncode):
    workflow = yaml.safe_load(WORKFLOW.read_text())
    step = next(
        step for step in workflow['jobs']['build']['steps']
        if step.get('name') == 'Verify Lean matches the pin'
    )
    (tmp_path / 'lean-toolchain').write_text('leanprover/lean4:v4.19.0\n')
    lean = tmp_path / 'lean'
    lean.write_text(f'#!/bin/sh\nprintf "Lean (version {reported_version} x86_64-linux)\\n"\n')
    lean.chmod(0o755)
    result = subprocess.run(
        ['bash', '-e', '-o', 'pipefail', '-c', step['run']],
        cwd=tmp_path,
        env={**os.environ, 'PATH': f'{tmp_path}{os.pathsep}{os.environ["PATH"]}'},
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected_returncode, result.stdout + result.stderr
