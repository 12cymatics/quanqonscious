#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

GATE_PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "${GITHUB_ACTIONS:-}" != true || "$(id -u)" == 0 || "$(uname -m)" != x86_64 ]]; then
  printf '%s\n' 'Bootstrap requires an unprivileged x86-64 GitHub-hosted Ubuntu runner.' >&2
  exit 1
fi
source /etc/os-release
if [[ "$ID" != ubuntu || "$VERSION_ID" != 26.04 ]]; then
  printf '%s\n' 'This hosted profile requires the explicit ubuntu-26.04 runner.' >&2
  exit 1
fi
if [[ -n "${OPENAI_API_KEY:-}" || -n "${GATE_AUTHORITY_KEY_B64:-}" ]]; then
  printf '%s\n' 'Build and preflight must run without API or signing secrets.' >&2
  exit 1
fi
GATE_BUILD="$(mktemp -d "${RUNNER_TEMP:?}/verified-agent-gate-build.XXXXXXXX")"
sudo apt-get update
sudo apt-get install --yes --no-install-recommends \
  build-essential python3 python3-venv bubblewrap apparmor apparmor-utils \
  curl ca-certificates xz-utils zstd

GATE_BWRAP_VERSION="$(dpkg-query -W -f='${Version}' bubblewrap)"
dpkg --compare-versions "$GATE_BWRAP_VERSION" ge '0.11.1'
/usr/bin/python3 - <<'PY'
import subprocess
help_text = subprocess.check_output(['/usr/bin/bwrap', '--help'], text=True)
required = ['--ro-bind-fd', '--bind-fd', '--disable-userns', '--assert-userns-disabled', '--seccomp', '--die-with-parent']
if any(value not in help_text for value in required):
    raise SystemExit('Installed bubblewrap lacks a mandatory sandbox interface.')
PY

if [[ -r /sys/module/apparmor/parameters/enabled ]] && [[ "$(cat /sys/module/apparmor/parameters/enabled)" == Y ]]; then
  GATE_DISTRO_PROFILE=/etc/apparmor.d/bwrap-userns-restrict
  if [[ ! -f "$GATE_DISTRO_PROFILE" ]]; then
    printf '%s\n' 'The required distribution bubblewrap AppArmor profile is unavailable.' >&2
    exit 1
  fi
  dpkg-query -S "$GATE_DISTRO_PROFILE"
  sudo apparmor_parser --replace "$GATE_DISTRO_PROFILE"
fi

GATE_LEAN_ARCHIVE="$GATE_BUILD/lean-4.36.0-linux.tar.zst"
curl --fail --location --proto '=https' --tlsv1.2 \
  'https://github.com/leanprover/lean4/releases/download/v4.36.0/lean-4.36.0-linux.tar.zst' \
  --output "$GATE_LEAN_ARCHIVE"
printf '%s  %s\n' '2b7ab1ed8b92d2e6abfaf41344823b510064dbc9c0260cac73eb496ab2dea6eb' "$GATE_LEAN_ARCHIVE" | sha256sum --check --strict
tar --extract --file "$GATE_LEAN_ARCHIVE" --directory "$GATE_BUILD" --no-same-owner --no-same-permissions
GATE_LEAN_ROOT="$GATE_BUILD/lean-4.36.0-linux"
test -x "$GATE_LEAN_ROOT/bin/lean"
LEAN_TOOLCHAIN_ROOT="$GATE_LEAN_ROOT" bash "$GATE_PACKAGE/lean/build.sh"
make -C "$GATE_PACKAGE/native" clean all

/usr/bin/python3 -m venv "$GATE_BUILD/controller-venv"
GATE_PYTHON="$GATE_BUILD/controller-venv/bin/python"
"$GATE_PYTHON" -m pip install --disable-pip-version-check 'cryptography==50.0.0'
"$GATE_PYTHON" "$GATE_PACKAGE/lean/test_kernel.py"
"$GATE_PYTHON" "$GATE_PACKAGE/native/test_native.py"

"$GATE_PYTHON" - "$GATE_PACKAGE" "$GATE_BUILD/build-provenance.json" "$GATE_BWRAP_VERSION" <<'PY'
import hashlib, json, os, platform, subprocess, sys
from pathlib import Path
root, target, bwrap = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
paths = ['gate3.py', 'lean/Main.lean', 'lean/gate-kernel', 'native/launcher.c', 'native/gate-sandbox', 'native/libgate_seccomp.so', 'native/bootstrap.py']
record = {
    'schema': 'verified-agent-gate-hosted-build-v1',
    'runner_os': platform.platform(), 'architecture': platform.machine(),
    'glibc': platform.libc_ver(), 'system_python': subprocess.check_output(['/usr/bin/python3', '--version'], text=True).strip(),
    'bubblewrap_package': bwrap, 'lean_version': '4.36.0',
    'lean_archive_sha256': '2b7ab1ed8b92d2e6abfaf41344823b510064dbc9c0260cac73eb496ab2dea6eb',
    'commit': os.environ.get('GITHUB_SHA'), 'run_id': os.environ.get('GITHUB_RUN_ID'),
    'run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT'),
    'file_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths},
    'native_and_kernel_preflight': 'passed',
}
target.write_text(json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n', encoding='utf-8')
PY
printf 'GATE_PYTHON=%s\n' "$GATE_PYTHON" >> "${GITHUB_ENV:?}"
printf 'GATE_BUILD_PROVENANCE=%s\n' "$GATE_BUILD/build-provenance.json" >> "$GITHUB_ENV"
printf '%s\n' 'Mandatory Lean-kernel and native-sandbox preflight passed.'
