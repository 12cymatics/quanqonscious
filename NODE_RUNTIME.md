# Node.js runtime

The simulator build, diagnostics and JavaScript tests use Node.js 26.13.0 LTS,
pinned in `.nvmrc`. With nvm installed, run `nvm install` and `nvm use` in this
repository. `node tools/check_node_runtime.mjs` prints and verifies the runtime.
The package engine range is Node 26; CI uses the exact `.nvmrc` version.

Migrated simulator workflows use the official setup-node v7 action, pinned to commit
`949feb2413d6458794dcd2491c4babbbce0c15c1`. Its internal Node 24 runtime executes
the action itself; the simulator and subsequent shell commands run Node 26.

The rigid-cup reconstruction workflow runs focused regressions, continuum
checks, short free/pinned full-Navier–Stokes smoke cases, fresh WASM parity,
worker pools, browser checks and every suite from a freshly rebuilt unpacked
bundle. The 8-case long water sweep and historical pattern runs remain manual.
Saved-water report regeneration is manual and does not rerun physical cases.
No solver equations or numerical pass thresholds change with this migration.
Generated WASM/bundle/ZIP outputs remain CI artifacts, not published binaries.

The root React/lucide package manifest and lockfile already disagree on
versions. This runtime migration changes only their engine metadata, not their
dependency versions. The dependency-free simulator suites are the runtime
compatibility gates; root `npm test` remains the pre-existing placeholder.

Official references:
- https://nodejs.org/en/blog/release/v26.13.0
- https://github.com/actions/setup-node/tree/v7

The dormant eight-case sweep and unrelated PyPI workflow retain their prior
Node 22 setup pending separate workflow-upload authorization. Neither is run
by this migration.
