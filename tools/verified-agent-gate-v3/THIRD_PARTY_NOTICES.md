# Third-party components

The compiled Lean executable contains code from the Lean runtime and standard library. Lean's Apache 2.0 license and the distribution's additional notices are preserved in `licenses/Lean-LICENSE.txt` and `licenses/Lean-LICENSES.txt`.

The compiler provenance, official release URL, measured archive SHA-256, and generated artifact hashes are recorded in `lean/toolchain-provenance.json` and `lean/build-artifacts.json`. The full compiler distribution is not included in this package.

The controller uses the separately installed `cryptography` package. The native launcher uses the separately installed Linux kernel, libc, system Python, and bubblewrap. These system dependencies retain their own licenses; their complete distributions are not bundled here.

Primary implementation references:

- Lean language and compiler: https://lean-lang.org/doc/reference/latest/
- Pinned Lean release: https://github.com/leanprover/lean4/releases/tag/v4.36.0
- Linux seccomp filters: https://docs.kernel.org/userspace-api/seccomp_filter.html
- Bubblewrap: https://github.com/containers/bubblewrap
- Ed25519 interface: https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ed25519/
