# CI

One workflow gates this package: `.github/workflows/submit-pypi.yml`.

## What it runs

1. **Installs Lean 4** at the version pinned by `vedic_trainer/lean-toolchain`,
   and verifies `lean --version` matches that pin. Lean is a hard requirement
   of the suite, not an optional extra: the Lean mirror is the one independent
   cross-check of the exact-ℚ kernel, and its tests used to carry `skipif`
   guards that turned "the compiler is absent" into a green run — so the
   mirror sat broken, unable to resolve a toolchain at all, without anyone
   noticing.
2. **Builds** the `vedic_trainer` sdist + wheel.
3. **Bit-exact ℚ gate** — `scripts/verify_bit_exact.py`. Refuses to run at all
   against a missing reference rather than rebuilding one from the code under
   test.
4. **Full test suite** — `pytest -q`.
5. **README count gate** — `scripts/verify_counts.py --check`. This cannot
   live inside the suite: `verify_counts.py` runs pytest, so a test calling it
   would recurse. The suite exercises its judgment in isolation, and this step
   is the only place that judgment meets the real README. Its first run caught
   a defect in the gate itself.
6. **PyPI upload**, on tag pushes only (`refs/tags/v*`), when `PYPI_API_TOKEN`
   is configured.

If a step is added to the workflow and not to this list, the list is wrong and
nothing here will say so. Read the workflow when the two disagree.

`.github/workflows/python-app.yml` also exists. It targets the **parent**
repository, not this package: `main` only, CUDA + CuPy, and a bare `pytest` at
the repository root.

## The `submit-pypi-override` status, and why it is gone

An opaque `submit-pypi` check, produced by no workflow in this repository,
used to fail on every commit and block PRs informationally. Two pieces of
machinery grew around it: a `submit-pypi-override` Commit Status posted by our
own workflow, and an `external-submit-pypi-watchdog.yml` that watched for the
external check's failure, mirrored our internal result into that status, and
left one explanatory comment per PR.

Both are **deleted**. What replaced them is nothing: the repo-internal
`submit-pypi` job is the gate, and it reports as itself.

The removal is worth a note because a shadow status is a genuinely bad thing
to leave lying around. It is a second name for a verdict, posted by a
different job, on the same commit — so a reader has two greens and no way to
tell which one the branch actually requires, and any drift between them is
invisible. The watchdog had already been caught posting green while the
repo-internal gate had failed, and was patched to mirror it. A mechanism that
needs that patch is a mechanism whose default is to lie.

**One thing to check if PRs start hanging.** If branch protection was ever
configured to *require* the `submit-pypi-override` status, nothing posts it
any more, so it will sit pending forever and no PR will merge. Point branch
protection at this repository's own `submit-pypi` check instead. That is the
only foreseeable consequence of this deletion, and it is a settings change,
not a code one.
