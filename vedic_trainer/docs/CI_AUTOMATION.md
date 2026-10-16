# CI

Two workflows run this package's tests: `.github/workflows/submit-pypi.yml`,
described here, and `.github/workflows/python-app.yml`, whose bare `pytest` at
the repository root collects them too (see the end of this page).

## What it runs

0. **Installs the training stack** at the exact pins in `requirements.txt`
   (torch from the CPU wheel index). It used to install only the exact-ℚ
   kernel's dependencies, so the HF-integration layer (`scripts/train_lora.py`,
   `vedic/training/trainer.py`, the SCAN and COGS loaders, the eval scripts)
   was never imported in CI, and the trainer could not start under the
   transformers a fresh install resolved to while this job stayed green. Some tests read the
   SmolLM2 tokenizer at the pinned revision from the HF Hub.
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

### The `formal-comparator` job

A separate job in the same workflow checks the formal statements under
`formal/`, as rule C1 of `CODEX_OPENAI_MATH.md` (repository root) requires:

1. **Installs Lean v4.34.1**, read from `formal/lean-toolchain` — openai/math's
   pin, not this package's v4.10.0.
2. **Builds comparator, lean4export and landrun** at the commits pinned in the
   job's `env`, compiling the two Lean tools against that same Lean.
3. **Fetches Mathlib's prebuilt files** (`lake exe cache get`). It builds
   nothing of `formal/` itself first: Comparator's trust argument assumes the
   checker has not already compiled the solution it is judging.
4. **Runs Comparator on every `formal/ComparatorChallenges/*.json`**, and fails
   if that glob matches nothing rather than passing vacuously.

What the formal results do and do not cover is in `formal/docs/VedicResiduals.md`.

`.github/workflows/python-app.yml` also gates this package, which this page
used to deny ("One workflow gates this package"; "It targets the **parent**
repository"). It runs a bare `pytest` at the repository root on pushes to the
default branch and on every pull request, with CPU torch, and that run
collects this package's tests too, the Lean-mirror tests among them. It
therefore installs the same pinned training stack. This paragraph also said
"`main` only, CUDA + CuPy"; the repository has no `main` branch, and the CUDA
toolkit step was removed from that workflow.

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
