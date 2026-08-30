"""Every sutra in `primarysutra.py` runs, in every execution mode.

Nothing tested this file before. It hard-imports cirq, cudaq and torch (see
`_enforce_heavy_dependencies`), so on a machine without them it does not
import at all, and no test in the repository referenced it. That is how eight
of its sixteen sutras came to raise in QUANTUM and HYBRID mode without anyone
noticing:

  * `_nikhilam_quantum` and `_nikhilam_hybrid` were dispatched to and NEVER
    DEFINED -- sutra 2 had no working quantum path;
  * every cudaq rotation in the file passed its arguments in the wrong order
    (`ry(qubit, angle)`; the builder takes `ry(parameter, target)`), all seven
    of them, with no correct call anywhere to compare against;
  * `cudaq.inverseFQFT` was called twice and does not exist in CUDA-Q;
  * `cirq.TOFFOLI(q[i-1], q[i], q[i])` named its target as its own control;
  * `.controlled_by(...)` -- a cirq method -- was called on the return of a
    cudaq builder call, which is None;
  * `cirq.Rz(rads)`, `sample(shots=)` and `SampleResult.get_counts()` are all
    APIs that do not exist in the installed versions.

None of those can be reached without importing the module, so the gate here is
simply that it imports and that every sutra runs. The arithmetic identities
below then pin the answers that are exactly determined.
"""
import itertools
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import primarysutra as ps

MODES = list(ps.SutraMode)

# One call per public sutra, with arguments in range for each.
CALLS = {
    "ekadhikena_purvena":                lambda v, c: v.ekadhikena_purvena(7.0, iterations=2, ctx=c),
    "nikhilam_navatashcaramam_dashatah": lambda v, c: v.nikhilam_navatashcaramam_dashatah(97.0, base=100.0, ctx=c),
    "paravartya_yojayet":                lambda v, c: v.paravartya_yojayet(12.0, 4.0, ctx=c),
    "shunyam_samyasamuccaye":            lambda v, c: v.shunyam_samyasamuccaye(3.0, 3.0, ctx=c),
    "sunyam_samya_samuccaye":            lambda v, c: v.sunyam_samya_samuccaye(3.0, 3.0, ctx=c),
    "anurupyena":                        lambda v, c: v.anurupyena(8.0, 4.0, ctx=c),
    "sankalana_vyavakalanabhyam":        lambda v, c: v.sankalana_vyavakalanabhyam(9.0, 4.0, ctx=c),
    "purna_apurna_bhyam":                lambda v, c: v.purna_apurna_bhyam(0.7, ctx=c),
    "chalana_kalana":                    lambda v, c: v.chalana_kalana(2.0, steps=3, ctx=c),
    "yavadunam":                         lambda v, c: v.yavadunam(8.0, base=10.0, ctx=c),
    "vyashtisamanstih":                  lambda v, c: v.vyashtisamanstih(10.0, [2.0, 3.0, 5.0], ctx=c),
    "sesanyankena_caramena":             lambda v, c: v.sesanyankena_caramena([1.0, 2.0, 3.0], 2.0, ctx=c),
    "ekanyunena_purvena":                lambda v, c: v.ekanyunena_purvena(9.0, base=10.0, ctx=c),
    "gunitasamuccayah":                  lambda v, c: v.gunitasamuccayah(6.0, 7.0, ctx=c),
    "gunakasamuccayah":                  lambda v, c: v.gunakasamuccayah(6.0, 7.0, ctx=c),
    "samuccayagunitah":                  lambda v, c: v.samuccayagunitah(6.0, 7.0, ctx=c),
}


def test_the_class_exposes_sixteen_sutras():
    """Sixteen, not twenty-nine: this class has no sub-sutras.

    `CLAUDE.md` lists this file alongside the 29 and calls it "the main
    VedicSutras class", which reads as though the 29 live here. They do not --
    the 13 sub-sutras are in `core/operators/sutra_ops.py` and
    `vedic_trainer/vedic/kernel/sutras_canonical.py`.
    """
    import inspect
    public = sorted(
        name for name, _ in inspect.getmembers(ps.VedicSutras, predicate=inspect.isfunction)
        if not name.startswith("_")
    )
    assert public == sorted(CALLS), f"the public sutra set changed: {public}"
    assert len(public) == 16


def test_every_sutra_runs_in_every_mode():
    """80 (sutra, mode) pairs. 16 of them raised before this was written."""
    v = ps.VedicSutras()
    failures = []
    for mode, (name, call) in itertools.product(MODES, CALLS.items()):
        try:
            call(v, ps.SutraContext(mode=mode))
        except Exception as exc:                      # noqa: BLE001 - report, don't mask
            failures.append(f"{name} in {mode.name}: {type(exc).__name__}: {exc}")
    assert not failures, (
        f"{len(failures)} of {len(MODES) * len(CALLS)} (sutra, mode) pairs raised:\n  "
        + "\n  ".join(failures)
    )


def _check(fn, expected, *args, **kwargs):
    got = fn(*args, **kwargs)
    assert float(got) == float(expected), f"{fn.__name__}{args}: got {got}, expected {expected}"


def test_the_exactly_determined_identities_hold_in_every_mode():
    """Four sutras have one right answer, so they are pinned across modes.

    Each was wrong in QUANTUM and HYBRID: ekadhikena returned 256 for
    7 + 1 + 1 (it read the 2**n state-vector amplitudes as if they were the n
    bits of the answer), ekanyunena returned x rather than x - 1 (its
    decrementer complemented all n qubits and un-complemented only n - 1, and
    rippled low-to-high so each carry control was read after being modified),
    nikhilam raised, and yavadunam returned 0 (bare CNOTs as carries).
    """
    v = ps.VedicSutras()
    for mode in MODES:
        ctx = ps.SutraContext(mode=mode)
        for x, iters in ((7, 2), (15, 1), (0, 3)):
            _check(v.ekadhikena_purvena, x + iters, float(x), iterations=iters, ctx=ctx)
        for x in (1, 5, 9, 16, 31):
            _check(v.ekanyunena_purvena, x - 1, float(x), base=10.0, ctx=ctx)
        for base in (10, 100):
            for x in (0, base // 2, base):
                _check(v.nikhilam_navatashcaramam_dashatah, base - x,
                       float(x), base=float(base), ctx=ctx)
                _check(v.yavadunam, base - x, float(x), base=float(base), ctx=ctx)


def test_the_inverse_qft_helper_inverts():
    """`cudaq.inverseFQFT` does not exist; this is what replaced it.

    Asserted as a genuine inverse rather than merely as something that runs:
    QFT followed by it must return every basis state unchanged.
    """
    import cudaq
    import numpy as np

    n = 3
    for basis in range(2 ** n):
        kernel = cudaq.make_kernel()
        q = kernel.qalloc(n)
        for bit in range(n):
            if (basis >> bit) & 1:
                kernel.x(q[bit])
        for i in range(n):                                   # forward QFT
            kernel.h(q[i])
            for j in range(i + 1, n):
                kernel.cr1(np.pi / float(2 ** (j - i)), [q[j]], q[i])
        for i in range(n // 2):
            kernel.swap(q[i], q[n - 1 - i])
        ps._cudaq_inverse_qft(kernel, q, n)
        kernel.mz(q)

        counts = dict(cudaq.sample(kernel, shots_count=200).items())
        assert len(counts) == 1, f"|{basis}> did not come back to a basis state: {counts}"
        top = next(iter(counts))
        recovered = sum(int(top[b]) << b for b in range(n))
        assert recovered == basis, f"QFT then inverse took |{basis}> to |{recovered}>"


if __name__ == "__main__":
    failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"  ✓ {_name}")
            except AssertionError as exc:
                failures += 1
                print(f"  ✗ {_name}\n      {exc}")
    print(f"\n{'FAILED' if failures else 'OK'}: {failures} failure(s)")
    sys.exit(1 if failures else 0)
