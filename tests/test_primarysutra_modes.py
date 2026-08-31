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
  * `cudaq.inverseFQFT` was called twice and does not exist in CUDA-Q --
    both call sites have since been removed outright, because both were
    approximations (an 8-bit phase estimate) rather than arithmetic;
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
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

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
    """All 16 sutras x all 5 modes. 16 of the 80 pairs raised before this.

    80 pairs, but not 80 distinct code paths. `SutraMode` declares five
    members and every dispatcher in `primarysutra.py` branches on only three:
    `MAYA_ILLUSION` and `SULBA` appear at lines 135-136 of that file and
    nowhere else in it, so both fall through to the classical body. 32 of
    these 80 pairs therefore re-run code the CLASSICAL 16 already cover, and
    the distinct-path count is 48. Both modes are still exercised here,
    because "declared and silently aliased to CLASSICAL" is a fact about the
    file worth pinning rather than hiding -- but the coverage claim is 48.
    """
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
        # Values either side of 127. Every bit read out of a cirq measurement
        # is an `np.int8`, and three of the four places this file accumulates
        # `sum(bit * 2**i)` were missing the `int()` cast that the fourth
        # (line 672) has. On numpy 2 that raises OverflowError at x = 128; on
        # numpy 1 it wrapped silently to a negative. The old bounds here all
        # sat below the cliff, so the suite could not see it.
        for x in (1, 5, 9, 16, 31, 128, 200, 255, 300):
            _check(v.ekanyunena_purvena, x - 1, float(x), base=10.0, ctx=ctx)
        for base in (10, 100):
            for x in (0, base // 2, base):
                _check(v.nikhilam_navatashcaramam_dashatah, base - x,
                       float(x), base=float(base), ctx=ctx)
                _check(v.yavadunam, base - x, float(x), base=float(base), ctx=ctx)


def test_the_claude_md_example_actually_runs():
    """Execute the Python block in CLAUDE.md, verbatim, out of the file.

    Every line of the block that stood there before was wrong -- it failed on
    its own first line (`ExecutionMode` does not exist; the enum is
    `SutraMode`), and each of its seven elements named an API that is not
    there: the `VedicSutras(mode=...)` constructor, the `use_quantum` and
    `cache_results` context fields, the `n=`/`context=` argument names,
    passing the engine to `HybridQuantumClassicalSimulator`, a `run_serial()`
    taking no argument, and `report.summary()`. It was documentation written
    from impression, and nothing ever ran it.

    Reading the block out of the file rather than copying it here is the whole
    point: a copy would drift, and this cannot.
    """
    md_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CLAUDE.md")
    with open(md_path, encoding="utf-8") as fh:
        md = fh.read()
    match = re.search(
        r"```python\n(from primarysutra import VedicSutras, SutraContext, "
        r"SutraMode.*?)```", md, re.S)
    assert match, "the documented usage block is gone from CLAUDE.md"
    block = match.group(1)
    assert "run_serial" in block, "the block no longer exercises the simulator"
    exec(compile(block, "CLAUDE.md", "exec"), {})


def test_the_engine_constructs_in_every_mode():
    """`VedicSutras(SutraContext(mode=...))` must work for all five modes.

    Every other test here builds the engine with `VedicSutras()` -- default
    CLASSICAL -- and passes the mode per call as `ctx`. That is the shape the
    file's own examples use, and it left one branch of `__init__` unreached:
    when the mode is QUANTUM or HYBRID *and* `quantum_backend` is None, the
    constructor called `cudaq.get_platform()`, which does not exist in CUDA-Q
    (the accessor is `get_target()`, and `Target.name` is an attribute, not a
    method). Constructing the engine in quantum mode raised AttributeError
    before a single sutra ran.
    """
    for mode in MODES:
        v = ps.VedicSutras(ps.SutraContext(mode=mode))
        assert float(v.ekadhikena_purvena(7.0, iterations=2)) == 9.0, (
            f"{mode.name}: engine built in this mode does not compute 7 + 2"
        )


def test_the_repaired_arithmetic_sutras_agree_in_every_mode():
    """Six sutras whose quantum path computed something other than the sutra.

    Each is exactly determined -- there is one right answer -- and each was
    wrong in QUANTUM and HYBRID until the register arithmetic below replaced
    the hand-written circuits:

      * sankalana's adder cancelled its own carry with a repeated Toffoli, so
        the carry-in stayed pinned at a[0]: 9 + 4 -> 17, 2 + 3 -> 1;
      * gunitasamuccayah XORed partial products, giving a carry-less GF(2)
        product read in reverse bit order: 6 * 7 -> 18;
      * samuccayagunitah scaled a shot-frequency reading by an unrelated
        max_val**2, converging to ~265 where (6 + 7)**2 is 169;
      * chalana_kalana walked a cyclic CNOT ring, which is not an increment;
      * shunyam fired on a close to b instead of a close to -b, cut by a
        hardcoded 0.8 sitting inside the sampling noise;
      * sesanyankena's codomain was 8 multiples of Sum|c|/8, so the true 17
        was not in the range of the function at all.

    The values below span both sides of 127, which is where the np.int8
    measurement bits used to overflow. That cliff is now pinned by value
    rather than by "does not raise": the earlier gate could only assert the
    absence of an OverflowError, because the adder was wrong at every
    magnitude and a value assertion would have redded for that reason and
    blamed the int() cast.
    """
    v = ps.VedicSutras()
    for mode in MODES:
        ctx = ps.SutraContext(mode=mode)
        for a, b in ((9, 4), (0, 1), (6, 7), (100, 27), (200, 50), (255, 1)):
            _check(v.sankalana_vyavakalanabhyam, a + b, float(a), float(b), ctx=ctx)
            _check(v.sankalana_vyavakalanabhyam, a - b, float(a), float(b),
                   operation='subtract', ctx=ctx)
            _check(v.gunitasamuccayah, a * b, float(a), float(b), ctx=ctx)
            _check(v.samuccayagunitah, (a + b) ** 2, float(a), float(b), ctx=ctx)
            _check(v.samuccayagunitah, a * a + b * b, float(a), float(b),
                   operation='sum_product', ctx=ctx)
            _check(v.shunyam_samyasamuccaye, a + b, float(a), float(b), ctx=ctx)
        for x, steps, direction in ((2, 3, 1), (5, 2, 1), (13, 7, -1), (0, 3, 1)):
            _check(v.chalana_kalana, x + steps * direction, float(x),
                   steps=steps, direction=direction, ctx=ctx)
        # The last three evaluate NEGATIVE. The accumulator register is
        # unsigned, so `_quantum_polynomial` offsets it by the total of the
        # negative terms and subtracts that back at the end; without a
        # negative-result case here that offset can be deleted and every
        # other polynomial still passes, which is how it first got past me.
        for coeffs, x in (([1, 2, 3], 2), ([1, 1, 1, 1, 1], 2), ([3, -2, 1], 5),
                          ([-5], 1), ([1, -3], 4), ([-2, -1], 3)):
            expected = sum(c * x ** i for i, c in enumerate(coeffs))
            _check(v.sesanyankena_caramena, expected,
                   [float(c) for c in coeffs], float(x), ctx=ctx)


def test_inputs_with_no_register_encoding_reach_the_classical_body():
    """Non-integral scalars must not be silently truncated into a register.

    `bin(int(x))` drops the fractional part, so before the domain guard the
    quantum paths answered a different question than the one asked:
    `gunitasamuccayah(6.5, 7.5)` returned 18 where the product is 48.75, and
    `chalana_kalana(2.5, 1)` returned 7 where the answer is 3.5.
    """
    v = ps.VedicSutras()
    for mode in MODES:
        ctx = ps.SutraContext(mode=mode)
        _check(v.gunitasamuccayah, 48.75, 6.5, 7.5, ctx=ctx)
        _check(v.sankalana_vyavakalanabhyam, 14.0, 9.5, 4.5, ctx=ctx)
        _check(v.chalana_kalana, 3.5, 2.5, steps=1, ctx=ctx)

def test_nikhilam_hybrid_uses_the_circuit_at_every_width():
    """No magnitude cap silently swaps the algorithm.

    `_nikhilam_hybrid` briefly carried `hybrid_base_limit = 1024`, above which
    it called the classical complement instead. Both branches return `base -
    x`, so no value assertion could ever see it -- the tell is the return
    type: `int` out of the circuit, `float` out of the classical body. That is
    what this pins, at a base above where the cap used to sit.
    """
    v = ps.VedicSutras()
    for mode in (ps.SutraMode.QUANTUM, ps.SutraMode.HYBRID):
        got = v.nikhilam_navatashcaramam_dashatah(
            5.0, base=1025.0, ctx=ps.SutraContext(mode=mode))
        assert float(got) == 1020.0, f"{mode.name}: got {got}, expected 1020"
        assert isinstance(got, (int, np.integer)) and not isinstance(got, bool), (
            f"{mode.name}: nikhilam returned {type(got).__name__}, so the "
            f"classical body ran where the circuit should have"
        )


def test_the_last_two_sutras_are_exact_in_every_mode():
    """gunakasamuccayah and paravartya_yojayet -- the two that needed a decision.

    Both were held back while the other seven were repaired, because neither
    was settled by its own docstring:

      * `gunakasamuccayah`\'s headline names no computable function of two
        scalars, so the spec taken here is the inline comment at the classical
        body and the body itself: (a + b)(a - b), i.e. a**2 - b**2. Its old
        circuit returned `p_11 * max_val**2` -- a probability times a square --
        so its codomain was [0, 196] and it could never return the -13 the
        sutra gives for (6, 7).
      * `paravartya_yojayet` is division, whose answer is generally not an
        integer and so does not fit a register. "Exact" is taken to mean exact
        integer quotient and remainder from the register, recombined as the
        rational q + r/d. Its old circuit estimated a reciprocal to 8 bits --
        quantising every answer to a multiple of 1/256 -- and returned 1.5 or
        4.5 at random where 12/4 is 3.

    The division cases below are deliberately mostly NOT divisible, because a
    remainder of zero would not exercise the recombination at all.
    """
    v = ps.VedicSutras()
    for mode in MODES:
        ctx = ps.SutraContext(mode=mode)
        for a, b in ((6, 7), (0, 1), (12, 5), (200, 255), (7, 7), (255, 1)):
            _check(v.gunakasamuccayah, (a + b) * (a - b), float(a), float(b), ctx=ctx)
        for x, d in ((12, 4), (1, 3), (7, 2), (-22, 7), (5, -8), (4095, 7), (0, 9)):
            _check(v.paravartya_yojayet, x / d, float(x), float(d), ctx=ctx)


def test_quantum_divmod_returns_the_true_quotient_and_remainder():
    """The quotient must be right, which the division result cannot show.

    `_paravartya_yojayet_quantum` recombines as q + r/d where r is computed as
    n - q*d. That is algebraically an identity: for ANY q the result is n/d,
    because an error in q cancels exactly against the r derived from it. So a
    wrong quotient is invisible to every value assertion on the sutra -- I
    found this by injecting "drop the highest quotient bit" and watching the
    whole suite stay green.

    What makes q meaningful is the invariant 0 <= r < |d| checked inside
    `_quantum_divmod`. This asserts the contract directly, against Python\'s
    own divmod, so the quotient is pinned where the sutra cannot pin it.
    """
    for n in (0, 1, 7, 12, 100, 4095, -22, -1):
        for d in (1, 2, 3, 4, 7, 1234, -5):
            got = ps._quantum_divmod(n, d)
            assert got == divmod(abs(n), abs(d)), (
                f"_quantum_divmod({n}, {d}) = {got}, "
                f"expected {divmod(abs(n), abs(d))}"
            )
    try:
        ps._quantum_divmod(5, 0)
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("division by zero returned a quotient")


def test_an_unsimulable_register_refuses_instead_of_substituting():
    """Past 24 qubits the circuit cannot be simulated, so the answer is refused.

    A state vector is 2**n amplitudes: 24 qubits is 0.12 GiB, 31 is 16 GiB.
    The honest options at that width are to compute the value exactly or to
    decline; they do not include quietly computing it a different way and
    returning that as though the circuit had produced it. So this must raise,
    and CLASSICAL must still answer -- the choice of algorithm stays with the
    caller rather than being made silently on their behalf.
    """
    v = ps.VedicSutras()
    try:
        v.paravartya_yojayet(987654321.0, 1234.0,
                             ctx=ps.SutraContext(mode=ps.SutraMode.QUANTUM))
    except ArithmeticError as exc:
        assert "qubits" in str(exc), f"refusal does not name the width: {exc}"
    else:
        raise AssertionError(
            "a 31-qubit register was accepted; it cannot have been simulated, "
            "so some other path answered"
        )
    _check(v.paravartya_yojayet, 987654321 / 1234, 987654321.0, 1234.0,
           ctx=ps.SutraContext(mode=ps.SutraMode.CLASSICAL))


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
