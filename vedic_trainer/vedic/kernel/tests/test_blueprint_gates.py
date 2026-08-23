"""Gate A (exact base + certificates) and Gate B (exact Q4 topology).

Every assertion is a named obligation from the *Exact Kernel Evolution
Blueprint*, 31 July 2026.
"""
from __future__ import annotations

import random
from fractions import Fraction

import pytest

from vedic.kernel import q4_complex as Q4
from vedic.kernel.k2_field import C4, K2, Q2, phi, phi_cubed


def _rnd_k2(seed: int) -> K2:
    r = random.Random(seed)
    return K2.from_coords(*(Fraction(r.randint(-9, 9), r.randint(1, 7))
                            for _ in range(4)))


# ══════════════════════════════════════════════ Gate A — K2 = ℚ(√2, i)


def test_coordinate_form_is_four_rationals():
    z = K2.from_coords(1, 2, 3, 4)
    assert z.coords() == (Fraction(1), Fraction(2), Fraction(3), Fraction(4))
    assert all(isinstance(c, Fraction) for c in z.coords())


def test_sigma2_and_sigma_i_are_involutions():
    for s in range(6):
        z = _rnd_k2(s)
        assert z.sigma2().sigma2() == z
        assert z.sigma_i().sigma_i() == z


def test_the_two_involutions_commute_exactly():
    """σ₂(σᵢ(z)) = σᵢ(σ₂(z)) — a stated blueprint requirement."""
    for s in range(8):
        z = _rnd_k2(s)
        assert z.sigma_i().sigma2() == z.sigma2().sigma_i()


def test_involutions_are_field_automorphisms():
    for s in range(4):
        x, y = _rnd_k2(s), _rnd_k2(s + 50)
        for sig in ("sigma2", "sigma_i"):
            f = lambda z: getattr(z, sig)()  # noqa: E731
            assert f(x + y) == f(x) + f(y)
            assert f(x * y) == f(x) * f(y)


def test_sigma2_negates_sqrt2_and_fixes_i():
    from vedic.kernel.k2_field import I, SQRT2
    assert SQRT2.sigma2() == -SQRT2
    assert I.sigma2() == I


def test_sigma_i_negates_i_and_fixes_sqrt2():
    from vedic.kernel.k2_field import I, SQRT2
    assert I.sigma_i() == -I
    assert SQRT2.sigma_i() == SQRT2


def test_hermitian_norm_lands_in_q_sqrt2():
    """H(z) = z†z ∈ ℚ(√2) — generally NOT rational."""
    z = K2.from_coords(Fraction(1, 2), Fraction(3, 4), Fraction(-2, 5), Fraction(7, 3))
    H = z.hermitian_norm()
    assert isinstance(H, Q2)
    assert not H.is_rational(), "this z should have an irrational Hermitian norm"


def test_total_norm_lands_in_q():
    for s in range(6):
        n = _rnd_k2(s).total_norm()
        assert isinstance(n, Fraction)


def test_the_two_norms_are_different_objects():
    """The blueprint: an energy definition must NAME the selected norm."""
    z = K2.from_coords(Fraction(1, 2), Fraction(3, 4), Fraction(-2, 5), Fraction(7, 3))
    H, N = z.hermitian_norm(), z.total_norm()
    assert not (H.is_rational() and H.a == N), "H and N must not coincide here"


def test_total_norm_is_the_q2_norm_of_the_hermitian_norm():
    """N(z) = H(z)·σ₂(H(z)) — the Galois product closes."""
    for s in range(6):
        z = _rnd_k2(s)
        H = z.hermitian_norm()
        assert (H * H.conj2()).a == z.total_norm()


def test_norms_are_multiplicative():
    for s in range(4):
        x, y = _rnd_k2(s), _rnd_k2(s + 30)
        assert (x * y).total_norm() == x.total_norm() * y.total_norm()


# ══════════════════════════════════ Gate A — C4 = K2(√5), the φ extension


def test_phi_is_not_in_k2():
    """Treating √5 or φ as elements of K2 would be false (blueprint)."""
    assert not phi().is_in_k2()
    assert not phi_cubed().is_in_k2()


def test_phi_satisfies_its_defining_identity():
    p = phi()
    assert p * p == p + C4.from_rational(1)          # φ² = φ + 1


def test_phi_cubed_is_exactly_two_plus_sqrt5():
    """φ³ = 2 + √5 — an algebraic integer, never the float 4.236…"""
    assert phi_cubed() == C4(K2.from_rational(2), K2.from_rational(1))


def test_phi_cubed_renders_to_the_documented_float():
    """The 4.236 the design docs quote, produced by the package's own boundary.

    This test used to read `approx = 2 + 5 ** 0.5` and compare that to a
    literal. It imported nothing under test and could not fail on any change
    to this codebase — it verified that Python can add. It now goes through
    `C4.to_float`, which is the one place a float is allowed to appear.
    """
    assert abs(phi_cubed().to_float() - (4.23606797749979 + 0j)) < 1e-12


def float_offenders(source: str, boundary: str = "to_float") -> list[str]:
    """Functions other than `boundary` that touch a float literal or constant.

    Pure, so `test_gates_reject.py` can prove it rejects a leaked float
    without editing k2_field.py on disk.
    """
    import ast as _ast
    out = []
    for node in _ast.walk(_ast.parse(source)):
        if isinstance(node, _ast.FunctionDef) and node.name != boundary:
            for sub in _ast.walk(node):
                if isinstance(sub, _ast.Constant) and isinstance(sub.value, float):
                    out.append(f"{node.name}: {sub.value}")
                if (isinstance(sub, _ast.Name)
                        and sub.id in {"_SQRT2_F", "_SQRT5_F"}):
                    out.append(f"{node.name} reads {sub.id}")
    return out


def test_the_render_boundary_is_the_only_float_in_the_field_module():
    """`to_float` is a boundary only if nothing else in the module uses floats."""
    import ast
    import inspect

    from vedic.kernel import k2_field

    offenders = float_offenders(inspect.getsource(k2_field))
    assert not offenders, (
        "float literals or float constants outside to_float: " + str(offenders))


def test_exact_and_rendered_values_disagree_as_they_must():
    """phi_cubed() is exactly 2+sqrt5; its float is not, and cannot be."""
    from fractions import Fraction
    rendered = phi_cubed().to_float().real
    assert Fraction(rendered) != Fraction(2) + Fraction(5) ** 1  # not exact
    assert phi_cubed() == C4(K2.from_rational(2), K2.from_rational(1))


def test_c4_embeds_k2():
    z = _rnd_k2(1)
    assert C4.from_k2(z).is_in_k2()


def test_c4_multiplication_uses_sqrt5_squared_equals_five():
    from vedic.kernel.k2_field import SQRT5
    assert SQRT5 * SQRT5 == C4.from_rational(5)


# ══════════════════════════════════════════ Gate B — exact Q4 topology


def test_toggle_is_an_involution():
    for v in range(Q4.N_VERTICES):
        for j in range(Q4.N_AXES):
            assert Q4.toggle(Q4.toggle(v, j), j) == v


def test_distinct_coordinate_toggles_commute():
    for v in range(Q4.N_VERTICES):
        for j in range(Q4.N_AXES):
            for k in range(Q4.N_AXES):
                if j != k:
                    assert Q4.toggle(Q4.toggle(v, j), k) == Q4.toggle(Q4.toggle(v, k), j)


def test_axis_to_neighbour_map_is_injective():
    for v in range(Q4.N_VERTICES):
        nb = [Q4.toggle(v, j) for j in range(Q4.N_AXES)]
        assert len(set(nb)) == Q4.N_AXES


def test_exactly_four_distinct_unsigned_neighbours_per_vertex():
    for v in range(Q4.N_VERTICES):
        nb = Q4.neighbours(v)
        assert len(set(nb)) == 4
        assert all(bin(v ^ w).count("1") == 1 for w in nb)


def test_exactly_eight_oriented_axis_incidences_per_vertex():
    for v in range(Q4.N_VERTICES):
        inc = Q4.oriented_incidences(v)
        assert len(inc) == 8
        assert len(set(inc)) == 8


def test_edge_count_is_thirty_two():
    """A 4-cube has 4·2⁴/2 = 32 edges."""
    assert len(Q4.edges()) == 32


def test_plaquette_count_is_twenty_four():
    """A 4-cube has C(4,2)·2² = 24 square faces."""
    assert len(Q4.plaquettes()) == 24


def test_toggle_rejects_out_of_range_arguments():
    with pytest.raises(ValueError):
        Q4.toggle(16, 0)
    with pytest.raises(ValueError):
        Q4.toggle(0, 4)


# ------------------------------------------------------- d² = 0


def test_d1_of_d0_is_zero_for_every_zero_cochain():
    """d¹(d⁰f) = 0 — the blueprint's stated Gate B obligation."""
    for s in range(8):
        r = random.Random(s)
        f = tuple(Fraction(r.randint(-9, 9), r.randint(1, 7))
                  for _ in range(Q4.N_VERTICES))
        dd = Q4.d1(Q4.d0(f))
        assert all(v == 0 for v in dd.values()), "d² must vanish"


def test_d0_is_antisymmetric_on_oriented_edges():
    r = random.Random(0)
    f = tuple(Fraction(r.randint(-9, 9)) for _ in range(Q4.N_VERTICES))
    g = Q4.d0(f)
    for (u, w), val in g.items():
        assert g[(w, u)] == -val


def test_d0_annihilates_constants():
    g = Q4.d0(Q4.constant(Fraction(7, 3)))
    assert all(v == 0 for v in g.values())


# ------------------------------------------------ four-axis Laplacian


def test_laplacian_annihilates_constants():
    for c in (0, 1, Fraction(-5, 3)):
        assert all(v == 0 for v in Q4.laplacian(Q4.constant(c)))


def test_laplacian_total_sum_is_zero():
    """Σ_v (Δf)(v) = 0 for every f — each edge contributes twice, oppositely."""
    for s in range(8):
        r = random.Random(s)
        f = tuple(Fraction(r.randint(-9, 9), r.randint(1, 7))
                  for _ in range(Q4.N_VERTICES))
        assert Q4.total(Q4.laplacian(f)) == 0


def test_laplacian_is_linear():
    r = random.Random(3)
    f = tuple(Fraction(r.randint(-9, 9)) for _ in range(16))
    g = tuple(Fraction(r.randint(-9, 9)) for _ in range(16))
    s = tuple(a + b for a, b in zip(f, g))
    assert Q4.laplacian(s) == tuple(a + b for a, b in
                                    zip(Q4.laplacian(f), Q4.laplacian(g)))


def test_laplacian_is_exact_rational():
    r = random.Random(5)
    f = tuple(Fraction(r.randint(-9, 9), r.randint(1, 7)) for _ in range(16))
    assert all(isinstance(x, Fraction) for x in Q4.laplacian(f))


def test_laplacian_matches_the_kernel_diffusive_stencil():
    """The DIFF operator's target is the edge mean; Δ = 4·(edgeMean − f)."""
    from vedic.kernel import sutras_canonical as K
    r = random.Random(11)
    f = tuple(Fraction(r.randint(-9, 9), r.randint(1, 7)) for _ in range(16))
    lap = Q4.laplacian(f)
    for v in range(16):
        assert lap[v] == 4 * (K.edge_mean(f, v) - f[v])
