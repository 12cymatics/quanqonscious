"""Exact field division, the full Q4 cochain complex, and all Walsh modes.

Fixed coordinate and signed-boundary expectations complement algebraic
identities: an implementation that simply returns zero cannot satisfy these
checks. These tests do not assert validity of the engine's separate dynamics.
"""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest
import numpy as np

from vedic.kernel import q4_complex as Q4
from vedic.kernel.k2_field import C4, I, K2, Q2, SQRT2, SQRT5, phi
from vedic.kernel.wht import HADAMARD_16_Q, walsh_inverse_exact, walsh_transform_exact


def test_q2_inverse_has_independent_coordinates():
    assert Q2(1, 1).inverse() == Q2(-1, 1)
    assert Q2(0, 1).inverse() == Q2(0, Fraction(1, 2))
    assert Q2(Fraction(2, 3), 0).inverse() == Q2(Fraction(3, 2), 0)
    assert Q2(1, 1) / Q2(1, -1) == Q2(-3, -2)


def test_k2_inverse_uses_hermitian_norm_without_replacing_it_with_total_norm():
    z = K2.from_coords(1, 1, 1, -1)
    assert z.hermitian_norm() == Q2(6, 0)
    assert z.total_norm() == Fraction(36)
    expected = K2.from_coords(Fraction(1, 6), Fraction(1, 6), Fraction(-1, 6), Fraction(1, 6))
    assert z.inverse() == expected
    assert K2.from_coords(1, 1, 0, 0).inverse() == K2.from_coords(-1, 1, 0, 0)
    assert I.inverse() == -I


def test_c4_inverse_has_independent_coordinates_including_complex_components():
    half = Fraction(1, 2)
    assert phi().inverse() == C4(K2.from_rational(-half), K2.from_rational(half))
    assert SQRT5.inverse() == C4(K2(), K2.from_rational(Fraction(1, 5)))

    # (1+i+√5)^(-1) = (-3-7i)/29 + (5+2i)√5/29.
    z = C4(K2.from_coords(1, 0, 1, 0), K2.from_rational(1))
    expected = C4(
        K2.from_coords(Fraction(-3, 29), 0, Fraction(-7, 29), 0),
        K2.from_coords(Fraction(5, 29), 0, Fraction(2, 29), 0),
    )
    assert z.inverse() == expected

    # (√2+i√5)^(-1) = (√2-i√5)/7, retaining both extensions.
    assert C4(SQRT2, I).inverse() == C4(
        K2.from_coords(0, Fraction(1, 7), 0, 0),
        K2.from_coords(0, 0, Fraction(-1, 7), 0),
    )


@pytest.mark.parametrize(
    "value,one,zero",
    [
        (Q2(Fraction(7, 3), Fraction(-2, 5)), Q2(1, 0), Q2()),
        (K2.from_coords(Fraction(1, 3), 2, Fraction(-4, 7), -5), K2.from_rational(1), K2()),
        (
            C4(K2.from_coords(1, 2, 3, 4), K2.from_coords(-5, 6, -7, 8)),
            C4.from_rational(1),
            C4(),
        ),
    ],
)
def test_field_inverse_and_division_retain_every_coordinate(value, one, zero):
    inverse = value.inverse()
    assert value * inverse == one
    assert inverse * value == one
    assert one / value == inverse
    assert value / value == one
    assert zero / value == zero
    numerator = value * value + one
    assert (numerator / value) * value == numerator
    with pytest.raises(ZeroDivisionError):
        zero.inverse()
    with pytest.raises(ZeroDivisionError):
        value / zero
    with pytest.raises(ZeroDivisionError):
        zero / zero
    with pytest.raises(TypeError):
        value / 1.0


@pytest.mark.parametrize("value", [0.5, True, "1/2", Decimal("0.5"), complex(1, 0)])
def test_exact_field_constructors_reject_non_rational_domain_inputs(value):
    with pytest.raises(TypeError):
        Q2(value, 0)
    with pytest.raises(TypeError):
        Q2(0, value)
    for index in range(4):
        coordinates = [0, 0, 0, 0]
        coordinates[index] = value
        with pytest.raises(TypeError):
            K2.from_coords(*coordinates)
    with pytest.raises(TypeError):
        K2.from_rational(value)
    with pytest.raises(TypeError):
        C4.from_rational(value)


@pytest.mark.parametrize("value", [0, Fraction(0), 0.0])
def test_nested_field_constructors_require_the_declared_field(value):
    with pytest.raises(TypeError):
        K2(value, Q2())
    with pytest.raises(TypeError):
        K2(Q2(), value)
    with pytest.raises(TypeError):
        C4(value, K2())
    with pytest.raises(TypeError):
        C4(K2(), value)
    with pytest.raises(TypeError):
        C4.from_k2(value)


def test_exact_field_integer_inputs_become_fraction_coordinates():
    assert all(isinstance(q, Fraction) for q in K2.from_coords(1, 2, 3, 4).coords())
    assert all(isinstance(q, Fraction) for q in C4.from_rational(3).u.coords())


def test_fixed_width_integer_inputs_use_unbounded_rational_coordinates():
    large = np.int64(1 << 62)
    value = Q2(large, 0)
    assert type(value.a.numerator) is int
    assert type(value.a.denominator) is int
    assert (value * value).a == Fraction(1 << 124)

    # Both results exceed the signed 64-bit input range.
    vertex_values = (large,) + (0,) * 15
    assert Q4.laplacian(vertex_values)[0] == Fraction(-(1 << 64))
    coefficients = walsh_transform_exact((large,) * 16)
    assert coefficients == (Fraction(1 << 66),) + (Fraction(0),) * 15
    assert all(type(q.numerator) is int and type(q.denominator) is int for q in coefficients)


def test_q4_includes_every_canonical_cell_in_every_degree():
    assert Q4.N_VERTICES == 16
    assert len(set(Q4.edges())) == 32
    assert len(set(Q4.plaquettes())) == 24
    assert set(Q4.cubes()) == {
        (0, 0, 1, 2), (0, 0, 1, 3), (0, 0, 2, 3), (0, 1, 2, 3),
        (1, 1, 2, 3), (2, 0, 2, 3), (4, 0, 1, 3), (8, 0, 1, 2),
    }
    assert Q4.four_cells() == ((0, 0, 1, 2, 3),)
    for cells in (Q4.plaquettes(), Q4.cubes(), Q4.four_cells()):
        for base, *axes in cells:
            assert axes == sorted(axes)
            assert all((base & (1 << axis)) == 0 for axis in axes)


def test_d1_independent_nonzero_edge_boundary_anchor():
    edge = {key: Fraction(int(key == (0, 1))) for key in Q4.edges()}
    expected = {
        key: Fraction(int(key in {(0, 0, 1), (0, 0, 2), (0, 0, 3)}))
        for key in Q4.plaquettes()
    }
    assert Q4.d1(edge) == expected


def test_d2_independent_nonzero_face_boundary_anchor():
    face = {key: Fraction(int(key == (0, 0, 1))) for key in Q4.plaquettes()}
    expected = {
        key: Fraction(-1 if key in {(0, 0, 1, 2), (0, 0, 1, 3)} else 0)
        for key in Q4.cubes()
    }
    assert Q4.d2(face) == expected


@pytest.mark.parametrize(
    "cube,sign",
    [
        ((0, 0, 1, 2), 1), ((8, 0, 1, 2), -1),
        ((0, 0, 1, 3), -1), ((4, 0, 1, 3), 1),
        ((0, 0, 2, 3), 1), ((2, 0, 2, 3), -1),
        ((0, 1, 2, 3), -1), ((1, 1, 2, 3), 1),
    ],
)
def test_d3_each_of_the_eight_signed_boundary_faces(cube, sign):
    cochain = {key: Fraction(int(key == cube)) for key in Q4.cubes()}
    assert Q4.d3(cochain) == {(0, 0, 1, 2, 3): Fraction(sign)}


@pytest.mark.parametrize("vertex", range(16))
def test_d1_d0_on_every_vertex_basis_vector(vertex):
    values = tuple(Fraction(int(v == vertex)) for v in range(16))
    assert Q4.d1(Q4.d0(values)) == {key: Fraction(0) for key in Q4.plaquettes()}


@pytest.mark.parametrize("edge", Q4.edges())
def test_d2_d1_on_every_edge_basis_vector(edge):
    values = {key: Fraction(int(key == edge)) for key in Q4.edges()}
    assert Q4.d2(Q4.d1(values)) == {key: Fraction(0) for key in Q4.cubes()}


@pytest.mark.parametrize("face", Q4.plaquettes())
def test_d3_d2_on_every_face_basis_vector(face):
    values = {key: Fraction(int(key == face)) for key in Q4.plaquettes()}
    assert Q4.d3(Q4.d2(values)) == {(0, 0, 1, 2, 3): Fraction(0)}


def test_d1_accepts_canonical_edges_and_consistent_reverse_orientations():
    canonical = {key: Fraction(index - 10, index + 1) for index, key in enumerate(Q4.edges())}
    oriented = dict(canonical)
    oriented.update({(w, v): -value for (v, w), value in canonical.items()})
    assert len(oriented) == 64
    assert Q4.d1(oriented) == Q4.d1(canonical)
    oriented[(1, 0)] += 1
    with pytest.raises(ValueError, match="reverse edge"):
        Q4.d1(oriented)


@pytest.mark.parametrize("operator,cells", [(Q4.d1, Q4.edges()), (Q4.d2, Q4.plaquettes()), (Q4.d3, Q4.cubes())])
def test_coboundaries_require_complete_exact_canonical_cell_data(operator, cells):
    values = {key: Fraction(0) for key in cells}
    missing = dict(values)
    del missing[cells[0]]
    with pytest.raises(ValueError, match="missing"):
        operator(missing)
    extra = dict(values)
    extra[(16,) + cells[0][1:]] = Fraction(0)
    with pytest.raises(ValueError, match="invalid"):
        operator(extra)
    floating = dict(values)
    floating[cells[0]] = 0.0
    with pytest.raises(TypeError, match="exact rational"):
        operator(floating)
    floating_key = dict(values)
    del floating_key[cells[0]]
    floating_key[(float(cells[0][0]),) + cells[0][1:]] = Fraction(0)
    with pytest.raises(TypeError, match="indices"):
        operator(floating_key)
    assert all(isinstance(value, Fraction) for value in operator({key: 1 for key in cells}).values())


@pytest.mark.parametrize("value", [0.5, True, "1/2", Decimal("0.5")])
def test_vertex_cochains_reject_non_rational_domain_inputs(value):
    values = [0] * 16
    values[7] = value
    for operator in (Q4.d0, Q4.laplacian, Q4.total):
        with pytest.raises(TypeError):
            operator(values)
    with pytest.raises(TypeError):
        Q4.constant(value)


def test_vertex_cochains_require_all_sixteen_values_and_integral_indices():
    for operator in (Q4.d0, Q4.laplacian, Q4.total):
        for size in (0, 15, 17):
            with pytest.raises(ValueError):
                operator([0] * size)
    with pytest.raises(TypeError):
        Q4.toggle(0.0, 1)
    with pytest.raises(TypeError):
        Q4.toggle(0, True)
    with pytest.raises(ValueError):
        Q4.toggle(16, 0)
    with pytest.raises(ValueError):
        Q4.toggle(0, 4)
    assert all(isinstance(value, Fraction) for value in Q4.d0(range(16)).values())
    assert all(isinstance(value, Fraction) for value in Q4.laplacian(range(16)))
    assert isinstance(Q4.total(range(16)), Fraction)


def test_full_walsh_matrix_has_all_pairwise_orthogonality_relations():
    assert len(HADAMARD_16_Q) == 16
    assert all(len(row) == 16 for row in HADAMARD_16_Q)
    for r in range(16):
        for s in range(16):
            inner = sum((HADAMARD_16_Q[r][v] * HADAMARD_16_Q[s][v] for v in range(16)), Fraction(0))
            assert inner == Fraction(16 if r == s else 0)


@pytest.mark.parametrize("character", range(16))
def test_full_walsh_transform_and_inverse_retain_every_character(character):
    values = tuple(Fraction(-1 if (character & v).bit_count() % 2 else 1) for v in range(16))
    expected = tuple(Fraction(16 if r == character else 0) for r in range(16))
    assert walsh_transform_exact(values) == expected
    assert walsh_inverse_exact(expected) == values
    assert Q4.laplacian(values) == tuple(-2 * character.bit_count() * value for value in values)


def test_walsh_has_independent_coefficients_for_the_vertex_label_function():
    expected = (120, -8, -16, 0, -32, 0, 0, 0, -64, 0, 0, 0, 0, 0, 0, 0)
    coefficients = walsh_transform_exact(tuple(range(16)))
    assert coefficients == tuple(Fraction(value) for value in expected)
    assert all(isinstance(value, Fraction) for value in coefficients)
    restored = walsh_inverse_exact(expected)
    assert restored == tuple(Fraction(v) for v in range(16))
    assert all(isinstance(value, Fraction) for value in restored)


def test_full_walsh_roundtrip_and_parseval_on_nonintegral_data():
    values = tuple(Fraction((v - 7) * (v + 3), v + 2) for v in range(16))
    coefficients = walsh_transform_exact(values)
    assert walsh_inverse_exact(coefficients) == values
    assert sum((c * c for c in coefficients), Fraction(0)) == 16 * sum((v * v for v in values), Fraction(0))


@pytest.mark.parametrize("operator", [walsh_transform_exact, walsh_inverse_exact])
def test_exact_walsh_requires_all_sixteen_exact_rational_values(operator):
    for size in (0, 15, 17):
        with pytest.raises(ValueError):
            operator([0] * size)
    for bad in (0.5, True, "1/2", Decimal("0.5")):
        values = [0] * 16
        values[15] = bad
        with pytest.raises(TypeError):
            operator(values)


@pytest.mark.parametrize("operator", [Q4.d0, Q4.laplacian, Q4.total, walsh_transform_exact, walsh_inverse_exact])
@pytest.mark.parametrize(
    "values",
    [
        {v: Fraction(100 - v, 7) for v in range(16)},
        set(range(16)),
        "0123456789abcdef",
        bytes(range(16)),
        bytearray(range(16)),
    ],
)
def test_exact_state_apis_reject_unordered_and_textual_containers(operator, values):
    with pytest.raises(TypeError, match="ordered sequence"):
        operator(values)
