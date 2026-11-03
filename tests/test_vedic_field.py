from fractions import Fraction

import pytest
import sympy as sp

from vedic_field import FieldBinding


@pytest.mark.parametrize("sid", range(1, 30))
def test_graph_mapping_matches_the_full_matrix_polynomial(sid):
    binding = FieldBinding.from_parameters(sid)
    psi = tuple(Fraction(i - 2, i + 1) for i in range(8))
    neighbors = tuple(tuple(i ^ (1 << bit) for bit in range(3)) for i in range(8))
    laplacian = sp.zeros(8)
    for i, row in enumerate(neighbors):
        laplacian[i, i] = len(row)
        for j in row:
            laplacian[i, j] -= 1
    matrix = sp.eye(8)
    for k, c in enumerate(binding.result.spectral_coefficients, 1):
        matrix += sp.Rational(1, 435) * c * laplacian ** k
    expected = tuple(matrix * sp.Matrix(psi))
    actual = binding.apply_hypercube(psi, Fraction(1, 435))
    assert actual == expected
    assert sum(actual) == sum(psi)
    assert all(isinstance(v, Fraction) for v in actual)
    assert binding.apply_hypercube(psi, Fraction(0)) == psi


def test_torus_retains_edges_with_their_multiplicity():
    binding = FieldBinding.from_parameters(9, {"p": [0, 1]})
    assert binding.apply_torus([Fraction(1), Fraction(0)], (2,)) == (3, -2)
    assert binding.apply_torus([Fraction(1)], (1,)) == (1,)


def test_operand_changes_reach_field_values():
    values = [Fraction(1)] + [Fraction(0)] * 15
    a = FieldBinding.from_parameters(2, {"a": 98, "b": 97, "base": 100})
    b = FieldBinding.from_parameters(2, {"a": 98, "b": 96, "base": 100})
    assert a.apply_hypercube(values) != b.apply_hypercube(values)


def test_all_zero_book_results_produce_the_prescribed_identity():
    binding = FieldBinding.from_parameters(9, {"p": [7]})
    values = tuple(Fraction(i, 7) for i in range(16))
    assert binding.result.spectral_coefficients == (0,)
    assert binding.apply_hypercube(values) == values


def test_invalid_field_shapes_are_rejected():
    binding = FieldBinding.from_parameters(2)
    with pytest.raises(ValueError):
        binding.apply_hypercube([1, 2, 3])
    with pytest.raises(ValueError):
        binding.apply_torus([1, 2, 3], (2, 2))
    with pytest.raises(ValueError):
        binding.apply_graph([1, 2], [(1,), (2,)])
    with pytest.raises(TypeError):
        binding.apply_hypercube([0.1, 0.2])


def test_complex_values_retain_both_components():
    binding = FieldBinding.from_parameters(9, {"p": [0, 1]})
    assert binding.apply_graph([sp.I, sp.S.One], [(1,), (0,)]) == (2 * sp.I - 1, 2 - sp.I)


@pytest.mark.parametrize("value", [sp.oo, -sp.oo, sp.zoo, sp.nan])
def test_nonfinite_algebraic_values_are_rejected(value):
    binding = FieldBinding.from_parameters(9, {"p": [0, 1]})
    with pytest.raises(TypeError):
        binding.apply_graph([value, 0], [(1,), (0,)])


@pytest.mark.parametrize("sid", range(1, 30))
def test_edge_mapping_is_exactly_unitary_and_uses_complete_book_output(sid):
    binding = FieldBinding.from_parameters(sid)
    adjacency = ((1, 1, 2), (0, 2), (0, 1, 2))
    strength = Fraction(1, 100)
    columns = [binding.rotate_graph(tuple(Fraction(i == j) for i in range(3)), adjacency, strength)
               for j in range(3)]
    matrix = sp.Matrix.hstack(*(sp.Matrix(column) for column in columns))
    assert matrix.T * matrix == sp.eye(3)
    expected = sp.eye(3)
    for c in binding.result.spectral_coefficients:
        t = c / 100
        for i, row in enumerate(adjacency):
            for j in row:
                if i == j:
                    continue
                a, z = sorted((i, j))
                generator = sp.zeros(3)
                generator[a, z], generator[z, a] = -t, t
                rotation = (sp.eye(3) - generator).inv() * (sp.eye(3) + generator)
                expected = rotation * expected
    assert matrix == expected
    state = (Fraction(2, 3), Fraction(-7, 5), Fraction(1, 9973))
    output = binding.rotate_graph(state, adjacency, strength)
    assert sum(x * x for x in output) == sum(x * x for x in state)
    assert binding.rotate_graph(state, adjacency, Fraction(0)) == state


def _first_fit_matchings(adjacency):
    matchings = []
    for i, row in enumerate(adjacency):
        for j in row:
            if i == j:
                continue
            edge = tuple(sorted((i, j)))
            for matching in matchings:
                if not set(edge) & {v for e in matching for v in e}:
                    matching.append(edge)
                    break
            else:
                matchings.append([edge])
    return matchings


@pytest.mark.parametrize("sid", range(1, 30))
def test_matched_mapping_is_the_product_of_commuting_matchings(sid):
    binding = FieldBinding.from_parameters(sid)
    adjacency = ((1, 1, 2, 3), (0, 2, 0), (0, 1, 2, 3), (2, 0))
    strength = Fraction(1, 100)
    columns = [binding.rotate_matched(tuple(Fraction(i == j) for i in range(4)), adjacency, strength)
               for j in range(4)]
    matrix = sp.Matrix.hstack(*(sp.Matrix(column) for column in columns))
    assert matrix.T * matrix == sp.eye(4)
    expected = sp.eye(4)
    for c in binding.result.spectral_coefficients:
        t = c / 100
        for matching in _first_fit_matchings(adjacency):
            step = sp.eye(4)
            for a, z in matching:
                generator = sp.zeros(4)
                generator[a, z], generator[z, a] = -t, t
                step = (sp.eye(4) - generator).inv() * (sp.eye(4) + generator) * step
            expected = step * expected
    assert matrix == expected
    assert binding.rotate_matched(tuple(Fraction(k + 1, 7) for k in range(4)), adjacency, Fraction(0)) == \
        tuple(Fraction(k + 1, 7) for k in range(4))


def test_matchings_share_no_vertex_and_keep_every_adjacency_entry():
    from vedic_field import edge_matchings
    coords = [(a, b, c) for a in range(4) for b in range(4) for c in range(4)]
    index = {p: i for i, p in enumerate(coords)}
    adjacency = tuple(tuple(index[tuple((p[k] + s) % 4 if k == axis else p[k] for k in range(3))]
                            for axis in range(3) for s in (-1, 1)) for p in coords)
    matchings = edge_matchings(adjacency)
    for matching in matchings:
        vertices = [v for edge in matching for v in edge]
        assert len(vertices) == len(set(vertices))
    entries = sorted(tuple(sorted((i, j))) for i, row in enumerate(adjacency) for j in row if i != j)
    assert sorted(edge for matching in matchings for edge in matching) == entries


def test_matched_mapping_preserves_the_exact_norm_and_bounds_denominators():
    from vedic_field import cayley_parts, edge_matchings
    coords = [(a, b, c) for a in range(4) for b in range(4) for c in range(4)]
    index = {p: i for i, p in enumerate(coords)}
    adjacency = tuple(tuple(index[tuple((p[k] + s) % 4 if k == axis else p[k] for k in range(3))]
                            for axis in range(3) for s in (-1, 1)) for p in coords)
    from math import gcd
    from core.state import RationalComplex
    field = tuple(RationalComplex(Fraction(i % 7 - 3, 11 + i), Fraction(5 - i % 5, 13 + 2 * i)) for i in range(64))
    for sid in (1, 2, 12, 16):
        binding = FieldBinding.from_parameters(sid)
        out = binding.rotate_matched(field, adjacency, Fraction(1, 100))
        assert sum(v.real * v.real + v.imag * v.imag for v in out) == \
            sum(v.real * v.real + v.imag * v.imag for v in field)
        d0 = 1
        for v in field:
            for part in (v.real, v.imag):
                d0 = d0 * part.denominator // gcd(d0, part.denominator)
        bound = d0
        for c in binding.result.spectral_coefficients:
            c = sp.Rational(c)
            bound *= cayley_parts(Fraction(1, 100) * Fraction(int(c.p), int(c.q)))[0] ** len(edge_matchings(adjacency))
        for v in out:
            for part in (v.real, v.imag):
                assert bound % part.denominator == 0


def test_execution_receipts_distinguish_inputs_and_operands():
    first = FieldBinding.from_parameters(9, {"p": [0, 1]})
    second = FieldBinding.from_parameters(9, {"p": [99, 1]})
    assert first.metadata() != second.metadata()
    adjacency = ((1,), (0,))
    before = (Fraction(1), Fraction(0))
    after = first.rotate_graph(before, adjacency, Fraction(1, 3))
    receipt = first.receipt(before, after, adjacency, Fraction(1, 3), "edge_rotations")
    assert receipt["input_sha256"] != receipt["output_sha256"]
    assert receipt["operands"] == {"p": ("0", "1")}
    assert receipt["strength"] == "1/3"
    assert receipt["topology_sha256"] != first.receipt(before, after, ((1, 1), (0,)), Fraction(1, 3), "edge_rotations")["topology_sha256"]


def test_rotation_mapping_keeps_a_long_coefficient_tail():
    first = FieldBinding.from_parameters(9, {"p": [0] * 1025 + [Fraction(1, 1025)]})
    second = FieldBinding.from_parameters(9, {"p": [0] * 1026})
    assert len(first.result.spectral_coefficients) == 1025
    assert first.rotate_graph([1, 0], ((1,), (0,)), Fraction(1, 3)) != second.rotate_graph([1, 0], ((1,), (0,)), Fraction(1, 3))
