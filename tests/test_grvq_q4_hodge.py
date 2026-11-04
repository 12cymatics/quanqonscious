"""Independent coordinates and complete cell bases for the Euclidean Hodge asset."""
from decimal import Decimal
from fractions import Fraction as Q

import pytest
import sympy as sp

from grvq_q4_hodge import hodge_star6, q4_hodge_diagnostics, split_hodge6


# Explicit face groups and bases, independent of q4_complex.plaquettes().
FACE_GROUPS = (
    ((0, 1), (0, 4, 8, 12)),
    ((0, 2), (0, 2, 8, 10)),
    ((0, 3), (0, 2, 4, 6)),
    ((1, 2), (0, 1, 8, 9)),
    ((1, 3), (0, 1, 4, 5)),
    ((2, 3), (0, 1, 2, 3)),
)
FACES = tuple((base, first, second) for (first, second), bases in FACE_GROUPS for base in bases)
PAIRS = tuple(pair for pair, _ in FACE_GROUPS)
ANCHOR = (1, -2, 3, -4, 5, -6)


def _zero_faces():
    return {face: Q(0) for face in FACES}


def _pair_constant_faces(values=ANCHOR):
    return {
        (base, first, second): Q(value)
        for value, ((first, second), bases) in zip(values, FACE_GROUPS)
        for base in bases
    }


# Expected columns of star, 2P+, and 2P- on all six independent basis vectors.
BASIS_EXPECTED = (
    ((0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, -1)),
    ((0, 0, 0, 0, -1, 0), (0, 1, 0, 0, -1, 0), (0, 1, 0, 0, 1, 0)),
    ((0, 0, 0, 1, 0, 0), (0, 0, 1, 1, 0, 0), (0, 0, 1, -1, 0, 0)),
    ((0, 0, 1, 0, 0, 0), (0, 0, 1, 1, 0, 0), (0, 0, -1, 1, 0, 0)),
    ((0, -1, 0, 0, 0, 0), (0, -1, 0, 0, 1, 0), (0, 1, 0, 0, 1, 0)),
    ((1, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 1), (-1, 0, 0, 0, 0, 1)),
)


@pytest.mark.parametrize("index,expected", enumerate(BASIS_EXPECTED))
def test_every_bivector_basis_has_the_declared_oriented_image(index, expected):
    field = tuple(Q(int(position == index)) for position in range(6))
    star, plus_twice, minus_twice = expected
    result = split_hodge6(field)
    assert hodge_star6(field) == star
    assert result["star6"] == star
    assert result["self_dual6"] == tuple(Q(v, 2) for v in plus_twice)
    assert result["anti_self_dual6"] == tuple(Q(v, 2) for v in minus_twice)
    assert result["plus_triple"] == tuple(Q(v, 2) for v in plus_twice[:3])
    assert result["minus_triple"] == tuple(Q(v, 2) for v in minus_twice[:3])
    assert result["input_norm_squared"] == 1
    assert result["star_norm_squared"] == 1
    assert result["self_dual_norm_squared"] == Q(1, 2)
    assert result["anti_self_dual_norm_squared"] == Q(1, 2)
    assert result["reconstruction6"] == field


def test_nonzero_anchor_retains_both_chiralities_and_original_orientation():
    result = split_hodge6(ANCHOR)
    assert result["input6"] == ANCHOR
    assert result["star6"] == (-6, -5, -4, 3, 2, 1)
    assert result["plus_triple"] == (Q(-5, 2), Q(-7, 2), Q(-1, 2))
    assert result["minus_triple"] == (Q(7, 2), Q(3, 2), Q(7, 2))
    assert result["self_dual6"] == (Q(-5, 2), Q(-7, 2), Q(-1, 2), Q(-1, 2), Q(7, 2), Q(-5, 2))
    assert result["anti_self_dual6"] == (Q(7, 2), Q(3, 2), Q(7, 2), Q(-7, 2), Q(3, 2), Q(-7, 2))
    assert result["input_norm_squared"] == result["star_norm_squared"] == 91
    assert result["self_dual_norm_squared"] == Q(75, 2)
    assert result["anti_self_dual_norm_squared"] == Q(107, 2)
    assert result["chirality_inner_product"] == 0
    assert result["reconstruction6"] == ANCHOR
    assert result["signature"] == (1, 1, 1, 1)
    assert result["orientation"] == (0, 1, 2, 3)
    assert result["axis_pairs"] == PAIRS


@pytest.mark.parametrize("field", (
    ANCHOR,
    (Q(1, 3), Q(-2, 5), Q(7, 11), Q(-4, 9), Q(5, 7), Q(-6, 13)),
    (2, 3, 5, -5, 3, -2),  # Pure anti-self-dual; the plus triple is genuinely zero.
    (2, 3, 5, 5, -3, 2),   # Pure self-dual.
    (0, 0, 0, 0, 0, 0),
))
def test_involution_eigenspaces_orthogonality_and_reconstruction(field):
    result = split_hodge6(field)
    assert hodge_star6(hodge_star6(field)) == field
    assert hodge_star6(result["self_dual6"]) == result["self_dual6"]
    assert hodge_star6(result["anti_self_dual6"]) == tuple(-v for v in result["anti_self_dual6"])
    assert result["chirality_inner_product"] == 0
    assert result["self_dual_norm_squared"] + result["anti_self_dual_norm_squared"] == result["input_norm_squared"]
    assert result["reconstruction6"] == field
    assert all(isinstance(v, Q) for v in result["input6"])


@pytest.mark.parametrize("face", FACES)
def test_every_face_basis_reaches_exactly_its_four_incident_vertices(face):
    cochain = _zero_faces()
    cochain[face] = Q(1)
    result = q4_hodge_diagnostics(cochain)
    base, first, second = face
    expected_vertices = {base, base + 2 ** first, base + 2 ** second, base + 2 ** first + 2 ** second}
    pair_index = PAIRS.index((first, second))
    assert result["face_count"] == 24
    assert result["vertex_count"] == 16
    assert result["face_square_sum"] == 1
    assert result["source_pressure"] == Q(1, 48)
    assert len(result["vertices"]) == 16
    assert result["plaquette_cochain"] == cochain
    for vertex, local in enumerate(result["vertices"]):
        active = vertex in expected_vertices
        expected = tuple(Q(int(active and index == pair_index)) for index in range(6))
        assert local["vertex"] == vertex
        assert local["local_bivector6"] == expected
        assert local["split"]["input6"] == expected
        assert local["split"]["reconstruction6"] == expected
        assert (face in local["incident_faces"]) == active
        assert len(local["incident_faces"]) == 6
        assert local["source_normalization"] == Q(1, 6)
        assert local["source_strength_squared"] == (Q(1, 72) if active else Q(0))
        assert local["source_strength"] == (sp.sqrt(2) / 12 if active else sp.S.Zero)
        plus_twice, minus_twice = BASIS_EXPECTED[pair_index][1:]
        assert local["source_plus_triple"] == tuple(Q(v, 12) if active else Q(0) for v in plus_twice[:3])
        assert local["source_minus_triple"] == tuple(Q(v, 12) if active else Q(0) for v in minus_twice[:3])


def test_all_vertices_keep_source_six_face_mean_and_normalized_pressure():
    result = q4_hodge_diagnostics(_pair_constant_faces())
    assert result["face_square_sum"] == 364
    assert result["source_pressure"] == Q(91, 12)
    assert result["signature"] == (1, 1, 1, 1)
    assert result["orientation"] == (0, 1, 2, 3)
    assert result["axis_pairs"] == PAIRS
    assert result["source"]["commit"] == "9f8af8bb6ed3c3d6ab064f71effdcc7bfc576ef1"
    assert result["source"]["path"] == "vedic_v18.24_full_kernel.html"
    for local in result["vertices"]:
        assert local["local_bivector6"] == ANCHOR
        assert local["source_plus_triple"] == (Q(-5, 12), Q(-7, 12), Q(-1, 12))
        assert local["source_minus_triple"] == (Q(7, 12), Q(1, 4), Q(7, 12))
        assert local["source_strength_squared"] == Q(91, 72)
        assert local["source_strength"] == sp.sqrt(182) / 12
        assert local["source_strength"] ** 2 == sp.Rational(91, 72)


def test_anti_self_dual_field_remains_active_in_source_strength():
    result = q4_hodge_diagnostics(_pair_constant_faces((2, 3, 5, -5, 3, -2)))
    for local in result["vertices"]:
        assert local["source_plus_triple"] == (0, 0, 0)
        assert local["source_minus_triple"] == (Q(1, 3), Q(1, 2), Q(5, 6))
        assert local["source_strength_squared"] == Q(19, 18)
        assert local["source_strength"] == sp.sqrt(38) / 6
        assert local["split"]["reconstruction6"] == (2, 3, 5, -5, 3, -2)


def test_each_face_value_is_retained_and_used_at_four_vertices():
    cochain = {face: Q((-1) ** index * (index + 1), index + 2) for index, face in enumerate(FACES)}
    result = q4_hodge_diagnostics(cochain)
    square_sum = sum((value ** 2 for value in cochain.values()), Q(0))
    assert result["plaquette_cochain"] == cochain
    assert result["face_square_sum"] == square_sum
    assert result["source_pressure"] == square_sum / 48
    incidence_counts = {face: 0 for face in FACES}
    for local in result["vertices"]:
        assert tuple(face[1:] for face in local["incident_faces"]) == PAIRS
        for face, value in zip(local["incident_faces"], local["local_bivector6"]):
            incidence_counts[face] += 1
            assert value == cochain[face]
        assert local["source_strength_squared"] == sum((value ** 2 for value in local["local_bivector6"]), Q(0)) / 72
        assert local["source_strength"] ** 2 == sp.Rational(local["source_strength_squared"].numerator, local["source_strength_squared"].denominator)
    assert set(incidence_counts.values()) == {4}
    assert sum((local["source_strength_squared"] for local in result["vertices"]), Q(0)) == square_sum / 18


def test_outputs_are_detached_from_mutable_input_cochain():
    cochain = _pair_constant_faces()
    result = q4_hodge_diagnostics(cochain)
    cochain[(0, 0, 1)] = Q(999)
    assert result["plaquette_cochain"][(0, 0, 1)] == 1
    assert result["vertices"][0]["local_bivector6"] == ANCHOR


@pytest.mark.parametrize("operation", (hodge_star6, split_hodge6))
@pytest.mark.parametrize("invalid", (None, 1, "123456", b"123456", bytearray(b"123456"), {0: 1}, {0, 1, 2, 3, 4, 5}))
def test_bivectors_require_ordered_sequences(operation, invalid):
    with pytest.raises(TypeError):
        operation(invalid)


@pytest.mark.parametrize("operation", (hodge_star6, split_hodge6))
@pytest.mark.parametrize("invalid", ((), (1,), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5, 6, 7)))
def test_bivectors_require_exactly_six_coordinates(operation, invalid):
    with pytest.raises(ValueError):
        operation(invalid)


@pytest.mark.parametrize("operation", (hodge_star6, split_hodge6))
@pytest.mark.parametrize("invalid", (True, False, 0.0, "0", Decimal(0), 0j, sp.Float(0)))
def test_every_bivector_coordinate_obeys_exact_rational_contract(operation, invalid):
    for index in range(6):
        values = [Q(0)] * 6
        values[index] = invalid
        with pytest.raises(TypeError):
            operation(values)


@pytest.mark.parametrize("invalid", (None, [], [0] * 24, "24 faces"))
def test_q4_faces_require_a_mapping(invalid):
    with pytest.raises(TypeError):
        q4_hodge_diagnostics(invalid)


@pytest.mark.parametrize("face", FACES)
def test_no_face_can_be_omitted_even_from_a_zero_cochain(face):
    cochain = _zero_faces()
    del cochain[face]
    with pytest.raises(ValueError, match="missing cells"):
        q4_hodge_diagnostics(cochain)


@pytest.mark.parametrize("invalid", (True, False, 0.0, "0", Decimal(0), 0j, sp.Float(0)))
def test_every_face_is_validated_even_when_its_value_would_be_zero(invalid):
    for face in FACES:
        cochain = _zero_faces()
        cochain[face] = invalid
        with pytest.raises(TypeError):
            q4_hodge_diagnostics(cochain)


@pytest.mark.parametrize("key,error", (
    ((False, 0, 1), TypeError),
    ((0, False, 1), TypeError),
    ((0.0, 0, 1), TypeError),
    ((0, 0, 1.0), TypeError),
    ((0, 1, 0), ValueError),
    ((0, 0, 0), ValueError),
    ((1, 0, 1), ValueError),
    ((16, 0, 1), ValueError),
    ((0, 0), ValueError),
    ((0, 0, 1, 2), ValueError),
    ("face", ValueError),
))
def test_noncanonical_or_invalid_face_indices_are_rejected(key, error):
    cochain = _zero_faces()
    del cochain[(0, 0, 1)]
    cochain[key] = Q(0)
    with pytest.raises(error):
        q4_hodge_diagnostics(cochain)


def test_extra_faces_are_rejected_without_discarding_them():
    cochain = _zero_faces()
    cochain[(16, 0, 1)] = Q(0)
    with pytest.raises(ValueError, match="invalid cells"):
        q4_hodge_diagnostics(cochain)
