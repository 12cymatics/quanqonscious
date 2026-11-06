"""Executable scalar contracts; execution belongs to the GitHub validation job.

These tests make no claim about the undefined full GRVQ model or formal proof.
The nonzero oracle is constructed from the user-supplied equation independently
of the implementation's component builder.
"""
from dataclasses import replace
from fractions import Fraction

import pytest
import sympy as sp

from grvq_canonical import (
    ABSENT_PARAMETER_IDS, CANONICAL_EXPRESSION, COORDINATE_SYMBOLS, CanonicalField,
    FIELD_SUTRA_IDS, GRVQParameters, GRVQPoint, MissingFieldComponentsError,
    PARAMETER_IDS, PARAMETER_SYMBOLS, SCALAR_PARAMETER_IDS,
    UNRESOLVED_REQUIREMENTS, UnresolvedGRVQModelError, apply_field_sutra,
    ekadhikena_purvena, evaluate_canonical_field, exact_real, nikhilam,
    require_complete_model, urdhva_tiryak,
)


def test_exact_origin_includes_the_large_reciprocal_factors():
    field = evaluate_canonical_field(GRVQPoint(0, 0, 0))
    assert field.value == sp.Rational(19999999700000001, 2)
    assert field.components["product_term1"] == -99999999
    assert field.components["product_term2"] == -199999999
    assert field.radial_term == 1
    assert field.components["f_vedic"] == sp.Rational(1, 2)
    assert field.components["turyavrtti_modulation"] == 1
    assert not field.expression.atoms(sp.Float)
    assert not field.value.atoms(sp.Float)


def test_field_constructor_computes_derived_state_and_rejects_forged_amplitudes():
    point, parameters = GRVQPoint(0, 0, 0), GRVQParameters()
    field = CanonicalField(point, parameters)
    assert field.value == sp.Rational(19999999700000001, 2)
    for forged in (sp.Integer(0), sp.Float("0.1")):
        with pytest.raises(TypeError):
            CanonicalField(point, parameters, value=forged)
    with pytest.raises(TypeError):
        CanonicalField(point, parameters, components={"radial_term": sp.Float("0.1")})
    with pytest.raises(TypeError):
        CanonicalField(point, parameters, expression=sp.Integer(0))
    with pytest.raises(TypeError):
        CanonicalField(point, parameters, literal_components={})
    with pytest.raises(TypeError):
        CanonicalField({"r": 0, "theta": 0, "phi": 0}, parameters)
    with pytest.raises(TypeError):
        CanonicalField(point, {"gamma": sp.Float("0.1")})


def test_nonzero_independent_oracle_preserves_every_factor():
    field = evaluate_canonical_field(GRVQPoint(1, 1, sp.Rational(1, 2)))
    S1 = sp.sin(1) * sp.cos(sp.Rational(1, 2)) * sp.exp(-sp.Rational(1, 10))
    S2 = sp.cos(1) * sp.sin(sp.Rational(1, 2)) * sp.exp(-sp.Rational(1, 20))
    carrier = sp.sin(sp.Rational(5, 2)) + sp.Rational(1, 2) * sp.cos(5)
    expected = (
        (1 - 1 / (sp.Abs(S1) + sp.Rational(1, 100000000)))
        * (1 - 2 / (sp.Abs(S2) + sp.Rational(1, 100000000)))
        * sp.Rational(1, 2) * carrier * sp.Rational(3, 2)
    )
    assert field.components["S1"] == S1
    assert field.components["S2"] == S2
    assert field.radial_term == sp.Rational(1, 2)
    assert field.components["f_vedic"] == carrier
    assert field.components["turyavrtti_modulation"] == sp.Rational(3, 2)
    assert field.value == expected


def test_nondefault_parameters_reach_the_actual_full_formula():
    parameters = GRVQParameters(
        r0_squared=2, shape_damping_S1=Fraction(1, 7),
        shape_damping_S2=Fraction(1, 11), k_vedic=3,
        epsilon=Fraction(1, 19), gamma=Fraction(2, 3),
    )
    field = evaluate_canonical_field(GRVQPoint(Fraction(3, 2), 1, Fraction(1, 2)), parameters)
    S1 = sp.sin(1) * sp.cos(sp.Rational(1, 2)) * sp.exp(-sp.Rational(3, 14))
    S2 = sp.cos(1) * sp.sin(sp.Rational(1, 2)) * sp.exp(-sp.Rational(9, 44))
    expected = (
        (1 - 1 / (sp.Abs(S1) + sp.Rational(1, 19)))
        * (1 - 2 / (sp.Abs(S2) + sp.Rational(1, 19)))
        * sp.Rational(8, 17)
        * (sp.sin(9) + sp.Rational(1, 2) * sp.cos(18))
        * (1 + sp.sqrt(2) / 3)
    )
    assert field.components["S1"] == S1
    assert field.components["S2"] == S2
    assert field.radial_term == sp.Rational(8, 17)
    assert field.value == expected


def test_both_negative_shape_functions_retain_the_absolute_values():
    field = evaluate_canonical_field(GRVQPoint(1, 3 * sp.pi / 4, 3 * sp.pi / 4))
    positive_S1 = sp.exp(-sp.Rational(1, 10)) / 2
    positive_S2 = sp.exp(-sp.Rational(1, 20)) / 2
    epsilon = sp.Rational(1, 100000000)
    expected = (
        (1 - 1 / (positive_S1 + epsilon))
        * (1 - 2 / (positive_S2 + epsilon))
        * sp.Rational(1, 2)
        * (-sp.cos(1) - sp.cos(2) / 2)
        * (1 + sp.sin(9 * sp.pi ** 3 / 16) / 2)
    )
    assert field.components["S1"] == -positive_S1
    assert field.components["S2"] == -positive_S2
    assert field.components["product_term1"] == 1 - 1 / (positive_S1 + epsilon)
    assert field.components["product_term2"] == 1 - 2 / (positive_S2 + epsilon)
    assert field.value == expected


def test_literal_expression_retains_the_given_product_and_radial_subtraction():
    field = evaluate_canonical_field(GRVQPoint(1, 1, Fraction(1, 2)))
    names = ("product_term1", "product_term2", "radial_term", "f_vedic", "turyavrtti_modulation")
    assert field.expression.func is sp.Mul
    assert field.expression.args == tuple(field.literal_components[name] for name in names)
    r_squared = sp.Pow(sp.Integer(1), sp.Integer(2), evaluate=False)
    expected_radial = sp.Add(
        sp.Integer(1),
        sp.Mul(sp.Integer(-1), r_squared,
               sp.Pow(sp.Add(r_squared, sp.Integer(1), evaluate=False), sp.Integer(-1), evaluate=False),
               evaluate=False),
        evaluate=False,
    )
    assert field.literal_components["radial_term"] == expected_radial
    assert field.expression.doit(deep=True) == field.value


def test_parameter_activity_is_structural_even_at_the_origin():
    field = evaluate_canonical_field(GRVQPoint(0, 0, 0))
    expected_present = {
        "r0_squared", "shape_damping_S1", "shape_damping_S2",
        "k_vedic", "epsilon", "gamma",
    }
    expected_absent = {"omega_vedic", "lambda_maya", "lambda_QBR", "kappa_GR"}
    report = field.parameter_activity()
    assert set(report) == set(PARAMETER_IDS)
    assert len(report) == 10
    assert set(SCALAR_PARAMETER_IDS) == expected_present
    assert set(ABSENT_PARAMETER_IDS) == expected_absent
    assert {name for name, item in report.items() if item["occurs_in_scalar"]} == expected_present
    for name, symbol in PARAMETER_SYMBOLS.items():
        assert report[name]["occurs_in_scalar"] == (symbol in CANONICAL_EXPRESSION.free_symbols)
        assert report[name]["declared_value"] == getattr(field.parameters, name)
    assert COORDINATE_SYMBOLS["t"] not in CANONICAL_EXPRESSION.free_symbols
    assert sp.diff(CANONICAL_EXPRESSION, COORDINATE_SYMBOLS["t"]) == 0


def test_time_and_four_undefined_couplings_do_not_gain_invented_effects():
    parameters = GRVQParameters()
    original = evaluate_canonical_field(GRVQPoint(1, 1, Fraction(1, 2)), parameters)
    changed = replace(parameters, omega_vedic=7, lambda_maya=11, lambda_QBR=13, kappa_GR=17)
    later = evaluate_canonical_field(GRVQPoint(1, 1, Fraction(1, 2), t=19), changed)
    assert later.expression == original.expression
    assert later.value == original.value
    assert later.parameters != original.parameters
    assert later.point.t == 19


@pytest.mark.parametrize("bad", [True, 0.1, "1/10", sp.Float("0.1"), sp.sin(sp.Float("0.2"), evaluate=False)])
def test_exact_scalar_inputs_reject_booleans_text_and_floating_data(bad):
    with pytest.raises(TypeError):
        exact_real(bad)
    with pytest.raises(TypeError):
        GRVQParameters(gamma=bad)


@pytest.mark.parametrize("bad", [sp.oo, -sp.oo, sp.nan, sp.I, sp.Symbol("x", real=True)])
def test_exact_scalar_inputs_require_closed_finite_real_values(bad):
    with pytest.raises(ValueError):
        exact_real(bad)


def test_exact_chart_and_parameter_domains_are_explicit():
    point = GRVQPoint(Fraction(1, 3), sp.pi, sp.pi / 2, -7)
    assert point.r == sp.Rational(1, 3)
    assert point.theta == sp.pi
    for point_args in ((-1, 0, 0), (1, -1, 0), (1, 2 * sp.pi, 0), (1, 0, -1), (1, 0, 2 * sp.pi)):
        with pytest.raises(ValueError):
            GRVQPoint(*point_args)
    for overrides in ({"r0_squared": 0}, {"epsilon": 0}, {"epsilon": -1}):
        with pytest.raises(ValueError):
            GRVQParameters(**overrides)
    field = evaluate_canonical_field({"r": 0, "theta": 0, "phi": 0}, {"gamma": Fraction(2, 3)})
    assert field.parameters.gamma == sp.Rational(2, 3)
    with pytest.raises(TypeError):
        evaluate_canonical_field({"r": 0, "theta": 0, "phi": 0}, {"invented_coupling": 1})


def test_the_three_field_operations_have_their_distinct_supplied_formulas():
    field = evaluate_canonical_field(GRVQPoint(1, 1, Fraction(1, 2)))
    assert ekadhikena_purvena(7, field) == 8 + field.value
    assert nikhilam(7, 10, field) == sp.Rational(3, 2)
    matrix = ((1, 2, 3), (4, 5, 6))
    expected = 2 * 5 * 1 + 2 * 7 * 2 + 2 * 11 * 3 + 3 * 5 * 4 + 3 * 7 * 5 + 3 * 11 * 6
    assert urdhva_tiryak((2, 3), (5, 7, 11), field, components=matrix) == expected
    mapping = {(i, j): value for i, row in enumerate(matrix) for j, value in enumerate(row)}
    assert apply_field_sutra("urdhva_tiryak", {"a": (2, 3), "b": (5, 7, 11)}, field, components=mapping) == expected
    assert apply_field_sutra("ekadhikena_purvena", {"value": 7}, field) == 8 + field.value
    assert apply_field_sutra("nikhilam", {"value": 7, "base": 10}, field) == sp.Rational(3, 2)


def test_urdhva_validates_every_component_even_when_coefficients_are_zero():
    field = evaluate_canonical_field(GRVQPoint(0, 0, 0))
    with pytest.raises(MissingFieldComponentsError):
        urdhva_tiryak((0,), (0,), field)
    with pytest.raises(MissingFieldComponentsError):
        urdhva_tiryak((0,), (0,), field, components={})
    with pytest.raises(TypeError):
        urdhva_tiryak((0,), (0,), field, components=((0.1,),))
    with pytest.raises(ValueError):
        urdhva_tiryak((0,), (0,), field, components={(False, 0): 1})
    assert urdhva_tiryak((0,), (0,), field, components=((13,),)) == 0
    assert urdhva_tiryak((), (), field, components=()) == 0


def test_field_operation_inputs_do_not_inherit_unrelated_book_defaults():
    field = evaluate_canonical_field(GRVQPoint(0, 0, 0))
    with pytest.raises(TypeError):
        apply_field_sutra("nikhilam", None, field)
    with pytest.raises(ValueError):
        apply_field_sutra("nikhilam", {"value": 7}, field)
    with pytest.raises(ValueError):
        apply_field_sutra("ekadhikena_purvena", {"value": 7, "base": 10}, field)
    with pytest.raises(ValueError):
        apply_field_sutra("nikhilam", {"value": 7, "base": 10}, field, components=((1,),))
    with pytest.raises(ValueError):
        urdhva_tiryak((1,), (1,), field, components={(0, 0): 1, (0, 1): 2})
    with pytest.raises(TypeError):
        ekadhikena_purvena(7, object())


def test_full_model_and_unsupplied_field_sutras_fail_with_the_actual_obligations():
    assert FIELD_SUTRA_IDS == ("ekadhikena_purvena", "nikhilam", "urdhva_tiryak")
    field = evaluate_canonical_field(GRVQPoint(0, 0, 0))
    for request in (require_complete_model, field.require_complete_model):
        with pytest.raises(UnresolvedGRVQModelError) as caught:
            request()
        assert caught.value.requirements == UNRESOLVED_REQUIREMENTS
        for missing in ("omega_vedic", "lambda_maya", "lambda_QBR", "kappa_GR", "4x4", "26 of the 29"):
            assert missing in str(caught.value)
    with pytest.raises(UnresolvedGRVQModelError, match="no supplied formula"):
        apply_field_sutra("gunita_samuccaya", {"value": 7}, field)
