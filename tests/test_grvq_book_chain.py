"""Independent Q4 calculations, full-stage evidence, and visible failures."""
from collections import Counter
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction

import pytest
import sympy as sp

import grvq_book_chain as chain
import vedic_book
import vedic_field


# Explicit small operands keep all three full29 executions tractable. These
# are valid test inputs for each actual application, never runtime presets.
OPERANDS = {
    1: {"denominator": 9},
    2: {"a": 1, "b": 1, "base": 2},
    3: {"a": [1, 1], "b": [1, 1]},
    4: {"dividend": [1, 2], "divisor": [1]},
    5: {"a": 1, "b": 1, "c": 1, "d": 1},
    6: {"a": 1, "b": 1, "c": 1, "d": 2, "e": 1, "f": 1},
    7: {"a": 2, "b": 1, "c": 1, "d": 0},
    8: {"a": 1, "b": 0, "c": -1},
    9: {"p": [0, 1, 1]},
    10: {"n": 1, "base": 2},
    11: {"a": 0, "b": 0, "total": 2},
    12: {"denominator": 3},
    13: {"first": [1, 1], "difference": [0, 1]},
    14: {"n": 1, "places": 1},
    15: {"factors": [[1, 1]]},
    16: {"factors": [[1, 1]]},
    17: {"a": 1, "b": 1, "reference_base": 2, "ratio": Fraction(1, 2)},
    18: {"dividend": 1, "divisors": [2]},
    19: {"quadratic": [1, 2, 1], "factor": [1, 1]},
    20: {"numerator": 1},
    21: {"n": 7, "divisor": 7},
    22: {"n": 1, "base": 2},
    23: {"n": 1, "base": 2},
    24: {"a": 1, "b": 9},
    25: {"a": [1, 1], "b": [2, 1], "common": [0, 1],
         "numerator_last": 1, "denominator_last": 2},
    26: {"a": [1, 1], "b": [1]},
    27: {"a": 1, "b": 1, "c": 1, "xy": 2, "yz": 2, "zx": 2},
    28: {"total": 2, "split": 1},
    29: {"factors": [[1, 1]]},
}
INITIAL = tuple(Fraction((-1) ** vertex * (vertex + 1), 7) for vertex in range(16))


def stages():
    return [{"sutra_id": sid, "operands": deepcopy(OPERANDS[sid]), "strength": Fraction(1, 3)}
            for sid in range(1, 30)]


def rational(value):
    if isinstance(value, sp.Rational):
        return Fraction(int(value.p), int(value.q))
    return Fraction(value)


def adjacency():
    return tuple(tuple(vertex ^ (2 ** axis) for axis in range(4)) for vertex in range(16))


def ordered_edges():
    return tuple(tuple(sorted((vertex, adjacent)))
                 for vertex, row in enumerate(adjacency()) for adjacent in row)


def independent_matchings():
    groups = []
    for edge in ordered_edges():
        for group in groups:
            if all(set(edge).isdisjoint(other) for other in group):
                group.append(edge)
                break
        else:
            groups.append([edge])
    return tuple(tuple(group) for group in groups)


def independent_polynomial(values, coefficients, strength):
    # Full dense matrix, derived from Hamming distance rather than the source
    # adjacency-row helper. Keep every column and every coefficient.
    matrix = tuple(tuple(4 if row == column else -1 if (row ^ column).bit_count() == 1 else 0
                         for column in range(16)) for row in range(16))
    power, result = tuple(values), tuple(values)
    for coefficient in coefficients:
        power = tuple(sum((matrix[row][column] * power[column] for column in range(16)), Fraction(0))
                      for row in range(16))
        result = tuple(value + strength * coefficient * term for value, term in zip(result, power))
    return result


def independent_rotation(values, coefficients, strength, mapping):
    edges = ordered_edges() if mapping == "ordered_edge_rotations" else \
        tuple(edge for group in independent_matchings() for edge in group)
    result = list(values)
    for coefficient in coefficients:
        t = strength * coefficient
        denominator = 1 + t * t
        for left, right in edges:
            u, v = result[left], result[right]
            result[left] = ((1 - t * t) * u - 2 * t * v) / denominator
            result[right] = (2 * t * u + (1 - t * t) * v) / denominator
    return tuple(result)


def independent_stage(values, coefficients, strength, mapping):
    if mapping == "laplacian_polynomial":
        return independent_polynomial(values, coefficients, strength)
    return independent_rotation(values, coefficients, strength, mapping)


@pytest.mark.parametrize("mapping", chain.MAPPINGS)
def test_full_chain_retains_every_computed_stage_and_sequential_state(mapping, monkeypatch):
    calls = []
    original = vedic_field.evaluate

    def observed(sid, **operands):
        calls.append(sid)
        return original(sid, **operands)

    monkeypatch.setattr(vedic_field, "evaluate", observed)
    requested = stages()
    execution = chain.execute_all_book_q4(requested, INITIAL, mapping=mapping)
    assert calls == list(range(1, 30))
    assert len(execution["stages"]) == 29
    assert execution["initial_values"] == INITIAL
    expected = INITIAL
    for index, record in enumerate(execution["stages"], 1):
        arithmetic = vedic_book.evaluate(index, **requested[index - 1]["operands"])
        coefficients = tuple(rational(value) for value in arithmetic.spectral_coefficients)
        assert record["sutra_id"] == index
        assert record["arithmetic_value"] == arithmetic.value
        assert record["spectral_coefficients"] == coefficients
        assert record["before"] == expected
        expected = independent_stage(expected, coefficients, Fraction(1, 3), mapping)
        assert record["after"] == expected
        assert len(record["before"]) == len(record["after"]) == 16
        assert all(isinstance(value, Fraction) for value in record["after"])
        assert record["norm_squared_before"] == sum(value * value for value in record["before"])
        assert record["norm_squared_after"] == sum(value * value for value in expected)
        assert record["norm_squared_change"] == record["norm_squared_after"] - record["norm_squared_before"]
        assert record["neighbors"] == adjacency()
        assert record["mapping_metadata"]["coefficient_omissions"] == 0
        assert record["mapping_metadata"]["field_model_authority"] == "new_construction"
        assert record["mapping_metadata"]["operand_preset"] == "explicit"
        if mapping == "laplacian_polynomial":
            assert record["edge_order"] == record["matchings"] == ()
        else:
            expected_edges = ordered_edges() if mapping == "ordered_edge_rotations" else \
                tuple(edge for group in independent_matchings() for edge in group)
            assert record["edge_order"] == tuple((m, left, right) for m in range(len(coefficients))
                                                  for left, right in expected_edges)
            assert record["norm_squared_change"] == 0
    assert execution["final_values"] == expected
    assert any(record["before"] != record["after"] for record in execution["stages"])


def test_polynomial_stage_retains_the_full_three_coefficient_anchor():
    record = chain.execute_book_q4(3, {"a": [1, 2], "b": [3, 4]}, [1] + [0] * 15,
                                   strength=1, mapping="laplacian_polynomial")
    assert record["spectral_coefficients"] == (3, 10, 8)
    assert record["after"] == (
        1109, -547, -547, 212, -547, 212, 212, -48,
        -547, 212, 212, -48, 212, -48, -48, 0,
    )
    assert record["norm_squared_before"] == 1
    assert record["norm_squared_after"] > 1


@pytest.mark.parametrize("mapping", chain.MAPPINGS[1:])
def test_rotation_order_retains_each_coefficient_and_both_edge_incidences(mapping):
    # Derivative coefficients are (1,0,1), including the prescribed zero.
    record = chain.execute_book_q4(9, {"p": [0, 1, 0, Fraction(1, 3)]}, INITIAL,
                                   strength=Fraction(1, 3), mapping=mapping)
    assert record["spectral_coefficients"] == (1, 0, 1)
    assert record["after"] == independent_rotation(INITIAL, (1, 0, 1), Fraction(1, 3), mapping)
    assert len(record["edge_order"]) == 3 * 64
    for m in range(3):
        occurrences = Counter((left, right) for coefficient, left, right in record["edge_order"]
                              if coefficient == m)
        assert len(occurrences) == 32
        assert set(occurrences.values()) == {2}
    if mapping == "matched_edge_rotations":
        assert record["matchings"] == independent_matchings()
        for group in record["matchings"]:
            vertices = [vertex for edge in group for vertex in edge]
            assert len(vertices) == len(set(vertices))
    else:
        assert record["matchings"] == ()
    without_tail = chain.execute_book_q4(9, {"p": [0, 1, 0, 0]}, INITIAL,
                                         strength=Fraction(1, 3), mapping=mapping)
    assert record["after"] != without_tail["after"]


def test_declared_defaults_are_resolved_from_the_function_and_recorded():
    record = chain.execute_book_q4(28, {"total": 2, "split": 1}, INITIAL,
                                   strength=1, mapping="laplacian_polynomial")
    assert record["defaulted_operands"] == ("numerator", "denominator", "sign")
    assert record["operands"] == {
        "total": Fraction(2), "split": Fraction(1), "numerator": (Fraction(0), Fraction(1)),
        "denominator": (Fraction(1), Fraction(0)), "sign": Fraction(1),
    }
    explicit = chain.execute_book_q4(28, record["operands"], INITIAL,
                                     strength=1, mapping="laplacian_polynomial")
    assert explicit["defaulted_operands"] == ()
    assert explicit["after"] == record["after"]


def test_changed_explicit_operands_reach_later_full_states():
    requested = stages()
    first = chain.execute_all_book_q4(requested, INITIAL, mapping="laplacian_polynomial")
    requested[28]["operands"] = {"factors": [[1, 2]]}
    second = chain.execute_all_book_q4(requested, INITIAL, mapping="laplacian_polynomial")
    assert first["stages"][:28] == second["stages"][:28]
    assert first["stages"][28]["before"] == second["stages"][28]["before"]
    assert first["stages"][28]["spectral_coefficients"] == (1, 1)
    assert second["stages"][28]["spectral_coefficients"] == (1, 2)
    assert first["final_values"] != second["final_values"]


@pytest.mark.parametrize("count", [0, 28, 30])
def test_incomplete_or_extra_stage_sequences_are_rejected(count, monkeypatch):
    calls = []
    original = vedic_field.evaluate

    def observed(sid, **operands):
        calls.append(sid)
        return original(sid, **operands)

    monkeypatch.setattr(vedic_field, "evaluate", observed)
    requested = stages()
    requested = requested[:count] if count <= 29 else requested + [deepcopy(requested[-1])]
    with pytest.raises(ValueError, match="exactly twenty-nine"):
        chain.execute_all_book_q4(requested, INITIAL, mapping="laplacian_polynomial")
    assert calls == []


@pytest.mark.parametrize("defect, expected", [
    ("wrong_id", "requires sutra 29 at stage 29"),
    ("extra_key", "exactly sutra_id, operands, and strength"),
    ("missing_operand", "missing a required argument"),
    ("extra_operand", "unexpected keyword argument"),
    ("inexact_nested_operand", "exact integer or rational"),
    ("inexact_strength", "exact integer or rational"),
])
def test_late_preflight_defects_reject_before_any_application_runs(defect, expected, monkeypatch):
    calls = []
    original = vedic_field.evaluate

    def observed(sid, **operands):
        calls.append(sid)
        return original(sid, **operands)

    monkeypatch.setattr(vedic_field, "evaluate", observed)
    requested = stages()
    if defect == "wrong_id":
        requested[-1] = {"sutra_id": 28, "operands": deepcopy(OPERANDS[28]), "strength": 1}
    elif defect == "extra_key":
        requested[-1]["extra"] = 1
    elif defect == "missing_operand":
        requested[-1]["operands"] = {}
    elif defect == "extra_operand":
        requested[-1]["operands"]["unknown"] = 1
    elif defect == "inexact_nested_operand":
        requested[-1]["operands"]["factors"] = [[1, 0.25]]
    elif defect == "inexact_strength":
        requested[-1]["strength"] = 0.25
    with pytest.raises(chain.BookQ4StageError, match=expected) as raised:
        chain.execute_all_book_q4(requested, INITIAL, mapping="laplacian_polynomial")
    error = raised.value
    assert calls == []
    assert error.stage_index == 29
    assert error.phase == "preflight"
    assert error.completed_stages == ()
    assert error.input_values == INITIAL
    assert isinstance(error.__cause__, (ValueError, TypeError))
    if defect in {"missing_operand", "extra_operand", "inexact_nested_operand", "inexact_strength"}:
        assert error.sutra_id == 29


def test_mathematical_stage_failure_keeps_completed_states_and_original_cause(monkeypatch):
    calls = []
    original = vedic_field.evaluate

    def observed(sid, **operands):
        calls.append(sid)
        return original(sid, **operands)

    monkeypatch.setattr(vedic_field, "evaluate", observed)
    requested = stages()
    requested[4]["operands"] = {"a": 1, "b": 2, "c": 3, "d": 5}
    with pytest.raises(chain.BookQ4StageError, match="stage 5") as raised:
        chain.execute_all_book_q4(requested, INITIAL, mapping="laplacian_polynomial")
    error = raised.value
    assert calls == [1, 2, 3, 4, 5]
    assert error.sutra_id == error.stage_index == 5
    assert error.phase == "execution"
    assert len(error.completed_stages) == 4
    assert isinstance(error.__cause__, ValueError)
    assert "independent-term products must be equal" in str(error.__cause__)
    expected = INITIAL
    for record in error.completed_stages:
        assert record["before"] == expected
        expected = independent_polynomial(expected, record["spectral_coefficients"], record["strength"])
        assert record["after"] == expected
    assert error.input_values == expected


@pytest.mark.parametrize("bad", [True, 0.25, "1/4", Decimal("0.25"), complex(1, 0), sp.Float("0.25")])
@pytest.mark.parametrize("location", ["value", "operand", "strength"])
def test_exact_boundaries_reject_coercion(bad, location):
    values = list(INITIAL)
    operands = {"p": [0, 1]}
    strength = Fraction(1)
    if location == "value":
        values[15] = bad
    elif location == "operand":
        operands["p"][1] = bad
    else:
        strength = bad
    with pytest.raises((TypeError, chain.BookQ4StageError)):
        chain.execute_book_q4(9, operands, values, strength=strength, mapping="laplacian_polynomial")


@pytest.mark.parametrize("mapping, method", [
    ("ordered_edge_rotations", "rotate_graph"),
    ("matched_edge_rotations", "rotate_matched"),
])
def test_rotation_norm_failure_is_visible(mapping, method, monkeypatch):
    def altered(self, values, neighbors, strength):
        return tuple(2 * value for value in values)

    monkeypatch.setattr(vedic_field.FieldBinding, method, altered)
    with pytest.raises(chain.BookQ4StageError, match="squared norm changed") as raised:
        chain.execute_book_q4(9, {"p": [0, 1]}, INITIAL, strength=1, mapping=mapping)
    assert isinstance(raised.value.__cause__, ArithmeticError)
    assert raised.value.input_values == INITIAL
    assert raised.value.completed_stages == ()


def test_prescribed_zero_coefficient_is_executed_and_retained(monkeypatch):
    calls = []
    original = vedic_field.FieldBinding.rotate_graph

    def observed(self, values, neighbors, strength):
        calls.append(self.result.spectral_coefficients)
        return original(self, values, neighbors, strength)

    monkeypatch.setattr(vedic_field.FieldBinding, "rotate_graph", observed)
    record = chain.execute_book_q4(9, {"p": [7]}, INITIAL, strength=1,
                                   mapping="ordered_edge_rotations")
    assert calls == [(0,)]
    assert record["spectral_coefficients"] == (0,)
    assert record["after"] == INITIAL
    assert len(record["edge_order"]) == 64


@pytest.mark.parametrize("bad", [None, "", "best", 1])
def test_mapping_is_explicit_and_unknown_choices_do_not_select_a_default(bad):
    with pytest.raises((TypeError, ValueError)):
        chain.execute_book_q4(9, {"p": [0, 1]}, INITIAL, strength=1, mapping=bad)
