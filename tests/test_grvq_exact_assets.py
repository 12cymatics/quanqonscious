"""Integration values, provenance boundaries, and complete-domain regressions."""
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path

import pytest
import sympy as sp

from core.exact_codec import decode_exact
import grvq_exact_assets as assets
import vedic_book
from vedic_book_cases import EXAMPLE_PARAMETERS


def test_canonical_scalar_receipt_keeps_exact_origin_and_missing_dependencies_visible():
    execution = assets.evaluate_canonical_field(0, 0, 0)
    assert execution.value == sp.Rational(19999999700000001, 2)
    assert execution.receipt["calculation"] == "canonical_GRVQ_scalar"
    activity = execution.receipt["details"]["parameter_activity"]
    assert len(activity) == 10
    assert {name for name, detail in activity.items() if detail["occurs_in_scalar"]} == {
        "r0_squared", "shape_damping_S1", "shape_damping_S2", "k_vedic", "epsilon", "gamma",
    }
    assert execution.receipt["details"]["unresolved_full_model_requirements"]["items"]


def test_canonical_scalar_retains_all_factors_at_a_nontrivial_point():
    execution = assets.evaluate_canonical_field(1, 1, Fraction(1, 2))
    s1 = sp.sin(1) * sp.cos(sp.Rational(1, 2)) * sp.exp(sp.Rational(-1, 10))
    s2 = sp.cos(1) * sp.sin(sp.Rational(1, 2)) * sp.exp(sp.Rational(-1, 20))
    expected = ((1 - 1 / (sp.Abs(s1) + sp.Rational(1, 10 ** 8)))
                * (1 - 2 / (sp.Abs(s2) + sp.Rational(1, 10 ** 8)))
                * sp.Rational(1, 2) * (sp.sin(sp.Rational(5, 2)) + sp.cos(5) / 2)
                * sp.Rational(3, 2))
    assert execution.value == expected


def test_same_sutra_name_keeps_arithmetic_and_field_formulas_separate():
    arithmetic = assets.evaluate_book_sutra(2, {"a": 98, "b": 97, "base": 100})
    field = assets.evaluate_field_sutra("nikhilam", {"value": 98, "base": 100}, 1, 1, Fraction(1, 2))
    assert arithmetic.value == 9506
    assert field.value == 1
    assert arithmetic.receipt["calculation"] != field.receipt["calculation"]


def test_field_crosswise_includes_every_explicit_component():
    execution = assets.evaluate_field_sutra(
        "urdhva_tiryak", {"a": [2, 3], "b": [5, 7]}, 0, 0, 0,
        components=[[11, 13], [17, 19]],
    )
    assert execution.value == 2 * 5 * 11 + 2 * 7 * 13 + 3 * 5 * 17 + 3 * 7 * 19


def test_json_canonical_scalar_executes_exact_named_pi():
    execution = assets.execute_request({"operation": "canonical_field", "arguments": {
        "r": 0, "theta": 0, "phi": {"constant": "pi"},
    }})
    assert execution.value == sp.Rational(19999999700000001, 2)


def test_book_arithmetic_has_explicit_operands_and_separate_identity():
    execution = assets.evaluate_book_sutra(2, {"a": 98, "b": 97, "base": 100})
    assert execution.value == 9506
    assert execution.receipt["calculation"] == "vedic_book_arithmetic"
    assert decode_exact(execution.receipt["details"]["spectral_coefficients"]) == (2, 3, 9500, 6)
    assert "Not supplied" in execution.receipt["details"]["canonical_field_coupling"]


@pytest.mark.parametrize("sid", range(1, 30))
def test_every_book_application_remains_callable_through_the_facade(sid, monkeypatch):
    calls = []
    original = vedic_book.evaluate

    def observed(actual_sid, **operands):
        calls.append((actual_sid, operands))
        return original(actual_sid, **operands)

    monkeypatch.setattr(vedic_book, "evaluate", observed)
    explicit_operands = EXAMPLE_PARAMETERS[sid]
    execution = assets.evaluate_book_sutra(sid, explicit_operands)
    assert calls == [(sid, explicit_operands)]
    assert decode_exact(execution.receipt["inputs"]["sutra_id"]) == sid
    assert execution.receipt["details"]["application"] == vedic_book.SUTRAS[sid - 1].application
    assert execution.receipt["details"]["spectral_coefficients"]["items"]


@pytest.mark.parametrize("missing", [None, [], "example"])
def test_missing_operands_never_select_a_preset(missing):
    with pytest.raises(TypeError, match="explicit operand mapping"):
        assets.evaluate_book_sutra(2, missing)
    with pytest.raises(TypeError, match="explicit operand mapping"):
        assets.apply_book_sutra_q4(2, missing, [0] * 16, strength=1)


def test_authored_mapping_retains_every_coefficient_and_laplacian_sign():
    # For delta_0, L delta has shell values (4,-1,0,0,0), L^2 has
    # (20,-8,2,0,0), L^3 has (112,-58,24,-6,0), from Q4 walk counts.
    # Coefficients of (1+2x)(3+4x) are (3,10,8).
    execution = assets.apply_book_sutra_q4(
        3, {"a": [1, 2], "b": [3, 4]}, [1] + [0] * 15, strength=1,
    )
    assert execution.value == (
        1109, -547, -547, 212, -547, 212, 212, -48,
        -547, 212, 212, -48, 212, -48, -48, 0,
    )
    details = execution.receipt["details"]
    assert decode_exact(details["spectral_coefficients"]) == (3, 10, 8)
    assert details["mapping_receipt"]["operand_preset"] == "explicit"
    assert details["mapping_receipt"]["field_model_authority"] == "new_construction"
    assert "-q4_complex.laplacian" in details["laplacian_sign"]


def test_q4_retains_the_highest_weight_walsh_mode():
    parity = tuple(Fraction((-1) ** bin(v).count("1")) for v in range(16))
    execution = assets.q4_diagnostics(parity)
    assert execution.value["values"] == parity
    assert execution.value["walsh_coefficients"] == (Fraction(0),) * 15 + (Fraction(16),)
    assert execution.value["walsh_reconstruction"] == parity
    assert execution.value["laplacian"] == tuple(-8 * value for value in parity)
    assert len(execution.value["gradient"]) == 64
    assert len(execution.value["gradient_curl"]) == 24
    assert set(execution.value["gradient_curl"].values()) == {0}
    assert decode_exact(execution.receipt["details"]["cell_counts"]) == (16, 32, 24, 8, 1)
    source = next(item for item in execution.receipt["implementation_sources"]
                  if item["path"].endswith("q4_complex.py"))
    assert "d2" not in source["functions"]
    assert "d3" not in source["functions"]


def test_independent_cochains_do_not_inherit_zero_gradient_curl():
    q4 = assets.q4_complex
    edges = q4.d0((Fraction(0),) * 16)
    edges[(0, 1)], edges[(1, 0)] = Fraction(1), Fraction(-1)
    faces = {face: Fraction(0) for face in q4.plaquettes()}
    faces[(0, 0, 1)] = Fraction(1)
    cubes = {cube: Fraction(0) for cube in q4.cubes()}
    cubes[(0, 1, 2, 3)] = Fraction(1)
    result = assets.q4_diagnostics([0] * 16, edge_cochain=edges,
                                   plaquette_cochain=faces, cube_cochain=cubes).value
    assert {key: value for key, value in result["edge_curl"].items() if value} == {
        (0, 0, 1): 1, (0, 0, 2): 1, (0, 0, 3): 1,
    }
    assert set(result["edge_curl_coboundary"].values()) == {0}
    assert {key: value for key, value in result["plaquette_coboundary"].items() if value} == {
        (0, 0, 1, 2): -1, (0, 0, 1, 3): -1,
    }
    assert tuple(result["plaquette_double_coboundary"].values()) == (0,)
    assert tuple(result["cube_coboundary"].values()) == (-1,)


def test_k2_division_uses_the_hermitian_norm_in_its_own_field():
    # 1/(1+sqrt(2)) = -1+sqrt(2); H=3+2sqrt(2), total norm=1.
    execution = assets.divide_k2((1, 0, 0, 0), (1, 1, 0, 0))
    assert execution.value.coords() == (-1, 1, 0, 0)
    details = execution.receipt["details"]
    assert decode_exact(details["denominator_hermitian_norm"]["coordinates"]) == (3, 2)
    assert decode_exact(details["denominator_total_norm"]) == 1
    assert details["hermitian_norm_codomain"] == "Q(sqrt(2))"
    assert details["total_norm_codomain"] == "Q"


def test_k2_mixed_complex_inverse_has_independent_coordinates():
    result = assets.divide_k2((1, 0, 0, 0), (1, 1, 1, -1)).value
    assert result.coords() == (Fraction(1, 6), Fraction(1, 6), Fraction(-1, 6), Fraction(1, 6))


def test_c4_division_retains_the_golden_ratio_extension():
    one = ((1, 0, 0, 0), (0, 0, 0, 0))
    phi = ((Fraction(1, 2), 0, 0, 0), (Fraction(1, 2), 0, 0, 0))
    result = assets.divide_c4(one, phi).value
    assert result.u.coords() == (Fraction(-1, 2), 0, 0, 0)
    assert result.v.coords() == (Fraction(1, 2), 0, 0, 0)


def test_division_by_zero_is_visible():
    with pytest.raises(ZeroDivisionError):
        assets.divide_k2((1, 0, 0, 0), (0, 0, 0, 0))
    with pytest.raises(ZeroDivisionError):
        assets.divide_c4(((1, 0, 0, 0), (0, 0, 0, 0)), ((0, 0, 0, 0), (0, 0, 0, 0)))


@pytest.mark.parametrize("value", [0.1, sp.Float("0.1"), True, "1/3", complex(1)])
def test_exact_asset_boundaries_reject_inexact_or_coerced_values(value):
    with pytest.raises(TypeError):
        assets.q4_diagnostics([value] + [0] * 15)
    with pytest.raises(TypeError):
        assets.divide_k2((value, 0, 0, 0), (1, 0, 0, 0))
    with pytest.raises(TypeError):
        assets.evaluate_book_sutra(2, {"a": value, "b": 3, "base": 10})


@pytest.mark.parametrize("count", [0, 5, 15, 17])
def test_q4_requires_the_complete_vertex_domain(count):
    with pytest.raises(ValueError, match="sixteen"):
        assets.q4_diagnostics([0] * count)


def test_receipt_is_a_detached_snapshot_of_explicit_operands():
    operands = {"a": [1, 2], "b": [3, 4]}
    execution = assets.evaluate_book_sutra(3, operands)
    saved = json.dumps(execution.receipt, sort_keys=True)
    operands["a"][1] = 999
    assert execution.value == (3, 10, 8)
    assert json.dumps(execution.receipt, sort_keys=True) == saved


def test_receipt_digest_binds_the_output_and_runtime_source_bytes():
    receipt = assets.evaluate_book_sutra(10, {"n": 299792458, "base": 300000000}).receipt
    payload = dict(receipt)
    expected = payload.pop("receipt_sha256")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert hashlib.sha256(encoded).hexdigest() == expected
    root = Path(assets.__file__).resolve().parent
    for source in receipt["implementation_sources"]:
        assert hashlib.sha256((root / source["path"]).read_bytes()).hexdigest() == source["sha256"]
    assert decode_exact(receipt["value"]) == 89875517873681764


def test_large_rationals_remain_unbounded_by_decimal_serialization():
    large = Fraction(2 ** 20000 + 7, 3 ** 1000)
    execution = assets.q4_diagnostics((large,) + (Fraction(0),) * 15)
    assert execution.value["walsh_reconstruction"][0] == large
    assert decode_exact(execution.receipt["inputs"]["values"])[0] == large
    assert len(json.dumps(execution.receipt)) > 20000


@pytest.mark.parametrize("rational_denominator", [False, True])
def test_symbolic_receipts_preserve_large_exact_coefficient_atoms(rational_denominator):
    # Gamma is a permitted exact input even when its literal occurrence is
    # multiplied by sin(0). Keeping that literal factor must not impose a
    # decimal-printing bound on the coefficient.
    numerator = 2 ** 20000 + 1
    denominator = numerator + 2 if rational_denominator else 1
    try:
        execution = assets.evaluate_canonical_field(
            0, 0, 0, parameters={"gamma": Fraction(numerator, denominator)},
        )
    except ValueError as error:
        raise AssertionError("The receipt rejected a permitted exact coefficient during serialization") from error
    assert execution.value == sp.Rational(19999999700000001, 2)
    literal = execution.receipt["details"]["literal_expression"]["srepr"]
    expected_atom = (f"Rational({hex(numerator)}, {hex(denominator)})" if rational_denominator
                     else f"Integer({hex(numerator)})")
    assert expected_atom in literal
    assert decode_exact(execution.receipt["inputs"]["parameters"]["gamma"]) == Fraction(numerator, denominator)


def test_symbolic_receipts_preserve_function_arguments_and_symbol_assumptions():
    symbol = sp.Symbol("x", real=True)
    argument = sp.Add(symbol, sp.Rational(1, 3), evaluate=False)
    expression = sp.sin(argument, evaluate=False)
    assert assets._encode(expression) == {
        "$exact": "symbolic",
        "srepr": "sin(Add(Symbol('x', real=True), Rational(0x1, 0x3)))",
    }


def test_json_entrypoint_executes_real_arithmetic(monkeypatch, capsys):
    request = {"operation": "book_sutra", "arguments": {
        "sutra_id": 2, "operands": {"a": {"rational": [2, 3]}, "b": 9, "base": 10},
    }}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(request)))
    assert assets.main(["-"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert decode_exact(output["value"]) == 6


@pytest.mark.parametrize("value", [0.5, True])
def test_json_entrypoint_reports_rejected_exact_inputs(monkeypatch, capsys, value):
    request = {"operation": "q4", "arguments": {"values": [value] + [0] * 15}}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(request)))
    assert assets.main(["-"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "TypeError"


def test_unknown_operations_are_errors():
    with pytest.raises(ValueError, match="Unknown"):
        assets.execute_request({"operation": "assume_full_engine", "arguments": {}})


def test_json_cochain_encoding_preserves_complete_independent_cells():
    cochain = [[list(key), 1 if key == (0, 0, 1) else 0] for key in assets.q4_complex.plaquettes()]
    request = {"operation": "q4", "arguments": {
        "values": [0] * 16, "plaquette_cochain": {"cochain": cochain},
    }}
    execution = assets.execute_request(request)
    assert execution.value["plaquette_coboundary"][(0, 0, 1, 2)] == -1
    assert tuple(execution.value["plaquette_double_coboundary"].values()) == (0,)
    source = next(item for item in execution.receipt["implementation_sources"]
                  if item["path"].endswith("q4_complex.py"))
    assert "d2" in source["functions"] and "d3" in source["functions"]


def test_json_cochain_encoding_rejects_duplicate_or_noninteger_cell_keys():
    with pytest.raises(ValueError, match="repeats"):
        assets._decode_request({"cochain": [[[0, 1], 1], [[0, 1], 2]]})
    with pytest.raises(TypeError, match="integers"):
        assets._decode_request({"cochain": [[[False, 1], 1]]})
