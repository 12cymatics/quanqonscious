"""Execution and provenance checks for the source-defined continuation."""
from collections import Counter
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path

import pytest
import sympy as sp

from core.exact_codec import decode_exact
import grvq_exact_assets as assets
from vedic_book_cases import EXAMPLE_PARAMETERS


def explicit_stages():
    # This is a named source fixture explicitly supplied by tests, never an API default.
    return [{"sutra_id": sid, "operands": dict(EXAMPLE_PARAMETERS[sid]),
             "strength": Fraction(1, sid + 1)} for sid in range(1, 30)]


def independent_edge_rotation(values, coefficients, strength, matched):
    edges = [(min(v, v ^ (1 << axis)), max(v, v ^ (1 << axis)))
             for v in range(16) for axis in range(4)]
    groups = []
    if matched:
        for edge in edges:
            for group in groups:
                if all(not (set(edge) & set(other)) for other in group):
                    group.append(edge)
                    break
            else:
                groups.append([edge])
        edges = [edge for group in groups for edge in group]
    state = list(map(Fraction, values))
    for coefficient in coefficients:
        t = strength * coefficient
        c, s = (1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)
        for left, right in edges:
            u, v = state[left], state[right]
            state[left], state[right] = c * u - s * v, s * u + c * v
    return tuple(state)


def test_transition_facade_preserves_distinct_orders_and_all_edge_occurrences():
    initial = tuple(range(16))
    outputs = []
    for mapping, matched in (("ordered_edge_rotations", False), ("matched_edge_rotations", True)):
        execution = assets.execute_book_q4(
            3, {"a": [1, 1], "b": [1]}, initial, strength=Fraction(1, 2), mapping=mapping,
        )
        expected = independent_edge_rotation(initial, (1, 1), Fraction(1, 2), matched)
        assert execution.value == expected
        assert execution.value != initial
        transition = execution.receipt["details"]["transition"]
        order = decode_exact(transition["edge_order"])
        assert len(order) == 128
        for index in (0, 1):
            counts = Counter((left, right) for coefficient, left, right in order if coefficient == index)
            assert len(counts) == 32
            assert set(counts.values()) == {2}
        assert decode_exact(transition["norm_squared_before"]) == 1240
        assert decode_exact(transition["norm_squared_after"]) == 1240
        assert transition["mapping_metadata"]["field_model_authority"] == "new_construction"
        assert transition["mapping_metadata"]["operand_preset"] == "explicit"
        outputs.append(execution.value)
    assert outputs[0] != outputs[1]


def test_all29_receipt_retains_complete_state_chain_and_sources():
    initial = tuple(range(16))
    execution = assets.execute_all_book_q4(explicit_stages(), initial, mapping="laplacian_polynomial")
    details = execution.receipt["details"]
    # Some book results are exact symbolic solution sets. Decode the rational
    # transition fields here without attempting to evaluate those expressions.
    encoded_stages = details["stages"]["items"]
    stages = [{key: decode_exact(stage[key]) for key in
               ("sutra_id", "before", "after", "spectral_coefficients")}
              for stage in encoded_stages]
    assert len(stages) == 29
    assert tuple(stage["sutra_id"] for stage in stages) == tuple(range(1, 30))
    assert stages[0]["before"] == initial
    for previous, current in zip(stages, stages[1:]):
        assert previous["after"] == current["before"]
    assert all(len(stage["before"]) == len(stage["after"]) == 16 for stage in stages)
    assert all(stage["spectral_coefficients"] for stage in stages)
    assert all("arithmetic_value" in stage for stage in encoded_stages)
    assert encoded_stages[4]["arithmetic_value"]["$exact"] == "symbolic"
    assert execution.value == stages[-1]["after"]
    assert execution.value != initial
    assert decode_exact(details["execution_order"]) == tuple(range(1, 30))
    assert decode_exact(details["stage_count"]) == 29
    source = next(record for record in execution.receipt["implementation_sources"]
                  if record["path"] == "vedic_book.py")
    assert len(source["functions"]) == 30  # evaluate and each actual arithmetic operation
    assert "missing canonical field sutras" in details["canonical_field_coupling"]
    for record in execution.receipt["implementation_sources"]:
        content = (Path(assets.__file__).resolve().parent / record["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == record["sha256"]


def test_hodge_keeps_both_chiralities_and_source_normalization():
    pair_values = {(0, 1): 1, (0, 2): 2, (0, 3): 3, (1, 2): 4, (1, 3): 5, (2, 3): 6}
    faces = {cell: pair_values[cell[1:]] for cell in assets.q4_complex.plaquettes()}
    execution = assets.evaluate_q4_hodge(faces)
    value = execution.value
    assert value["plaquette_cochain"] == faces
    assert value["face_count"] == 24
    assert value["vertex_count"] == 16
    assert value["source_pressure"] == Fraction(91, 12)
    for vertex, record in enumerate(value["vertices"]):
        assert record["vertex"] == vertex
        assert record["local_bivector6"] == (1, 2, 3, 4, 5, 6)
        assert record["split"]["star6"] == (6, -5, 4, 3, -2, 1)
        assert record["split"]["reconstruction6"] == (1, 2, 3, 4, 5, 6)
        assert record["source_plus_triple"] == (Fraction(7, 12), Fraction(-1, 4), Fraction(7, 12))
        assert record["source_minus_triple"] == (Fraction(-5, 12), Fraction(7, 12), Fraction(-1, 12))
        assert record["source_strength_squared"] == Fraction(91, 72)
        assert record["source_strength"] == sp.sqrt(182) / 12
    assert "Euclidean" in execution.receipt["details"]["metric_scope"]


def test_hodge_cli_and_existing_full_q4_diagnostics_execute_same_faces():
    faces = {cell: Fraction(index - 11, 3) for index, cell in enumerate(assets.q4_complex.plaquettes())}
    request = {"operation": "q4_hodge", "arguments": {
        "plaquette_cochain": {"cochain": [
            [list(cell), {"rational": [value.numerator, value.denominator]}]
            for cell, value in faces.items()
        ]},
    }}
    direct = assets.evaluate_q4_hodge(faces)
    assert assets.execute_request(request).value == direct.value
    assert assets.q4_diagnostics(range(16), plaquette_cochain=faces).value["plaquette_hodge"] == direct.value
    sources = {record["path"] for record in assets.q4_diagnostics(range(16)).receipt["implementation_sources"]}
    assert "grvq_q4_hodge.py" not in sources


def test_hodge_six_component_request_retains_anti_self_dual_input():
    initial = (1, 2, 3, -3, 2, -1)
    execution = assets.execute_request({"operation": "hodge6", "arguments": {"values": list(initial)}})
    assert execution.value["self_dual6"] == (0,) * 6
    assert execution.value["anti_self_dual6"] == initial
    assert execution.value["reconstruction6"] == initial
    sources = {record["path"] for record in execution.receipt["implementation_sources"]}
    assert "vedic_trainer/vedic/kernel/q4_complex.py" in sources


def test_chain_cli_rejects_missing_stage_without_a_success_receipt(monkeypatch, capsys):
    request = {"operation": "book_q4_chain", "arguments": {
        "stages": [{"sutra_id": sid, "operands": dict(EXAMPLE_PARAMETERS[sid]), "strength": 1}
                   for sid in range(1, 29)],
        "values": list(range(16)), "mapping": "laplacian_polynomial",
    }}
    payload = json.dumps(request, default=lambda value: {"rational": [value.numerator, value.denominator]})
    monkeypatch.setattr(assets.sys, "stdin", io.StringIO(payload))
    assert assets.main(["-"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "twenty-nine" in json.loads(output.err)["message"]


def test_chain_cli_keeps_failed_stage_cause_and_completed_full_states(monkeypatch, capsys):
    stages = explicit_stages()
    stages[3]["operands"] = {"dividend": [1, 2, 1], "divisor": [0]}
    request = {"operation": "book_q4_chain", "arguments": {
        "stages": stages, "values": list(range(16)), "mapping": "laplacian_polynomial",
    }}
    payload = json.dumps(request, default=lambda value: {"rational": [value.numerator, value.denominator]})
    monkeypatch.setattr(assets.sys, "stdin", io.StringIO(payload))
    assert assets.main(["-"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    failure = json.loads(output.err)
    assert failure["error"] == "BookQ4StageError"
    assert failure["stage_index"] == failure["sutra_id"] == 4
    assert failure["phase"] == "execution"
    assert failure["cause"] == {"type": "ValueError", "message": "The leading divisor coefficient must be nonzero"}
    completed = decode_exact(failure["completed_stages"])
    assert tuple(stage["sutra_id"] for stage in completed) == (1, 2, 3)
    assert all(len(stage["before"]) == len(stage["after"]) == 16 for stage in completed)
    assert decode_exact(failure["input_values"]) == completed[-1]["after"]


@pytest.mark.parametrize("operation,arguments", [
    ("hodge6", {"values": [0, 0, 0, 0, 0, 1.0]}),
    ("book_q4_transition", {"sutra_id": 3, "operands": {"a": [1], "b": [1]},
                            "values": list(range(16)), "strength": 0.5,
                            "mapping": "ordered_edge_rotations"}),
])
def test_continuation_json_boundaries_reject_float_data(operation, arguments):
    with pytest.raises(TypeError):
        assets.execute_request({"operation": operation, "arguments": arguments})
