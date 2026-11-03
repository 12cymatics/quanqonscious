from fractions import Fraction
import copy
import json

from core.exact_codec import encode_exact, decode_exact
from core.lattice import create_3d_lattice
from core.operators.base import OperatorContext, OperatorTrace
from core.operators.sutra_ops import get_sutra_by_number
from core.state import RationalComplex, create_zero_field, state_digest
from core.trace import EvolutionTrace, TraceReplayer
from vedic_book import BOOK
from vedic_field import FieldBinding


def field(tail=Fraction(1, 2**5000)):
    state = create_zero_field(create_3d_lattice(2, 1, 1))
    state.set_by_coords((0, 0, 0), RationalComplex(1, 0))
    state.set_by_coords((1, 0, 0), RationalComplex(tail, 0))
    return state


def test_large_rational_states_have_distinct_full_digests_and_traces():
    traces = []
    for numerator in (1, 2):
        trace = OperatorTrace()
        context = OperatorContext(trace=trace, dt=Fraction(1, 3),
                                  parameters={"sutra_operands": {9: {"p": [0, 1]}}})
        get_sutra_by_number(9)(field(Fraction(numerator, 2**5000)), context)
        traces.append(trace)
    assert state_digest(field()) != state_digest(field(Fraction(2, 2**5000)))
    assert not traces[0].verify_determinism(traces[1])


def test_recursive_exact_codec_preserves_large_values_and_mapping_keys():
    value = {9: {"p": [Fraction(1, 2**20000), 2**20000, (Fraction(-3, 7),)]},
             "9": {"$exact": "a user key"}, "float": 0.1}
    assert decode_exact(json.loads(json.dumps(encode_exact(value)))) == value


def test_trace_roundtrip_replays_explicit_operands_context_and_receipts():
    initial = field()
    trace = EvolutionTrace()
    trace.start(initial)
    operands = {9: {"p": [Fraction(1, 3), 1, Fraction(1, 7)]}}
    context = OperatorContext(dt=Fraction(2, 7), trace=trace.operator_trace,
                              parameters={"sutra_operands": operands})
    operator = get_sutra_by_number(9)
    final = operator(initial, context)
    trace.finish(final)
    restored = EvolutionTrace.from_json(trace.to_json())
    assert trace.operator_trace.verify_determinism(restored.operator_trace)
    assert restored.operator_trace.entries[0].parameters["sutra_operands"] == operands
    operands[9]["p"][1] = 1000
    assert trace.operator_trace.entries[0].parameters["sutra_operands"][9]["p"][1] == 1
    replayer = TraceReplayer()
    replayer.register_operator(operator)
    output, verified, errors = replayer.replay(initial, restored)
    assert verified, errors
    assert state_digest(output) == state_digest(final)
    restored.operator_trace.entries[0].parameters["sutra_execution"]["binding_sha256"] = "invalid"
    _, verified, errors = replayer.replay(initial, restored)
    assert not verified
    assert any("receipt mismatch" in error for error in errors)


def test_composite_trace_depth_survives_json_replay():
    operator = get_sutra_by_number(9) >> get_sutra_by_number(10)
    initial = field(Fraction(1, 7))
    trace = EvolutionTrace()
    trace.start(initial)
    final = operator(initial, OperatorContext(trace=trace.operator_trace))
    trace.finish(final)
    restored = EvolutionTrace.from_json(trace.to_json())
    assert [entry.depth for entry in restored.operator_trace.entries] == [1, 1, 0]
    replayer = TraceReplayer()
    replayer.register_operator(operator)
    output, verified, errors = replayer.replay(initial, restored)
    assert verified, errors
    assert state_digest(output) == state_digest(final)


def test_receipt_uses_the_same_canonical_topology_as_execution():
    binding = FieldBinding.from_parameters(9, {"p": [0, 1]})
    adjacency = ((Fraction(1),), (Fraction(0),))
    result = binding.rotate_graph((1, 0), adjacency)
    first = binding.receipt((1, 0), result, adjacency, 1, "edge_rotations")
    second = binding.receipt((1, 0), result, ((1,), (0,)), 1, "edge_rotations")
    assert first == second


def test_receipt_owns_a_snapshot_of_book_metadata():
    binding = FieldBinding.from_parameters(9)
    receipt = binding.receipt((1,), (1,), ((),), 1, "edge_rotations")
    saved = copy.deepcopy(receipt)
    author = BOOK["author"]
    try:
        BOOK["author"] = "changed"
        assert receipt == saved
    finally:
        BOOK["author"] = author
