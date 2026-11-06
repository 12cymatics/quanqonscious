"""Complete, explicit book arithmetic stages on the existing rational Q4 maps.

These are the authored mappings in ``vedic_field.FieldBinding``. They do not
define the missing field-coupled GRVQ sutras, a metric, or geodesic evolution.
The polynomial, ordered rotations, and matched rotations retain their separate
equations and orders. Every supplied coefficient and adjacency incidence is
used by the unchanged source methods.

The complete-chain interface requires stages 1 through 29 in that order. Each
stage supplies its own arithmetic operands and strength; only its full Q4
output becomes the following stage's Q4 input. The whole request is checked
for exact values and argument/schema errors before any arithmetic application
runs. Mathematical domain failures remain failures at their actual stage.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from inspect import signature

import vedic_book
import vedic_field
from vedic_trainer.vedic.kernel import q4_complex


MAPPINGS = (
    "laplacian_polynomial",
    "ordered_edge_rotations",
    "matched_edge_rotations",
)
_METADATA_MAPPING = {
    "laplacian_polynomial": "laplacian_polynomial",
    "ordered_edge_rotations": "edge_rotations",
    "matched_edge_rotations": "matched_edge_rotations",
}


class BookQ4StageError(RuntimeError):
    """A stage failed; its completed predecessors are evidence, not success.

    ``stage_index`` is one-based. ``sutra_id`` is None when it could not be
    read as an exact integer. ``completed_stages`` contains full stage records;
    ``input_values`` is the full state entering the failed stage. The original
    exception is retained as ``__cause__`` by every raising call site.
    """

    def __init__(self, stage_index, sutra_id, phase, mapping, completed_stages,
                 input_values, error):
        self.stage_index = stage_index
        self.sutra_id = sutra_id
        self.phase = phase
        self.mapping = mapping
        self.completed_stages = tuple(completed_stages)
        self.input_values = tuple(input_values)
        identity = "unvalidated" if sutra_id is None else str(sutra_id)
        super().__init__(
            f"Book Q4 stage {stage_index} (sutra {identity}) failed during "
            f"{phase}: {type(error).__name__}: {error}"
        )


def _mapping(value):
    if not isinstance(value, str):
        raise TypeError("Supply an explicit mapping name")
    if value not in MAPPINGS:
        raise ValueError(f"Unknown book Q4 mapping; expected one of {MAPPINGS}")
    return value


def _fraction(value):
    rational = vedic_book.rational(value)
    return Fraction(int(rational.p), int(rational.q))


def _values(values):
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError("Supply an ordered sequence of sixteen exact rational Q4 values")
    if len(values) != 16:
        raise ValueError("Q4 requires all sixteen vertex values")
    return tuple(_fraction(value) for value in values)


def _operand_value(value):
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError("Book operands must be exact rationals or ordered sequences of them")
    if isinstance(value, Sequence):
        return tuple(_operand_value(item) for item in value)
    return _fraction(value)


def _prepare_stage(sutra_id, operands, strength):
    sid = vedic_book.integer(sutra_id)
    if not 1 <= sid <= 29:
        raise ValueError("The book sutra id must be between one and twenty-nine")
    if not isinstance(operands, Mapping) or not all(isinstance(k, str) for k in operands):
        raise TypeError("Supply an explicit operand mapping with string names")
    supplied = {name: _operand_value(value) for name, value in operands.items()}
    operation_signature = signature(vedic_book.OPERATIONS[sid - 1])
    bound = operation_signature.bind(**supplied)
    defaulted = tuple(name for name in operation_signature.parameters if name not in bound.arguments)
    bound.apply_defaults()
    resolved = {name: _operand_value(value) for name, value in bound.arguments.items()}
    return {
        "sutra_id": sid,
        "operands": resolved,
        "defaulted_operands": defaulted,
        "strength": _fraction(strength),
    }


def _preflight_stage(raw_stage, index, mapping, initial_values, *, required_id=None):
    sid = None
    try:
        if not isinstance(raw_stage, Mapping):
            raise TypeError("Every book stage must be an explicit mapping")
        if set(raw_stage) != {"sutra_id", "operands", "strength"}:
            raise ValueError("Every book stage requires exactly sutra_id, operands, and strength")
        sid = vedic_book.integer(raw_stage["sutra_id"])
        prepared = _prepare_stage(sid, raw_stage["operands"], raw_stage["strength"])
        if required_id is not None and sid != required_id:
            raise ValueError(f"The complete chain requires sutra {required_id} at stage {index}")
        return prepared
    except Exception as error:
        raise BookQ4StageError(index, sid, "preflight", mapping, (), initial_values, error) from error


def _execute_stage(prepared, before, mapping, index, completed_stages):
    sid = prepared["sutra_id"]
    try:
        binding = vedic_field.FieldBinding.from_parameters(sid, prepared["operands"])
        coefficients = tuple(_fraction(c) for c in binding.result.spectral_coefficients)
        neighbors = tuple(q4_complex.neighbours(v) for v in range(16))
        strength = prepared["strength"]
        matchings = ()
        edge_order = ()
        if mapping == "laplacian_polynomial":
            after = binding.apply_graph(before, neighbors, strength)
        elif mapping == "ordered_edge_rotations":
            after = binding.rotate_graph(before, neighbors, strength)
            edges = tuple((min(v, adjacent), max(v, adjacent))
                          for v, row in enumerate(neighbors) for adjacent in row if v != adjacent)
            edge_order = tuple((m, left, right) for m in range(len(coefficients)) for left, right in edges)
        elif mapping == "matched_edge_rotations":
            after = binding.rotate_matched(before, neighbors, strength)
            matchings = vedic_field.edge_matchings(neighbors)
            edge_order = tuple((m, left, right) for m in range(len(coefficients))
                               for matching in matchings for left, right in matching)
        else:
            raise ValueError("A prepared stage has an unknown mapping")
        after = _values(after)
        norm_before = sum((value * value for value in before), Fraction(0))
        norm_after = sum((value * value for value in after), Fraction(0))
        if mapping != "laplacian_polynomial" and norm_after != norm_before:
            raise ArithmeticError(f"The exact rational squared norm changed in {mapping}")
        return {
            "sutra_id": sid,
            "operands": dict(prepared["operands"]),
            "defaulted_operands": prepared["defaulted_operands"],
            "strength": strength,
            "mapping": mapping,
            "arithmetic_value": binding.result.value,
            "spectral_coefficients": coefficients,
            "before": before,
            "after": after,
            "norm_squared_before": norm_before,
            "norm_squared_after": norm_after,
            "norm_squared_change": norm_after - norm_before,
            "neighbors": neighbors,
            "edge_order": edge_order,
            "matchings": matchings,
            "mapping_metadata": binding.metadata(_METADATA_MAPPING[mapping]),
        }
    except Exception as error:
        raise BookQ4StageError(index, sid, "execution", mapping, completed_stages, before, error) from error


def execute_book_q4(sutra_id, operands, values, *, strength, mapping):
    """Execute one existing authored map with full native exact evidence.

    ``edge_order`` lists every applied rotation as
    ``(zero_based_coefficient_index, lower_vertex, higher_vertex)``. Thus Q4
    contains all 64 adjacency incidences for each coefficient, including both
    entries of every undirected edge. Polynomial stages have no rotations and
    return an empty edge order; their complete adjacency is in ``neighbors``.
    """
    mapping = _mapping(mapping)
    before = _values(values)
    stage = _preflight_stage({"sutra_id": sutra_id, "operands": operands, "strength": strength},
                             1, mapping, before)
    return _execute_stage(stage, before, mapping, 1, ())


def execute_all_book_q4(stages, values, *, mapping):
    """Execute all 29 explicit arithmetic/mapping stages sequentially.

    Every stage must contain exactly ``sutra_id``, ``operands``, and
    ``strength``. Declared function defaults are resolved only from that
    operation's actual signature, then included in ``operands`` and named in
    ``defaulted_operands``. No example operand set is selected. A failed
    stage raises with the full completed-stage trace and its original cause.
    """
    mapping = _mapping(mapping)
    initial_values = _values(values)
    if isinstance(stages, (str, bytes, bytearray)) or not isinstance(stages, Sequence):
        raise TypeError("Supply the ordered sequence of all twenty-nine explicit book stages")
    if len(stages) != 29:
        raise ValueError("The complete book chain requires exactly twenty-nine stages")
    prepared = tuple(_preflight_stage(stage, index, mapping, initial_values, required_id=index)
                     for index, stage in enumerate(stages, 1))
    completed = []
    current = initial_values
    for index, stage in enumerate(prepared, 1):
        record = _execute_stage(stage, current, mapping, index, completed)
        completed.append(record)
        current = record["after"]
    return {
        "mapping": mapping,
        "initial_values": initial_values,
        "final_values": current,
        "stages": tuple(completed),
    }
