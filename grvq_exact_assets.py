"""Executable, exact asset integration with equation-specific execution receipts.

The book's arithmetic applications, its authored graph polynomial, cubical
cochains, and algebraic coefficient fields retain their separate definitions.
None of these operations supplies missing GRVQ dynamics. Public arithmetic
inputs are explicit; an omitted operand mapping never selects example data.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import sympy as sp
from sympy.printing.repr import ReprPrinter

from core.exact_codec import encode_exact
import grvq_canonical
import vedic_book
from vedic_field import FieldBinding
from vedic_trainer.vedic.kernel import k2_field, q4_complex, wht


ROOT = Path(__file__).resolve().parent
SCHEMA = "quanqonscious.exact-asset-execution.v1"


@dataclass(frozen=True)
class AssetExecution:
    """A computed value and a detached, JSON-compatible receipt of that call.

    Source hashes identify the implementation bytes; they are not proof of
    equation correctness or physical validity. ``receipt`` is a snapshot.
    """

    value: Any
    receipt: dict


class _ExactReprPrinter(ReprPrinter):
    """Keep symbolic integer atoms exact without decimal conversion limits."""

    def _print_Integer(self, expression):
        return f"Integer({hex(int(expression.p))})"

    def _print_Rational(self, expression):
        return f"Rational({hex(int(expression.p))}, {hex(int(expression.q))})"

    def _print_int(self, value):
        return repr(value) if isinstance(value, bool) else hex(value)


def _exact_srepr(value):
    # Preserve the literal argument order and the inherited symbol assumptions.
    # This emits a representation; no incoming string is parsed or evaluated.
    return _ExactReprPrinter({"order": "none"}).doprint(value)


def _encode(value):
    """Encode exact values without decimal conversion or evaluating strings."""
    if isinstance(value, sp.Rational):
        return encode_exact(Fraction(int(value.p), int(value.q)))
    if isinstance(value, sp.Basic):
        if value.atoms(sp.Float):
            raise TypeError("A symbolic execution result contains binary floating-point data")
        return {"$exact": "symbolic", "srepr": _exact_srepr(value)}
    if isinstance(value, k2_field.Q2):
        return {"$exact": "Q2", "coordinates": _encode((value.a, value.b))}
    if isinstance(value, k2_field.K2):
        return {"$exact": "K2", "coordinates": _encode(value.coords())}
    if isinstance(value, k2_field.C4):
        return {"$exact": "C4", "u": _encode(value.u), "v": _encode(value.v)}
    if isinstance(value, float):
        raise TypeError("Binary floating-point results are outside the exact asset interface")
    if isinstance(value, tuple):
        return {"$exact": "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value) and "$exact" not in value:
            return {key: _encode(item) for key, item in value.items()}
        return {"$exact": "mapping", "items": [[_encode(key), _encode(item)] for key, item in value.items()]}
    return encode_exact(value)


def _execution(calculation, value, inputs, details, implementations, entrypoint):
    sources = {"grvq_exact_assets.py": (entrypoint, "_execution", "_encode"),
               "core/exact_codec.py": ("encode_exact",)}
    sources.update(implementations)
    evidence = []
    for relative_path, functions in sorted(sources.items()):
        content = (ROOT / relative_path).read_bytes()
        evidence.append({"path": relative_path, "functions": list(functions),
                         "sha256": hashlib.sha256(content).hexdigest()})
    receipt = {
        "schema": SCHEMA,
        "calculation": calculation,
        "scope": "Execution of the named component calculation; no full-engine certification",
        "inputs": _encode(inputs),
        "value": _encode(value),
        "details": _encode(details),
        "implementation_sources": evidence,
        "symbolic_library": {"name": "sympy", "version": sp.__version__},
    }
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    receipt["receipt_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return AssetExecution(value, receipt)


def _fraction(value):
    q = vedic_book.rational(value)
    return Fraction(int(q.p), int(q.q))


def _q4_values(values):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("A sequence of sixteen exact rational vertex values is required")
    if len(values) != 16:
        raise ValueError("Q4 requires all sixteen vertex values")
    return tuple(_fraction(value) for value in values)


def _operands(operands):
    if not isinstance(operands, Mapping):
        raise TypeError("Supply an explicit operand mapping; no example preset is selected")
    if not all(isinstance(key, str) for key in operands):
        raise TypeError("Operand names must be strings")
    return dict(operands)


def _canonical_field(r, theta, phi, t, parameters):
    if parameters is None:
        parameters = grvq_canonical.GRVQParameters()
    elif isinstance(parameters, Mapping):
        parameters = grvq_canonical.GRVQParameters(**dict(parameters))
    elif not isinstance(parameters, grvq_canonical.GRVQParameters):
        raise TypeError("Canonical parameters must be an explicit mapping or GRVQParameters")
    point = grvq_canonical.GRVQPoint(r, theta, phi, t)
    return grvq_canonical.evaluate_canonical_field(point, parameters)


def evaluate_canonical_field(r, theta, phi, t=0, parameters=None):
    """Execute the supplied scalar ansatz with its literal factors intact.

    A declared parameter absent from that equation remains reported as absent.
    Its missing coupling is never replaced with an invented dependence.
    """
    field = _canonical_field(r, theta, phi, t, parameters)
    return _execution(
        "canonical_GRVQ_scalar", field.value,
        {"point": asdict(field.point), "parameters": asdict(field.parameters)},
        {"literal_expression": field.expression, "components": dict(field.components),
         "parameter_activity": field.parameter_activity(),
         "time_dependence": "No explicit t occurs in the supplied scalar formula",
         "unresolved_full_model_requirements": field.unresolved_requirements},
        {"grvq_canonical.py": ("GRVQPoint", "GRVQParameters", "evaluate_canonical_field",
                               "CanonicalField.parameter_activity")},
        "evaluate_canonical_field",
    )


def evaluate_field_sutra(sutra_id, operands, r, theta, phi, t=0, parameters=None, components=None):
    """Execute one of the three explicitly supplied field-coupled formulas."""
    field = _canonical_field(r, theta, phi, t, parameters)
    operands = _operands(operands)
    value = grvq_canonical.apply_field_sutra(sutra_id, operands, field, components=components)
    return _execution(
        "specified_GRVQ_field_sutra", value,
        {"sutra_id": sutra_id, "operands": operands, "point": asdict(field.point),
         "parameters": asdict(field.parameters), "components": components},
        {"field_amplitude": field.amplitude, "radial_term": field.radial_term,
         "defined_field_sutras": grvq_canonical.FIELD_SUTRA_IDS,
         "component_source": "Caller-supplied matrix for Urdhva; no component law is inferred",
         "unresolved_full_model_requirements": field.unresolved_requirements},
        {"grvq_canonical.py": ("GRVQPoint", "GRVQParameters", "evaluate_canonical_field",
                               "apply_field_sutra")},
        "evaluate_field_sutra",
    )


def evaluate_book_sutra(sutra_id, operands):
    """Execute one of the existing 29 book arithmetic applications.

    The name of an arithmetic application does not identify it with a
    separately specified field-coupled operation bearing that sutra name.
    """
    operands = _operands(operands)
    result = vedic_book.evaluate(sutra_id, **operands)
    return _execution(
        "vedic_book_arithmetic", result.value,
        {"sutra_id": result.sutra.id, "operands": operands},
        {"sutra_name": result.sutra.name, "application": result.sutra.application,
         "book": dict(vedic_book.BOOK), "pages": result.sutra.pages,
         "spectral_coefficients": result.spectral_coefficients,
         "canonical_field_coupling": "Not supplied by this arithmetic application"},
        {"vedic_book.py": ("evaluate", vedic_book.OPERATIONS[result.sutra.id - 1].__name__)},
        "evaluate_book_sutra",
    )


def apply_book_sutra_q4(sutra_id, operands, values, *, strength):
    """Apply the existing authored book polynomial on every Q4 vertex.

    This executes FieldBinding's exact equation, with every coefficient:
    psi_out = psi + strength * sum(c[m] * L**(m+1) * psi).
    Its L=degree-adjacency equals minus q4_complex's Delta. This API does
    not identify the polynomial with the user's canonical GRVQ field.
    """
    operands = _operands(operands)
    values, strength = _q4_values(values), _fraction(strength)
    binding = FieldBinding.from_parameters(sutra_id, operands)
    neighbors = tuple(q4_complex.neighbours(v) for v in range(16))
    result = binding.apply_graph(values, neighbors, strength)
    mapping_receipt = binding.receipt(values, result, neighbors, strength, "laplacian_polynomial")
    return _execution(
        "authored_book_q4_laplacian_polynomial", result,
        {"sutra_id": binding.result.sutra.id, "operands": operands,
         "values": values, "strength": strength},
        {"arithmetic_value": binding.result.value,
         "spectral_coefficients": binding.result.spectral_coefficients,
         "mapping_receipt": mapping_receipt, "neighbors": neighbors,
         "laplacian_sign": "L = degree - adjacency = -q4_complex.laplacian",
         "canonical_field_coupling": "This is the existing authored graph polynomial"},
        {"vedic_book.py": ("evaluate", vedic_book.OPERATIONS[binding.result.sutra.id - 1].__name__),
         "vedic_field.py": ("FieldBinding.from_parameters", "FieldBinding.apply_graph",
                            "laplacian_rows", "FieldBinding.receipt"),
         "vedic_trainer/vedic/kernel/q4_complex.py": ("neighbours", "toggle")},
        "apply_book_sutra_q4",
    )


def q4_diagnostics(values, *, edge_cochain=None, plaquette_cochain=None, cube_cochain=None):
    """Compute complete rational Q4 diagnostics, retaining every supplied cell.

    An exact gradient has zero curl. An independently supplied edge cochain
    has its own curl; this function never infers it from the gradient result.
    Walsh coefficients are a reversible diagnostic of all sixteen values.
    """
    values = _q4_values(values)
    gradient = q4_complex.d0(values)
    coefficients = wht.walsh_transform_exact(values)
    result = {
        "values": values,
        "gradient": gradient,
        "gradient_curl": q4_complex.d1(gradient),
        "laplacian": q4_complex.laplacian(values),
        "walsh_coefficients": coefficients,
        "walsh_reconstruction": wht.walsh_inverse_exact(coefficients),
    }
    inputs = {"values": values}
    invoked_cochain_functions = ["d0", "d1", "laplacian", "edges", "plaquettes", "cubes", "four_cells"]
    if edge_cochain is not None:
        curl = q4_complex.d1(edge_cochain)
        inputs["edge_cochain"] = dict(edge_cochain)
        result["edge_curl"] = curl
        result["edge_curl_coboundary"] = q4_complex.d2(curl)
        invoked_cochain_functions.append("d2")
    if plaquette_cochain is not None:
        flux = q4_complex.d2(plaquette_cochain)
        inputs["plaquette_cochain"] = dict(plaquette_cochain)
        result["plaquette_coboundary"] = flux
        result["plaquette_double_coboundary"] = q4_complex.d3(flux)
        invoked_cochain_functions.extend(("d2", "d3"))
    if cube_cochain is not None:
        inputs["cube_cochain"] = dict(cube_cochain)
        result["cube_coboundary"] = q4_complex.d3(cube_cochain)
        invoked_cochain_functions.append("d3")
    return _execution(
        "complete_rational_q4_diagnostics", result, inputs,
        {"cell_counts": (len(values), len(q4_complex.edges()), len(q4_complex.plaquettes()),
                         len(q4_complex.cubes()), len(q4_complex.four_cells())),
         "laplacian": "Delta f(v) = sum_j (f(v xor 2**j) - f(v))",
         "walsh_convention": "Unnormalized sixteen-character transform; inverse divides by sixteen",
         "state_reduction": "None; all sixteen input values and coefficients are retained",
         "physical_curvature_claim": "None; coboundaries of the explicitly supplied cochains are reported"},
        {"vedic_trainer/vedic/kernel/q4_complex.py": tuple(dict.fromkeys(invoked_cochain_functions)),
         "vedic_trainer/vedic/kernel/wht.py": ("walsh_transform_exact", "walsh_inverse_exact")},
        "q4_diagnostics",
    )


def _k2_coordinates(values):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != 4:
        raise TypeError("Supply four exact rational coordinates (a,b,c,d) for a+b*sqrt(2)+i*(c+d*sqrt(2))")
    return k2_field.K2.from_coords(*(_fraction(value) for value in values))


def divide_k2(numerator, denominator):
    """Divide K2 coordinate tuples; report both distinct denominator norms."""
    left, right = _k2_coordinates(numerator), _k2_coordinates(denominator)
    quotient = left / right
    return _execution(
        "exact_K2_division", quotient,
        {"numerator": left, "denominator": right},
        {"field": "Q(sqrt(2),i)", "denominator_hermitian_norm": right.hermitian_norm(),
         "hermitian_norm_codomain": "Q(sqrt(2))", "denominator_total_norm": right.total_norm(),
         "total_norm_codomain": "Q", "quotient_times_denominator": quotient * right},
        {"vedic_trainer/vedic/kernel/k2_field.py": ("K2.__truediv__", "K2.inverse",
                                                  "K2.hermitian_norm", "K2.total_norm")},
        "divide_k2",
    )


def divide_c4(numerator, denominator):
    """Divide C4 pairs of K2 coordinate tuples, retaining the sqrt(5) part."""
    def make(value):
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
            raise TypeError("Supply two K2 coordinate tuples (u,v) for u+v*sqrt(5)")
        return k2_field.C4(_k2_coordinates(value[0]), _k2_coordinates(value[1]))

    left, right = make(numerator), make(denominator)
    quotient = left / right
    return _execution(
        "exact_C4_division", quotient,
        {"numerator": left, "denominator": right},
        {"field": "Q(sqrt(2),i,sqrt(5))", "quotient_times_denominator": quotient * right},
        {"vedic_trainer/vedic/kernel/k2_field.py": ("C4.__truediv__", "C4.inverse", "K2.inverse")},
        "divide_c4",
    )


def _decode_request(value):
    """Decode JSON rationals and pi; never evaluate a supplied expression."""
    if isinstance(value, (bool, float)):
        raise TypeError("Exact requests reject booleans and binary floating-point numbers")
    if isinstance(value, list):
        return [_decode_request(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"cochain"}:
            entries = value["cochain"]
            if not isinstance(entries, list):
                raise TypeError("A cochain requires a list of [integer-cell-key, exact-value] pairs")
            result = {}
            for entry in entries:
                if (not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], list)
                        or any(type(index) is not int for index in entry[0])):
                    raise TypeError("Every cochain cell key must be a list of integers")
                key = tuple(entry[0])
                if key in result:
                    raise ValueError("A cochain repeats a cell key")
                result[key] = _decode_request(entry[1])
            return result
        if set(value) == {"rational"}:
            pair = value["rational"]
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(type(item) is not int for item in pair)):
                raise TypeError("A rational requires an integer numerator and denominator")
            return Fraction(*pair)
        if set(value) == {"constant"}:
            if value["constant"] != "pi":
                raise ValueError("The only named request constant is exact pi")
            return sp.pi
        return {key: _decode_request(item) for key, item in value.items()}
    return value


def execute_request(request):
    """Run a named, explicit calculation from a JSON-compatible request."""
    if not isinstance(request, dict) or set(request) != {"operation", "arguments"}:
        raise ValueError("A request must contain exactly operation and arguments")
    if not isinstance(request["arguments"], dict):
        raise TypeError("Request arguments must be a mapping")
    operations = {
        "canonical_field": evaluate_canonical_field,
        "field_sutra": evaluate_field_sutra,
        "book_sutra": evaluate_book_sutra,
        "book_q4": apply_book_sutra_q4,
        "q4": q4_diagnostics,
        "k2_divide": divide_k2,
        "c4_divide": divide_c4,
    }
    if not isinstance(request["operation"], str) or request["operation"] not in operations:
        raise ValueError("Unknown exact asset operation")
    return operations[request["operation"]](**_decode_request(request["arguments"]))


def main(argv=None):
    """Execute one JSON request from a file or stdin; failures exit nonzero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="JSON request path, or - for stdin")
    args = parser.parse_args(argv)
    try:
        if args.request == "-":
            request = json.load(sys.stdin)
        else:
            with open(args.request, encoding="utf-8") as source:
                request = json.load(source)
        result = execute_request(request)
        print(json.dumps(result.receipt, indent=2, ensure_ascii=False))
    except (TypeError, ValueError, ArithmeticError, OSError,
            grvq_canonical.UnresolvedGRVQModelError) as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
