"""Exact Euclidean four-dimensional Hodge diagnostics for the complete Q4.

The ordered basis is (01, 02, 03, 12, 13, 23), the orientation is (0, 1, 2, 3),
and the metric signature is (+, +, +, +). These are the geometry declared by
the preserved vertex source, not a definition of the missing GRVQ spacetime
metric. Both chiralities and all six input coordinates remain available.

Source: the Hodge tables HODGE_SD and HODGE_ASD of vedic_v18.51.1_exact_phi.html,
whose self-dual and anti-self-dual triples are (F + *F)/2 and (F - *F)/2 under
the star below. The per-vertex diagnostic divides both triples by six and uses
the normalized pressure sum(face**2)/(2*24). These conventions are stated
explicitly; they do not supply a physical permeability or units.
"""
from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import sympy as sp

from vedic_trainer.vedic.kernel import q4_complex


AXIS_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
SIGNATURE = (1, 1, 1, 1)
ORIENTATION = (0, 1, 2, 3)
SOURCE = {
    "path": "vedic_v18.51.1_exact_phi.html",
    "tables": ("HODGE_SD", "HODGE_ASD"),
    "geometry": "Euclidean four-space with orientation (0, 1, 2, 3)",
    "metric_authority": "The preserved vertex diagnostic; the GRVQ metric remains unspecified",
    "pressure_units": "Normalized simulation quantity; no physical permeability is inferred",
}


def _check6(values):
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise TypeError("A bivector must be an ordered sequence of six exact rationals")
    if len(values) != 6:
        raise ValueError("A bivector needs all six coordinates in (01, 02, 03, 12, 13, 23) order")
    return tuple(
        q4_complex._rational(value, f"bivector[{index}]")
        for index, value in enumerate(values)
    )


def _norm_squared(values):
    return sum((value * value for value in values), Fraction(0))


def hodge_star6(values):
    """Apply the Euclidean Hodge star to all six oriented bivector entries.

    (F01, F02, F03, F12, F13, F23) maps to
    (F23, -F13, F12, F03, -F02, F01).
    """
    a, b, c, d, e, f = _check6(values)
    return (f, -e, d, c, -b, a)


def split_hodge6(values):
    """Compute F+, F-, both triples, reconstruction, and exact squared norms.

    F+ = (F + star(F))/2 and F- = (F - star(F))/2. The triples are coordinates
    for their respective three-dimensional eigenspaces; neither replaces F.
    Squared norms use the original orthonormal six-coordinate basis.
    """
    field = _check6(values)
    star = hodge_star6(field)
    self_dual = tuple((value + dual) / 2 for value, dual in zip(field, star))
    anti_self_dual = tuple((value - dual) / 2 for value, dual in zip(field, star))
    plus = self_dual[:3]
    minus = anti_self_dual[:3]
    reconstruction = (
        plus[0] + minus[0],
        plus[1] + minus[1],
        plus[2] + minus[2],
        plus[2] - minus[2],
        minus[1] - plus[1],
        plus[0] - minus[0],
    )
    return {
        "input6": field,
        "star6": star,
        "self_dual6": self_dual,
        "anti_self_dual6": anti_self_dual,
        "plus_triple": plus,
        "minus_triple": minus,
        "reconstruction6": reconstruction,
        "input_norm_squared": _norm_squared(field),
        "star_norm_squared": _norm_squared(star),
        "self_dual_norm_squared": _norm_squared(self_dual),
        "anti_self_dual_norm_squared": _norm_squared(anti_self_dual),
        "chirality_inner_product": sum(
            (positive * negative for positive, negative in zip(self_dual, anti_self_dual)),
            Fraction(0),
        ),
        "signature": SIGNATURE,
        "orientation": ORIENTATION,
        "axis_pairs": AXIS_PAIRS,
    }


def q4_hodge_diagnostics(plaquette_cochain):
    """Evaluate the source diagnostic on every face and all sixteen vertices.

    Every canonical oriented face must be supplied, including genuine zeros.
    Its six incident faces give a local bivector at each vertex. Both source
    triples have the specified factor 1/6, so their combined squared strength
    equals the local six-coordinate squared norm divided by 72. Strength is
    computed as an exact symbolic square root of that rational result.

    Validation reuses Q4's complete cell and rational contracts. The returned
    cochain has canonical integer keys and detached exact rational values.
    """
    cells = q4_complex.plaquettes()
    checked = q4_complex._check_cells(plaquette_cochain, cells, "2-cochain")
    cochain = {cell: checked[cell] for cell in cells}
    vertices = []
    for vertex in range(q4_complex.N_VERTICES):
        incident_faces = tuple(
            (vertex & ~((1 << first) | (1 << second)), first, second)
            for first, second in AXIS_PAIRS
        )
        local_bivector = tuple(cochain[face] for face in incident_faces)
        split = split_hodge6(local_bivector)
        source_plus = tuple(value / 6 for value in split["plus_triple"])
        source_minus = tuple(value / 6 for value in split["minus_triple"])
        source_strength_squared = _norm_squared(source_plus) + _norm_squared(source_minus)
        source_strength = sp.sqrt(sp.Rational(
            source_strength_squared.numerator, source_strength_squared.denominator,
        ))
        vertices.append({
            "vertex": vertex,
            "incident_faces": incident_faces,
            "local_bivector6": local_bivector,
            "split": split,
            "source_plus_triple": source_plus,
            "source_minus_triple": source_minus,
            "source_strength_squared": source_strength_squared,
            "source_strength": source_strength,
            "source_normalization": Fraction(1, 6),
        })
    face_square_sum = _norm_squared(cochain.values())
    return {
        "plaquette_cochain": cochain,
        "face_count": len(cells),
        "vertex_count": len(vertices),
        "face_square_sum": face_square_sum,
        "source_pressure": face_square_sum / (2 * len(cells)),
        "vertices": tuple(vertices),
        "signature": SIGNATURE,
        "orientation": ORIENTATION,
        "axis_pairs": AXIS_PAIRS,
        "source": dict(SOURCE),
    }
