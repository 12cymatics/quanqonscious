"""Exact scalar reference for the user's supplied GRVQ product expression.

This module evaluates the specified scalar and the three supplied field-coupled
operations. It does not define a metric, dynamics, the other 26 field operations,
or a map from the scalar to indexed field components. Those missing equations
are explicit obligations, not functions replaced by invented calculations.

``expression`` retains the literal factor tree. ``value`` performs exact
arithmetic and exact special-function evaluation; unevaluated transcendental
values remain symbolic. No decimal conversion or numerical evaluation is used.
All public scalar inputs must be finite exact real numbers, not open symbolic
equations. This is not the missing equation-to-coordinate deformulation stage.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field, fields
from fractions import Fraction
from numbers import Integral
from types import MappingProxyType

import sympy as sp


PARAMETER_IDS = (
    "r0_squared", "shape_damping_S1", "shape_damping_S2", "k_vedic",
    "omega_vedic", "epsilon", "gamma", "lambda_maya", "lambda_QBR", "kappa_GR",
)
FIELD_SUTRA_IDS = ("ekadhikena_purvena", "nikhilam", "urdhva_tiryak")
FIELD_SUTRA_NAMES = MappingProxyType({
    "ekadhikena_purvena": "Ekadhikena Purvena (एकाधिकेन पूर्वेण)",
    "nikhilam": "Nikhilam Navatashcaramam Dashatah (निखिलम् नवतश्चरमं दशतः)",
    "urdhva_tiryak": "Urdhva-Tiryagbhyam (ऊर्ध्वा तिर्यग्भ्याम्)",
})

UNRESOLVED_REQUIREMENTS = (
    "omega_vedic: no time-dependent term or field evolution equation is supplied; t is absent from the scalar RHS",
    "lambda_maya: no Maya coupling equation is supplied",
    "lambda_QBR: no quantum-backreaction equation, including its all-orders definition, is supplied",
    "kappa_GR: no gravitational deformation or coupling equation is supplied",
    "metric: the full 4x4 g_mu_nu, its signature and relation to the scalar, and its evolution equation are undefined",
    "geodesics: the metric connection, initial data, and compatible exact/symplectic evolution specification are missing",
    "boundary: an evolution equation and a mathematical absorbing condition at r -> infinity are missing",
    "field components: get_component(i,j) has no supplied definition; Urdhva requires explicitly supplied components",
    "field sutras: 26 of the 29 required field-coupled formulas and their operand domains are not supplied",
    "sutra chain: the heterogeneous operand bindings and field evolution between its 29 stages are undefined",
    "deformulation: the equation-to-spherical-coordinate extraction map is undefined",
)


class UnresolvedGRVQModelError(NotImplementedError):
    """A full-model request lacks the actual required equations."""

    def __init__(self, request="complete GRVQ model"):
        self.request = request
        self.requirements = UNRESOLVED_REQUIREMENTS
        super().__init__(
            f"Cannot execute {request}: required definitions are unresolved:\n- "
            + "\n- ".join(self.requirements)
        )


class MissingFieldComponentsError(ValueError):
    """The specified cross-product has no supplied component matrix."""


def require_complete_model(request="complete GRVQ model"):
    """Reject a request whose equations are absent, without returning a result."""
    raise UnresolvedGRVQModelError(request)


def exact_real(value, *, name="value"):
    """Accept exact closed real expressions without parsing strings or floats."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an exact real scalar, not a boolean")
    if isinstance(value, Integral):
        result = sp.Integer(int(value))
    elif isinstance(value, Fraction):
        result = sp.Rational(value.numerator, value.denominator)
    elif isinstance(value, sp.Expr):
        result = value
    else:
        raise TypeError(f"{name} must be an integer, Fraction, or exact SymPy real expression; strings and floats are rejected")
    if result.atoms(sp.Float):
        raise TypeError(f"{name} contains floating-point data")
    if result.free_symbols:
        raise ValueError(f"{name} must be a closed scalar expression; free symbols are outside this interface")
    if result.is_real is not True or result.is_finite is not True:
        raise ValueError(f"{name} must be a provably finite real scalar")
    return result


@dataclass(frozen=True)
class GRVQParameters:
    """All ten declared parameters, including the four absent from the RHS.

    Positivity of r0_squared and epsilon makes the given denominators defined
    throughout the spherical chart. Other parameters may be any finite exact
    real number; their defaults reproduce the supplied specification.
    """

    r0_squared: sp.Expr = sp.Integer(1)
    shape_damping_S1: sp.Expr = sp.Rational(1, 10)
    shape_damping_S2: sp.Expr = sp.Rational(1, 20)
    k_vedic: sp.Expr = sp.Integer(1)
    omega_vedic: sp.Expr = sp.Integer(1)
    epsilon: sp.Expr = sp.Rational(1, 100000000)
    gamma: sp.Expr = sp.Rational(1, 2)
    lambda_maya: sp.Expr = sp.Rational(3, 10)
    lambda_QBR: sp.Expr = sp.Rational(1, 10)
    kappa_GR: sp.Expr = sp.Rational(1, 100)

    def __post_init__(self):
        for item in fields(self):
            object.__setattr__(self, item.name, exact_real(getattr(self, item.name), name=item.name))
        for name in ("r0_squared", "epsilon"):
            if getattr(self, name).is_positive is not True:
                raise ValueError(f"{name} must be provably positive for the complete scalar chart")

    def as_dict(self):
        return {name: getattr(self, name) for name in PARAMETER_IDS}


@dataclass(frozen=True)
class GRVQPoint:
    """Exact spherical-chart coordinates, with t retained as an inactive input.

    The scalar is evaluated on this chart; no assertion of single-valuedness at
    the spherical coordinate identifications is made. Its supplied angular
    factors do not, by themselves, establish a global smooth scalar field.
    """

    r: sp.Expr
    theta: sp.Expr
    phi: sp.Expr
    t: sp.Expr = sp.Integer(0)

    def __post_init__(self):
        for item in fields(self):
            object.__setattr__(self, item.name, exact_real(getattr(self, item.name), name=item.name))
        checks = (
            (self.r.is_nonnegative, "r must satisfy r >= 0"),
            (self.theta.is_nonnegative, "theta must satisfy theta >= 0"),
            ((sp.pi - self.theta).is_nonnegative, "theta must satisfy theta <= pi"),
            (self.phi.is_nonnegative, "phi must satisfy phi >= 0"),
            ((2 * sp.pi - self.phi).is_positive, "phi must satisfy phi < 2*pi"),
        )
        for valid, message in checks:
            if valid is not True:
                raise ValueError(message + " with an exactly decidable bound")

    def as_dict(self):
        return {name: getattr(self, name) for name in ("r", "theta", "phi", "t")}


def _add(*terms):
    return sp.Add(*terms, evaluate=False)


def _mul(*factors):
    return sp.Mul(*factors, evaluate=False)


def _pow(base, exponent):
    return sp.Pow(base, exponent, evaluate=False)


def _literal_components(r, theta, phi, parameters):
    """Build the supplied expression literally; no formula transformation."""
    r_squared = _pow(r, sp.Integer(2))
    radial_term = _add(
        sp.Integer(1),
        _mul(sp.Integer(-1), r_squared,
             _pow(_add(r_squared, parameters["r0_squared"]), sp.Integer(-1))),
    )
    S1 = _mul(
        sp.sin(theta, evaluate=False), sp.cos(phi, evaluate=False),
        sp.exp(_mul(sp.Integer(-1), parameters["shape_damping_S1"], r), evaluate=False),
    )
    S2 = _mul(
        sp.cos(theta, evaluate=False), sp.sin(phi, evaluate=False),
        sp.exp(_mul(sp.Integer(-1), parameters["shape_damping_S2"], r_squared), evaluate=False),
    )
    product_term1 = _add(
        sp.Integer(1),
        _mul(sp.Integer(-1), _pow(_add(sp.Abs(S1, evaluate=False), parameters["epsilon"]), sp.Integer(-1))),
    )
    product_term2 = _add(
        sp.Integer(1),
        _mul(sp.Integer(-2), _pow(_add(sp.Abs(S2, evaluate=False), parameters["epsilon"]), sp.Integer(-1))),
    )
    phase = _add(r, theta, phi)
    f_vedic = _add(
        sp.sin(_mul(parameters["k_vedic"], phase), evaluate=False),
        _mul(sp.Rational(1, 2),
             sp.cos(_mul(sp.Integer(2), parameters["k_vedic"], phase), evaluate=False)),
    )
    turyavrtti_modulation = _add(
        sp.Integer(1),
        _mul(parameters["gamma"], sp.sin(_mul(sp.pi, r, theta, phi), evaluate=False)),
    )
    return {
        "S1": S1, "S2": S2,
        "product_term1": product_term1, "product_term2": product_term2,
        "radial_term": radial_term, "f_vedic": f_vedic,
        "turyavrtti_modulation": turyavrtti_modulation,
    }


FACTOR_IDS = (
    "product_term1", "product_term2", "radial_term", "f_vedic", "turyavrtti_modulation",
)
PARAMETER_SYMBOLS = MappingProxyType({name: sp.Symbol(name, real=True) for name in PARAMETER_IDS})
COORDINATE_SYMBOLS = MappingProxyType({name: sp.Symbol(name, real=True) for name in ("r", "theta", "phi", "t")})
CANONICAL_COMPONENTS = MappingProxyType(_literal_components(
    COORDINATE_SYMBOLS["r"], COORDINATE_SYMBOLS["theta"], COORDINATE_SYMBOLS["phi"], PARAMETER_SYMBOLS,
))
CANONICAL_EXPRESSION = _mul(*(CANONICAL_COMPONENTS[name] for name in FACTOR_IDS))
# Occurrence is measured in the actual unspecialized expression, not at a point.
SCALAR_PARAMETER_IDS = tuple(name for name in PARAMETER_IDS if PARAMETER_SYMBOLS[name] in CANONICAL_EXPRESSION.free_symbols)
ABSENT_PARAMETER_IDS = tuple(name for name in PARAMETER_IDS if name not in SCALAR_PARAMETER_IDS)


@dataclass(frozen=True)
class CanonicalField:
    """A scalar state derived only from validated point and parameter inputs.

    Callers cannot supply an amplitude or components that disagree with the
    canonical calculation. Derived fields are computed by this constructor.
    """

    point: GRVQPoint
    parameters: GRVQParameters
    expression: sp.Expr = dataclass_field(init=False)
    value: sp.Expr = dataclass_field(init=False)
    components: Mapping = dataclass_field(init=False)
    literal_components: Mapping = dataclass_field(init=False)

    def __post_init__(self):
        if not isinstance(self.point, GRVQPoint):
            raise TypeError("point must be a validated GRVQPoint")
        if not isinstance(self.parameters, GRVQParameters):
            raise TypeError("parameters must be validated GRVQParameters")
        literal = _literal_components(
            self.point.r, self.point.theta, self.point.phi, self.parameters.as_dict(),
        )
        expression = _mul(*(literal[name] for name in FACTOR_IDS))
        evaluated = {name: component.doit(deep=True) for name, component in literal.items()}
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "value", expression.doit(deep=True))
        object.__setattr__(self, "components", MappingProxyType(evaluated))
        object.__setattr__(self, "literal_components", MappingProxyType(literal))

    @property
    def amplitude(self):
        """The reference interface names its computed scalar F 'amplitude'."""
        return self.value

    @property
    def radial_term(self):
        return self.components["radial_term"]

    @property
    def unresolved_requirements(self):
        return UNRESOLVED_REQUIREMENTS

    def parameter_activity(self):
        """Structural occurrence, explicitly distinct from local sensitivity."""
        free = CANONICAL_EXPRESSION.free_symbols
        return {
            name: {
                "declared_value": getattr(self.parameters, name),
                "symbol": PARAMETER_SYMBOLS[name],
                "occurs_in_scalar": PARAMETER_SYMBOLS[name] in free,
                "status": "present_in_supplied_scalar" if PARAMETER_SYMBOLS[name] in free else "declared_but_coupling_equation_absent",
            }
            for name in PARAMETER_IDS
        }

    def require_complete_model(self):
        require_complete_model()


def evaluate_canonical_field(point, parameters=None):
    """Evaluate every literal scalar factor with exact finite real inputs.

    Mapping inputs are accepted only as named constructor arguments; they are
    never interpreted as code or parsed symbolic text.
    """
    if isinstance(point, Mapping):
        point = GRVQPoint(**point)
    if not isinstance(point, GRVQPoint):
        raise TypeError("point must be a GRVQPoint or an explicit coordinate mapping")
    if parameters is None:
        parameters = GRVQParameters()
    elif isinstance(parameters, Mapping):
        parameters = GRVQParameters(**parameters)
    if not isinstance(parameters, GRVQParameters):
        raise TypeError("parameters must be GRVQParameters or a declared parameter mapping")
    return CanonicalField(point, parameters)


def _field(field):
    if not isinstance(field, CanonicalField):
        raise TypeError("field must be a computed CanonicalField")
    return field


def ekadhikena_purvena(value, field):
    """The supplied field operation: value + 1 + F."""
    value = exact_real(value)
    return _add(value, sp.Integer(1), _field(field).amplitude).doit(deep=True)


def nikhilam(value, base, field):
    """The supplied field operation: (base - value) * radial_term."""
    value, base = exact_real(value), exact_real(base, name="base")
    return _mul(_add(base, _mul(sp.Integer(-1), value)), _field(field).radial_term).doit(deep=True)


def _exact_sequence(values, name):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be an explicit sequence of exact real scalars")
    return tuple(exact_real(value, name=f"{name}[{index}]") for index, value in enumerate(values))


def field_component_matrix(components, rows, columns):
    """Validate every supplied (i,j), including entries with zero coefficients.

    No scalar-to-component map is inferred. An explicit rectangular matrix or
    a complete mapping keyed by (i,j) is the input to the supplied double sum.
    """
    if components is None:
        raise MissingFieldComponentsError(
            "Urdhva requires every explicit get_component(i,j) value; the canonical scalar does not define this map"
        )
    if isinstance(components, Mapping):
        expected = {(i, j) for i in range(rows) for j in range(columns)}
        if any(not isinstance(key, tuple) or len(key) != 2
               or any(type(index) is not int for index in key) for key in components):
            raise ValueError("Field component mapping keys must be pairs of integer indices")
        if set(components) != expected:
            raise MissingFieldComponentsError("Field component keys must match the complete operand rectangle exactly")
        return tuple(tuple(exact_real(components[(i, j)], name=f"components[{i},{j}]")
                           for j in range(columns)) for i in range(rows))
    if isinstance(components, (str, bytes)) or not isinstance(components, Sequence):
        raise TypeError("Field components must be an explicit matrix or complete (i,j) mapping")
    if len(components) != rows:
        raise MissingFieldComponentsError("Field component matrix must have one row per a operand")
    result = []
    for i, row in enumerate(components):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != columns:
            raise MissingFieldComponentsError("Every field component row must have one entry per b operand")
        result.append(tuple(exact_real(value, name=f"components[{i},{j}]") for j, value in enumerate(row)))
    return tuple(result)


def urdhva_tiryak(a, b, field, *, components=None):
    """The full supplied double sum, using explicitly supplied field components."""
    _field(field)
    a, b = _exact_sequence(a, "a"), _exact_sequence(b, "b")
    matrix = field_component_matrix(components, len(a), len(b))
    terms = []
    for i in range(len(a)):
        for j in range(len(b)):
            terms.append(_mul(a[i], b[j], matrix[i][j]))
    # An empty index set has the prescribed empty-sum value zero.
    return _add(*terms).doit(deep=True)


def apply_field_sutra(sutra_id, operands, field, *, components=None):
    """Execute one of the three supplied formulas with explicit named operands."""
    if sutra_id not in FIELD_SUTRA_IDS:
        raise UnresolvedGRVQModelError(f"field sutra {sutra_id!r}: no supplied formula")
    if not isinstance(operands, Mapping):
        raise TypeError("An explicit operand mapping is required")
    expected = {"ekadhikena_purvena": {"value"}, "nikhilam": {"value", "base"}, "urdhva_tiryak": {"a", "b"}}[sutra_id]
    if set(operands) != expected:
        raise ValueError(f"{sutra_id} requires exactly the operands {sorted(expected)}")
    if sutra_id != "urdhva_tiryak" and components is not None:
        raise ValueError("Only the supplied Urdhva formula consumes indexed field components")
    if sutra_id == "ekadhikena_purvena":
        return ekadhikena_purvena(operands["value"], field)
    if sutra_id == "nikhilam":
        return nikhilam(operands["value"], operands["base"], field)
    return urdhva_tiryak(operands["a"], operands["b"], field, components=components)
