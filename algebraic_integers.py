"""Utilities for working with algebraic integers without resorting to floats.

This module provides a lightweight wrapper around SymPy expressions that enforces
that every stored value is an algebraic integer.  The implementation validates
each expression by inspecting its minimal polynomial and only accepting numbers
whose polynomial is monic with integer coefficients.  The resulting
``AlgebraicInteger`` objects support the standard arithmetic operations required
by the standalone simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import add, mul
from typing import Iterable, Union

import sympy
from sympy import Integer, Matrix, Poly, Symbol, conjugate, nsimplify, simplify
from sympy.polys.numberfields import minimal_polynomial

SympyCompatible = Union[int, sympy.Expr, "AlgebraicInteger"]

_VALIDATION_CACHE: dict[sympy.Expr, sympy.Expr] = {}


class AlgebraicValidationError(ValueError):
    """Raised when a value fails the algebraic-integer validation."""


def _ensure_symbol() -> Symbol:
    return Symbol("_ai_symbol", rational=True)


def _is_integral_polynomial(poly: Poly) -> bool:
    coeffs = poly.all_coeffs()
    return all(coef.is_Integer for coef in coeffs) and poly.LC() == 1


@dataclass(frozen=True)
class AlgebraicInteger:
    """Immutable representation of an algebraic integer."""

    expr: sympy.Expr

    def __post_init__(self) -> None:  # type: ignore[override]
        simplified = nsimplify(self.expr, rational=True)
        validated = self._validate_expr(simplified)
        object.__setattr__(self, "expr", validated)

    @classmethod
    def _create(cls, expr: sympy.Expr, validate: bool = True) -> "AlgebraicInteger":
        instance = object.__new__(cls)
        simplified = nsimplify(expr, rational=True)
        if validate:
            simplified = cls._validate_expr(simplified)
        object.__setattr__(instance, "expr", simplified)
        return instance

    @staticmethod
    def _validate_expr(value: sympy.Expr) -> sympy.Expr:
        simplified = simplify(value)
        cached = _VALIDATION_CACHE.get(simplified)
        if cached is not None:
            return cached
        if not simplified.is_algebraic:
            raise AlgebraicValidationError(f"Expression {value!r} is not algebraic.")
        symbol = _ensure_symbol()
        poly_expr = minimal_polynomial(simplified, symbol)
        poly = Poly(poly_expr, symbol)
        if not _is_integral_polynomial(poly):
            raise AlgebraicValidationError(
                f"Expression {value!r} does not have a monic integer minimal polynomial."
            )
        _VALIDATION_CACHE[simplified] = simplified
        return simplified

    @classmethod
    def _coerce(cls, value: SympyCompatible) -> sympy.Expr:
        if isinstance(value, AlgebraicInteger):
            return value.expr
        if isinstance(value, sympy.Expr):
            return cls._create(value).expr
        return Integer(value)

    def _operate(self, other: SympyCompatible, op) -> "AlgebraicInteger":
        other_expr = self._coerce(other)
        return self._create(op(self.expr, other_expr), validate=False)

    def __add__(self, other: SympyCompatible) -> "AlgebraicInteger":
        return self._operate(other, add)

    def __radd__(self, other: SympyCompatible) -> "AlgebraicInteger":
        return self._operate(other, add)

    def __sub__(self, other: SympyCompatible) -> "AlgebraicInteger":
        return self._operate(other, lambda a, b: a - b)

    def __rsub__(self, other: SympyCompatible) -> "AlgebraicInteger":
        other_expr = self._coerce(other)
        return self._create(other_expr - self.expr, validate=False)

    def __mul__(self, other: SympyCompatible) -> "AlgebraicInteger":
        return self._operate(other, mul)

    def __rmul__(self, other: SympyCompatible) -> "AlgebraicInteger":
        return self._operate(other, mul)

    def __pow__(self, power: int) -> "AlgebraicInteger":
        if power < 0:
            raise ValueError("Negative exponents are not supported for algebraic integers.")
        return self._create(self.expr**power, validate=False)

    def __neg__(self) -> "AlgebraicInteger":
        return self._create(-self.expr, validate=False)

    def conjugate(self) -> "AlgebraicInteger":
        return self._create(conjugate(self.expr), validate=False)

    def norm(self) -> "AlgebraicInteger":
        return self * self.conjugate()

    def trace(self) -> "AlgebraicInteger":
        return self._create(self.expr + conjugate(self.expr), validate=False)

    def to_integer(self) -> int:
        simplified = sympy.simplify(self.expr)
        if not simplified.is_Integer:
            raise AlgebraicValidationError(f"Expression {simplified!r} is not an integer.")
        return int(simplified)

    def __str__(self) -> str:
        return str(self.expr)

    def __repr__(self) -> str:
        return f"AlgebraicInteger({self.expr!r})"

    def as_expr(self) -> sympy.Expr:
        return self.expr


def ensure_vector(values: Iterable[SympyCompatible]) -> list[AlgebraicInteger]:
    return [value if isinstance(value, AlgebraicInteger) else AlgebraicInteger(value) for value in values]


def sum_integers(values: Iterable[AlgebraicInteger]) -> AlgebraicInteger:
    return reduce(lambda acc, val: acc + val, values, AlgebraicInteger(0))


def matrix_from_rows(rows: Iterable[Iterable[SympyCompatible]]) -> Matrix:
    validated_rows = []
    for row in rows:
        row_exprs = []
        for value in row:
            if isinstance(value, AlgebraicInteger):
                row_exprs.append(value.as_expr())
            else:
                row_exprs.append(AlgebraicInteger(value).as_expr())
        validated_rows.append(row_exprs)
    return Matrix(validated_rows)

