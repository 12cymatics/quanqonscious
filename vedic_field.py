from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import gcd
import hashlib
import json

import sympy as sp

from vedic_book import BOOK, Evaluation, evaluate, integer, rational
from vedic_book_cases import EXAMPLE_PARAMETERS


def as_fraction(value):
    value = rational(value)
    return Fraction(int(value.p), int(value.q))


def exact_field_value(value):
    if isinstance(value, (int, Fraction, sp.Rational)):
        return as_fraction(value)
    if isinstance(value, sp.Expr) and value.is_number is True and value.is_finite is True and not value.has(sp.Float):
        return value
    if isinstance(getattr(value, "real", None), Fraction) and isinstance(getattr(value, "imag", None), Fraction):
        return value
    raise TypeError("Field values must be exact rationals or exact complex/algebraic numbers")


def laplacian_rows(values, rows, start):
    return tuple(len(row) * values[start + i]
                 - sum((values[j] for j in row), values[start + i] * 0)
                 for i, row in enumerate(rows))


def frozen_operand(value):
    if isinstance(value, (tuple, list)):
        return tuple(frozen_operand(item) for item in value)
    return str(rational(value))


def exact_digest(values):
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, (int, Fraction, sp.Rational)):
            parts = (as_fraction(value),)
        elif isinstance(getattr(value, "real", None), Fraction) and isinstance(getattr(value, "imag", None), Fraction):
            parts = (value.real, value.imag)
        else:
            text = sp.srepr(value).encode()
            digest.update(len(text).to_bytes(8, "big") + text)
            continue
        digest.update(len(parts).to_bytes(1, "big"))
        for part in parts:
            for integer_part in (part.numerator, part.denominator):
                encoded = integer_part.to_bytes((integer_part.bit_length() + 8) // 8, "big", signed=True)
                digest.update(len(encoded).to_bytes(8, "big") + encoded)
    return digest.hexdigest()


def rotate_fraction_channels(channels, neighbors, coefficients, strength):
    denominator = 1
    for channel in channels:
        for value in channel:
            denominator = denominator * value.denominator // gcd(denominator, value.denominator)
    numerators = [[value.numerator * (denominator // value.denominator) for value in channel]
                  for channel in channels]
    n = len(neighbors)
    for coefficient in coefficients:
        t = strength * as_fraction(coefficient)
        p, q = t.numerator, t.denominator
        base, cosine, sine = q * q + p * p, q * q - p * p, 2 * p * q
        common = gcd(gcd(base, cosine), sine)
        base, cosine, sine = base // common, cosine // common, sine // common
        exponents, powers = [0] * n, {0: 1}

        def power(exponent):
            if exponent not in powers:
                powers[exponent] = base ** exponent
            return powers[exponent]

        for i, row in enumerate(neighbors):
            for j in row:
                left, right = min(i, j), max(i, j)
                if left != right:
                    exponent = max(exponents[left], exponents[right])
                    left_scale = power(exponent - exponents[left])
                    right_scale = power(exponent - exponents[right])
                    for channel in numerators:
                        u, v = channel[left] * left_scale, channel[right] * right_scale
                        channel[left], channel[right] = cosine * u - sine * v, sine * u + cosine * v
                    exponents[left] = exponents[right] = exponent + 1
        exponent = max(exponents)
        for channel in numerators:
            for i in range(n):
                channel[i] *= power(exponent - exponents[i])
        denominator *= power(exponent)
        common = denominator
        for channel in numerators:
            for numerator in channel:
                common = gcd(common, numerator)
        denominator //= common
        numerators = [[numerator // common for numerator in channel] for channel in numerators]
    return tuple(tuple(Fraction(numerator, denominator) for numerator in channel) for channel in numerators)


def edge_matchings(neighbors):
    matchings, used = [], []
    for i, row in enumerate(neighbors):
        for j in row:
            if i == j:
                continue
            edge = (min(i, j), max(i, j))
            for matching, vertices in zip(matchings, used):
                if edge[0] not in vertices and edge[1] not in vertices:
                    matching.append(edge)
                    vertices.update(edge)
                    break
            else:
                matchings.append([edge])
                used.append(set(edge))
    return tuple(tuple(matching) for matching in matchings)


def cayley_parts(t):
    p, q = t.numerator, t.denominator
    base, cosine, sine = q * q + p * p, q * q - p * p, 2 * p * q
    common = gcd(gcd(base, cosine), sine)
    return base // common, cosine // common, sine // common


def rotate_matched_fraction_channels(channels, neighbors, coefficients, strength):
    denominator = 1
    for channel in channels:
        for value in channel:
            denominator = denominator * value.denominator // gcd(denominator, value.denominator)
    numerators = [[value.numerator * (denominator // value.denominator) for value in channel]
                  for channel in channels]
    matchings = edge_matchings(neighbors)
    for coefficient in coefficients:
        base, cosine, sine = cayley_parts(strength * as_fraction(coefficient))
        for matching in matchings:
            for c, channel in enumerate(numerators):
                rotated = [value * base for value in channel]
                for left, right in matching:
                    u, v = channel[left], channel[right]
                    rotated[left], rotated[right] = cosine * u - sine * v, sine * u + cosine * v
                numerators[c] = rotated
            denominator *= base
        common = denominator
        for channel in numerators:
            for numerator in channel:
                common = gcd(common, numerator)
                if common == 1:
                    break
            if common == 1:
                break
        if common > 1:
            denominator //= common
            numerators = [[numerator // common for numerator in channel] for channel in numerators]
    return tuple(tuple(Fraction(numerator, denominator) for numerator in channel) for channel in numerators)


@dataclass(frozen=True)
class FieldBinding:
    result: Evaluation
    parameters: tuple = ()
    preset: str = "explicit"

    @classmethod
    def from_parameters(cls, sid, parameters=None):
        sid = integer(sid)
        if not 1 <= sid <= 29:
            raise ValueError("The sutra id must be between one and twenty-nine")
        preset = "explicit"
        if parameters is None:
            parameters = EXAMPLE_PARAMETERS[sid]
            preset = f"tirthaji_application_example_{sid}_v1"
        return cls(evaluate(sid, **parameters),
                   tuple((name, frozen_operand(value)) for name, value in sorted(parameters.items())), preset)

    def rotate_graph(self, values, neighbors, strength=Fraction(1)):
        values = tuple(exact_field_value(value) for value in values)
        neighbors = tuple(tuple(integer(j) for j in row) for row in neighbors)
        strength = as_fraction(strength)
        if not values or len(values) != len(neighbors):
            raise ValueError("A nonempty field and one adjacency row per vertex are required")
        if any(j < 0 or j >= len(values) for row in neighbors for j in row):
            raise ValueError("An adjacency index is outside the field")
        if all(isinstance(value, Fraction) for value in values):
            return rotate_fraction_channels((values,), neighbors, self.result.spectral_coefficients, strength)[0]
        if all(isinstance(getattr(value, "real", None), Fraction)
               and isinstance(getattr(value, "imag", None), Fraction) for value in values):
            channels = tuple(tuple(getattr(value, component) for value in values) for component in ("real", "imag"))
            real, imag = rotate_fraction_channels(channels, neighbors, self.result.spectral_coefficients, strength)
            return tuple(type(value)(r, i) for value, r, i in zip(values, real, imag))
        result = list(values)
        for coefficient in self.result.spectral_coefficients:
            t = strength * as_fraction(coefficient)
            denominator = 1 + t * t
            cosine, sine = (1 - t * t) / denominator, 2 * t / denominator
            for i, row in enumerate(neighbors):
                for j in row:
                    left, right = min(i, j), max(i, j)
                    if left != right:
                        u, v = result[left], result[right]
                        result[left] = cosine * u - sine * v
                        result[right] = sine * u + cosine * v
        return tuple(result)

    def rotate_matched(self, values, neighbors, strength=Fraction(1)):
        values = tuple(exact_field_value(value) for value in values)
        neighbors = tuple(tuple(integer(j) for j in row) for row in neighbors)
        strength = as_fraction(strength)
        if not values or len(values) != len(neighbors):
            raise ValueError("A nonempty field and one adjacency row per vertex are required")
        if any(j < 0 or j >= len(values) for row in neighbors for j in row):
            raise ValueError("An adjacency index is outside the field")
        coefficients = self.result.spectral_coefficients
        if all(isinstance(value, Fraction) for value in values):
            return rotate_matched_fraction_channels((values,), neighbors, coefficients, strength)[0]
        if all(isinstance(getattr(value, "real", None), Fraction)
               and isinstance(getattr(value, "imag", None), Fraction) for value in values):
            channels = tuple(tuple(getattr(value, component) for value in values) for component in ("real", "imag"))
            real, imag = rotate_matched_fraction_channels(channels, neighbors, coefficients, strength)
            return tuple(type(value)(r, i) for value, r, i in zip(values, real, imag))
        result = list(values)
        matchings = edge_matchings(neighbors)
        for coefficient in coefficients:
            t = strength * as_fraction(coefficient)
            denominator = 1 + t * t
            cosine, sine = (1 - t * t) / denominator, 2 * t / denominator
            for matching in matchings:
                for left, right in matching:
                    u, v = result[left], result[right]
                    result[left] = cosine * u - sine * v
                    result[right] = sine * u + cosine * v
        return tuple(result)

    def receipt(self, before, after, neighbors, strength, mapping):
        neighbors = tuple(tuple(integer(j) for j in row) for row in neighbors)
        record = self.metadata(mapping)
        record["strength"] = str(rational(strength))
        record["topology_sha256"] = hashlib.sha256(json.dumps(neighbors, separators=(",", ":")).encode()).hexdigest()
        record["input_sha256"] = exact_digest(before)
        record["output_sha256"] = exact_digest(after)
        record["binding_sha256"] = hashlib.sha256(json.dumps(self.metadata(mapping), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return record

    def apply_graph(self, values, neighbors, strength=Fraction(1), executor=None, chunk_size=256):
        values = tuple(exact_field_value(value) for value in values)
        neighbors = tuple(tuple(integer(j) for j in row) for row in neighbors)
        strength = as_fraction(strength)
        if not values or len(values) != len(neighbors):
            raise ValueError("A nonempty field and one adjacency row per vertex are required")
        if any(j < 0 or j >= len(values) for row in neighbors for j in row):
            raise ValueError("An adjacency index is outside the field")
        chunk_size = integer(chunk_size)
        if chunk_size < 1:
            raise ValueError("The row chunk size must be positive")
        power, result = values, values
        for coefficient in self.result.spectral_coefficients:
            if executor is None:
                power = laplacian_rows(power, neighbors, 0)
            else:
                jobs = [executor.submit(laplacian_rows, power, neighbors[start:start + chunk_size], start)
                        for start in range(0, len(values), chunk_size)]
                power = tuple(value for job in jobs for value in job.result())
            weight = strength * as_fraction(coefficient)
            result = tuple(previous + weight * term for previous, term in zip(result, power))
        return result

    def apply_hypercube(self, values, strength=Fraction(1)):
        values = tuple(values)
        n = len(values)
        if n < 2 or n & (n - 1):
            raise ValueError("A hypercube field must have a power-of-two vertex count >= 2")
        neighbors = tuple(tuple(i ^ (1 << bit) for bit in range(n.bit_length() - 1)) for i in range(n))
        return self.apply_graph(values, neighbors, strength)

    def apply_torus(self, values, shape, strength=Fraction(1)):
        shape = tuple(integer(n) for n in shape)
        if not shape or any(n < 1 for n in shape):
            raise ValueError("All torus dimensions must be positive")
        coordinates = tuple(product(*(range(n) for n in shape)))
        values = tuple(values)
        if len(values) != len(coordinates):
            raise ValueError("The field size does not match the complete torus")
        indices = {point: i for i, point in enumerate(coordinates)}
        neighbors = []
        for point in coordinates:
            row = []
            for axis, n in enumerate(shape):
                for sign in (-1, 1):
                    adjacent = list(point)
                    adjacent[axis] = (adjacent[axis] + sign) % n
                    row.append(indices[tuple(adjacent)])
            neighbors.append(tuple(row))
        return self.apply_graph(values, neighbors, strength)

    def metadata(self, mapping="laplacian_polynomial"):
        record = {
            "book": dict(BOOK),
            "sutra": self.result.sutra.id,
            "application": self.result.sutra.application,
            "pages": self.result.sutra.pages,
            "mapping": "authored_complete_graph_laplacian_polynomial",
            "equation": "psi_out = psi + strength*sum(c[m]*L**(m+1)*psi for m in range(len(c)))",
            "laplacian": "(L*psi)[i] = len(neighbors[i])*psi[i] - sum(psi[j] for j in neighbors[i])",
            "coefficients": [str(c) for c in self.result.spectral_coefficients],
            "coefficient_omissions": 0,
            "input_parameters": "explicit_application_operands_or_named_example_preset",
            "field_model_authority": "new_construction",
            "mapping_version": 1,
            "operands": dict(self.parameters),
            "operand_preset": self.preset,
        }
        if mapping == "edge_rotations":
            record.update({
                "mapping": "authored_complete_ordered_edge_rotations",
                "equation": "(u,v) -> ((1-t*t)*u-2*t*v, 2*t*u+(1-t*t)*v)/(1+t*t); t=strength*c[m]",
                "edge_order": "coefficient_order_then_vertex_order_then_adjacency_order",
                "edge_orientation": "lower_vertex_index_to_higher_vertex_index",
                "edge_multiplicity": "every_adjacency_entry_is_applied",
                "self_edges": "identity",
                "norm": "sum(abs(psi[i])**2) is invariant",
            })
        elif mapping == "matched_edge_rotations":
            record.update({
                "mapping": "authored_complete_matched_edge_rotations",
                "equation": "(u,v) -> ((1-t*t)*u-2*t*v, 2*t*u+(1-t*t)*v)/(1+t*t); t=strength*c[m]",
                "edge_order": "coefficient_order_then_greedy_first_fit_matchings_of_vertex_then_adjacency_order",
                "edge_orientation": "lower_vertex_index_to_higher_vertex_index",
                "edge_multiplicity": "every_adjacency_entry_is_applied",
                "matching": "edges in one matching share no vertex, so their rotations commute",
                "self_edges": "identity",
                "norm": "sum(abs(psi[i])**2) is invariant",
            })
        elif mapping != "laplacian_polynomial":
            raise ValueError("Unknown authored field mapping")
        return record
