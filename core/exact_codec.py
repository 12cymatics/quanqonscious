from fractions import Fraction
import math


def encode_exact(value):
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return {"$exact": "integer", "hex": hex(value)}
    if isinstance(value, Fraction):
        return {"$exact": "rational", "numerator": hex(value.numerator),
                "denominator": hex(value.denominator)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Trace values must be finite")
        return {"$exact": "binary64", "hex": value.hex()}
    if isinstance(value, tuple):
        return {"$exact": "tuple", "items": [encode_exact(item) for item in value]}
    if isinstance(value, list):
        return [encode_exact(item) for item in value]
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value) and "$exact" not in value:
            return {key: encode_exact(item) for key, item in value.items()}
        return {"$exact": "mapping", "items": [[encode_exact(key), encode_exact(item)]
                                               for key, item in value.items()]}
    if getattr(value, "is_Rational", False):
        return encode_exact(Fraction(int(value.p), int(value.q)))
    raise TypeError(f"Unsupported exact trace value: {type(value).__name__}")


def decode_exact(value):
    if isinstance(value, list):
        return [decode_exact(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("$exact")
    if kind is None:
        return {key: decode_exact(item) for key, item in value.items()}
    if kind == "integer":
        return int(value["hex"], 16)
    if kind == "rational":
        return Fraction(int(value["numerator"], 16), int(value["denominator"], 16))
    if kind == "binary64":
        result = float.fromhex(value["hex"])
        if not math.isfinite(result):
            raise ValueError("Trace values must be finite")
        return result
    if kind == "tuple":
        return tuple(decode_exact(item) for item in value["items"])
    if kind == "mapping":
        return {decode_exact(key): decode_exact(item) for key, item in value["items"]}
    raise ValueError(f"Unknown exact trace encoding: {kind}")


def fraction_text(value):
    return f"{hex(value.numerator)}/{hex(value.denominator)}"


def digest_integer(digest, value):
    encoded = value.to_bytes((value.bit_length() + 8) // 8, "big", signed=True)
    digest.update(len(encoded).to_bytes(8, "big") + encoded)
