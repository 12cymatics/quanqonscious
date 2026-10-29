import argparse
import json
from fractions import Fraction

import vedic_book as book
from vedic_book_cases import EXAMPLE_PARAMETERS
from vedic_plate import PlateField


class VedicSutraEngine:
    evaluate = staticmethod(book.evaluate)

    @staticmethod
    def nikhilam(a, b, base):
        result = book.nikhilam(a, b, base)
        return (result.value,) + result.spectral_coefficients

    @staticmethod
    def urdhva(a_digits, b_digits):
        if not a_digits or not b_digits:
            raise ValueError("Both digit sequences must be nonempty")
        length = max(len(a_digits), len(b_digits))
        a = [0] * (length - len(a_digits)) + list(a_digits)
        b = [0] * (length - len(b_digits)) + list(b_digits)
        return list(book.urdhva(a, b).value)

    @staticmethod
    def yavadunam(n, base):
        result = book.yavadunam(n, base)
        deficiency, left, right = result.spectral_coefficients
        return result.value, abs(deficiency), left / book.rational(base), right

    sopantya = staticmethod(book.sopantya)


def exact_parameters(value):
    if isinstance(value, dict):
        return {key: exact_parameters(item) for key, item in value.items()}
    if isinstance(value, list):
        return [exact_parameters(item) for item in value]
    if isinstance(value, str):
        return Fraction(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("JSON numbers must be integers; use strings for exact fractions")
    return value


def encoded_parameters(value):
    if isinstance(value, dict):
        return {key: encoded_parameters(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encoded_parameters(item) for item in value]
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return str(book.rational(value))


def visualize_sutra_cymatics(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sutra", type=int, choices=range(1, 30), action="append")
    parser.add_argument("--parameters")
    parser.add_argument("--intervals", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    ids = args.sutra or list(range(1, 30))
    if args.parameters is not None and len(ids) != 1:
        parser.error("--parameters requires exactly one --sutra")
    records = []
    for sid in ids:
        try:
            parameters = EXAMPLE_PARAMETERS[sid] if args.parameters is None else exact_parameters(json.loads(args.parameters))
            if not isinstance(parameters, dict):
                raise ValueError("--parameters must be a JSON object")
            result = book.evaluate(sid, **parameters)
            field = PlateField.from_evaluation(result)
            if args.intervals is not None:
                field.validate_grid(args.intervals)
            record = {
                "sutra": sid,
                "name": result.sutra.name,
                "application": result.sutra.application,
                "book": book.BOOK,
                "pages": result.sutra.pages,
                "parameters": encoded_parameters(parameters),
                "value": str(result.value),
                "field": field.metadata(),
            }
            if args.json:
                records.append(record)
            else:
                print(f"S{sid}: {result.sutra.name}")
                print(f"{result.sutra.application}: {result.value}")
                print(f"pp. {', '.join(map(str, result.sutra.pages))}")
                print(field.render(args.intervals))
                print()
        except (ArithmeticError, TypeError, ValueError) as exc:
            parser.error(f"S{sid}: {exc}")
    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    visualize_sutra_cymatics()
