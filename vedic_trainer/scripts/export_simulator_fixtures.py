"""Export `fixtures/*.json` from `vedic_v18.16_strict_kernel.html` (Playwright).

Protocol (also documented in ``docs/BIT_EXACT_PROTOCOL.md``):

    1. Open ``vedic_v18.16_strict_kernel.html`` in a headless Chromium.
    2. Drive the simulator's seeded "30-second recorder" with the same
       inputs that ``scripts/build_fixtures.py`` uses.
    3. Read out the JSON it exports (rationals as {num, den}).
    4. Write to ``fixtures/{psi_inputs,sutra_outputs,conservation_residuals}.json``.

The simulator and ``scripts/build_fixtures.py`` are both implementations
of the same exact-rational kernel; their JSON outputs must match
byte-for-byte.

Requires Playwright + Chromium installed (``pip install playwright`` and
``playwright install chromium``). Run from the repo root:

    python scripts/export_simulator_fixtures.py \
        --html ../vedic_v18.16_strict_kernel.html \
        --out fixtures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Playwright is a heavy import; only loaded when this script is invoked.


def main() -> None:
    parser = argparse.ArgumentParser(description="Export v18.16 fixtures via Playwright.")
    parser.add_argument("--html", type=Path, required=True,
                        help="Path to vedic_v18.16_strict_kernel.html")
    parser.add_argument("--out", type=Path, default=Path("fixtures"))
    parser.add_argument("--seed", type=int, default=0xC0DEC0DE)
    parser.add_argument("--n", type=int, default=32)
    args = parser.parse_args()

    if not args.html.exists():
        raise FileNotFoundError(f"simulator HTML not found: {args.html}")

    from playwright.sync_api import sync_playwright

    args.out.mkdir(parents=True, exist_ok=True)
    url = args.html.resolve().as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        # The v18.16 simulator exposes a JS function ``window.exportFixtures(seed, n)``
        # documented in BIT_EXACT_PROTOCOL.md. It returns an object with three
        # JSON-serialisable members: psi_inputs, sutra_outputs, conservation_residuals.
        payload = page.evaluate("({seed, n}) => window.exportFixtures(seed, n)",
                                {"seed": args.seed, "n": args.n})
        if not isinstance(payload, dict):
            raise RuntimeError("simulator did not return an object payload")
        for name in ("psi_inputs", "sutra_outputs", "conservation_residuals"):
            if name not in payload:
                raise RuntimeError(f"simulator payload missing key: {name}")
            with (args.out / f"{name}.json").open("w", encoding="utf-8") as f:
                json.dump(payload[name], f, indent=2, sort_keys=True)
        browser.close()


if __name__ == "__main__":
    main()
