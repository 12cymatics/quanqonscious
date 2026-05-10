"""Generate `fixtures/*.json` from the ℚ kernel.

These JSON files are the bit-exact reference that every other consumer
(``test_simulator_match.py``, the v18.16 simulator export, the
verify_bit_exact gate) compares against. The Fraction kernel is the
authoritative simulator: when the user runs the v18.16 HTML through
``scripts/export_simulator_fixtures.py`` (Playwright) on their Mac, the
JSON byte-stream MUST match what this script produced.

Rationals are serialised as ``{"num": int, "den": int}`` so there is no
float round-trip and no precision loss.

Run:
    python scripts/build_fixtures.py
"""
from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from pathlib import Path

from vedic.kernel import conservation_exact as ce
from vedic.kernel import sutras_exact as se
from vedic.kernel.q import Q16


def _frac_to_obj(x: Fraction) -> dict[str, int]:
    return {"num": x.numerator, "den": x.denominator}


def _q_to_obj(psi: Q16) -> list[dict[str, int]]:
    return [_frac_to_obj(x) for x in psi]


def _random_q16(rng: random.Random, denom_max: int = 1000) -> Q16:
    out = []
    for _ in range(16):
        num = rng.randint(-denom_max, denom_max)
        den = rng.randint(1, denom_max)
        out.append(Fraction(num, den))
    return tuple(out)


def build(seed: int, n_inputs: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    inputs: list[Q16] = [_random_q16(rng) for _ in range(n_inputs)]

    psi_inputs_path = out_dir / "psi_inputs.json"
    sutra_outputs_path = out_dir / "sutra_outputs.json"
    cons_path = out_dir / "conservation_residuals.json"

    with psi_inputs_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "n": n_inputs,
                "denom_max": 1000,
                "inputs": [_q_to_obj(p) for p in inputs],
            },
            f,
            indent=2,
            sort_keys=True,
        )

    sutra_records: list[dict[str, object]] = []
    for psi in inputs:
        rec: dict[str, object] = {"input": _q_to_obj(psi)}
        rec["S1"] = _q_to_obj(se.s1_eka_adhikena(psi))
        rec["S2"] = _q_to_obj(se.s2_nikhilam(psi))
        rec["S4"] = _q_to_obj(se.s4_paravartya(psi))
        rec["S5"] = _q_to_obj(se.s5_shunyam_samya(psi))
        rec["S6"] = _q_to_obj(se.s6_anurupya_shunyam(psi))
        sym, anti = se.s7_sankalana_vyavakalana(psi)
        rec["S7_sym"] = _q_to_obj(sym)
        rec["S7_anti"] = _q_to_obj(anti)
        rec["S8"] = _q_to_obj(se.s8_puranapuranabhyam_fill(psi))
        rec["S9"] = _q_to_obj(se.s9_chalana_kalanabhyam(psi))
        rec["S10"] = _q_to_obj(se.s10_yavadunam_tavadunikrtya(psi))
        rec["S11"] = _q_to_obj(se.s11_vyasti_samasti(psi))
        rec["S12"] = _q_to_obj(se.s12_shesanyankena_charamena(psi))
        rec["S13"] = _q_to_obj(se.s13_sopantyadvayamantyam_last2(psi))
        rec["S14"] = _q_to_obj(se.s14_ekanyunena_purvena(psi))
        rec["S15"] = _q_to_obj(se.s15_gunitasamucchaya_product(psi))
        rec["S16"] = _q_to_obj(se.s16_gunaka_samucchaya(psi))
        rec["S18"] = _frac_to_obj(se.s18_adyamadyena_antyamantyena(psi))
        rec["S19"] = _q_to_obj(se.s19_lopana_sthapanabhyam(psi))
        rec["S20"] = _q_to_obj(se.s20_vilokanam_spect(psi))
        rec["S21"] = _q_to_obj(se.s21_dhvajanka_flag(psi))
        rec["S22"] = [_frac_to_obj(x) for x in se.s22_parity_complement(psi)]
        rec["S24"] = _q_to_obj(se.s24_kevalaih_saptakam(psi))
        rec["S25"] = _q_to_obj(se.s25_vestana_circular(psi))
        rec["S26"] = _q_to_obj(se.s26_yavadunam_square(psi))
        rec["S27"] = _frac_to_obj(se.s27_samuccaya_gunitah(psi))
        rec["S28"] = _q_to_obj(se.s28_lopana_restore(psi))
        rec["S29"] = _q_to_obj(se.s29_mean_drive(psi))
        sutra_records.append(rec)

    with sutra_outputs_path.open("w", encoding="utf-8") as f:
        json.dump({"records": sutra_records}, f, indent=2, sort_keys=True)

    cons_records: list[dict[str, object]] = []
    for psi in inputs:
        # R1 only meaningful with an integer trace; include both zero and
        # multiple-of-435 to exercise both sides.
        for trace in (Fraction(0), Fraction(435), Fraction(7 * 435)):
            r1, r2, r3, r4 = ce.all_residuals(psi, trace)
            cons_records.append(
                {
                    "trace_sum": _frac_to_obj(trace),
                    "R1": _frac_to_obj(r1),
                    "R2": _frac_to_obj(r2),
                    "R3": _frac_to_obj(r3),
                    "R4": _frac_to_obj(r4),
                }
            )

    with cons_path.open("w", encoding="utf-8") as f:
        json.dump({"records": cons_records}, f, indent=2, sort_keys=True)

    print(f"wrote {psi_inputs_path}")
    print(f"wrote {sutra_outputs_path}")
    print(f"wrote {cons_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vedic_trainer ℚ fixtures.")
    parser.add_argument("--seed", type=int, default=0xC0DEC0DE)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("fixtures"))
    args = parser.parse_args()
    build(args.seed, args.n, args.out)


if __name__ == "__main__":
    main()
