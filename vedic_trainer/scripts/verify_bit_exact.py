"""Bit-exact verification gate.

Runs the ℚ kernel against the committed fixtures. Exits 0 on success and
1 on any mismatch with a unified diff. Training scripts call this via
``subprocess.check_call`` before any optimizer step is taken.

No floats. The kernel is ℚ throughout; the fixtures store {num, den}
rationals; mismatches are bit-exact integer-pair mismatches.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

from vedic.kernel import conservation_exact as ce
from vedic.kernel import z2_primitives as se
from vedic.kernel.q import Q16


REPO = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO / "fixtures"


def _require_fixtures() -> None:
    """Refuse to run if the committed reference is absent.

    This used to call ``build_fixtures.py`` when a fixture was missing --
    regenerating the reference *from the same kernel the gate then compares
    against*, so an empty ``fixtures/`` produced a confident
    "bit-exact" pass having verified nothing at all.

    The fixtures are a committed reference, not a build artefact. If they are
    missing the answer is that the gate cannot run, not that it should
    manufacture something to agree with.
    """
    missing = [p.name for p in (
        FIXTURE_DIR / "psi_inputs.json",
        FIXTURE_DIR / "sutra_outputs.json",
        FIXTURE_DIR / "conservation_residuals.json",
    ) if not p.exists()]
    if missing:
        raise SystemExit(
            f"fixtures/ is missing {', '.join(missing)}. These are the "
            f"committed reference this gate checks against; they are tracked "
            f"in git. Restore them (git checkout -- fixtures/) rather than "
            f"rebuilding: scripts/build_fixtures.py writes them from the same "
            f"kernel under test, so a rebuilt fixture cannot falsify anything."
        )


def _obj_to_frac(o: dict[str, int]) -> Fraction:
    return Fraction(int(o["num"]), int(o["den"]))


def _objs_to_q16(objs: list[dict[str, int]]) -> Q16:
    return tuple(_obj_to_frac(o) for o in objs)


def _load(name: str) -> dict[str, object]:
    with (FIXTURE_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def _frac_to_obj_cmp(x):
    return x


def main() -> int:
    _require_fixtures()

    psi_data = _load("psi_inputs.json")
    sutra_data = _load("sutra_outputs.json")
    cons_data = _load("conservation_residuals.json")

    inputs: list[Q16] = [_objs_to_q16(p) for p in psi_data["inputs"]]
    failures: list[str] = []

    # Exhaustive: every operator recorded in the fixture is recomputed and
    # compared. There is no spot-check and no representative subset -- a gate
    # that verifies a quarter of the kernel is not a gate.
    def _binary(psi):
        return se.s2_nikhilam(psi)

    vector_ops = {
        "S1": se.s1_eka_adhikena, "S2": se.s2_nikhilam,
        "S4": se.s4_paravartya, "S5": se.s5_shunyam_samya,
        "S6": se.s6_anurupya_shunyam, "S8": se.s8_puranapuranabhyam_fill,
        "S9": se.s9_chalana_kalanabhyam,
        "S10": se.s10_yavadunam_tavadunikrtya, "S11": se.s11_vyasti_samasti,
        "S12": se.s12_shesanyankena_charamena,
        "S13": se.s13_sopantyadvayamantyam_last2,
        "S14": se.s14_ekanyunena_purvena,
        "S15": se.s15_gunitasamucchaya_product,
        "S16": se.s16_gunaka_samucchaya, "S19": se.s19_lopana_sthapanabhyam,
        "S20": se.s20_vilokanam_spect, "S21": se.s21_dhvajanka_flag,
        "S24": se.s24_kevalaih_saptakam, "S25": se.s25_vestana_circular,
        "S26": se.s26_yavadunam_square, "S28": se.s28_lopana_restore,
        "S29": se.s29_mean_drive,
    }
    binary_ops = {
        "S3": se.s3_urdhva_tiryak,
        "S17": se.s17_anurupyena_proportion,
        "S23": se.s23_dwandwa_yoga,
    }
    scalar_ops = {
        "S18": se.s18_adyamadyena_antyamantyena,
        "S27": se.s27_samuccaya_gunitah,
    }

    checked_keys: set[str] = set()
    for i, rec in enumerate(sutra_data["records"]):
        psi = _objs_to_q16(rec["input"])
        if psi != inputs[i]:
            failures.append(f"input mismatch at idx {i}")

        for key, fn in vector_ops.items():
            if _objs_to_q16(rec[key]) != fn(psi):
                failures.append(f"{key} mismatch at idx {i}")
            checked_keys.add(key)

        for key, fn in binary_ops.items():
            if _objs_to_q16(rec[key]) != fn(psi, _binary(psi)):
                failures.append(f"{key} mismatch at idx {i}")
            checked_keys.add(key)

        for key, fn in scalar_ops.items():
            if _obj_to_frac(rec[key]) != fn(psi):
                failures.append(f"{key} mismatch at idx {i}")
            checked_keys.add(key)

        sym, anti = se.s7_sankalana_vyavakalana(psi)
        if _objs_to_q16(rec["S7_sym"]) != sym:
            failures.append(f"S7_sym mismatch at idx {i}")
        if _objs_to_q16(rec["S7_anti"]) != anti:
            failures.append(f"S7_anti mismatch at idx {i}")
        checked_keys.update({"S7_sym", "S7_anti"})

        if [_frac_to_obj_cmp(x) for x in se.s22_parity_complement(psi)] != \
                [(_obj_to_frac(o)) for o in rec["S22"]]:
            failures.append(f"S22 mismatch at idx {i}")
        checked_keys.add("S22")

        # Nothing in the fixture may go unverified.
        unchecked = set(rec) - checked_keys - {"input"}
        if unchecked:
            failures.append(f"fixture keys never verified at idx {i}: {sorted(unchecked)}")

    for i, rec in enumerate(cons_data["records"]):
        psi = inputs[i // 3]
        trace = _obj_to_frac(rec["trace_sum"])
        r1, r2, r3, r4 = ce.all_residuals(psi, trace)
        if _obj_to_frac(rec["R1"]) != r1:
            failures.append(f"R1 mismatch at idx {i}")
        if _obj_to_frac(rec["R2"]) != r2:
            failures.append(f"R2 mismatch at idx {i}")
        if _obj_to_frac(rec["R3"]) != r3:
            failures.append(f"R3 mismatch at idx {i}")
        if _obj_to_frac(rec["R4"]) != r4:
            failures.append(f"R4 mismatch at idx {i}")

    if failures:
        for f in failures[:20]:
            print(f"FAIL: {f}", file=sys.stderr)
        print(f"{len(failures)} bit-exact mismatches", file=sys.stderr)
        return 1
    print(
        f"OK — {len(inputs)} inputs, {len(sutra_data['records'])} sutra records, "
        f"{len(cons_data['records'])} conservation records bit-exact"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
