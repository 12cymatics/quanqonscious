"""Print the 29 canonical sutras: id, name, category, formula, coefficient.

    python scripts/show_sutras.py               # the full table
    python scripts/show_sutras.py --verify      # + structural checks
    python scripts/show_sutras.py --drift 50    # + drift ranking at strength 50
"""
from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vedic.kernel import sutras_canonical as K  # noqa: E402

FORMULA = {
    "MULT": "Ψ'ᵢ = Ψᵢ·(1 + α·Ψ_{i⊕1})",
    "REFL": "Ψ'ᵢ = blend(Ψᵢ, (Ψᵢ+Ψ_c)/2, α)",
    "CONV": "Ψ'ᵢ = blend(Ψᵢ, (Ψ⊛Ψ)ᵢ/16, α)",
    "DIV":  "Ψ'ᵢ = blend(Ψᵢ, m + hw(i)/4·(edge−m), α)",
    "DIFF": "Ψ'ᵢ = blend(Ψᵢ, edgeMean(i), α)",
    "PERM": "Ψ'ᵢ = blend(Ψᵢ, Ψ_{i⊕2^((id+1)&3)}, α)",
    "MOD":  "Ψ'ᵢ = blend(Ψᵢ, mean(Ψ), α)",
}
S5_FORMULA = "Ψ'ᵢ = blend(Ψᵢ, −Ψ_c, α)   [zero-sum]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--drift", type=int, default=None, metavar="STRENGTH")
    a = ap.parse_args()

    print(f"THE {K.N_SUTRAS} SUTRAS — canonical definitions (exact ℚ)")
    print(f"α(n) = (n/{K.SUTRA_SUM})·(strength/100),  "
          f"{K.SUTRA_SUM} = T(29) = 29·30/2")
    print(f"blend(c,t,w) = c + (t−c)·w\n")
    # Column width is measured from the data, so no name is ever cut. A
    # fixed 34 with a [:34] slice silently renamed the longer sutras in every
    # listing this script produced.
    name_w = max(len("name"), max(len(s.name) for s in K.SUTRAS))
    coef_w = max(len("coefficient"),
                 max(len(str(s.coefficient)) for s in K.SUTRAS))
    rule = 3 + 2 + name_w + 1 + 15 + 1 + 5 + 1 + coef_w
    print(f"{'id':>3}  {'name':{name_w}} {'category':15} {'kind':5} "
          f"{'coefficient':>{coef_w}}")
    print("─" * rule)
    for s in K.SUTRAS:
        print(f"{s.id:>3}  {s.name:{name_w}} {s.category:15} {s.kind:5} "
              f"{str(s.coefficient):>{coef_w}}")

    print("\nOPERATOR FORMULAE (seven structurally different couplings)")
    print("─" * 92)
    for kind in ("MULT", "REFL", "CONV", "DIV", "DIFF", "PERM", "MOD"):
        ids = [s.id for s in K.SUTRAS if s.kind == kind]
        print(f"  {kind:5} {FORMULA[kind]:46} S{ids}")
    print(f"  {'':5} {S5_FORMULA:46} S[5]")

    failures: list[str] = []

    if a.verify:
        psi = tuple(Fraction(v * v + 1, 7) for v in range(16))
        print("\nSTRUCTURAL VERIFICATION")
        print("─" * 92)

        def check(label: str, ok: bool) -> None:
            # Every one of these printed True or False and the script returned
            # 0 either way, so a red line was indistinguishable from a green
            # one to anything but a human reading the terminal.
            print(f"  {label:34}{ok}")
            if not ok:
                failures.append(label.strip())

        check("all 29 present",
              len(K.SUTRAS) == 29 and K.ALL == tuple(range(1, 30)))
        check("Σδ(1..29) = T(29) = 435",
              sum(s.delta for s in K.SUTRAS) == 435)
        check("α→0 ⇒ identity, all 29 (§12Y)",
              all(K.apply_sutra(s.id, psi, Fraction(0)) == psi
                  for s in K.SUTRAS))
        check("every sutra moves the field",
              all(K.apply_sutra(s.id, psi, Fraction(50)) != psi
                  for s in K.SUTRAS))
        check("exact ℚ throughout, no floats",
              all(isinstance(x, Fraction) for s in K.SUTRAS
                  for x in K.apply_sutra(s.id, psi, Fraction(50))))
        check("full cascade non-degenerate",
              any(x != 0 for x in K.apply_all(psi, Fraction(50))))

    if a.drift is not None:
        psi = tuple(Fraction(v * v + 1, 7) for v in range(16))
        st = Fraction(a.drift)
        print(f"\nDRIFT RANKING at strength {a.drift}  (§3.7: D_k = |Q(S_kΨ) − Q(Ψ)|)")
        print("─" * 92)
        for sid, d in K.rank_by_drift(psi, st):
            print(f"  S{sid:<3} {K.NAMES[sid]:{name_w}} {K.CATEGORY[sid]:15} "
                  f"{float(d):.6e}")
        q0 = K.norm_sq(psi)
        print("\n  conservation cores (§3.8), relative |ΔQ|/Q:")
        for label, core in (("wormhole", K.WORMHOLE_CORE),
                            ("symmetry", K.SYMMETRY_CORE),
                            ("all 29", K.ALL)):
            q = K.norm_sq(K.apply_all(psi, st, core))
            print(f"    {label:9} {float(abs(q - q0) / q0):.6f}")

    if failures:
        print(f"\n{len(failures)} structural check(s) FAILED: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
