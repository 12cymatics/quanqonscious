"""Check whether each of the four sutra auxiliary losses actually reaches Ψ.

An auxiliary loss with zero gradient w.r.t. Ψ cannot influence training: it
only adds a constant to the reported loss. Two of the four were in exactly
that state for the whole first ablation, and this script is what found it.

It previously ended with an unconditional ``return 0``. It printed the dead
list and exited successfully, so ``reproduce_ablation.sh`` -- which runs it
under ``set -euo pipefail`` -- sailed past a result that should have stopped
the pipeline. A detector that cannot fail detects nothing.

It also hardcoded the loss weights, so its ``weighted`` column described a
configuration that may not be the one being run. The weights now come from
the config under test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

from vedic.kernel.hessian import HessianModule
from vedic.kernel.sutras_torch import S5, S7, S11
from vedic.kernel.wht import wht_axis_torch
from vedic.training.losses import L_chi, L_cons, L_curv, L_dual

REPO = Path(__file__).resolve().parents[1]

_KEY = {"L_chi": "alpha_chi", "L_cons": "beta_cons",
        "L_curv": "gamma_curv", "L_dual": "delta_dual"}


def weights_from(config: Path) -> dict[str, float]:
    cfg = yaml.safe_load(config.read_text())
    try:
        w = cfg["loss_weights"]
    except KeyError:
        raise SystemExit(f"{config} declares no loss_weights")
    missing = [k for k in _KEY.values() if k not in w]
    if missing:
        raise SystemExit(f"{config} is missing loss weights: {missing}")
    return {name: float(w[key]) for name, key in _KEY.items()}


def probe(weights: dict[str, float]) -> dict:
    torch.manual_seed(0)
    s5, s7, s11, h = S5(), S7(), S11(), HessianModule()
    wht = wht_axis_torch(device="cpu")
    fns = {
        "L_chi": lambda p, t: L_chi(p, s7),
        "L_cons": lambda p, t: L_cons(p, t),
        "L_curv": lambda p, t: L_curv(p, h),
        "L_dual": lambda p, t: L_dual(p, wht, s5, s11),
    }
    out: dict = {"losses": {}}
    for name, fn in fns.items():
        psi = torch.randn(8, 16, requires_grad=True)
        ts = torch.arange(8, dtype=torch.long)
        v = fn(psi, ts)
        if not v.requires_grad:
            grad_l1 = 0.0
        else:
            g, = torch.autograd.grad(v, psi, allow_unused=True)
            grad_l1 = 0.0 if g is None else float(g.abs().sum())
        value = float(v.detach())
        out["losses"][name] = {
            "value": value,
            "weight": weights[name],
            "weighted": weights[name] * value,
            "grad_l1": grad_l1,
            "reaches_psi": grad_l1 > 0.0,
        }
    a, b = torch.randn(2, 16), torch.randn(2, 16)
    out["g_ab_constant_in_psi"] = bool(torch.equal(h(a), h(b)))
    out["dead"] = sorted(k for k, v in out["losses"].items()
                         if not v["reaches_psi"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=REPO / "configs" / "ablations" / "cpu_full.yaml",
                    help="config whose loss_weights to report against")
    ap.add_argument("--output", type=Path, default=None,
                    help="write the report as JSON here")
    args = ap.parse_args()

    config = args.config.resolve()
    if not config.exists():
        raise SystemExit(f"no such config: {args.config}")
    report = probe(weights_from(config))
    report["config"] = (str(config.relative_to(REPO))
                        if config.is_relative_to(REPO) else str(config))
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)

    if report["dead"]:
        print(f"\nDEAD (zero gradient w.r.t. Psi, cannot affect training): "
              f"{report['dead']}", file=sys.stderr)
        print("Training on this config would optimise fewer objectives than "
              "it reports. Fix the losses or change the config before "
              "running the ablation.", file=sys.stderr)
        return 1
    print("\nOK — all four auxiliary losses reach Psi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
