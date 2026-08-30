"""Check whether each of the four sutra auxiliary losses actually reaches Ψ.

An auxiliary loss with zero gradient w.r.t. Ψ cannot influence training: it
only adds a constant to the reported loss. Two of the four were in exactly
that state for the whole first ablation, and this script is what found it.

It previously ended with an unconditional ``return 0``. It printed the dead
list and exited successfully, so the shell driver that ran it under ``set
-euo pipefail`` sailed past a result that should have stopped the pipeline.
A detector that cannot fail detects nothing. It exits non-zero on a dead
loss now, whatever invokes it.

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


#: The eight-row Psi batch every loss is probed on. It was
#: ``torch.randn(8, 16)`` under a fixed manual seed -- deterministic, but
#: still a draw: a detector whose verdict depends on which vectors a PRNG
#: happened to produce is reporting on those vectors. These eight are
#: enumerated and each is a case. Every component is dyadic (k/2^m), so the
#: probe reads the same on any machine and needs no seed to fix it.
#:
#: A loss with a genuine gradient path to Psi is non-zero somewhere on a set
#: this varied; one with none is zero on all of it, which is what the probe
#: is looking for.
PROBE_ROWS: tuple[tuple[float, ...], ...] = (
    tuple(1.0 if v == 0 else 0.0 for v in range(16)),          # a low spike
    tuple(1.0 if v == 15 else 0.0 for v in range(16)),         # a high spike
    tuple(1.0 for _ in range(16)),                             # constant
    tuple(1.0 if v % 2 == 0 else -1.0 for v in range(16)),     # alternating
    tuple((v - 8) / 4.0 for v in range(16)),                   # a ramp
    tuple((8 - v) / 4.0 for v in range(16)),                   # its reverse
    tuple(1.0 / 2 ** (v % 8) for v in range(16)),              # decaying
    tuple((-1.0) ** v * (v + 1) / 8.0 for v in range(16)),     # signed fan
)

#: Two Psi that differ in every component, for the g_ab constancy check.
CONSTANCY_ROWS: tuple[tuple[float, ...], ...] = (
    tuple((v + 1) / 8.0 for v in range(16)),
    tuple((v + 1) / -4.0 for v in range(16)),
)


def _psi(rows: tuple[tuple[float, ...], ...], grad: bool = False) -> torch.Tensor:
    return torch.tensor([list(r) for r in rows],
                        dtype=torch.float32, requires_grad=grad)


def probe(weights: dict[str, float]) -> dict:
    s5, s7, s11, h = S5(), S7(), S11(), HessianModule()
    wht = wht_axis_torch(device="cpu")
    fns = {
        "L_chi": lambda p: L_chi(p, s7),
        "L_cons": lambda p: L_cons(p),
        "L_curv": lambda p: L_curv(p, h),
        "L_dual": lambda p: L_dual(p, wht, s5, s11),
    }
    out: dict = {"losses": {}}
    for name, fn in fns.items():
        psi = _psi(PROBE_ROWS, grad=True)
        v = fn(psi)
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
    a, b = _psi(CONSTANCY_ROWS[:1]), _psi(CONSTANCY_ROWS[1:])
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
