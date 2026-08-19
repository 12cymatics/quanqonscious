"""Check whether each of the four sutra auxiliary losses actually reaches Psi.

An auxiliary loss with zero gradient w.r.t. Psi cannot influence training:
it only adds a constant to the reported loss.
"""
import json, sys, torch
from vedic.kernel.sutras_torch import S5, S7, S11
from vedic.kernel.hessian import HessianModule
from vedic.kernel.wht import wht_axis_torch
from vedic.training.losses import L_chi, L_cons, L_curv, L_dual

W = {"L_chi": 0.10, "L_cons": 0.05, "L_curv": 0.02, "L_dual": 0.05}


def main() -> int:
    torch.manual_seed(0)
    s5, s7, s11, h = S5(), S7(), S11(), HessianModule()
    wht = wht_axis_torch(device="cpu")
    fns = {
        "L_chi":  lambda p, t: L_chi(p, s7),
        "L_cons": lambda p, t: L_cons(p, t),
        "L_curv": lambda p, t: L_curv(p, h),
        "L_dual": lambda p, t: L_dual(p, wht, s5, s11),
    }
    out = {}
    for name, fn in fns.items():
        psi = torch.randn(8, 16, requires_grad=True)
        ts = torch.arange(8, dtype=torch.long)
        v = fn(psi, ts)
        if not v.requires_grad:
            gn = 0.0
        else:
            g, = torch.autograd.grad(v, psi, allow_unused=True)
            gn = 0.0 if g is None else float(g.abs().sum())
        out[name] = {"value": float(v), "weight": W[name],
                     "weighted": W[name] * float(v), "grad_l1": gn,
                     "reaches_psi": gn > 0.0}

    a, b = torch.randn(2, 16), torch.randn(2, 16)
    out["g_ab_constant_in_psi"] = bool(torch.equal(h(a), h(b)))
    print(json.dumps(out, indent=2))
    dead = [k for k, v in out.items() if isinstance(v, dict) and not v["reaches_psi"]]
    print(f"\nDEAD (zero gradient, cannot affect training): {dead or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
