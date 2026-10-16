"""Derive the treatment arm's auxiliary weights from the run's initial state.

    python scripts/calibrate_aux_weights.py --control configs/scan_length/no_sutra_seed0.yaml

The rule, fixed in PREREGISTRATION_SCAN_LENGTH.md before any training step:

    α_k = FRACTION · ‖∇_LoRA L_CE‖ / ‖∇_LoRA L_k‖

with every gradient taken at the run's exact initial state (``build_run``,
the same construction ``train_lora.py`` trains), in eval mode so dropout does
not enter, as the gradient of the mean over the first CALIBRATION_RECORDS
training records in file order, in micro-batches of CALIBRATION_BATCH. Each
weighted auxiliary term then starts with a LoRA gradient one tenth the size
of cross-entropy's, so together they are at most 0.4 of it.

Why a rule, and why gradients
-----------------------------
The weights in the shipped configs made the auxiliary terms 94-99% of the loss
and 40-163x cross-entropy's LoRA-gradient norm at initialisation, depending on
the seed's projection draw and the batch (two independent measurements). With gradient
clipping active at every step, the "full" arm's update direction was then
about 98% auxiliary, and any difference between the arms would have measured
cross-entropy under-training rather than an effect of the terms. Clipping acts
on gradients, so the dominance that matters is the gradient's, not the loss
value's.

Why per seed
------------
The projection Ψ = W·pool(h) is a random orthogonal draw, and Ψ is almost
entirely one corpus-wide direction, so the auxiliary dose at initialisation
varied 2.5x across seeds through that draw alone. Calibrating each seed on its
own initial state gives every seed the same starting dose. It does not stop
the dose from moving during training (see the pre-registration).

Writes ``calibration_seed<k>.json`` (every measured quantity) and
``full_seed<k>.yaml`` (the control config with these weights and its own
output_dir) beside the control config.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vedic.provenance import environment, sha256_file  # noqa: E402
from vedic.training import load_yaml  # noqa: E402
from vedic.training.losses import L_chi, L_cons, L_curv, L_dual  # noqa: E402
from vedic.training.run import build_run  # noqa: E402

FRACTION = 0.1
CALIBRATION_RECORDS = 64
CALIBRATION_BATCH = 8
TERMS = {"L_chi": "alpha_chi", "L_cons": "beta_cons",
         "L_curv": "gamma_curv", "L_dual": "delta_dual"}


def measure(run) -> dict:
    from transformers import DataCollatorForLanguageModeling

    model, trainer = run.model, run.trainer
    model.eval()
    if len(run.train_ds) < CALIBRATION_RECORDS:
        raise ValueError("the training corpus is smaller than the calibration set")
    records = [run.train_ds[i] for i in range(CALIBRATION_RECORDS)]
    collate = DataCollatorForLanguageModeling(run.tokenizer, mlm=False)
    batches = [collate(records[i:i + CALIBRATION_BATCH])
               for i in range(0, CALIBRATION_RECORDS, CALIBRATION_BATCH)]
    n_batches = len(batches)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    w = trainer.tesseract_wm.proj.weight

    def terms(b):
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                    labels=b["labels"], output_hidden_states=True, use_cache=False)
        psi = trainer.tesseract_wm(out.hidden_states[-1], b["attention_mask"])
        return psi, {
            "L_CE": out.loss,
            "L_chi": L_chi(psi, trainer.s7),
            "L_cons": L_cons(psi),
            "L_curv": L_curv(psi, trainer.hessian),
            "L_dual": L_dual(psi, trainer.wht_axis, trainer.s5, trainer.s11),
        }

    def norm(params) -> float:
        return math.sqrt(sum(float(p.grad.double().pow(2).sum())
                             for p in params if p.grad is not None))

    measured: dict[str, dict] = {}
    psi_sq, psi_n = 0.0, 0
    for name in ("L_CE", *TERMS):
        model.zero_grad(set_to_none=True)
        value = 0.0
        for b in batches:
            psi, t = terms(b)
            (t[name] / n_batches).backward()
            value += float(t[name].detach()) / n_batches
            if name == "L_CE":
                psi_sq += float(psi.detach().double().pow(2).sum())
                psi_n += psi.numel()
        measured[name] = {"value": value, "grad_norm_lora": norm(lora),
                          "grad_norm_projection": norm([w])}
    model.zero_grad(set_to_none=True)
    g_ce = measured["L_CE"]["grad_norm_lora"]
    weights = {TERMS[k]: FRACTION * g_ce / measured[k]["grad_norm_lora"] for k in TERMS}
    return {"measured": measured, "weights": weights,
            "psi_rms": math.sqrt(psi_sq / psi_n)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", type=Path, required=True,
                    help="the no_sutra config of the seed being calibrated")
    a = ap.parse_args()

    cfg = load_yaml(a.control)
    if any(getattr(cfg.loss_weights, f) != 0.0 for f in TERMS.values()):
        raise ValueError(f"{a.control} is not a control config: its weights are not all 0")
    result = measure(build_run(cfg))

    seed = cfg.seed
    out_dir = a.control.parent
    record = {
        "rule": {"fraction": FRACTION, "records": CALIBRATION_RECORDS,
                 "batch": CALIBRATION_BATCH, "gradients_of": "LoRA parameters",
                 "mode": "eval (no dropout)"},
        "control_config": str(a.control),
        "control_config_sha256": sha256_file(a.control),
        "train_sha256": sha256_file(cfg.data.train_path),
        **result,
        "environment": environment(),
    }
    (out_dir / f"calibration_seed{seed}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")

    raw = yaml.safe_load(a.control.read_text(encoding="utf-8"))
    raw["output_dir"] = str(Path(raw["output_dir"]).parent / f"full_seed{seed}")
    raw["loss_weights"] = result["weights"]
    header = (f"# GENERATED by scripts/calibrate_aux_weights.py from {a.control.name}.\n"
              f"# Do not edit: regenerate. The weights follow the pre-registered rule\n"
              f"# alpha_k = {FRACTION} * |grad_LoRA L_CE| / |grad_LoRA L_k| at this\n"
              f"# seed's initial state; calibration_seed{seed}.json holds the measurement.\n")
    (out_dir / f"full_seed{seed}.yaml").write_text(
        header + yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    print(json.dumps({"seed": seed, **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
