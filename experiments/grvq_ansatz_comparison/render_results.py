"""Render completed GRVQ experiments as figures, HTML, and a runnable notebook.

Run from the repository root:
    python -m experiments.grvq_ansatz_comparison.render_results

The exact energies remain in results.json and complete traces. Decimal metrics
and logarithmic display floors below are presentation operations only. This
module never modifies a state, objective, parameter, or recorded result.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
import html
import io
import json
import os
from pathlib import Path
import statistics
import sys
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DISPLAY_FLOOR = 1e-15
LABELS = {
    "source_ry_cz": "Source Ry/CZ topology",
    "grvq_R2": "GRVQ R2 product",
    "grvq_R4": "GRVQ R4 product",
    "grvq_R2_cayley": "R2 + controlled Cayley",
    "grvq_R4_cayley": "R4 + controlled Cayley",
    "grvq_R4_reverse": "R4 + reversed order",
    "grvq_R4_local": "R4 + local rotations",
}
COLORS = ["#405277", "#3295a0", "#e38937", "#427bb8", "#b85653", "#8b6fac", "#639b62"]
FIGURES = [
    ("median_error", "Median objective error", "Each cell uses the median across the declared seeds. The color is log10(max(abs(E−E0),10⁻¹⁵)); the floor affects only this display."),
    ("convergence", "Convergence under the same objective budget", "Lines are median incumbent energy gaps. Shading is the minimum-to-maximum seed range, not a confidence interval. Numerical absolute-gap display floor: 10⁻¹⁵."),
    ("r2_r4_paired", "Paired R4−R2 energy differences", "Negative values favor R4; positive values favor R2. Each model and seed is paired at the same budget. Points are medians; bars show the seed range. The axis is symmetric logarithmic."),
    ("runtime_error", "Runtime and final objective error", "Every point is one complete optimization run. Runtime is measured wall time for the exact optimizer. This is a small-system implementation comparison, not a quantum speedup measurement."),
    ("method_calls", "All 29 methods in the coefficient compiler", "Actual function-entry counts in the source compiler that generates the simulation coefficients. Call counts establish execution, not accuracy or a performance advantage."),
]
COMPILER_FIGURE = (
    "compiler_cost", "Measured cost of conventional exact replacements",
    "Points show median coefficient-generation milliseconds and bars span the observed minimum and maximum across randomized repetition blocks. The vertical line is the original source median. Process startup, source loading and instrumentation are excluded. These measurements describe this compiler workload and machine.",
)


def figure_catalog(output):
    return FIGURES + ([COMPILER_FIGURE] if (output / "figures/compiler_cost.png").exists() else [])


def _median_fraction(values):
    values = sorted(values)
    n = len(values)
    if not n:
        raise ValueError("Cannot take a median without observations")
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


def _number(value, digits=6):
    return f"{float(value):.{digits}g}"


def summarize(report):
    """Aggregate observed runs; rank within models using exact energies."""
    groups = defaultdict(list)
    for run in report["runs"]:
        groups[(run["model"], run["family"])].append(run)
    families = report["configuration"]["families"]
    models = report["models"]
    expected = len(families) * len(models) * len(report["configuration"]["seeds"])
    if len(report["runs"]) != expected:
        raise ValueError(f"Expected {expected} completed model/family/seed runs, found {len(report['runs'])}")
    if len({(r["model"], r["family"], r["seed"]) for r in report["runs"]}) != expected:
        raise ValueError("Duplicate model/family/seed records")
    summaries = []
    winner_counts = Counter()
    exact_medians = {}
    for model in models:
        rows = []
        for family in families:
            runs = groups[(model["name"], family)]
            exact = _median_fraction([Fraction(r["energy_exact"]) for r in runs])
            exact_medians[(model["name"], family)] = exact
            rows.append({
                "model": model["name"], "family": family, "seeds": len(runs),
                "median_energy": float(exact),
                "median_ground_energy_error": statistics.median(r["ground_energy_error"] for r in runs),
                "median_fidelity": statistics.median(r["ground_subspace_fidelity"] for r in runs),
                "median_seconds": statistics.median(r["seconds"] for r in runs),
                "parameter_count": runs[0]["parameter_count"],
            })
        best = min(exact_medians[(model["name"], family)] for family in families)
        winners = [family for family in families if exact_medians[(model["name"], family)] == best]
        winner_counts.update(winners)
        for row in rows:
            row["best_median_in_model"] = row["family"] in winners
            summaries.append(row)
    paired = []
    run_index = {(r["model"], r["family"], r["seed"]): r for r in report["runs"]}
    for label, r2, r4 in (("Product only", "grvq_R2", "grvq_R4"),
                          ("Controlled Cayley", "grvq_R2_cayley", "grvq_R4_cayley")):
        if r2 not in families or r4 not in families:
            continue
        observations = []
        for model in models:
            for seed in report["configuration"]["seeds"]:
                difference = (Fraction(run_index[(model["name"], r4, seed)]["energy_exact"])
                              - Fraction(run_index[(model["name"], r2, seed)]["energy_exact"]))
                observations.append({"model": model["name"], "seed": seed,
                                     "energy_difference": float(difference),
                                     "exact_sign": (difference > 0) - (difference < 0)})
        paired.append({"comparison": label, "R2_family": r2, "R4_family": r4,
                       "R4_lower": sum(x["exact_sign"] < 0 for x in observations),
                       "R2_lower": sum(x["exact_sign"] > 0 for x in observations),
                       "exact_ties": sum(x["exact_sign"] == 0 for x in observations),
                       "observations": observations})
    return {"schema": "grvq-comparison-analysis/v1", "groups": summaries,
            "model_median_wins": dict(winner_counts), "paired_R2_R4": paired,
            "ranking": "Exact median Rayleigh energies within each declared model; ties retained",
            "display_floor": DISPLAY_FLOOR}


def _save_figure(fig, directory, name):
    fig.savefig(directory / (name + ".png"), dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / (name + ".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(report, analysis, directory):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#25324a",
                         "text.color": "#25324a", "figure.dpi": 120})
    families = report["configuration"]["families"]
    models = [m["name"] for m in report["models"]]
    color = {family: COLORS[i % len(COLORS)] for i, family in enumerate(families)}
    lookup = {(row["model"], row["family"]): row for row in analysis["groups"]}
    gaps = np.array([[lookup[(model, family)]["median_ground_energy_error"] for family in families] for model in models])
    logs = np.log10(np.maximum(np.abs(gaps), DISPLAY_FLOOR))
    fig, ax = plt.subplots(figsize=(12.6, max(4.5, .43 * len(models) + 2)))
    im = ax.imshow(logs, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(families)), [LABELS.get(f, f).replace(" + ", "\n+ ") for f in families], rotation=25, ha="right")
    ax.set_yticks(range(len(models)), models)
    midpoint = (logs.min() + logs.max()) / 2
    for i in range(len(models)):
        for j in range(len(families)):
            ax.text(j, i, f"{gaps[i,j]:.2e}", ha="center", va="center", fontsize=8,
                    color="white" if logs[i, j] < midpoint else "#15202b")
    ax.set_title("Median final energy gap to the numerical ground-state reference", pad=16)
    fig.colorbar(im, ax=ax, fraction=.025, pad=.025, label="log₁₀ |E − E₀| (display floor 10⁻¹⁵)")
    fig.tight_layout()
    _save_figure(fig, directory, "median_error")

    preferred = ["Q4_harmonic", "TFIM4_h1", "XXZ4_delta1", "vedic_spectral_2d"]
    representatives = [name for name in preferred if name in models]
    representatives += [name for name in models if name not in representatives][:4 - len(representatives)]
    ncols = 2 if len(representatives) > 1 else 1
    nrows = (len(representatives) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.3*nrows), squeeze=False)
    refs = {m["name"]: m["reference_ground_energy"] for m in report["models"]}
    budget = report["configuration"]["budget"]
    grid = np.arange(1, budget+1)
    for ax, model in zip(axes.flat, representatives):
        for family in families:
            series = []
            for run in report["runs"]:
                if run["model"] != model or run["family"] != family:
                    continue
                steps = np.array([x["evaluation"] for x in run["trace"]])
                values = np.array([x["energy"] - refs[model] for x in run["trace"]])
                indices = np.searchsorted(steps, grid, side="right") - 1
                if (indices < 0).any():
                    raise ValueError("A convergence trace lacks its first evaluation")
                series.append(values[indices])
            data = np.maximum(np.abs(np.array(series)), DISPLAY_FLOOR)
            ax.step(grid, np.median(data, axis=0), where="post", color=color[family],
                    label=LABELS.get(family, family), linewidth=1.6)
            if len(series) > 1:
                ax.fill_between(grid, data.min(axis=0), data.max(axis=0), step="post",
                                color=color[family], alpha=.10)
        ax.set_yscale("log")
        ax.set_title(model)
        ax.set_xlabel("Objective evaluations")
        ax.set_ylabel("Absolute numerical energy gap")
        ax.grid(alpha=.18, which="both")
    for ax in list(axes.flat)[len(representatives):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .11 if nrows == 1 else .075, 1, 1))
    _save_figure(fig, directory, "convergence")

    fig, ax = plt.subplots(figsize=(12, max(4.8, .43*len(models)+1.7)))
    if analysis["paired_R2_R4"]:
        for index, pairing in enumerate(analysis["paired_R2_R4"]):
            for i, model in enumerate(models):
                values = [x["energy_difference"] for x in pairing["observations"] if x["model"] == model]
                med = statistics.median(values)
                offset = (index - (len(analysis["paired_R2_R4"])-1)/2)*.26
                ax.errorbar(med, i+offset, xerr=[[med-min(values)], [max(values)-med]],
                            fmt="o", color=["#3295a0", "#b85653"][index], capsize=3,
                            label=pairing["comparison"] if i == 0 else None, markersize=5)
        ax.axvline(0, color="#536178", linewidth=1)
        ax.set_xscale("symlog", linthresh=1e-5)
        ax.set_yticks(range(len(models)), models)
        ax.invert_yaxis()
        ax.set_xlabel("E(R4) − E(R2), median and seed range; negative favors R4")
        ax.set_title("R4 compared with R2 under paired initialization and budget")
        ax.legend(frameon=False)
        ax.grid(axis="x", alpha=.2, which="both")
    else:
        ax.text(.5, .5, "This experiment contains no paired R2/R4 families", ha="center", transform=ax.transAxes)
        ax.set_axis_off()
    fig.tight_layout()
    _save_figure(fig, directory, "r2_r4_paired")

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    for family in families:
        subset = [run for run in report["runs"] if run["family"] == family]
        times = [r["seconds"] for r in subset]
        errors = [max(abs(r["ground_energy_error"]), DISPLAY_FLOOR) for r in subset]
        ax.scatter(times, errors, s=[18+5*r["nqubits"] for r in subset], alpha=.65,
                   color=color[family], label=LABELS.get(family, family), edgecolors="white", linewidth=.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Exact optimization wall time per run (seconds)")
    ax.set_ylabel("Absolute numerical ground-energy gap (display floor 10⁻¹⁵)")
    ax.set_title("Measured cost and achieved objective accuracy")
    ax.grid(alpha=.18, which="both")
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    _save_figure(fig, directory, "runtime_error")

    methods = report["polynomial_method_trace"]
    fig, ax = plt.subplots(figsize=(12, 11))
    counts = [m["calls"] for m in methods]
    ax.barh(range(len(methods)), counts, color=["#427bb8" if m["id"] <= 16 else "#e38937" for m in methods])
    ax.set_yticks(range(len(methods)), [f"S{m['id']:02d}  {m['name']}" for m in methods], fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    for i, count in enumerate(counts):
        ax.text(count*1.07, i, str(count), va="center", fontsize=8)
    ax.set_xlim(.8, max(counts)*1.8)
    ax.set_xlabel("Observed function entries during coefficient generation (log scale)")
    ax.set_title("All 16 main methods and 13 submethods execute in the compiler")
    ax.grid(axis="x", alpha=.18)
    fig.tight_layout()
    _save_figure(fig, directory, "method_calls")

    controls_path = directory.parent / "conventional_comparison/summary.json"
    if controls_path.exists():
        controls = json.loads(controls_path.read_text())
        variants = controls.get("variants", [])
        if variants and all("timing_min_ms" in item for item in variants):
            names = {item["id"]: item["name"] for item in methods}
            fig, ax = plt.subplots(figsize=(12, 11.5))
            labels = []
            for i, item in enumerate(variants):
                ids = item["classical_ids"]
                labels.append("All 29 conventional replacements" if len(ids) > 1
                              else f"S{ids[0]:02d}  {names[ids[0]]}")
                median = item["generation_ms"]
                ax.errorbar(median, i,
                            xerr=[[median-item["timing_min_ms"]], [item["timing_max_ms"]-median]],
                            fmt="o", capsize=3, markersize=5,
                            color="#b85653" if len(ids) > 1 else "#427bb8")
            ax.axvline(controls["baseline_generation_ms"], color="#25324a", linestyle="--",
                       label="Unchanged source median")
            ax.set_yticks(range(len(labels)), labels, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel(f"Compiler generation time (milliseconds); {controls.get('repeats', '?')} randomized repetitions")
            ax.set_title("Cost of exact coefficient generation with conventional replacements")
            ax.grid(axis="x", alpha=.18)
            ax.legend(frameon=False, loc="lower right")
            fig.tight_layout()
            _save_figure(fig, directory, "compiler_cost")


def _image_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _table(headers, rows, *, table_id="", sortable=False):
    attrs = f' id="{html.escape(table_id)}"' if table_id else ""
    attrs += ' class="sortable"' if sortable else ""
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in rows)
    return f"<div class='table-scroll'><table{attrs}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def make_html(report, analysis, output):
    esc = html.escape
    groups = analysis["groups"]
    winners = sorted(analysis["model_median_wins"].items(), key=lambda x: (-x[1], x[0]))
    leader = ", ".join(f"{LABELS.get(family,family)} ({count} model wins)" for family, count in winners[:3])
    config = report["configuration"]
    panels = []
    for name, title, caption in figure_catalog(output):
        panels.append(f"<section><h2>{esc(title)}</h2><p>{esc(caption)}</p>"
                      f"<img class='plot' alt='{esc(title)}' src='{_image_uri(output/'figures'/(name+'.png'))}'>"
                      f"<p class='downloads'><a href='figures/{name}.png'>PNG</a> · <a href='figures/{name}.svg'>SVG</a></p></section>")
    paired_rows = [(esc(p["comparison"]), str(p["R4_lower"]), str(p["R2_lower"]), str(p["exact_ties"]), str(len(p["observations"])))
                   for p in analysis["paired_R2_R4"]]
    model_rows = []
    for model in report["models"]:
        win = [g for g in groups if g["model"] == model["name"] and g["best_median_in_model"]]
        model_rows.append((esc(model["name"]), str(model["nqubits"]), str(model["dimension"]),
                           _number(model["reference_ground_energy"], 9), str(model["ground_subspace_dimension"]),
                           esc(", ".join(LABELS.get(g["family"], g["family"]) for g in win)),
                           _number(win[0]["median_ground_energy_error"]), _number(win[0]["median_fidelity"])))
    group_rows = [(esc(g["model"]), esc(LABELS.get(g["family"], g["family"])), str(g["parameter_count"]),
                   _number(g["median_energy"], 9), _number(g["median_ground_energy_error"]),
                   _number(g["median_fidelity"]), _number(g["median_seconds"]),
                   "Yes" if g["best_median_in_model"] else "") for g in groups]
    run_rows = []
    for r in report["runs"]:
        raw = f"<a href='{esc(r['state_file'])}'>state</a> · <a href='traces/{esc(r['complete_trace'])}'>trace</a>"
        run_rows.append((esc(r["model"]), esc(LABELS.get(r["family"], r["family"])), str(r["seed"]),
                         str(r["objective_evaluations"]), _number(r["energy"], 9), _number(r["ground_energy_error"]),
                         _number(r["ground_subspace_fidelity"]), _number(r["seconds"]), raw))
    config_rows = [(esc(str(k)), esc(", ".join(map(str, v)) if isinstance(v, list) else str(v)))
                   for k, v in config.items()]
    model_definitions = []
    for model in report["models"]:
        metadata = model["metadata"]
        definition = metadata.get("formula", metadata.get("construction", metadata.get("family", "See matrix metadata")))
        model_definitions.append((esc(model["name"]), esc(str(definition)),
                                  f"<a href='{esc(model['name'])}_matrices.json'>exact A and M</a>"))
    classical = ""
    control_path = output / "conventional_comparison" / "summary.json"
    if control_path.exists():
        data = json.loads(control_path.read_text())
        compared = data.get("variants", [])
        equal = sum(bool(x.get("exact_output_equality")) for x in compared)
        repeats = data.get("repeats", 1)
        names = {item["id"]: item["name"] for item in report["polynomial_method_trace"]}
        control_rows = []
        for item in compared:
            ids = item["classical_ids"]
            label = "All 29" if len(ids) > 1 else f"S{ids[0]:02d} {names[ids[0]]}"
            span = (_number(item["timing_min_ms"], 5) + "–" + _number(item["timing_max_ms"], 5)
                    if "timing_min_ms" in item else "One observation")
            ratio = item.get("relative_median_to_baseline", item["generation_ms"] / data["baseline_generation_ms"])
            control_rows.append((esc(label), str(bool(item["exact_output_equality"])),
                                 _number(item["generation_ms"], 5), span, _number(ratio, 5)))
        classical = (f"<section><h2>Conventional exact arithmetic controls</h2><p>{equal} of {len(compared)} observed compiler variants produced exactly identical mathematical outputs. "
                     "The variants replace each consumed polynomial extension individually and all 29 together with conventional exact algorithms. "
                     "Equivalent coefficient arithmetic therefore produces the same model; it does not create a new ground energy. "
                     f"The reported equality holds across {repeats} repetitions per variant.</p>"
                     f"<p>The unchanged source median is {_number(data['baseline_generation_ms'], 5)} ms. A ratio below one means the conventional replacement compiled this workload faster. "
                     "The ratio is the conventional median divided by the unchanged-source median; all raw timings and randomized block orders are retained.</p>"
                     + _table(["Conventional replacements", "All outputs exact-equal", "Median ms", "Observed range ms", "Ratio to source"], control_rows, sortable=True)
                     + "<p class='note'>" + esc(data.get("timing_note", "Single-run timings do not establish a general performance advantage.")) + "</p>"
                     +
                     "<p><a href='conventional_comparison/summary.json'>Complete control results and call counts</a></p></section>")
    qbraid = ""
    qbraid_path = output / "qbraid_results.json"
    if qbraid_path.exists():
        qdata = json.loads(qbraid_path.read_text())
        scalar = {k: v for k, v in qdata.items() if isinstance(v, (str, int, float, bool, type(None)))} if isinstance(qdata, dict) else {}
        circuit_rows = []
        for item in qdata.get("circuits", []):
            validation = item.get("validation", {})
            stats = validation.get("stats", {})
            circuit_rows.append((esc(item["model"]), esc(LABELS.get(item["family"], item["family"])),
                                 esc(item["basis"]), esc(str(validation.get("valid", "Unrecorded"))),
                                 esc(str(stats.get("depth", ""))), esc(str(stats.get("gateCount", ""))),
                                 _number(item["max_absolute_probability_error"]),
                                 _number(item["l1_probability_error"])))
        energy_rows = [(esc(item["model"]), esc(LABELS.get(item["family"], item["family"])),
                        _number(item["measured_basis_energy"], 12),
                        _number(item["local_exact_energy_projected"], 12),
                        _number(item["absolute_energy_error"]))
                       for item in qdata.get("energy_comparisons", [])]
        qbraid = ("<section><h2>Recorded qBraid results</h2><p>An observed qBraid result record accompanies this report. "
                  "Read its status and circuit-specific comparisons before interpreting agreement.</p>"
                  + _table(["Recorded field", "Value"], [(esc(k), esc(str(v))) for k, v in scalar.items()])
                  + ("<h3>Independent circuit probabilities</h3>"
                     + _table(["Model", "Family", "Basis", "Circuit valid", "Depth", "Gates", "Maximum |Δp|", "Sum |Δp|"], circuit_rows, sortable=True)
                     if circuit_rows else "")
                  + ("<h3>Energy reconstructed from measurement bases</h3><p>The values below use the complete required Pauli groups from ideal simulator distributions. They do not include finite-shot sampling or hardware noise.</p>"
                     + _table(["Model", "Family", "Basis-derived E", "Exact E, displayed numerically", "Absolute difference"], energy_rows, sortable=True)
                     if energy_rows else "")
                  + "<p><a href='qbraid_results.json'>Complete returned result and comparison data</a></p></section>")
    exports = []
    for group in report.get("exports", []):
        for file in group["files"]:
            exports.append((esc(group["model"]), esc(LABELS.get(group["family"], group["family"])), esc(file["basis"]),
                            f"<a href='{esc(file['qasm'])}'>OpenQASM</a>", f"<a href='{esc(file['expected'])}'>exact probabilities</a>"))
    source_commit = report["source_commit"]
    source_base = "https://github.com/12cymatics/quanqonscious/blob/" + quote(source_commit, safe="") + "/"
    template = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GRVQ ansatz comparison — observed results</title><style>
:root{--ink:#25324a;--muted:#59667b;--line:#dbe2eb;--paper:#f3f6fa;--accent:#286da1}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1240px;margin:auto;padding:38px 24px 70px}header{padding:25px 0 30px}h1{font-size:clamp(30px,5vw,52px);letter-spacing:-1.5px;line-height:1.1;margin:12px 0 18px}h2{font-size:26px;line-height:1.25;margin:0 0 14px}h3{margin-bottom:7px}.eyebrow{text-transform:uppercase;letter-spacing:2px;font-size:12px;font-weight:750;color:var(--accent)}.lede{font-size:19px;max-width:950px}.badge{display:inline-block;padding:5px 11px;background:#e3edf6;border-radius:7px;font-size:13px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-top:25px}.card{background:white;border:1px solid var(--line);padding:18px;border-radius:11px}.card strong{font-size:29px;display:block}.card span{color:var(--muted);font-size:13px}section{margin:24px 0;background:white;border:1px solid var(--line);border-radius:13px;padding:28px}p{margin:10px 0 17px}.plot{display:block;width:100%;height:auto}.downloads{font-size:13px}.table-scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:11px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:#f0f4f8;font-size:12px;white-space:nowrap}.sortable th{cursor:pointer}.sortable th:hover{background:#dfebf5}tbody tr:hover{background:#f5f8fc}a{color:#246eaa;text-underline-offset:3px}code,pre{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:13px}pre{padding:16px;background:#f3f6fa;overflow:auto;border-radius:8px}li{margin:8px 0}input{font:inherit;padding:10px 13px;border:1px solid #b9c9d9;border-radius:7px;width:min(100%,420px);margin:0 0 18px}details{margin-top:20px}summary{cursor:pointer;font-weight:650}.note{color:var(--muted);font-size:14px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:24px}@media(max-width:760px){main{padding:20px 12px}.cards{grid-template-columns:1fr 1fr}section{padding:18px}.grid2{grid-template-columns:1fr}h2{font-size:23px}}@media print{body{background:white}section{break-inside:avoid;border:0;padding:10px 0}.downloads,input{display:none}main{max-width:none;padding:0}}
</style></head><body><main><header><div class="eyebrow">DanCrab · source-derived experiment</div><h1>GRVQ ansatz comparison</h1>
<div class="badge">Exact optimizer · Numerical reference diagnostics</div><p class="lede">@@LEAD@@</p><p class="note">Observed run created @@DATE@@. Source snapshot <a href="@@COMMIT_URL@@">@@COMMIT@@</a>.</p>
<div class="cards"><div class="card"><strong>@@MODELS@@</strong><span>declared objective models</span></div><div class="card"><strong>@@RUNS@@</strong><span>complete optimization runs</span></div><div class="card"><strong>@@EVALUATIONS@@</strong><span>exact objective evaluations</span></div><div class="card"><strong>29 / 29</strong><span>methods used in coefficient generation</span></div></div></header>
<section><h2>What the comparison establishes</h2><p>@@LEADER@@</p><p>Ranking uses the exact median Rayleigh energy within each model. Models have different scales and interpretations, so raw energies are not pooled across models. Every run uses @@BUDGET@@ objective evaluations and the declared seeds @@SEEDS@@. Parameter counts differ between families and are reported below.</p><p>Optimization ran locally using exact classical state simulation. The recorded qBraid work, when present, independently checks selected optimized exported circuits and measurement energies; it does not execute the full training loop.</p>
@@PAIR_TABLE@@<p class="note">Paired signs above are computed from exact rational energies. Error and fidelity columns use a numerical eigensystem reference.</p></section>
@@PANELS@@
<section><h2>Per-model result</h2><p>Lowest median energy among the tested families, with ties retained. E₀ is the numerical reference for the exact finite matrix.</p>@@MODEL_TABLE@@</section>
<section><h2>Every family, every model</h2><p>Filter the tables by model or family. Click a column heading to sort. Display decimals do not alter the retained exact values.</p><input id="filter" aria-label="Filter result tables" placeholder="Filter model or family…">@@GROUP_TABLE@@<details><summary>All @@RUNS@@ individual runs and raw evidence</summary>@@RUN_TABLE@@</details></section>
@@CLASSICAL@@
<section><h2>Source selection and the update</h2><p>The audit selects <a href="@@SOURCE_CANONICAL@@">grvq_canonical.py</a> and <a href="@@SOURCE_CHAIN@@">grvq_book_chain.py</a> as the strongest recent exact source foundation. The benchmark develops the separately authored product construction in <a href="@@SOURCE_NOTEBOOK@@">Untitled5.ipynb</a>: <code>Ψᵢ = fᵢ Rₚ(rᵢ) ∏ⱼ(1+αⱼSⱼ(xᵢ))</code>. These scalar families are distinct, and this experiment does not assert that the product equals the canonical transcendental expression.</p><p>The defined radial factor is <code>Rₚ(r) = ∏ⱼ λⱼᵖ / (rᵖ + λⱼᵖ)</code>, with p=2 or p=4. Both variants use the same declared shell widths. The four-shell R2 is a matched control constructed for this experiment; it extends the source notebook's single-shell R2. Exact rational controlled Cayley rotations are added and tested in forward, reversed, and local-only variants. The four shape polynomials are actual cardinal polynomials generated by the source coefficient compiler, and the carrier is the nonconstant interpolant of <code>1+x</code>.</p>
<ul><li><a href="@@SOURCE_POLY@@">dns/vedic-polynomial.js</a> and <a href="@@SOURCE_OPERATORS@@">dns/vedic-operators.js</a> supply the arithmetic and consumed polynomial extensions.</li><li><a href="@@SOURCE_ANSATZ@@">ansatz.py</a> supplies the H/Ry/CZ circuit lineage. The experiment baseline explicitly uses two layers and a ring, with exact rational Cayley parameters.</li><li>The update is a declared finite variational experiment. Its explicit shape functions, shell widths, potentials, and couplings are experiment inputs, not inferred universal constants.</li></ul><p><a href="../source_audit/ansatz_selection.md">Complete source selection and file history</a> · <a href="../source_audit/other_repositories.json">Other original repositories audited</a></p>
<details><summary>Complete experiment configuration</summary>@@CONFIG_TABLE@@</details><details><summary>Objective definitions and exact matrix files</summary>@@DEFINITIONS@@</details></section>
<section><h2>Metric definitions</h2><div class="grid2"><div><h3>Exact computation</h3><p>Energy is <code>(ψᵀAψ)/(ψᵀMψ)</code>. Ordinary models use <code>M=I</code>; the spectral problems retain their exact mass matrix. Candidate comparisons and accepted optimizer updates use rational arithmetic. Original state vectors, parameters, matrices, and every objective evaluation are retained.</p><p>For ordinary models, exact energy variance is <code>||Hψ−Eψ||²/||ψ||²</code>. This tests eigenstate quality independently of energy reduction.</p></div><div><h3>Numerical diagnostics</h3><p>The ground-energy reference is <code>scipy.linalg.eigh(A,M)</code> applied to a numerical projection of the exact matrices. Ground-subspace fidelity sums overlaps over the complete numerically identified degenerate ground space, using the mass metric. Each model records its reference residual and eigenspace tolerance.</p><p>Plots, displayed decimals, fidelity, generalized residual diagnostics, and exported QASM angles are numerical boundary outputs. Log plots use a stated display-only floor.</p></div></div></section>
<section><h2>Scope of the result</h2><ul><li>The measured conclusions apply to these finite matrices, declared initializations, ansatz families, and optimization budgets. They do not establish universal superiority of R4, all Vedic methods, or a quantum speed advantage.</li><li>Exact arithmetic proves the represented operations are not rounded; it does not by itself prove the continuum physics, an optimizer's global convergence, or a runtime-to-Lean refinement.</li><li>The polynomial identities and conventional controls compare arithmetic implementations of the same model. Equivalent exact identities should agree on coefficients and objective values.</li><li>Q4, toroidal, and finite-grid models are defined experiments. Spin models use explicit Pauli conventions. No molecular H₂ result is asserted.</li><li>All seeds and families are shown. Wall time is measured for this implementation and machine; sample ranges are descriptive rather than statistical confidence intervals.</li></ul></section>
<section><h2>Run and inspect the complete experiment</h2><p>From the extracted repository root:</p><pre>python -m experiments.grvq_ansatz_comparison.run_controls
python -m experiments.grvq_ansatz_comparison.run --budget 385 --seeds 7,19,43
python -m experiments.grvq_ansatz_comparison.render_results</pre><p><a href="comparison.ipynb">Notebook with computed analysis outputs</a> · <a href="results.json">Complete results</a> · <a href="analysis_summary.json">Analysis summary</a> · <a href="polynomial_data.json">Exact compiler outputs</a> · <a href="polynomial_verification.json">Independent symbolic comparison</a> · <a href="arithmetic.json">All arithmetic cases</a></p></section>
<section><h2>qBraid-ready circuits</h2><p>The exported circuits include exact expected probabilities in multiple measurement bases. Circuit availability is distinct from a completed simulator or hardware job. Use the qBraid free built-in simulator for supported circuits; a provider submission or paid compute instance requires a separate spending decision.</p>@@EXPORT_TABLE@@</section>
@@QBRAID@@
<footer class="note">All plots and this report are generated from the accompanying observed result files. No external JavaScript, font, image, or stylesheet is required.</footer></main>
<script>
for (const table of document.querySelectorAll('table.sortable')) {
  for (const [column, head] of [...table.tHead.rows[0].cells].entries()) {
    let ascending = true;
    head.addEventListener('click', () => {
      const body = table.tBodies[0];
      const rows = [...body.rows];
      rows.sort((a,b) => {
        const x=a.cells[column].innerText.trim(),y=b.cells[column].innerText.trim();
        const nx=Number(x),ny=Number(y);
        const result=x!==''&&y!==''&&Number.isFinite(nx)&&Number.isFinite(ny)?nx-ny:x.localeCompare(y);
        return ascending?result:-result;
      });
      rows.forEach(row=>body.appendChild(row));ascending=!ascending;
    });
  }
}
document.querySelector('#filter').addEventListener('input', event => {
  const query=event.target.value.toLowerCase();
  for (const table of document.querySelectorAll('#group-results,#run-results'))
    for (const row of table.tBodies[0].rows) row.hidden=!row.innerText.toLowerCase().includes(query);
});
</script></body></html>"""
    values = {
        "LEAD": esc(f"{len(report['runs'])} completed runs compare {len(config['families'])} ansatz families on {len(report['models'])} finite objective models. The report separates exact computation from numerical reference metrics and shows the observed R2/R4 tradeoff."),
        "DATE": esc(report["created_utc"]), "COMMIT": esc(source_commit),
        "COMMIT_URL": f"https://github.com/12cymatics/quanqonscious/tree/{quote(source_commit,safe='')}",
        "MODELS": str(len(report["models"])), "RUNS": str(len(report["runs"])),
        "EVALUATIONS": f"{report['total_objective_evaluations']:,}",
        "LEADER": esc("Most per-model median wins in this run: " + leader + ". This count describes only the selected benchmark set."),
        "BUDGET": str(config["budget"]), "SEEDS": esc(", ".join(map(str, config["seeds"]))),
        "PAIR_TABLE": _table(["Paired families", "R4 lower", "R2 lower", "Exact ties", "Pairs"], paired_rows),
        "PANELS": "".join(panels),
        "MODEL_TABLE": _table(["Model", "Qubits", "Dimension", "Reference E₀", "Ground dimension", "Best median family", "Median gap", "Median fidelity"], model_rows, sortable=True),
        "GROUP_TABLE": _table(["Model", "Family", "Parameters", "Median E", "Median gap", "Median fidelity", "Median seconds", "Model best"], group_rows, table_id="group-results", sortable=True),
        "RUN_TABLE": _table(["Model", "Family", "Seed", "Evaluations", "Final E", "Energy gap", "Fidelity", "Seconds", "Evidence"], run_rows, table_id="run-results", sortable=True),
        "CLASSICAL": classical, "QBRAID": qbraid,
        "SOURCE_POLY": source_base+"dns/vedic-polynomial.js", "SOURCE_OPERATORS": source_base+"dns/vedic-operators.js", "SOURCE_ANSATZ": source_base+"ansatz.py",
        "SOURCE_CANONICAL": source_base+"grvq_canonical.py", "SOURCE_CHAIN": source_base+"grvq_book_chain.py", "SOURCE_NOTEBOOK": source_base+"Untitled5.ipynb",
        "CONFIG_TABLE": _table(["Setting", "Declared value"], config_rows),
        "DEFINITIONS": _table(["Model", "Definition", "Evidence"], model_definitions),
        "EXPORT_TABLE": _table(["Model", "Selected family", "Basis", "Circuit", "Reference probabilities"], exports),
    }
    for key, value in values.items():
        template = template.replace("@@" + key + "@@", value)
    if "@@" in template:
        raise ValueError("An HTML template field was not filled")
    (output / "results_report.html").write_text(template)


def make_notebook(report, output):
    """Execute the shown analysis cells and retain their actual MIME outputs.

    No optimizer rerun is implied. The final full-recompute cell is deliberately
    left unexecuted and is clearly marked as a user-triggered full experiment.
    """
    cells = []
    def markdown(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)})
    def code(source, manual=False):
        cells.append({"cell_type": "code", "metadata": {"tags": ["manual-full-recompute"]} if manual else {},
                      "source": source.splitlines(True), "execution_count": None, "outputs": []})
    markdown("# GRVQ ansatz comparison\n\nThis notebook contains computed analysis outputs from the accompanying complete experiment. "
             "All exact energies, matrices, states, and full evaluation traces remain in the package. "
             "The optimization ran locally with exact classical state simulation. qBraid records, when present, independently check selected exported circuits and measurement energies. "
             "The final cell can rerun the complete experiment from the extracted repository root.\n\n"
             "The ansatz retains $\\psi_i=f_iR_p(r_i)\\prod_j[1+\\alpha_jS_j(x_i)]$. "
             "Here $R_p(r)=\\prod_j\\lambda_j^p/(r^p+\\lambda_j^p)$ with $p=2$ or $4$ and the same declared shell widths. "
             "Energy is the exact Rayleigh quotient $\\psi^TA\\psi/(\\psi^TM\\psi)$. "
             "Eigensystem comparisons, plotting and QASM angles are explicitly numerical diagnostics.")
    code("from pathlib import Path\nimport json, sys, statistics\nfrom fractions import Fraction\nfrom collections import defaultdict\nfrom IPython.display import Image, HTML\n"
         "sys.set_int_max_str_digits(0)\n"
         "candidates = [Path.cwd() / 'results.json', Path.cwd() / 'results/results.json', Path.cwd() / 'experiments/grvq_ansatz_comparison/results/results.json']\n"
         "RESULTS = next(path for path in candidates if path.is_file())\n"
         "report = json.loads(RESULTS.read_text())\n"
         "assert 'elapsed_seconds' in report and 'total_objective_evaluations' in report, 'Incomplete experiment checkpoint'\n"
         "print(f\"Models: {len(report['models'])}; families: {len(report['configuration']['families'])}; completed runs: {len(report['runs'])}\")\n"
         "print(f\"Exact objective evaluations: {report['total_objective_evaluations']:,}; budget per run: {report['configuration']['budget']}\")\n"
         "print('Seeds:', report['configuration']['seeds'])\nprint('Source commit:', report['source_commit'])\n"
         "print('Arithmetic:', report['configuration']['state_arithmetic'])\n")
    markdown("## Every model and family\n\nThe following table uses the observed median numerical ground-energy gap, fidelity and wall time. "
             "Exact energies are available per run in `results.json`; the HTML report ranks medians using rational energies.")
    code("import html\ngroups = defaultdict(list)\nfor run in report['runs']:\n    groups[(run['model'], run['family'])].append(run)\n"
         "rows = []\nfor (model, family), runs in groups.items():\n"
         "    gap = statistics.median(r['ground_energy_error'] for r in runs)\n"
         "    fidelity = statistics.median(r['ground_subspace_fidelity'] for r in runs)\n"
         "    seconds = statistics.median(r['seconds'] for r in runs)\n"
         "    rows.append(f'<tr><td>{html.escape(model)}</td><td>{html.escape(family)}</td><td>{gap:.8g}</td><td>{fidelity:.8g}</td><td>{seconds:.6g}</td></tr>')\n"
         "display(HTML('<table><thead><tr><th>Model</th><th>Family</th><th>Median gap</th><th>Median fidelity</th><th>Median seconds</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'))\n")
    markdown("## Scientific figures\n\nLogarithmic error charts use `max(abs(E−E0), 1e−15)` only for display. "
             "The raw errors and exact objective values are retained unchanged. Convergence shading is the observed seed range, not a confidence interval. "
             "For the paired comparison, negative R4−R2 favors R4.")
    code("figure_names = ['median_error', 'convergence', 'r2_r4_paired', 'runtime_error', 'method_calls']\n"
         "if (RESULTS.parent / 'figures/compiler_cost.png').is_file():\n    figure_names.append('compiler_cost')\n"
         "for name in figure_names:\n    print(name)\n    display(Image(filename=str(RESULTS.parent / 'figures' / (name + '.png'))))\n")
    markdown("## Exact source-method execution and exported quantum circuits\n\n"
             "The 29 method counts come from the coefficient compiler used by the experiment. Counts do not prove an advantage. "
             "Exported QASM circuits can be inspected or used with the free qBraid built-in simulator. A paid provider job is a separate action; this notebook does not submit one.")
    code("for method in report['polynomial_method_trace']:\n    print(f\"S{method['id']:02d} {method['name']}: {method['calls']} calls\")\n"
         "control_path = RESULTS.parent / 'conventional_comparison/summary.json'\n"
         "if control_path.exists():\n"
         "    controls = json.loads(control_path.read_text())\n"
         "    print('\\nConventional exact controls:', sum(bool(x['exact_output_equality']) for x in controls['variants']), 'of', len(controls['variants']), 'exactly equal')\n"
         "    print('Repetitions per variant:', controls.get('repeats', 1))\n"
         "    print('Source median generation milliseconds:', controls['baseline_generation_ms'])\n"
         "    print('Timing scope:', controls.get('timing_note', 'Single-run timings'))\n"
         "print('\\nExported measurement circuits:')\nfor group in report.get('exports', []):\n"
         "    for item in group['files']:\n        print(group['model'], item['basis'], item['qasm'], 'expected:', item['expected'])\n"
         "qbraid_path = RESULTS.parent / 'qbraid_results.json'\n"
         "if qbraid_path.exists():\n"
         "    qbraid = json.loads(qbraid_path.read_text())\n"
         "    print('\\nObserved qBraid provider:', qbraid.get('provider'))\n"
         "    print('Execution:', qbraid.get('execution'))\n"
         "    print('Maximum absolute probability difference:', qbraid.get('max_absolute_probability_error'))\n"
         "    print('Maximum absolute energy difference:', qbraid.get('max_absolute_energy_error'))\n"
         "    print('Credits spent:', qbraid.get('credits_spent'))\n"
         "    rows = []\n"
         "    for item in qbraid.get('energy_comparisons', []):\n"
         "        rows.append(f\"<tr><td>{html.escape(item['model'])}</td><td>{html.escape(item['family'])}</td><td>{item['measured_basis_energy']:.12g}</td><td>{item['local_exact_energy_projected']:.12g}</td><td>{item['absolute_energy_error']:.6g}</td></tr>\")\n"
         "    if rows:\n"
         "        display(HTML('<table><thead><tr><th>Model</th><th>Family</th><th>Basis-derived E</th><th>Exact E numerical display</th><th>Absolute difference</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'))\n"
         "else:\n    print('\\nNo qBraid result record accompanies this snapshot.')\n")
    markdown("## Interpretation\n\n"
             "This is a complete finite experiment over the declared models and budgets. It does not establish universal R4 superiority, "
             "a quantum speedup, a complete continuum field solution, or formal runtime refinement. "
             "Different families use different parameter counts; those counts are retained. The ground-state fidelity uses a numerical ground-subspace projector, "
             "including degeneracy and the mass metric where required.\n\n"
             "## Recompute the full experiment\n\nThe next cell is intentionally not executed in this saved notebook. "
             "Run it from a notebook inside the extracted repository to execute the full sweep with 385 evaluations per run and seeds 7, 19, 43, "
             "then rebuild all report artifacts. It uses local computation and makes no paid qBraid submission.")
    code("import subprocess\n"
         "root = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'experiments/grvq_ansatz_comparison/run.py').is_file())\n"
         "subprocess.run([sys.executable, '-m', 'experiments.grvq_ansatz_comparison.run', '--budget', '385', '--seeds', '7,19,43'], cwd=root, check=True)\n"
         "subprocess.run([sys.executable, '-m', 'experiments.grvq_ansatz_comparison.render_results'], cwd=root, check=True)\n", manual=True)
    execution = 0
    namespace = {"__name__": "__notebook_analysis__"}
    old_cwd = Path.cwd()
    try:
        os.chdir(output)
        for cell in cells:
            if cell["cell_type"] != "code" or "manual-full-recompute" in cell["metadata"].get("tags", []):
                continue
            execution += 1
            captured = []
            def display(value):
                from IPython.display import Image, HTML
                if isinstance(value, Image):
                    raw = Path(value.filename).read_bytes() if value.filename else value.data
                    data = {"image/png": base64.b64encode(raw).decode("ascii"), "text/plain": "<computed scientific figure>"}
                elif isinstance(value, HTML):
                    data = {"text/html": value.data, "text/plain": "<computed results table>"}
                else:
                    data = {"text/plain": repr(value)}
                captured.append({"output_type": "display_data", "data": data, "metadata": {}})
            namespace["display"] = display
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(compile("".join(cell["source"]), "comparison.ipynb", "exec"), namespace)
            outputs = []
            if stdout.getvalue():
                outputs.append({"output_type": "stream", "name": "stdout", "text": stdout.getvalue().splitlines(True)})
            if stderr.getvalue():
                outputs.append({"output_type": "stream", "name": "stderr", "text": stderr.getvalue().splitlines(True)})
            outputs.extend(captured)
            cell["outputs"], cell["execution_count"] = outputs, execution
    finally:
        os.chdir(old_cwd)
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": sys.version.split()[0]},
                             "grvq_analysis": {"execution": "Shown analysis cells executed by Python with captured stdout and MIME outputs",
                                               "optimizer_rerun_during_render": False,
                                               "source_created_utc": report["created_utc"]}}}
    (output / "comparison.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "experiments/grvq_ansatz_comparison/results/results.json")
    args = parser.parse_args(argv)
    sys.set_int_max_str_digits(0)
    source = args.results.resolve()
    report = json.loads(source.read_text())
    if "elapsed_seconds" not in report or "total_objective_evaluations" not in report:
        raise ValueError("The result file is an incomplete checkpoint; wait for the experiment to finish")
    output = source.parent
    analysis = summarize(report)
    (output / "analysis_summary.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    make_figures(report, analysis, output / "figures")
    make_html(report, analysis, output)
    make_notebook(report, output)
    print(json.dumps({"report": str(output / "results_report.html"),
                      "notebook": str(output / "comparison.ipynb"),
                      "figures": len(figure_catalog(output)), "runs": len(report["runs"]),
                      "model_median_wins": analysis["model_median_wins"]}))


if __name__ == "__main__":
    main()
