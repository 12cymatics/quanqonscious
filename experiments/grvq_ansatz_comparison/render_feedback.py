"""Render completed quantum–GRVQ trajectories without changing engine values."""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from fractions import Fraction
import hashlib
import html
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
MODES = ("SERIES", "PARALLEL", "SYMMETRIC_CONCURRENT")
COLORS = {"SERIES": "#168b82", "PARALLEL": "#d56b31", "SYMMETRIC_CONCURRENT": "#7656b4"}
LABELS = {"SERIES": "Series", "PARALLEL": "Parallel", "SYMMETRIC_CONCURRENT": "Symmetric concurrent"}


def value(item):
    if item["purpose"] != "display only; never an arithmetic input":
        raise ValueError("Unexpected numerical projection provenance")
    return item["value"]


def interval(item):
    cert = item["certificate"]
    denominator = 1 << cert["denominator_power"]
    return Fraction(int(cert["lower_integer"]), denominator), Fraction(int(cert["upper_integer"]), denominator)


def energy_change_status(step):
    if step["energy_initial"] == step["energy_final"]:
        return "exactly_equal_expression"
    before, after = interval(step["display"]["energy_initial"]), interval(step["display"]["energy_final"])
    if after[1] < before[0]:
        return "certified_lower"
    if after[0] > before[1]:
        return "certified_higher"
    return "not_decided_by_summary_intervals"


def table(headers, rows):
    return "<table><thead><tr>" + "".join("<th>"+html.escape(str(x))+"</th>" for x in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>"+"".join("<td>"+html.escape(str(x))+"</td>" for x in row)+"</tr>" for row in rows) + "</tbody></table>"


def figure_html(path, caption):
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def render(directory=BASE / "latest_hybrid_results"):
    directory = Path(directory).resolve()
    result_path = directory / "results.json"
    results = json.loads(result_path.read_text())
    if results["schema"] != "grvq-quantum-feedback-sweep/v2" or results["status"] != "complete":
        raise ValueError("The report requires the completed feedback sweep")
    cases = results["cases"]
    if len(cases) != results["expected_cases"] or results["total_frames"] != results["expected_frames"]:
        raise ValueError("Completed-case or frame counts do not match the requested sweep")
    frames = []
    for case in cases:
        if case["status"] != "complete":
            raise ValueError("A trajectory is incomplete")
        path = directory / case["result_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["result_sha256"]:
            raise ArithmeticError("The trajectory result hash has changed")
        cfg = case["configuration"]
        for step in case["steps"]:
            if not step["all29_consumed"]:
                raise ArithmeticError("The report cannot count an unconsumed method as executed")
            frames.append({"case": case, "configuration": cfg, "step": step})
    figdir = directory / "figures"
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    grouped = defaultdict(list)
    for entry in frames:
        cfg, step = entry["configuration"], entry["step"]
        grouped[(cfg["composition"], cfg["strength"], cfg["feedback"], step["step"])].append(step)
    strengths = results["configuration"]["strengths"]
    modes = results["configuration"]["modes"]
    feedback_states = sorted({c["configuration"]["feedback"] for c in cases})
    steps = results["configuration"]["steps"]
    fig, axes = plt.subplots(1, len(strengths), figsize=(10.8, 4.6), squeeze=False)
    for column, strength in enumerate(strengths):
        ax = axes[0, column]
        for mode in modes:
            for enabled in feedback_states:
                initial = statistics.median(max(abs(value(x)) for x in s["display"]["lambda_in"])
                                            for s in grouped[mode, strength, enabled, 0])
                vals = [initial] + [statistics.median(max(abs(value(x)) for x in s["display"]["lambda_after_feedback"])
                         for s in grouped[mode, strength, enabled, t]) for t in range(steps)]
                ax.plot(range(steps+1), vals, marker="o", linestyle="-" if enabled else "--",
                        color=COLORS[mode], label=LABELS[mode]+(" · feedback" if enabled else " · control"),
                        linewidth=2 if enabled else 1.3, alpha=1 if enabled else .65)
        ax.set_yscale("log"); ax.set_xticks(range(steps+1)); ax.set_xlabel("Completed outer steps")
        ax.set_title("Each method strength = "+strength); ax.grid(alpha=.18)
        ax.set_ylabel("Median max |λq| · display projection")
    axes[0, -1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("Evolving coefficients — source dynamics remain unclamped", fontsize=14)
    fig.tight_layout(); fig.savefig(figdir/"lambda_trajectories.png", dpi=170, bbox_inches="tight")
    fig.savefig(figdir/"lambda_trajectories.svg", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    for mode_index, mode in enumerate(modes):
        selected = [x for x in frames if x["configuration"]["composition"] == mode]
        for step_index in range(steps):
            current = [x["step"] for x in selected if x["step"]["step"] == step_index]
            xs = [mode_index + (step_index-(steps-1)/2)*.23 for _ in current]
            gaps = [value(s["display"]["energy_gap"]) for s in current]
            fidelities = [float(Fraction(s["ground_fidelity_rational"])) for s in current]
            axes[0].scatter(xs, gaps, s=32, color=COLORS[mode], marker="o" if step_index==0 else "s", alpha=.65)
            axes[1].scatter(xs, fidelities, s=32, color=COLORS[mode], marker="o" if step_index==0 else "s", alpha=.65)
    axes[0].set_yscale("symlog", linthresh=1e-6); axes[0].set_ylabel("E − current exact ground energy")
    axes[1].set_ylim(-.03, 1.03); axes[1].set_ylabel("Exact ground-space probability · decimal display")
    for ax in axes:
        ax.set_xticks(range(len(modes)), [LABELS[mode].replace(" ", "\n") for mode in modes]); ax.grid(axis="y", alpha=.18)
    axes[0].set_title("Gap against the Hamiltonian of that step")
    axes[1].set_title("Ground-space fidelity")
    fig.suptitle("Variational quality · circles: first step, squares: later steps", fontsize=14)
    fig.tight_layout(); fig.savefig(figdir/"variational_quality.png", dpi=170, bbox_inches="tight")
    fig.savefig(figdir/"variational_quality.svg", bbox_inches="tight"); plt.close(fig)

    pairs = defaultdict(dict)
    for case in cases:
        cfg = case["configuration"]
        pairs[cfg["composition"], cfg["strength"], cfg["seed"]][cfg["feedback"]] = case
    paired = []
    for (mode, strength, seed), pair in sorted(pairs.items()):
        if set(pair) != {False, True}:
            if results["configuration"]["feedback"] == "both":
                raise ValueError("Feedback-control pairing is incomplete")
            continue
        off, on = pair[False], pair[True]
        first_identical = (off["steps"][0]["after_formula_field_digests"] == on["steps"][0]["after_formula_field_digests"])
        consumed = all(on["steps"][i]["input_field_digests"] == on["steps"][i-1]["next_field_digests"] for i in range(1, steps))
        if not first_identical or not consumed:
            raise ArithmeticError("The paired trajectory does not share its start or consume its saved feedback")
        last_off, last_on = off["steps"][-1], on["steps"][-1]
        delta = [value(a)-value(b) for a,b in zip(last_on["display"]["lambda_after_feedback"],last_off["display"]["lambda_after_feedback"])]
        paired.append({"composition":mode,"strength":strength,"seed":seed,
                       "same_first_formula_field":first_identical,"feedback_checkpoint_consumed":consumed,
                       "final_lambda_difference_display":delta,"max_absolute_difference_display":max(map(abs,delta)),
                       "display_difference_note":"Difference of decimal display projections, used only for plotting and summary; the exact values retain independent bounds.",
                       "final_field_root_digests_differ":last_on["next_field_digests"]!=last_off["next_field_digests"]})
    fig, ax = plt.subplots(figsize=(10.8, 4.0))
    xs, labels = [], []
    positive_differences = []
    for i, mode in enumerate(modes):
        for j, strength in enumerate(strengths):
            selected = [x["max_absolute_difference_display"] for x in paired if x["composition"]==mode and x["strength"]==strength]
            if not selected:
                continue
            position = i*(len(strengths)+1)+j
            xs.append(position); labels.append(LABELS[mode].replace(" ","\n")+"\n"+strength)
            offsets = [(k-(len(selected)-1)/2)*.09 for k in range(len(selected))]
            ax.scatter([position+x for x in offsets],selected,color=COLORS[mode],s=40,zorder=3)
            median = statistics.median(selected)
            ax.plot([position-.2,position+.2],[median,median],color="black",linewidth=1.8,zorder=4)
            positive_differences.extend(selected)
    ax.set_xticks(xs,labels,fontsize=9)
    if positive_differences:
        ax.set_yscale("log" if all(v>0 for v in positive_differences) else "symlog")
    ax.set_ylabel("Max |λfeedback − λcontrol| · display")
    ax.set_title("Paired feedback influence after the complete trajectory")
    if not paired:
        ax.text(.5,.5,"This experiment did not request paired feedback controls",ha="center",va="center",transform=ax.transAxes)
    ax.grid(axis="y",alpha=.18);fig.tight_layout()
    fig.savefig(figdir/"feedback_effect.png",dpi=170,bbox_inches="tight")
    fig.savefig(figdir/"feedback_effect.svg",bbox_inches="tight");plt.close(fig)

    changes = defaultdict(int)
    for entry in frames: changes[energy_change_status(entry["step"])] += 1
    by_mode = []
    for mode in modes:
        selected = [e for e in frames if e["configuration"]["composition"] == mode]
        gaps = [value(e["step"]["display"]["energy_gap"]) for e in selected]
        fidelities = [Fraction(e["step"]["ground_fidelity_rational"]) for e in selected]
        by_mode.append({"composition":mode,"trajectories":sum(c["configuration"]["composition"]==mode for c in cases),
                        "frames":len(selected),"core_method_calls":sum(e["step"]["core_method_calls"] for e in selected),
                        "distributed_grvq_updates":sum(e["step"]["distributed_grvq_updates"] for e in selected),
                        "median_energy_gap_display":statistics.median(gaps),"minimum_energy_gap_display":min(gaps),"maximum_energy_gap_display":max(gaps),
                        "median_ground_fidelity_exact":str(statistics.median(fidelities)),"median_ground_fidelity_display":float(statistics.median(fidelities)),
                        "minimum_ground_fidelity_exact":str(min(fidelities)),"maximum_ground_fidelity_exact":str(max(fidelities)),
                        "minimum_ground_fidelity_display":float(min(fidelities)),"maximum_ground_fidelity_display":float(max(fidelities)),
                        "exact_zero_gap_frames":sum(e["step"]["energy_gap_is_exactly_zero"] for e in selected)})
    summary = {"schema":"grvq-feedback-analysis/v1","status":"complete","sweep_results_sha256":hashlib.sha256(result_path.read_bytes()).hexdigest(),
               "trajectories":len(cases),"frames":len(frames),"all_frames_consumed_all29":all(e["step"]["all29_consumed"] for e in frames),
               "energy_change_status":dict(changes),"totals":results["totals"],"elapsed_seconds":results["elapsed_seconds"],
               "by_mode":by_mode,"paired_feedback":paired,
               "interpretation":"The fixed-current-Hamiltonian quantum solve is assessed separately at each step. Changing raw energies across steps do not establish fixed-objective convergence. Figures use display projections only."}
    (directory/"analysis_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    status_rows = [[LABELS[r["composition"]],r["trajectories"],r["frames"],r["core_method_calls"],r["distributed_grvq_updates"],
                    f'{r["median_energy_gap_display"]:.7g}',f'{r["median_ground_fidelity_display"]:.7g}',
                    f'{r["minimum_ground_fidelity_display"]:.4g}–{r["maximum_ground_fidelity_display"]:.4g}'] for r in by_mode]
    case_rows = []
    for case in sorted(cases,key=lambda c:c["case_id"]):
        cfg=case["configuration"];last=case["steps"][-1]
        case_rows.append([LABELS[cfg["composition"]],cfg["strength"],cfg["seed"],"on" if cfg["feedback"] else "control",
                          f'{value(last["display"]["energy_final"]):.9g}',f'{value(last["display"]["ground_energy"]):.9g}',
                          f'{value(last["display"]["energy_gap"]):.7g}',f'{float(Fraction(last["ground_fidelity_rational"])):.7g}'])
    composition_count = len(modes)
    strength_count = len(strengths)
    seed_count = len(results["configuration"]["seeds"])
    quantum_sweeps = results["configuration"]["sweeps"]
    paired_statement = (f"All {len(paired)} enabled/control pairs share their first formula field. "
                        + ("The enabled trajectory’s retained next field is consumed by its following step. " if steps > 1 else
                           "The final feedback field is retained as a checkpoint; this one-step selection has no following calculation. ")
                        + "The paired differences below show the resulting influence on the coefficients.") if paired else (
                        "This selected experiment contains one feedback setting and therefore has no paired enabled/control comparison.")
    qfile=directory/"qbraid_verification.json"
    if qfile.exists():
        verification = json.loads(qfile.read_text())
        if verification["status"] != "complete":
            raise ValueError("The provider verification record is incomplete")
        summary["qbraid_verification"] = {k:v for k,v in verification.items() if k!="comparisons"}
        summary["qbraid_verification_sha256"] = hashlib.sha256(qfile.read_bytes()).hexdigest()
        (directory/"analysis_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        provider = (f'<p><strong>{verification["circuits"]} actual circuit simulations</strong> cover the Z basis for all '
                    f'{verification["frames_checked_in_z"]} frames, plus selected X, Y and mixed-basis checks.</p>'
                    +table(["Comparison","Maximum absolute discrepancy"],[
                        ["Bitstring probabilities",f'{verification["maximum_probability_error"]:.4g}'],
                        ["Current Hamiltonian energy",f'{verification["maximum_energy_error"]:.4g}'],
                        ["Feedback increment",f'{verification["maximum_feedback_increment_error"]:.4g}']])
                    +f'<p>Recorded provider: {html.escape(verification["provider"])}. Credits spent: {verification["credits_spent"]}. '
                    '<a href="qbraid_verification.json">Complete verification record</a> · <a href="qbraid_results.json">Provider responses</a>.</p>')
    else:
        provider = "<p>The exact trajectory sweep is complete. A separate provider-verification summary has not been added to this report build.</p>"
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Quantum ansatz–GRVQ feedback sweep</title>
<style>body{{max-width:1160px;margin:auto;padding:28px 24px;color:#172a35;background:#f7fafb;font:16px/1.6 system-ui,sans-serif}}h1{{font-size:36px;line-height:1.2}}h2{{margin-top:2.3em}}.lead{{font-size:20px}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{background:white;border-top:4px solid #168b82;padding:18px;flex:1;min-width:160px;box-shadow:0 2px 14px #17324310}}.big{{font-size:32px;font-weight:700}}table{{border-collapse:collapse;width:100%;background:white;font-size:13px}}td,th{{border-bottom:1px solid #dbe4e8;text-align:left;padding:9px}}th{{background:#e7eff2}}figure{{background:white;margin:24px 0;padding:15px}}img{{width:100%;height:auto}}figcaption{{font-size:13px;color:#455b67}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:18px;font-size:12px}}.scroll{{overflow-x:auto}}a{{color:#08796e}}.note{{padding:15px;border-left:4px solid #d56b31;background:#fff4e9}}</style>
<h1>Quantum ansatz–GRVQ feedback trajectories</h1><p class="lead">The completed sweep executes all 29 field methods, solves the source quantum circuit for the changing Hamiltonian, and retains the exact quantum feedback for every subsequent requested field step.</p>
<div class="cards"><div class="card"><div class="big">{len(cases)}</div>completed trajectories</div><div class="card"><div class="big">{len(frames)}</div>complete outer steps</div><div class="card"><div class="big">{results['totals']['core_method_calls']:,}</div>core method applications</div><div class="card"><div class="big">{results['elapsed_seconds']:.1f} s</div>coordinator elapsed time</div></div>
<h2>What was run</h2><p>{composition_count} mathematical modes × {strength_count} declared method strengths × {seed_count} deterministic circuit starts × {len(feedback_states)} declared feedback settings. Every trajectory completes {steps} outer steps. Each quantum solve evaluates all 12 coordinates over {quantum_sweeps} sweeps on the declared rational grid. Full source formulas, exact expression graphs, quantum states, search records and checkpoints are retained.</p>
<p>The field follows the November 17 BOOK/GRVQ/MSTVQ formula. The quantum state follows the recovered three-layer, four-qubit Ry/CNOT chain. Its Hamiltonian is ΣλqZq + ½ΣZqZq+1. Feedback is exactly −⟨Zq⟩/1000 and the next step consumes the injected field.</p>
<h2>Completed calculations by mode</h2><div class="scroll">{table(['Mode','Trajectories','Frames','Core applications','Distributed GRVQ updates','Median current-H gap','Median ground fidelity','Fidelity range'],status_rows)}</div>
<p>All {len(frames)} frames consumed all 29 methods. The optimizer reduced the current Hamiltonian energy in {changes['certified_lower']} frames, as certified by disjoint exact bounds. Equal expression results: {changes['exactly_equal_expression']}. Summary comparisons unresolved by their retained bounds: {changes['not_decided_by_summary_intervals']}.</p>
<h2>Field and quantum trajectories</h2>{figure_html(figdir/'lambda_trajectories.png','Coefficients after each complete outer step, including the declared feedback/control update. Each line shows the median across the three starts. No coefficient clamp is applied.')}
<div class="note">Changing λ changes the Hamiltonian. Compare each quantum energy with that step’s exact ground energy. The plotted raw coefficient growth is part of the measured source dynamics, not a claim of physical stability or convergence.</div>
{figure_html(figdir/'variational_quality.png','Every point is a completed frame. Gaps compare the optimized circuit with the independent exact minimum of the current 16 diagonal levels. Fidelity is the circuit probability in the current ground space.')}
<h2>Does quantum feedback reach the next field calculation?</h2><p>{paired_statement}</p>
{figure_html(figdir/'feedback_effect.png','Paired differences after the full trajectory. Every dot is one deterministic start; the black line marks its group median. This chart uses decimal display projections; exact fields, bounds and feedback expressions remain in the saved data.')}
<h2>Final-frame results for every trajectory</h2><div class="scroll">{table(['Mode','Strength','Seed','Feedback','Final E','Current ground E','Current gap','Ground fidelity'],case_rows)}</div>
<h2>qBraid circuit cross-checks</h2>{provider}<p>The trajectory optimization is an exact classical state calculation. Independent qBraid checks use its free ideal state-vector simulator; they are not a QPU run or a provider-executed optimization loop.</p>
<h2>Restoration provenance</h2><p>This report is generated exclusively from the actual completed outputs of the included source. The restored quantum runner and coordinator have new provenance. Original local-run observations are kept separately in <a href="../recovery/completed_sweep_observation_2026-11-23.json">the historical observation record</a>; they do not replace this run’s trial data or its validation evidence.</p>\n<h2>Source changes and mathematical scope</h2><p>The earlier floating feedback conversion and small-probability skips are replaced by their intended exact expressions. The source’s arbitrary three-digit repetend guard is removed; the complete required repetends are calculated. BOOK formulas and their matrix outputs remain in the executed path. Complete commuting hypercube-axis rotations are evaluated with the exact Gaussian/tensor identity recorded in the source ledger; every coefficient multiplicity and every matrix entry is retained. Large rational values are stored as exact expression graphs, with certified bounds used only to establish decisions and display projections.</p>
<p>The selected field formula includes GRVQ and MSTVQ. Remaining browser MAG/TGCR/ZPE/Maya time-step subsystems are outside this trajectory experiment. The earlier fixed-objective R2/R4 comparison remains a separate benchmark; its radial widths are not identified with these four signed Hamiltonian coefficients.</p>
<h2>Files and reproducibility</h2><p>Start with <a href="../START_HERE.ipynb">START_HERE.ipynb</a> and <a href="../FEEDBACK_SWEEP.md">FEEDBACK_SWEEP.md</a>. Machine-readable results are in <a href="results.json">results.json</a> and <a href="analysis_summary.json">analysis_summary.json</a>. Each case directory contains exact graphs, full frame records, quantum outputs, QASM and a checkpoint. Source provenance is recorded in <a href="../source_audit/hybrid_lineage_audit.json">hybrid_lineage_audit.json</a>.</p>
<p>Source result SHA-256: <code>{summary['sweep_results_sha256']}</code></p></html>'''
    (directory/"feedback_report.html").write_text(body)
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--results",type=Path,default=BASE/"latest_hybrid_results")
    args=parser.parse_args();r=render(args.results)
    print(json.dumps({"report":str(args.results/"feedback_report.html"),"trajectories":r["trajectories"],"frames":r["frames"]},indent=2))
