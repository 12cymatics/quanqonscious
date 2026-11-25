"""Build the portable browser notebook from the actual packaged sources."""
from pathlib import Path
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import nbformat as nb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build():
    relative = ["vedic_v18.51.1_exact_phi.html"] + [
        "experiments/grvq_ansatz_comparison/" + name for name in
        ("run_latest_hybrid.py", "hybrid_quantum.cjs", "latest_hybrid_executor.cjs", "exact_dag.cjs", "ansatz.py")]
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in relative}
    cells = []
    def md(text): cells.append(nb.v4.new_markdown_cell(text))
    def code(text): cells.append(nb.v4.new_code_cell(text))
    md("""# Quantum ansatz–GRVQ feedback sweep

**Open, inspect, and rerun the complete selected 29-method quantum feedback experiment in your browser.**

The default comparison contains 36 trajectories and 72 outer steps: three mathematical modes, two method strengths, three deterministic circuit starts and paired quantum feedback enabled/disabled controls. Every field step evaluates all 29 methods, the source distributed GRVQ and MSTVQ stages, and the recovered three-layer, four-qubit Ry/CNOT circuit. The exact quantum expectation is injected into the next evolving field.

This package restores the declared experiment under new provenance after an execution environment was lost. The notebook reads only the actual completed outputs bundled with the current source. Historical observations are kept separately in `recovery/completed_sweep_observation_2026-11-23.json` and are not inserted into the current data.

## Start in qBraid Lab or Jupyter

Upload the complete `GRVQ_Quantum_Feedback_Sweep_2026-11-23.zip` and this notebook to the same directory, or open this notebook inside the extracted repository. Run the cells in order. The environment must provide Python 3.12+ and Node.js. Computation happens in that notebook environment; your own computer only needs a browser.

The saved report opens before any new experiment is launched. The **Run selected complete sweep** button starts the explicitly selected calculation and saves it to a fresh output directory. All 29 methods remain active. Mathematical mode and CPU scheduling have separate controls.

See `FEEDBACK_SWEEP.md` for the equations, source lineage and interpretation. The selected field formula is BOOK → GRVQ → MSTVQ; browser MAG/TGCR/ZPE/Maya auxiliary evolution is outside this experiment.""")
    code('''from pathlib import Path
import sys, json, hashlib, shutil, subprocess, zipfile
from fractions import Fraction
from datetime import datetime, timezone
from IPython.display import display, Markdown, HTML, FileLink, Image

ARCHIVE_PATH = Path.cwd() / "GRVQ_Quantum_Feedback_Sweep_2026-11-23.zip"
ENTRY = Path("experiments/grvq_ansatz_comparison/run_latest_hybrid.py")
candidates = [Path.cwd(), *Path.cwd().parents, Path.cwd()/"quanqonscious"]
REPO_ROOT = next((p for p in candidates if (p/ENTRY).is_file()), None)
if REPO_ROOT is None:
    if not ARCHIVE_PATH.is_file():
        raise FileNotFoundError("Upload the complete ZIP beside this notebook, set ARCHIVE_PATH above, or open the notebook inside the extracted repository.")
    destination = Path.cwd()/"grvq_feedback_bundle"
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        for item in archive.infolist():
            member = Path(item.filename)
            target = (destination/member).resolve()
            if member.is_absolute() or not target.is_relative_to(destination.resolve()):
                raise ValueError("The archive contains an unsafe extraction path")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("The archive contains a symbolic link")
        archive.extractall(destination)
    matches = list(destination.rglob("experiments/grvq_ansatz_comparison/run_latest_hybrid.py"))
    if len(matches) != 1:
        raise ValueError("Expected exactly one packaged feedback runner")
    REPO_ROOT = matches[0].parents[2]
REPO_ROOT = REPO_ROOT.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXPERIMENT = REPO_ROOT/"experiments/grvq_ansatz_comparison"
RECORDED = EXPERIMENT/"latest_hybrid_results"
if sys.version_info < (3,12):
    raise RuntimeError("Select a Python 3.12+ notebook environment")
if shutil.which("node") is None:
    raise RuntimeError("Node.js is required for the exact source executor. Install Node.js in this notebook environment before continuing.")
print("Project:", REPO_ROOT)
print("Python:", sys.version.split()[0])
print("Node:", subprocess.check_output(["node", "--version"], text=True).strip())
''')
    code("EXPECTED_SOURCE_SHA256 = "+repr(hashes)+'''
for relative, expected in EXPECTED_SOURCE_SHA256.items():
    actual = hashlib.sha256((REPO_ROOT/relative).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Source differs from this notebook's package: {relative}. Review the change and rebuild the notebook before rerunning.")
print("Verified", len(EXPECTED_SOURCE_SHA256), "packaged source files against their full SHA-256 hashes.")
''')
    md("""## Notebook packages

The exact field and circuit calculations use the included Python/Node.js source. `mpmath` formats circuit-export angles; `matplotlib` draws report figures; `ipywidgets` supplies the controls. Their numerical outputs do not enter the exact evolving state.

Set `INSTALL_NOTEBOOK_PACKAGES = True` only when setting up an environment, then run this cell. qBraid recommends `%pip` for the active notebook kernel. If installation requests a kernel restart, restart it and rerun the cells with installation switched off.""")
    code('''INSTALL_NOTEBOOK_PACKAGES = False
if INSTALL_NOTEBOOK_PACKAGES:
    %pip install mpmath==1.3.0 matplotlib==3.10.8 ipywidgets==8.1.9
import mpmath, matplotlib, ipywidgets as widgets
print("mpmath", mpmath.__version__, "| matplotlib", matplotlib.__version__, "| ipywidgets", widgets.__version__)
''')
    md("""## Inspect the completed comparison

Each frame’s energy gap is measured against the **exact ground energy of that frame’s Hamiltonian**. Its four signed coefficients change between outer steps, so raw energy values across steps do not establish convergence of a fixed objective. The feedback-enabled/control pairs isolate the actual influence of `−⟨Zq⟩/1000` on the subsequent field.

Saved state graphs, full method operands and outputs, parameter searches, 288 QASM exports and retained field/parameter checkpoints accompany the result summary. qBraid verification is a separate numerical circuit cross-check of the exact classical optimization.""")
    code('''recorded = json.loads((RECORDED/"results.json").read_text())
if recorded["status"] != "complete":
    raise RuntimeError("The supplied result is not a completed sweep")
assert len(recorded["cases"]) == recorded["expected_cases"]
assert recorded["total_frames"] == recorded["expected_frames"]
for case in recorded["cases"]:
    assert case["status"] == "complete"
    assert all(s["all29_consumed"] for s in case["steps"])
    assert hashlib.sha256((RECORDED/case["result_file"]).read_bytes()).hexdigest() == case["result_sha256"]
display(Markdown(f"**{len(recorded['cases'])} completed trajectories · {recorded['total_frames']} complete steps · {recorded['totals']['core_method_calls']:,} core method applications**"))
print(json.dumps(recorded["totals"], indent=2))
display(FileLink(str((RECORDED/"results.json").relative_to(Path.cwd())) if RECORDED.is_relative_to(Path.cwd()) else str(RECORDED/"results.json")))
''')
    code('''from experiments.grvq_ansatz_comparison.render_feedback import render
analysis = render(RECORDED)
display(HTML((RECORDED/"feedback_report.html").read_text()))
''')
    md("""## Configure a complete new sweep

**Mathematical composition** changes the source equation: SERIES uses its ordered 29-method composition; PARALLEL sums 29 branches from one common input; SYMMETRIC_CONCURRENT runs the source forward/reverse half-strength compositions, 58 applications per step.

**Independent trajectory workers** schedule separate cases concurrently. **PARALLEL branch backend** controls serial versus actual worker-thread evaluation of independent branches inside that mathematical mode. It does not reorder the dependent SERIES or SYMMETRIC_CONCURRENT calculations.

Every chosen strength applies to all 29 methods and must be a nonzero exact rational. The disabled-feedback control still evaluates the field, quantum circuit and expectations; only its declared applied feedback is zero. The default two outer steps demonstrate feedback entering the next calculation. Increasing steps extends the actual trajectory and can increase exact expression sizes substantially.

The rational Cayley grid is `−1, −(d−1)/d, …, 0, …, 1`, plus the current coordinate. Four quantum coordinate sweeps on the default denominator-eight grid evaluate 48 coordinate searches per frame. All parameter decisions use exact comparisons.""")
    code('''mode_control = widgets.SelectMultiple(options=["SERIES", "PARALLEL", "SYMMETRIC_CONCURRENT"], value=("SERIES", "PARALLEL", "SYMMETRIC_CONCURRENT"), description="Math modes", layout=widgets.Layout(width="470px"))
strength_control = widgets.Text(value="1/29,1/58", description="Strengths")
seed_control = widgets.Text(value="7,19,43", description="Seeds")
feedback_control = widgets.Dropdown(options=[("Enabled + control", "both"), ("Enabled only", "on"), ("Control only", "off")], value="both", description="Feedback")
step_control = widgets.BoundedIntText(value=2, min=1, max=1000, description="Outer steps")
sweep_control = widgets.BoundedIntText(value=4, min=1, max=1000, description="Q sweeps")
grid_control = widgets.BoundedIntText(value=8, min=1, max=1024, description="Grid denom")
workers_control = widgets.BoundedIntText(value=2, min=1, max=64, description="Case workers")
backend_control = widgets.Dropdown(options=[("Serial branch evaluation", "serial"), ("Worker-thread branches", "workers")], value="serial", description="PAR backend", layout=widgets.Layout(width="400px"))
branch_control = widgets.BoundedIntText(value=2, min=1, max=29, description="Branch workers")
run_button = widgets.Button(description="Run selected complete sweep", button_style="success", layout=widgets.Layout(width="290px"))
run_output = widgets.Output()

def selected_configuration():
    modes = list(mode_control.value)
    strengths = [str(Fraction(s.strip())) for s in strength_control.value.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in seed_control.value.split(",") if s.strip()]
    if not modes or not strengths or not seeds:
        raise ValueError("Choose at least one mode, strength and seed")
    if len(set(strengths)) != len(strengths) or len(set(seeds)) != len(seeds):
        raise ValueError("Strengths and seeds must be distinct")
    if any(not 0 < Fraction(s) <= 1 for s in strengths):
        raise ValueError("Every method strength must be in (0,1]")
    return dict(modes=modes, strengths=strengths, seeds=seeds, feedback=feedback_control.value,
                steps=step_control.value, sweeps=sweep_control.value, grid_denominator=grid_control.value,
                workers=workers_control.value, parallel_backend=backend_control.value,
                branch_workers=branch_control.value)

LAST_OUTPUT = RECORDED
def execute_selected_sweep(configuration=None):
    global LAST_OUTPUT
    from experiments.grvq_ansatz_comparison.run_latest_hybrid import run_sweep
    configuration = selected_configuration() if configuration is None else configuration
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = EXPERIMENT/"notebook_runs"/stamp
    print("Requested configuration:", json.dumps(configuration, indent=2))
    print("New output:", output)
    result = run_sweep(output, **configuration)
    if result["status"] != "complete":
        raise RuntimeError("Calculation did not finish; inspect its retained logs and checkpoints")
    summary = render(output)
    LAST_OUTPUT = output
    display(Markdown(f"**Completed {summary['trajectories']} trajectories and {summary['frames']} outer steps.**"))
    display(HTML((output/"feedback_report.html").read_text()))
    return result

def clicked(_):
    run_button.disabled = True
    try:
        with run_output:
            run_output.clear_output(wait=True)
            execute_selected_sweep()
    finally:
        run_button.disabled = False

run_button.on_click(clicked)
display(widgets.VBox([mode_control, widgets.HBox([strength_control, seed_control]),
    widgets.HBox([feedback_control, step_control]), widgets.HBox([sweep_control, grid_control]),
    widgets.HTML("<b>CPU scheduling (separate from mathematical composition)</b>"),
    widgets.HBox([workers_control, branch_control]), backend_control, run_button, run_output]))
print("Default control configuration:", json.dumps(selected_configuration(), indent=2))
''')
    md("""### Direct notebook API

For a scriptable run, edit the explicit configuration below and set `RUN_EXPLICIT_CONFIGURATION = True`. This calls the same complete engine as the button. Existing recorded results are preserved in their original directory. The result is written to a unique timestamped directory; errors retain the actual failed case and logs rather than returning substitute results.""")
    code('''RUN_EXPLICIT_CONFIGURATION = False
configuration = dict(modes=["SERIES", "PARALLEL", "SYMMETRIC_CONCURRENT"],
    strengths=["1/29", "1/58"], seeds=[7,19,43], feedback="both", steps=2,
    sweeps=4, grid_denominator=8, workers=2, parallel_backend="serial", branch_workers=2)
if RUN_EXPLICIT_CONFIGURATION:
    explicit_result = execute_selected_sweep(configuration)
''')
    md("""## Inspect a trajectory and its exact quantum state

Change the case index and step index to inspect any saved trajectory. The exact graph stores field expressions, and the full compressed frame stores every method’s operands, coefficients and consumed outputs. Decimal values below are only display projections. The exact quantum state uses rational amplitudes.""")
    code('''CASE_INDEX, STEP_INDEX = 0, 1
active = json.loads((LAST_OUTPUT/"results.json").read_text())
case = active["cases"][CASE_INDEX]
step = case["steps"][min(STEP_INDEX, len(case["steps"])-1)]
case_path = (LAST_OUTPUT/case["result_file"]).parent
quantum = json.loads((case_path/step["quantum_file"]).read_text())
print("Case:", case["case_id"], "| step:", step["step"])
print("All 29 methods consumed:", step["all29_consumed"])
print("Core applications:", step["core_method_calls"], "| distributed GRVQ:", step["distributed_grvq_updates"])
print("Applied feedback (exact):", step["feedback_deltas"])
print("Ground fidelity (exact):", step["ground_fidelity_rational"])
print("Current-H energy gap (display):", step["display"]["energy_gap"]["value"])
print("Exact state norm:", sum(Fraction(a)**2 for a in quantum["amplitudes"]))
print("Full method frame:", case_path/step["frame_file"])
print("Exact expression graph:", case_path/"exact_graph.json.gz")
print("Checkpoint:", case_path/"checkpoint.json")
print("Circuit exports:", case_path/"qasm_manifest.json")
''')
    md("""## Export a new result

The delivered ZIP already includes the recorded complete comparison. To package a newly run experiment, set `EXPORT_CURRENT_RESULTS = True`. The notebook writes the report, every trajectory, exact graph, quantum state, QASM and checkpoint into a new ZIP. Download that ZIP from the qBraid/Jupyter file browser, or use the displayed link.

For reproducibility, retain the original source package together with this result ZIP; each run records its code hashes and selected configuration.""")
    code('''EXPORT_CURRENT_RESULTS = False
if EXPORT_CURRENT_RESULTS:
    result_archive = EXPERIMENT/("feedback_results_"+LAST_OUTPUT.name+".zip")
    with zipfile.ZipFile(result_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(LAST_OUTPUT.rglob("*")):
            if path.is_file():
                archive.write(path, Path(LAST_OUTPUT.name)/path.relative_to(LAST_OUTPUT))
    print("Exported:", result_archive, "| bytes:", result_archive.stat().st_size)
    print("SHA-256:", hashlib.sha256(result_archive.read_bytes()).hexdigest())
    link_path = result_archive.relative_to(Path.cwd()) if result_archive.is_relative_to(Path.cwd()) else result_archive
    display(FileLink(str(link_path)))
''')
    notebook = nb.v4.new_notebook(cells=cells, metadata={
        "kernelspec":{"display_name":"Python 3", "language":"python", "name":"python3"},
        "language_info":{"name":"python","version":"3.12"},
        "grvq_package":{"schema":"feedback-browser-notebook/v1", "source_sha256":hashes,
                        "automatic_new_sweep":False}})
    path = HERE/"START_HERE.ipynb"
    nb.write(notebook, path)
    return path


def execute_inprocess(notebook):
    """Execute real IPython cells without a network-bound kernel process."""
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    shell=InteractiveShell.instance()
    executed=0
    with contextlib.chdir(HERE):
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            started=datetime.now(timezone.utc).isoformat()
            with capture_output(stdout=True,stderr=True,display=True) as captured:
                result=shell.run_cell(cell.source,store_history=True)
            cell.outputs=[]
            if captured.stdout:
                cell.outputs.append(nb.v4.new_output("stream",name="stdout",text=captured.stdout))
            if captured.stderr:
                cell.outputs.append(nb.v4.new_output("stream",name="stderr",text=captured.stderr))
            for output in captured.outputs:
                cell.outputs.append(nb.v4.new_output("display_data",data=output.data,metadata=output.metadata))
            cell.execution_count=result.execution_count
            cell.metadata["execution"]={"started_utc":started,"completed_utc":datetime.now(timezone.utc).isoformat()}
            if not result.success:
                raise RuntimeError("Notebook cell execution failed") from (result.error_in_exec or result.error_before_exec)
            executed+=1
    import ipywidgets
    notebook.metadata["widgets"]={"application/vnd.jupyter.widget-state+json":ipywidgets.Widget.get_manager_state()}
    notebook.metadata["grvq_execution"]={"status":"complete","engine":"IPython InteractiveShell in-process",
        "code_cells_executed":executed,"reason":"The explicitly selected in-process IPython executor uses no socket-based kernel.",
        "calculation":"Actual notebook code executed; saved trajectory results loaded and checked. A new sweep is explicitly launched by the notebook controls."}


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true")
    parser.add_argument("--executor",choices=("inprocess","jupyter"),default="inprocess")
    args=parser.parse_args();path=build()
    if args.execute:
        notebook=nb.read(path,as_version=4)
        if args.executor == "inprocess":
            execute_inprocess(notebook)
        else:
            from nbclient import NotebookClient
            NotebookClient(notebook,timeout=240,kernel_name="python3",resources={"metadata":{"path":str(HERE)}}).execute()
        nb.write(notebook,path)
    print(path)
