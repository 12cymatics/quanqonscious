"""Create the complete feedback-sweep deliverable without dropping exact graph data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "grvq_ansatz_comparison"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "dist" / "GRVQ_Quantum_Feedback_Sweep_2026-11-23.zip")
    args = p.parse_args()
    result_path = EXP / "latest_hybrid_results" / "results.json"
    r = json.loads(result_path.read_text())
    if r.get("status") != "complete" or len(r.get("cases", [])) != 36 or r.get("total_frames") != 72:
        raise RuntimeError("The full 36-trajectory / 72-frame sweep must finish before export.")
    required_files = [
        ROOT / "vedic_v18.51.1_exact_phi.html",
        EXP / "exact_dag.cjs",
        EXP / "latest_hybrid_executor.cjs",
        EXP / "hybrid_quantum.cjs",
        EXP / "run_latest_hybrid.py",
        EXP / "START_HERE.ipynb",
        EXP / "FEEDBACK_SWEEP.md",
        EXP / "latest_hybrid_results" / "feedback_report.html",
        EXP / "latest_hybrid_results" / "independent_output_audit.json",
        ROOT / "tests" / "test_exact_expression_dag.py",
        ROOT / "tests" / "test_feedback_trajectories.py",
    ]
    for f in required_files:
        if not f.is_file():
            raise FileNotFoundError(f"Required deliverable file missing: {f}")
    selected = set(required_files)
    for f in EXP.iterdir():
        if f.is_file() and f.suffix in {".py", ".cjs", ".md", ".txt", ".ipynb"}:
            selected.add(f)
    for name in ["source_audit", "recovery", "latest_hybrid_results", "latest_hybrid_validation", "latest_hybrid_backend_validation"]:
        directory = EXP / name
        if not directory.is_dir():
            raise FileNotFoundError(f"Required deliverable directory missing: {directory}")
        for f in directory.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                selected.add(f)
    for name in ["LICENSE", "LICENSE.md", "LICENSE.txt"]:
        f = ROOT / name
        if f.is_file():
            selected.add(f)
    entries = []
    for f in sorted(selected):
        data = f.read_bytes()
        entries.append({"path":f.relative_to(ROOT).as_posix(), "bytes":len(data),
                        "sha256":hashlib.sha256(data).hexdigest()})
    manifest = {
        "schema":"grvq-feedback-export/v1", "status":"complete",
        "trajectories":36, "frames":72, "files":entries,
        "total_uncompressed_file_bytes":sum(e["bytes"] for e in entries),
        "results_sha256":hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "scope":"Complete recovered-source ansatz–GRVQ feedback sweep, including exact rational expression graphs, full per-case records, all QASM exports, code, notebook, report and retained validation.",
        "provider_verification":"See actual provider status in the report. Exporting QASM alone does not establish provider or QPU execution.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for e in entries:
            f = ROOT / e["path"]
            z.write(f, e["path"], compress_type=zipfile.ZIP_STORED if f.suffix == ".gz" else zipfile.ZIP_DEFLATED)
        z.writestr("EXPORT_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
        z.writestr("OPEN_ME.md", "# Completed quantum ansatz–GRVQ feedback sweep\n\nOpen experiments/grvq_ansatz_comparison/START_HERE.ipynb in qBraid/Jupyter. For the self-contained report, open experiments/grvq_ansatz_comparison/latest_hybrid_results/feedback_report.html in a browser. Equations, scope and reproduction commands are in experiments/grvq_ansatz_comparison/FEEDBACK_SWEEP.md.\n\nAll 36 trajectories, 72 coupled frames and all 29 source methods are included. EXPORT_MANIFEST.json records every file's SHA256. Exact graphs and journals are retained in full.\n")
    with zipfile.ZipFile(args.output) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
        for e in entries:
            if hashlib.sha256(z.read(e["path"])).hexdigest() != e["sha256"]:
                raise RuntimeError(f"ZIP content SHA256 mismatch: {e['path']}")
    digest = hashlib.sha256()
    with args.output.open("rb") as f:
        while block := f.read(1024*1024):
            digest.update(block)
    summary = {"path":str(args.output), "bytes":args.output.stat().st_size,
               "sha256":digest.hexdigest(), "files":len(entries), "verified":"CRC and every file SHA256"}
    args.output.with_suffix(".sha256").write_text(digest.hexdigest() + "  " + args.output.name + "\n")
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
if __name__ == "__main__":
    main()
