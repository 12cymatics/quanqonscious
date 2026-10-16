"""Run the pre-registered SCAN length experiment, end to end, as registered.

    python scripts/run_scan_length_experiment.py

Every step is the one ``PREREGISTRATION_SCAN_LENGTH.md`` names, with its
flags. The script builds the corpus, trains the six configs under
``configs/scan_length/`` (control and treatment for each seed, run side by
side so each pair shares conditions), then evaluates the base model and all
six adapters on both registered metrics.

Restarts, not resumes
---------------------
The pre-registration allows a run that dies to be restarted from scratch
with its unchanged config, recorded as a restart. That is what happens here:
a training run counts as done only once its output directory holds
``vedic_config.json``, which ``train_lora.py`` writes after the final save. A
directory without it is deleted and the run starts over. Nothing is resumed
from a checkpoint. Every restart is appended to ``restarts.jsonl``. An
evaluation counts as done only if its record exists and covers the whole
split, so a partial record is redone from scratch too.

Records
-------
Results go to ``results/scan_length/``, tracked in git: one held-out record
and one exact-match record per model, plus each run's provenance and its
Trainer log history (``trainer_state.json``). The log history is where psi_rms
and the per-term values live. Adapters stay in ``checkpoints/``, which is not
tracked. Each evaluation record carries the adapter's SHA-256, so a re-run can
be checked against it.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
SEEDS = (0, 1, 2)
ARMS = ("no_sutra", "full")
THREADS = "2"
RESULTS = REPO / "results" / "scan_length"
CHECKPOINTS = REPO / "checkpoints" / "scan_length"
DATA = REPO / "data" / "scan_length"
N_TEST = 3920


def _log_restart(kind: str, name: str, why: str) -> None:
    with (RESULTS / "restarts.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": kind, "name": name, "why": why}) + "\n")


def _py(*args: str, log: Path) -> subprocess.Popen:
    out = log.open("w", encoding="utf-8")
    return subprocess.Popen([sys.executable, *args], cwd=REPO, stdout=out,
                            stderr=subprocess.STDOUT)


def _train_pair(seed: int) -> None:
    procs = []
    for arm in ARMS:
        name = f"{arm}_seed{seed}"
        out = CHECKPOINTS / name
        if (out / "vedic_config.json").exists():
            continue
        if out.exists():
            _log_restart("train", name, "output directory without vedic_config.json")
            shutil.rmtree(out)
        procs.append((name, _py("scripts/train_lora.py", "--config",
                                f"configs/scan_length/{name}.yaml",
                                log=RESULTS / f"train_{name}.log")))
    for name, p in procs:
        if p.wait() != 0:
            raise SystemExit(f"training {name} failed; see results/scan_length/train_{name}.log")
    for arm in ARMS:
        name = f"{arm}_seed{seed}"
        for f in ("run_provenance", "trainer_state"):
            shutil.copy(CHECKPOINTS / name / f"{f}.json", RESULTS / f"{f}_{name}.json")


def _done(record: Path, key: str) -> bool:
    if not record.exists():
        return False
    data = json.loads(record.read_text(encoding="utf-8"))
    if key == "heldout":
        return data.get("heldout", {}).get("n_examples") == N_TEST
    return data.get("scan", {}).get("length", {}).get("n_total") == N_TEST


def _evaluate(name: str) -> None:
    adapter = [] if name == "base" else ["--adapter", str(CHECKPOINTS / name)]
    common = ["--base-model", MODEL, "--revision", REVISION, "--num-threads", THREADS]
    jobs = [
        ("heldout", ["scripts/eval_heldout.py", *common, *adapter,
                     "--heldout", str(DATA / "test.jsonl"), "--batch-size", "8"]),
        ("em", ["scripts/eval_benchmarks.py", "--benchmark", "scan", "--split", "length",
                *common, *adapter]),
    ]
    procs = []
    for kind, argv in jobs:
        record = RESULTS / f"{kind}_{name}.json"
        if _done(record, kind):
            continue
        if record.exists():
            _log_restart(kind, name, "record does not cover the whole split")
            record.unlink()
        procs.append((kind, _py(*argv, "--output", str(record),
                                log=RESULTS / f"{kind}_{name}.log")))
    for kind, p in procs:
        if p.wait() != 0:
            raise SystemExit(f"{kind} for {name} failed; see results/scan_length/{kind}_{name}.log")


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "scripts/build_scan_corpus.py", "--split", "length",
                    "--output-dir", str(DATA)], cwd=REPO, check=True)
    for seed in SEEDS:
        _train_pair(seed)
    for name in ["base"] + [f"{arm}_seed{s}" for s in SEEDS for arm in ARMS]:
        _evaluate(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
