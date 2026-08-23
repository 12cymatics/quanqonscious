"""Single authority for every ablation number this package reports.

Why this exists
---------------
The ablation results had three owners: ``ABLATION_RESULTS.md``, the pull
request body, and whatever was said in conversation. Three owners means
three chances to drift, and no way to tell which one is wrong when they
disagree. The numbers themselves were only ever in one place that
*computed* them -- the JSON files under ``runs/`` -- and nothing checked
that the prose agreed with those files.

This script makes ``runs/*.json`` the sole authority and makes every
document that quotes a number checkable against it.

The check is strict in one specific way: **an unchecked numeric cell is a
failure.** A table column this script does not know how to verify is
reported as unverifiable rather than passed over, because a number nobody
checks is exactly how the previous drift happened.

Usage
-----
    python scripts/verify_ablation.py            # print measured tables
    python scripts/verify_ablation.py --check    # verify ABLATION_RESULTS.md
                                                 # exits 1 on any mismatch

``--check`` is the gate. Run it before quoting an ablation number anywhere.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
DOC = REPO / "ABLATION_RESULTS.md"

SEEDS = (42, 1, 2)


# ─────────────────────────────────────────────────────────── run sets

@dataclass(frozen=True)
class RunSet:
    """One reported ablation, and the files that back it.

    ``columns`` names each table column after the leading seed column, so
    every cell has a declared meaning. Recognised names:

    ``base``/``full``  held-out CE for that arm
    ``delta``          full - base
    ``rel``            (full - base) / base, as a percentage
    ``rel@KEY``        the ``rel`` column of another run set (a cross-reference)
    ``same@KEY``       the literal ``yes`` iff both arms reproduce KEY exactly
    """
    key: str
    heading: str
    base: str      # filename template, {} = seed
    full: str
    columns: tuple[str, ...]
    base_adapter: str = r"no_sutra$"
    full_adapter: str = r"(?<!no_)(?<!no_su)full$"

    # Known limitation, stated rather than papered over: the run JSONs do not
    # record their seed, so the seed-to-file binding rests on the filename
    # alone. The *arm* does not -- it is cross-checked against the adapter
    # each file records, so a base/full swap cannot pass. Runs written from
    # here on should record the seed; these were not.


RUN_SETS: tuple[RunSet, ...] = (
    RunSet(
        key="initial",
        heading="## Held-out CE (the discriminating measure)",
        base="{}_no_sutra.json",
        full="{}_full.json",
        columns=("base", "full", "delta", "rel"),
    ),
    RunSet(
        key="rerun",
        heading="## Re-run on the reworked kernel (operands + composition algebra)",
        base="rerun_seed{}_base.json",
        full="rerun_seed{}_full.json",
        columns=("base", "full", "delta", "rel", "same@initial"),
    ),
    RunSet(
        key="fixed",
        heading="## After fixing the structural defects",
        base="fixed_seed{}_base.json",
        full="fixed_seed{}_full.json",
        columns=("base", "full", "delta", "rel", "rel@initial"),
    ),
    RunSet(
        key="scaled",
        heading="## After rescaling the auxiliary weights",
        base="fixed_seed{}_base.json",   # the zero-weight arm is unchanged
        full="scaled_seed{}.json",
        columns=("base", "full", "delta", "rel", "rel@fixed"),
        full_adapter=r"scaled\d*_full$",
    ),
)

# The initial run stored seed 42 under a different name than seeds 1 and 2.
_INITIAL_STEM = {42: "eval", 1: "heldout_seed1", 2: "heldout_seed2"}


def _path(rs: RunSet, arm: str, seed: int) -> Path:
    template = rs.base if arm == "base" else rs.full
    stem = _INITIAL_STEM[seed] if rs.key == "initial" else str(seed)
    return RUNS / template.format(stem)


def ce(path: Path, expect_adapter: str) -> float:
    """Held-out cross-entropy, read from the run's own JSON.

    The arm is confirmed against the ``adapter`` the file itself records,
    never inferred from the filename. ``fixed_seed42_base.json`` records
    ``checkpoints/cpu_no_sutra``: "base" here means the zero-weight arm, not
    the untuned base model (that is ``eval_base.json``, adapter ``base``).
    With names that close to each other, a swapped or mislabelled file has to
    be caught by content.
    """
    rec = json.loads(path.read_text())
    adapter = rec.get("adapter")
    if adapter is None:
        raise ValueError(f"{path.name} records no adapter; its arm is unknowable")
    if not re.search(expect_adapter, adapter):
        raise ValueError(
            f"{path.name} is used as the {expect_adapter!r} arm but records "
            f"adapter {adapter!r} -- the file and its slot disagree")
    return rec["heldout"]["ce_loss"]


@dataclass
class Measured:
    seeds: dict[int, tuple[float, float]]      # seed -> (base, full)

    @property
    def base(self) -> list[float]:
        return [self.seeds[s][0] for s in SEEDS if s in self.seeds]

    @property
    def full(self) -> list[float]:
        return [self.seeds[s][1] for s in SEEDS if s in self.seeds]

    def delta(self, seed: int) -> float:
        b, f = self.seeds[seed]
        return f - b

    def rel(self, seed: int) -> float:
        b, f = self.seeds[seed]
        return 100.0 * (f - b) / b

    @property
    def mean_base(self) -> float:
        return statistics.mean(self.base)

    @property
    def mean_full(self) -> float:
        return statistics.mean(self.full)

    @property
    def mean_delta(self) -> float:
        return self.mean_full - self.mean_base

    @property
    def mean_rel(self) -> float:
        return 100.0 * self.mean_delta / self.mean_base

    def sd(self, arm: str) -> float:
        """Sample standard deviation (n-1), matching the reported figures.

        ``arm`` is one of base / full / delta / rel -- the spread of the
        per-seed deltas is reported too, and every reported spread must be
        derivable here or the check will refuse to pass it.
        """
        series = {
            "base": self.base,
            "full": self.full,
            "delta": [self.delta(s) for s in SEEDS if s in self.seeds],
            "rel": [self.rel(s) for s in SEEDS if s in self.seeds],
        }[arm]
        return statistics.stdev(series)


def load(rs: RunSet) -> Measured | None:
    """Measured values, or None if the run has not been executed yet."""
    seeds: dict[int, tuple[float, float]] = {}
    for s in SEEDS:
        b, f = _path(rs, "base", s), _path(rs, "full", s)
        if not (b.exists() and f.exists()):
            return None
        seeds[s] = (ce(b, rs.base_adapter), ce(f, rs.full_adapter))
    return Measured(seeds)


# ─────────────────────────────────────────────────────────── reporting

def render(rs: RunSet, m: Measured) -> str:
    out = [f"  {rs.key}", "    seed        base       full      delta       rel"]
    for s in SEEDS:
        b, f = m.seeds[s]
        out.append(f"    {s:>4}  {b:>10.6f} {f:>10.6f} {m.delta(s):>+10.6f} "
                   f"{m.rel(s):>+8.2f}%")
    out.append(f"    mean  {m.mean_base:>10.6f} {m.mean_full:>10.6f} "
               f"{m.mean_delta:>+10.6f} {m.mean_rel:>+8.2f}%")
    out.append(f"    sd    {m.sd('base'):>10.6f} {m.sd('full'):>10.6f}")
    return "\n".join(out)


# ─────────────────────────────────────────────────────────── the check

_CELL = re.compile(r"^\**([+-]?\d+\.?\d*)\**%?$")


def _quoted(cell: str) -> tuple[float, int] | None:
    """Parse a table cell into (value, decimal places), or None if not numeric."""
    text = cell.strip().replace("**", "").replace("%", "").strip()
    m = _CELL.match(text)
    if not m:
        return None
    body = m.group(1)
    places = len(body.split(".")[1]) if "." in body else 0
    return float(body), places


def _rounds_to(computed: float, quoted: float, places: int) -> bool:
    """True iff `quoted` is `computed` correctly rounded to `places` decimals."""
    return abs(computed - quoted) <= 0.5 * 10 ** (-places) + 1e-12


def _rows(text: str, heading: str) -> list[list[str]]:
    """Table rows belonging to a section, up to the next same-level heading."""
    start = text.find(heading)
    if start < 0:
        return []
    rest = text[start + len(heading):]
    end = rest.find("\n## ")
    body = rest if end < 0 else rest[:end]
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("|") and not set(line) <= set("|-: "):
            rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def check(text: str, sets: dict[str, Measured]) -> list[str]:
    problems: list[str] = []
    for rs in RUN_SETS:
        m = sets.get(rs.key)
        rows = _rows(text, rs.heading)
        if m is None:
            if rows:
                problems.append(
                    f"[{rs.key}] the document has a table but the runs are "
                    f"missing from runs/ — nothing backs those numbers")
            continue
        if not rows:
            problems.append(
                f"[{rs.key}] measured runs exist in runs/ but the document "
                f"has no section {rs.heading!r} reporting them")
            continue

        seen: set[str] = set()
        for row in rows:
            label = row[0].replace("**", "").strip()
            cells = row[1:]

            if label.isdigit() and int(label) in SEEDS:
                seed = int(label)
                expect: dict[str, float] = {
                    "base": m.seeds[seed][0], "full": m.seeds[seed][1],
                    "delta": m.delta(seed), "rel": m.rel(seed),
                }
                for other in sets:
                    expect[f"rel@{other}"] = sets[other].rel(seed)
                seen.add(f"seed{seed}")
            elif label == "mean":
                seed = None
                expect = {"base": m.mean_base, "full": m.mean_full,
                          "delta": m.mean_delta, "rel": m.mean_rel}
                for other in sets:
                    expect[f"rel@{other}"] = sets[other].mean_rel
                seen.add("mean")
            elif label == "sd":
                seed = None
                expect = {name: m.sd(name)
                          for name in ("base", "full", "delta", "rel")}
                seen.add("sd")
            else:
                continue

            for i, cell in enumerate(cells):
                name = rs.columns[i] if i < len(rs.columns) else None
                q = _quoted(cell)

                if name is not None and name.startswith("same@"):
                    other = sets.get(name.split("@")[1])
                    if seed is None or other is None:
                        continue
                    ok = m.seeds[seed] == other.seeds[seed]
                    claimed = cell.strip().lower().replace("**", "") == "yes"
                    if claimed != ok:
                        problems.append(
                            f"[{rs.key}] seed {seed} claims "
                            f"{'reproduction' if claimed else 'no reproduction'} "
                            f"of {name.split('@')[1]}, measured "
                            f"{'identical' if ok else 'different'}")
                    continue

                if q is None:
                    # A column declared to hold a number must hold one. "~28%"
                    # or "n/a" in a numeric column would otherwise slip past
                    # unchecked, which is the failure mode this gate exists for.
                    if name in expect and cell.strip():
                        problems.append(
                            f"[{rs.key}] row {label!r} {name} is declared "
                            f"numeric but reads {cell.strip()!r}")
                    continue          # genuinely a prose column
                value, places = q

                if name is None or name not in expect:
                    problems.append(
                        f"[{rs.key}] row {label!r} column {i + 1} quotes "
                        f"{cell.strip()!r} but no column is declared for it — "
                        f"unverifiable numbers are not allowed")
                    continue

                if not _rounds_to(expect[name], value, places):
                    problems.append(
                        f"[{rs.key}] row {label!r} {name} says {cell.strip()}, "
                        f"measured {expect[name]:.6f}")

        for required in [f"seed{s}" for s in SEEDS] + ["mean", "sd"]:
            if required not in seen:
                problems.append(f"[{rs.key}] table has no {required} row")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify ABLATION_RESULTS.md matches; exit 1 on mismatch")
    args = ap.parse_args()

    sets = {rs.key: m for rs in RUN_SETS if (m := load(rs)) is not None}

    for rs in RUN_SETS:
        if rs.key in sets:
            print(render(rs, sets[rs.key]))
        else:
            print(f"  {rs.key}\n    not run (missing files under runs/)")
        print()

    if not args.check:
        return 0

    problems = check(DOC.read_text(encoding="utf-8"), sets)
    if problems:
        print("MISMATCH:")
        for p in problems:
            print(f"  - {p}")
        print(f"\n{len(problems)} ablation claims do not match runs/.")
        return 1
    print("OK — every documented ablation number matches runs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
