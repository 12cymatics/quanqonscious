"""Re-import and independently validate a generated guitar CAD package."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import cadquery as cq


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "11string_22edo_acoustic_electric_FULL_CAD.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("guitar_22edo_generator", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generator: {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate(build_dir: Path) -> dict[str, object]:
    generator = load_generator()
    step = build_dir / "11string_22edo_acoustic_electric_MASTER.step"
    stl = build_dir / "11string_22edo_acoustic_electric_MASTER.stl"
    if not step.is_file() or step.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty master STEP: {step}")
    if not stl.is_file() or stl.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty master STL: {stl}")

    imported = cq.importers.importStep(str(step))
    solids = imported.solids().vals()
    if not solids:
        raise AssertionError("OpenCascade re-import produced no solids")

    proof = generator.verify_exact_geometry()
    scale = float(generator.dec("scale"))
    origin = (0.0, 0.0, float(generator.dec("rear_line_z0")))
    fret_22 = tuple(float(value) for value in generator.rear_fret_point(22))
    saddle = tuple(
        float(value)
        for value in generator.rear_point_at_distance(generator.dec("scale"))
    )
    rear_fret_distance = math.dist(origin, fret_22)
    rear_scale_distance = math.dist(origin, saddle)
    assert math.isclose(rear_fret_distance, scale / 2, abs_tol=1e-10)
    assert math.isclose(rear_scale_distance, scale, abs_tol=1e-10)

    report = {
        "step_reimport": "PASS",
        "solid_count": len(solids),
        "aggregate_solid_volume_mm3": sum(solid.Volume() for solid in solids),
        "front_tuners": len(generator.FRONT_TUNER_CENTERS),
        "rear_tuners": 1,
        "total_tuners": len(generator.FRONT_TUNER_CENTERS) + 1,
        "front_fret_22_mm": scale / 2,
        "rear_3d_fret_22_path_mm": rear_fret_distance,
        "rear_3d_scale_mm": rear_scale_distance,
        "exact_decimal_proof": proof,
    }
    output = build_dir / "data" / "validation_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.build_dir.resolve()), indent=2))
