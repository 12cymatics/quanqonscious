"""Generate the manufacturing package for an 11-string, 22-EDO guitar.

CadQuery 2.8.x; dimensions are millimetres.  The front scale is measured on
X.  The rear scale is a three-dimensional speaking line (not its X
projection).  Coordinate calculations use Decimal so exported tables are
deterministic and are checked before any CAD is constructed.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 50
D = Decimal


@dataclass(frozen=True)
class Parameters:
    scale: str = "711.20"
    edo: int = 22
    fret_count: int = 22
    a4_hz: str = "432.0"
    nut_width: str = "63.50"
    fretboard_width_22: str = "74.40"
    fretboard_radius: str = "508.0"
    fretboard_thickness: str = "6.50"
    front_nut_spacing: str = "6.25"
    front_bridge_spacing: str = "8.50"
    front_strings: int = 10
    rear_strings: int = 1
    rear_line_z0: str = "-31.5"
    rear_line_dz: str = "-6.0"
    rear_rail_width: str = "20.0"
    rear_rail_thickness: str = "8.0"
    headstock_length: str = "180.0"
    headstock_thickness: str = "14.0"
    body_length: str = "510.0"
    body_depth: str = "115.0"
    side_thickness: str = "3.0"
    top_thickness: str = "3.2"
    back_thickness: str = "4.0"
    soundhole_diameter: str = "100.0"
    soundhole_x: str = "520.0"
    bridge_width: str = "112.0"
    bridge_depth: str = "40.0"
    bridge_thickness: str = "9.5"


P = Parameters()


def dec(name: str) -> Decimal:
    return D(getattr(P, name))


def pow2_fraction(exponent: Decimal) -> Decimal:
    """Return 2**exponent in the active, 50-digit Decimal context."""
    return (D(2).ln() * exponent).exp()


def fret_distance(n: int) -> Decimal:
    if not 0 <= n <= P.fret_count:
        raise ValueError(f"fret must be in [0, {P.fret_count}]")
    return dec("scale") * (D(1) - pow2_fraction(-D(n) / D(P.edo)))


def rear_x_projection() -> Decimal:
    return (dec("scale") ** 2 - dec("rear_line_dz") ** 2).sqrt()


def rear_point_at_distance(distance: Decimal) -> tuple[Decimal, Decimal, Decimal]:
    q = distance / dec("scale")
    return rear_x_projection() * q, D(0), dec("rear_line_z0") + dec("rear_line_dz") * q


def rear_fret_point(n: int) -> tuple[Decimal, Decimal, Decimal]:
    return rear_point_at_distance(fret_distance(n))


def width_at_x(x: Decimal) -> Decimal:
    joint = dec("scale") / 2
    return dec("nut_width") + (dec("fretboard_width_22") - dec("nut_width")) * x / joint


def squared_distance(a: tuple[Decimal, ...], b: tuple[Decimal, ...]) -> Decimal:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def verify_exact_geometry() -> dict[str, str | int]:
    """Prove the defining identities before toleranced B-rep operations.

    Squared-distance identities are compared exactly in Decimal arithmetic;
    this avoids hiding an erroneous X projection behind a float tolerance.
    """
    origin = (D(0), D(0), dec("rear_line_z0"))
    saddle = rear_point_at_distance(dec("scale"))
    octave = rear_fret_point(P.edo)
    scale2 = dec("scale") ** 2
    octave2 = (dec("scale") / 2) ** 2
    assert squared_distance(origin, saddle) == scale2
    assert squared_distance(origin, octave) == octave2
    assert fret_distance(P.edo) == dec("scale") / 2
    assert P.front_strings + P.rear_strings == 11
    return {
        "front_strings": P.front_strings,
        "rear_thumb_strings": P.rear_strings,
        "total_strings": P.front_strings + P.rear_strings,
        "fret_22_distance_mm": str(fret_distance(P.edo)),
        "rear_scale_squared_mm2": str(squared_distance(origin, saddle).normalize()),
        "specified_scale_squared_mm2": str(scale2.normalize()),
    }


FRONT_TUNER_CENTERS = tuple(
    (x, y)
    for y in (-24.5, 24.5)
    for x in (-24.0, -51.0, -78.0, -105.0, -132.0)
)
REAR_TUNER_CENTER = (-155.0, 0.0)


def cq_module():
    import cadquery as cq

    return cq


def cylinder_between(cq, a, b, radius):
    av, bv = cq.Vector(*a), cq.Vector(*b)
    vector = bv - av
    return cq.Workplane(obj=cq.Solid.makeCylinder(radius, vector.Length, av, vector.normalized()))


def body_outline(cq):
    j = float(dec("scale") / 2)
    points = [
        (j, 47), (j + 35, 118), (j + 92, 155), (j + 165, 136),
        (j + 223, 122), (j + 305, 174), (j + 370, 205), (j + 445, 193),
        (j + 510, 165), (j + 510, -165), (j + 445, -193), (j + 370, -205),
        (j + 305, -174), (j + 250, -128), (j + 205, -108), (j + 165, -94),
        (j + 128, -88), (j + 95, -80), (j + 65, -64), (j + 42, -43),
        (j + 22, -38), (j, -42),
    ]
    return cq.Workplane("XY").moveTo(*points[0]).spline(points[1:]).close()


def make_parts(cq):
    scale, joint = float(dec("scale")), float(dec("scale") / 2)
    outer = body_outline(cq).extrude(-float(dec("body_depth")))
    inner = (body_outline(cq).offset2D(-float(dec("side_thickness")))
             .extrude(-float(dec("body_depth") - dec("top_thickness") - dec("back_thickness")))
             .translate((0, 0, -float(dec("top_thickness")))))
    soundhole = (cq.Workplane("XY").center(float(dec("soundhole_x")), 0)
                 .circle(float(dec("soundhole_diameter") / 2)).extrude(-6).translate((0, 0, 1)))
    body = outer.cut(inner).cut(soundhole)

    fb = (cq.Workplane("XY").polyline([
        (0, -float(dec("nut_width") / 2)), (joint, -float(dec("fretboard_width_22") / 2)),
        (joint, float(dec("fretboard_width_22") / 2)), (0, float(dec("nut_width") / 2)),
    ]).close().extrude(-float(dec("fretboard_thickness"))))
    radius = float(dec("fretboard_radius"))
    fb = fb.intersect(cq.Workplane("YZ", origin=(-2, 0, -radius)).circle(radius).extrude(joint + 4))

    neck = (cq.Workplane("XY").polyline([
        (0, -30), (joint, -35.7), (joint, 35.7), (0, 30)
    ]).close().extrude(-26.5).translate((0, 0, -float(dec("fretboard_thickness")))))
    hs_len, hs_t = float(dec("headstock_length")), float(dec("headstock_thickness"))
    headstock = cq.Workplane("XY").polyline([
        (0, -31.75), (-25, -34), (-145, -36), (-hs_len, -25),
        (-hs_len, 25), (-145, 36), (-25, 34), (0, 31.75),
    ]).close().extrude(-hs_t)
    for x, y in (*FRONT_TUNER_CENTERS, REAR_TUNER_CENTER):
        headstock = headstock.cut(cq.Workplane("XY").center(x, y).circle(5).extrude(-(hs_t + 2)).translate((0, 0, 1)))

    rear_x = float(rear_x_projection())
    # Rotate using atan2 from the standard library; Decimal established length first.
    import math
    angle = -math.degrees(math.atan2(float(dec("rear_line_dz")), rear_x))
    rail = (cq.Workplane("XY").box(scale / 2, float(dec("rear_rail_width")),
            float(dec("rear_rail_thickness")), centered=(False, True, False))
            .translate((0, 0, float(dec("rear_line_z0") + D("1.55"))))
            .rotate((0, 0, 0), (0, 1, 0), angle))

    frets = []
    for n in range(1, P.fret_count + 1):
        x = float(fret_distance(n)); width = float(width_at_x(fret_distance(n)))
        frets.append(cq.Workplane("YZ", origin=(x, 0, 0)).rect(width, 1.1).extrude(1.1).val())
    front_frets = cq.Workplane(obj=cq.Compound.makeCompound(frets))

    bridge = (cq.Workplane("XY").center(scale + 4, 0)
              .box(float(dec("bridge_depth")), float(dec("bridge_width")),
                   float(dec("bridge_thickness")), centered=(True, True, False)))
    front_strings = []
    nut_span = (P.front_strings - 1) * float(dec("front_nut_spacing"))
    bridge_span = (P.front_strings - 1) * float(dec("front_bridge_spacing"))
    for i in range(P.front_strings):
        yn = -nut_span / 2 + i * float(dec("front_nut_spacing"))
        yb = -bridge_span / 2 + i * float(dec("front_bridge_spacing"))
        front_strings.append(cylinder_between(cq, (0, yn, 1.4), (scale, yb, 13), .2).val())
    rear_saddle = tuple(float(v) for v in rear_point_at_distance(dec("scale")))
    return {
        "body_shell": body, "neck": neck, "front_fretboard": fb,
        "front_frets": front_frets, "rear_thumb_rail": rail,
        "headstock": headstock, "front_bridge": bridge,
        "front_strings_reference": cq.Workplane(obj=cq.Compound.makeCompound(front_strings)),
        "rear_string_reference": cylinder_between(cq, (0, 0, float(dec("rear_line_z0"))), rear_saddle, .45),
    }


def write_dxf(path: Path, lines):
    records = ["0", "SECTION", "2", "HEADER", "9", "$INSUNITS", "70", "4", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]
    for layer, x1, y1, x2, y2 in lines:
        records += ["0", "LINE", "8", layer, "10", f"{x1:.8f}", "20", f"{y1:.8f}", "30", "0", "11", f"{x2:.8f}", "21", f"{y2:.8f}", "31", "0"]
    records += ["0", "ENDSEC", "0", "EOF"]
    path.write_text("\n".join(records) + "\n", encoding="ascii")


def write_data(outdir: Path):
    data, drawings = outdir / "data", outdir / "drawings"
    data.mkdir(parents=True, exist_ok=True); drawings.mkdir(parents=True, exist_ok=True)
    proof = verify_exact_geometry()
    rows = []
    for n in range(1, P.fret_count + 1):
        s = fret_distance(n); x, _, z = rear_fret_point(n)
        rows.append((n, s, x, z))
    with (data / "fret_coordinates.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("fret", "front_x_mm", "rear_x_mm", "rear_z_mm", "rear_path_distance_mm"))
        for n, s, x, z in rows:
            writer.writerow((n, f"{s:.12f}", f"{x:.12f}", f"{z:.12f}", f"{s:.12f}"))
    front = [("FRETS", float(s), -float(width_at_x(s) / 2), float(s), float(width_at_x(s) / 2)) for _, s, _, _ in rows]
    rear = [("FRETS", float(x), -float(dec("rear_rail_width") / 2), float(x), float(dec("rear_rail_width") / 2)) for _, _, x, _ in rows]
    write_dxf(drawings / "front_fretboard_slots.dxf", front)
    write_dxf(drawings / "rear_thumb_fret_stations_projected.dxf", rear)
    manifest = {"parameters": asdict(P), "derived": {
        "rear_scale_x_projection_mm": str(rear_x_projection()),
        "rear_saddle_xyz_mm": [str(v) for v in rear_point_at_distance(dec("scale"))],
    }, "proof": proof}
    (data / "master_parameters.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return proof


def build(outdir: Path):
    proof = write_data(outdir)
    cq = cq_module()
    from cadquery import exporters
    parts = make_parts(cq); part_dir = outdir / "parts"; part_dir.mkdir(exist_ok=True)
    assembly = cq.Assembly(name="11String_22EDO_AcousticElectric")
    for name, part in parts.items():
        assembly.add(part, name=name)
        exporters.export(part, str(part_dir / f"{name}.step"))
    assembly.save(str(outdir / "11string_22edo_acoustic_electric_MASTER.step"), exportType="STEP")
    exporters.export(assembly.toCompound(), str(outdir / "11string_22edo_acoustic_electric_MASTER.stl"), tolerance=.12, angularTolerance=.15)
    print("PASS exact geometry:", json.dumps(proof, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="cad_build")
    parser.add_argument("--verify-only", action="store_true", help="verify mathematics and write data/DXF without CadQuery")
    arguments = parser.parse_args()
    destination = Path(arguments.output_dir).resolve()
    if arguments.verify_only:
        print("PASS exact geometry:", json.dumps(write_data(destination), sort_keys=True))
    else:
        build(destination)
