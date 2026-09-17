# 11-String 22-EDO Acoustic-Electric Guitar — FreeCAD Release

## Generated deliverables

Running

```bash
python 11string_22edo_acoustic_electric_FULL_CAD.py --output-dir cad_build
```

produces:

- Master STEP: `cad_build/11string_22edo_acoustic_electric_MASTER.step`
- Master STL: `cad_build/11string_22edo_acoustic_electric_MASTER.stl`
- Individual STEP parts: `cad_build/parts/`
- DXF drawings: `cad_build/drawings/`
- Fret-coordinate CSV and parameter/proof JSON: `cad_build/data/`

The output directory is deliberately not committed: all manufacturing files
are reproducibly derived from the source parameters.

## Exact geometry contract

The front scale is **711.20 mm** along X. The rear-thumb string is a straight
three-dimensional speaking line of exactly **711.20 mm**, with a Z displacement
of **−6.00 mm**. Its X projection is therefore derived, never assumed:

\[
x_{\mathrm{rear}}=\sqrt{711.20^2-6.00^2}
=711.1746902133118\ldots\ \mathrm{mm}.
\]

For fret \(n\), the speaking distance and rear coordinates are

\[
s_n=711.20\left(1-2^{-n/22}\right),\qquad
(x_n,y_n,z_n)=\left(\frac{x_{\mathrm{rear}}s_n}{711.20},0,
-31.5-\frac{6s_n}{711.20}\right).
\]

Consequently, by direct substitution,

\[
x_n^2+(z_n+31.5)^2
=\left(\frac{s_n}{711.20}\right)^2
\left(x_{\mathrm{rear}}^2+6^2\right)=s_n^2.
\]

At fret 22, \(2^{-22/22}=1/2\), so both the front position and rear
three-dimensional speaking-path distance are exactly **355.60 mm**. The
generator checks these identities before creating any B-rep geometry and
writes the proof values to `cad_build/data/master_parameters.json`.

## Hardware count contract

| Item | Count |
| --- | ---: |
| Front strings | 10 |
| Rear-thumb strings | 1 |
| Total strings | 11 |
| Front tuners | 10 |
| Rear-thumb tuners | 1 |
| Total tuners | 11 |
| Front bridge pins | 10 |
| Independent rear anchor | 1 |
| Total string terminations | 11 |
| Rear-thumb frets | 22 |

Front tuner centres, in millimetres, are the Cartesian product

```text
X = (-24, -51, -78, -105, -132)
Y = (-24.5, 24.5)
```

and the independent rear tuner centre is `(-155, 0)`.

The ten compensated front bridge-pin centres are:

```text
(727.0000000000000, -38.25)
(727.5777777777778, -29.75)
(728.1555555555556, -21.25)
(728.7333333333333, -12.75)
(729.3111111111111,  -4.25)
(729.8888888888889,   4.25)
(730.4666666666667,  12.75)
(731.0444444444445,  21.25)
(731.6222222222223,  29.75)
(732.2000000000000,  38.25)
```

## Fret coordinates

The canonical full-precision table is generated at
`cad_build/data/fret_coordinates.csv`. Selected invariant stations are:

| Fret | Front X (mm) | Rear X (mm) | Rear Z (mm) | Rear path (mm) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 22.058242161422 | 22.057457165073 | −31.686093156592 | 22.058242161422 |
| 11 | 208.305657220127 | 208.298244155237 | −33.257359312881 | 208.305657220127 |
| 22 | 355.600000000000 | 355.587345106656 | −34.500000000000 | 355.600000000000 |

Decimal digits displayed by CSV/DXF/STEP serializers are rounded
representations. The defining squared-distance and octave identities above,
not equality of rounded display strings, are the mathematical invariants.

## Validation

Run the dependency-free mathematical validation and table generation with:

```bash
python 11string_22edo_acoustic_electric_FULL_CAD.py \
  --verify-only --output-dir cad_build
```

Run the complete CAD generation and OpenCascade re-import check with:

```bash
python 11string_22edo_acoustic_electric_FULL_CAD.py --output-dir cad_build
python scripts/validate_11string_cad.py cad_build
```

The validator reports the actual solid count and aggregate volume from the
generated STEP rather than embedding unverified release-specific values in
this document. It also proves:

- the STEP can be re-imported by CadQuery/OpenCascade;
- the master STEP and STL are non-empty;
- fret 22 is 355.60 mm from the nut;
- the rear fret-22 3-D distance is 355.60 mm;
- the full rear speaking length is 711.20 mm; and
- the declared string and tuner counts are internally consistent.

## FreeCAD

This runtime does not provide FreeCAD, so no native `.FCStd` file is falsely
claimed. Open the master STEP directly in FreeCAD, or execute
`FreeCAD_IMPORT_11STRING.FCMacro`, select the `cad_build` directory, and save
the imported document as `.FCStd`.
