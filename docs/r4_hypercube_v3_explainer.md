<!-- SPDX-License-Identifier: Apache-2.0 -->
# R4 Hypercube v3 — Sutra Mode Formulas and Benchmarks

## Overview
This update replaces the placeholder interactive HTML with a full R⁴ Tesseract Cymatic Engine that embeds explicit sutra mode algebra for **ISOLATED**, **SERIES**, **PARALLEL**, **CONCURRENT**, **INVERSE**, and **COMPOSITE** execution. The sutra core formulas are expressed in exact rational arithmetic and applied via deterministic mode operators to support hybrid quantum‑classical scheduling constraints.

## Mode Formula Canon
The engine now exposes the explicit mathematical rule set for each mode and the sutra base formula, displayed per cube selection inside the UI. This couples the mode algebra to the selected sutra and ensures all 29 sutras are executed under the same operational semantics:

- **ISOLATED**: one-step sutra application.
- **SERIES**: four-stage serial pipeline with rotation and staged r increments.
- **PARALLEL**: four rotated lanes merged by exact rational averaging.
- **CONCURRENT**: fused average of SERIES and PARALLEL outputs.
- **INVERSE**: apply sutra to inverted state and invert outputs.
- **COMPOSITE**: averaged merge of ISOLATED, SERIES, PARALLEL, INVERSE.

## Figure — Sutra Mode Pipeline (ASCII)
```
λ, r  ──►  ISOLATED  ──►  λ', r'
  │
  ├─► SERIES (4 stages with rotation) ──► λ^ser, r^ser
  │
  ├─► PARALLEL (4 lanes averaged) ──► λ^par, r^par
  │
  ├─► CONCURRENT = (λ^ser + λ^par)/2, (r^ser + r^par)/2
  │
  └─► COMPOSITE = average(ISO, SER, PAR, INV)
```

## Deterministic Algorithmic Benchmarks
The following table is a static, deterministic benchmark derived from the mode definitions in the engine. It records the number of core sutra applications and rational merges per mode (these are exact counts derived from the algorithm, not stochastic measurements).

| Mode | Core Sutra Applications | Rational State Merges | Rotation Steps |
| --- | --- | --- | --- |
| ISOLATED | 1 | 0 | 0 |
| SERIES | 4 | 0 | 4 |
| PARALLEL | 4 | 1 | 8 (rotate + unrotate) |
| CONCURRENT | 8 (SERIES + PARALLEL) | 2 | 12 |
| INVERSE | 1 | 0 | 0 |
| COMPOSITE | 10 (ISO + SERIES + PARALLEL + INVERSE) | 3 | 12 |

## Files Updated
- `docs/sutraws_interactive.html` — full engine with explicit sutra mode formulas and exact rational execution.
- `docs/r4_hypercube_v3_explainer.md` — explainer with figure and benchmarks.
