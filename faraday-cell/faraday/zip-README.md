# faraday-cell

A Faraday-wave simulation of a circular cell: a 24.25 mm disc holding 3 mm of
water, shaken vertically, forming the standing surface patterns Chladni and
Faraday drew. Two solvers and a page that draws them.

Everything here runs on `node` and a browser. There is nothing to install: no
npm packages, no build step, no compiler.

## What is in here

| | |
|---|---|
| `faraday-cell-standalone.html` | **Open this first.** One file, double-click it. No server, no siblings, nothing fetched. |
| `cymatic.html` | The same page as it lives in the repository, loading its scripts from `faraday/` and `dns/`. Needs a local server (below). |
| `dns/run-cell3d.mjs` | The three-dimensional solver in a terminal, no browser: an ASCII plan view of the surface, the energy split, and the clocks. |
| `run-checks.mjs` | Runs every suite in this directory, in one command. |
| `dns/PLAN-cell3d.md` | The working record: what each stage measured, what is still open, and every expectation measurement has overturned. |

The solvers:

| | |
|---|---|
| `faraday/kernel.js` | The linear theory: Bessel modes, viscous damping, the Kumar–Tuckerman-style threshold problem, the mode map the page draws its labels from. |
| `dns/faraday-disc.js` | A two-dimensional axisymmetric Navier–Stokes solver, **one azimuthal mode at a time**, linear in the surface amplitude. Its Floquet growth rates are the reference the three-dimensional solver is measured against. |
| `dns/faraday-cell3d.js` | The three-dimensional **nonlinear** solver: every azimuthal mode at once and coupled, in a domain that follows the surface. This is what the page draws. |
| `dns/faraday_disc.cpp`, `dns/faraday_disc.wasm` | The two-dimensional solver again in C++, compiled to WebAssembly, bit for bit against the JavaScript. `dns/build-wasm.sh` rebuilds it. |
| `dns/faraday_cell3d.cpp`, `dns/faraday_cell3d.wasm` | **The three-dimensional solver in C++**, likewise compiled to WebAssembly and likewise bit for bit — operator by operator and then over a whole drive period, 894 steps at 12 × 24 × 8, every element by `Object.is`. `dns/build-wasm-cell3d.sh` rebuilds it; neither needs Emscripten, only `clang` with the wasm32 target and `lld`. |
| `dns/faraday-floquet.js`, `dns/faraday-dns.js` | The Floquet machinery and the periodic-box solver the disc one was built from. |
| `boundary/boundarykernel.js`, `fsi/coupled-affine-benchmark.js` | A boundary-condition kernel and an exact fluid–structure benchmark in rational arithmetic. |

## Running it

**The single file.** Double-click `faraday-cell-standalone.html`. The
three-dimensional solver runs in a worker; pick it under *surface drawn from*.

**The repository page.** `cymatic.html` loads its scripts by relative path and
its compiled WebAssembly by fetch, and a page opened from `file://` has no
origin a fetch can satisfy. Serve the directory instead:

```
python3 -m http.server 8000
```

then open <http://localhost:8000/cymatic.html>.

**The terminal.** No browser needed:

```
node dns/run-cell3d.mjs
node dns/run-cell3d.mjs --nr 16 --nth 32 --nz 10 --m 4 --accel 12 --periods 4
node dns/run-cell3d.mjs --engine cpp
```

Options: `nr nth nz` (grid), `m` (azimuthal mode seeded), `rel` (seed amplitude
as a fraction of the depth), `accel` and `freq` (the shaker; with no `freq` it
drives at twice the mode's own frequency, which is the subharmonic resonance
Faraday waves live on), `periods`, `frames`, `contact` (`free` or `pinned`),
`R h rho nu gamma g`, `rStretch zStretch`, `width`, and `engine` (`js` or
`cpp`). An unknown option is refused rather than ignored.

`--engine cpp` is not a faster-but-looser mode. The C++ is a transcription of the
JavaScript — same discretisation, same flux forms, same conjugate gradient with
the same tolerance and cap, same order of operations down to the grouping of each
sum, and no fused multiply-add, because an FMA rounds once where JavaScript rounds
twice. It computes no transcendental at all: the `cos` in the drive is evaluated in
JavaScript and passed in, which is what makes bit-for-bit parity achievable rather
than approximate. Measured on the container this was built on: **3.2× at 10 × 24 × 8
and 2.7× at 16 × 24 × 10**, with identical fields. If the module cannot be loaded it
refuses and says so, rather than running the JavaScript under the C++ engine's
name and reporting a time that means something else.

**The suites.**

```
node run-checks.mjs --list      # what there is
node run-checks.mjs --fast      # everything but the three-dimensional solver gate
node run-checks.mjs             # everything
```

`dns/check-cell3d.mjs` and `dns/check-cell3d-wasm.mjs` take minutes — the first
integrates the solver over many grids to measure convergence orders, the second
rebuilds the C++ and holds it to the JavaScript over a whole drive period — which
is why `--fast` exists and why those two are the only things it leaves out. `dns/check-page.mjs` drives a real Chrome over
the DevTools protocol; it **refuses** rather than skipping if it cannot find one,
and `CHROME=/path/to/chrome` tells it where to look.

## Two things to know before you file a bug

**It does not run at real time, and it cannot.** On the cloud container this was
built on, a 10 × 24 × 8 grid takes 64.3 ms per step in JavaScript and 23.5 ms in
C++, and the step is set by the capillary limit: 1278× slower than real time for
the first and 489× for the second. The page shows
you the physics clock, the wall clock and their ratio rather than hiding the
gap; `dns/run-cell3d.mjs` prints the same three numbers. Your machine will give a
different ratio and the number it prints is measured on it, not estimated.

**The physics is in double precision on the CPU, and stays there.** No GPU on an
Intel Mac has a double-precision type — Metal has no `double`, so WGSL has no
`f64` — so the GPU is used for *drawing*, where single precision is correct
because the output is an 8-bit pixel, and never for the field.

**Drawing on the GPU.** Pick **GPU** under *draw on*. The surface is then shaded by
a WebGL2 fragment shader (`faraday/render-gl.js`) instead of a per-pixel JavaScript
loop that costs 11.6–22.2 ms a frame on the CPU this was built on — more than a
whole 60 Hz frame on its own. The page remembers the choice. The deck names the
processor that drew the frame and what it cost the page. Where WebGL2 is not
available, or the browser takes the GPU's context away, drawing pauses with GPU
still selected and the line under *draw on* says why: choose CPU to draw there, or
GPU again to retry once the context is back. It never draws with the CPU under the
GPU's name, and never switches to the CPU for you.

The two are held to the same picture: `dns/check-page.mjs` draws every view both
ways, at the page's amplitude and at full amplitude in both phases, with the
texture noise off and on, and again after the field is rebuilt, and requires every
byte to agree **to within one level out of 255**. That bound is derived rather
than tuned — the inputs are identical and the two evaluations differ by parts in
ten million, which can move a byte only at a rounding boundary and only by one
level. Because a byte check alone cannot see an error smaller than one level — a
wrong coefficient that moved pixels by half a level passed it — the values
*before* rounding are compared too, against a bound of 0.02 of a level derived
from the GLSL precision rules; measured, they agree to 5×10⁻⁴. The grain simulation stays on the CPU: it draws from one seeded random
sequence in grain order, and a parallel version would be a different algorithm.

## What this is, and what it is not

What has been measured is in `dns/PLAN-cell3d.md`, stage by stage, with the
numbers. In summary: the three-dimensional solver's discrete operators converge
at second order against calculus; its projection drives the divergence to the
solve tolerance; its advection telescopes exactly and conserves energy in the
inviscid limit; its curvature is the exact variational derivative of its own
discrete surface area; and at small amplitude its growth rates agree with the
two-dimensional solver next door — 2.06 against 2.09 on the driven Floquet
problem, and a quarter-period energy decay agreeing to −0.04%.

What has **not** been done, stated plainly because the list matters more than the
agreements:

- **No comparison against an experiment.** Not one.
- **No comparison against any other published code or benchmark.** Both solvers
  here were written by the same author, so their agreement rules out arithmetic
  slips and rules out nothing about a shared misconception.
- **The grids are coarse** — 16 × 24 × 10 to 32 × 48 × 20. At `nth = 16` the
  m = 5 mode reads 32.6% low, which is the grid and not the physics, and the
  honest remedy is more cells.
- **The surface is single-valued and the solver refuses when the layer breaks.**
  No jets, no droplet pinch-off, no air entrainment, no overturning. A real cell
  driven hard does all four.
- **Open order losses are recorded, not hidden.** The surface row's accuracy is
  limited by a reconstruction whose stencil changes between adjacent columns;
  measured orders on four probes are 2.41/1.74, 1.42/1.17, 2.35/1.99 and
  2.02/1.59 where second order is wanted throughout. The pinned contact line is
  first order at the rim. Both are written up in the plan with the fix each needs.

It is not the most accurate Faraday-wave simulation in existence and nothing here
should be read as claiming that.

## The one rule the suites are built on

A test that has never been seen to fail is not known to work. Every gate in here
was proven able to fail by injecting the defect it claims to catch **on a
disposable copy**, requiring red, then removing it and requiring green — and the
record of which defect, for which gate, is in `dns/PLAN-cell3d.md`. Several of
those injections found that a gate which had been passing all along could not
see its own defect. Nothing skips, nothing is marked expected-to-fail, and no
tolerance was widened to make something green.
