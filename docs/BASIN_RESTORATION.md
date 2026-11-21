# Original stone basin and controls

The original basin source was recovered from the Cymatica Circular Hybrid Studio Site at source commit `dca62dc0e6890d4e0da8d80046dad5535bd4dd76`. Its stone vertex shader, fragment shader and seven material palettes agree with the earlier 6.2 and 6.6 releases. The renderer uses a solid cylindrical base, an annular wall, a water body and the computed free surface.

`faraday/scene3d.js` carries the recovered stone shaders and material palettes. It constructs the original 256-segment stone base and wall in the current scene coordinate system. The liquid surface and tracers still come from the current nonlinear solver observations. The WebGL2 and Canvas2D paths use the same frame and basin geometry. Camera moves, material changes and microscopy change the view, not the physical state.

`cymatic.html` and `faraday/basin-controls.js` restore the basin workspace and bind its controls to the current solver. The original frequency and stroke inputs use hertz and peak micrometres. The conversion to the solver's peak acceleration is

\[
a=A_{\mu\mathrm m}\,10^{-6}(2\pi f)^2.
\]

At 4,392 Hz, a 3 µm peak stroke is 2,284.5762325192195 m/s². Changing frequency with fixed stroke therefore changes the acceleration as well. Water-property controls convert displayed mPa·s and mN/m to Pa·s and N/m. The heritage-stone selection and wall/base thickness describe a rigid vessel; they do not add an unimplemented elastic stone response.

Valid changes to the physical experiment retain the user's running or paused intent. A running experiment restarts with its new parameters. Display changes leave the current computation running. The microscope crops the same computed surface at the selected physical field of view. Export captures that actual view and state.

## Numerical resolution and time

The restored original mesh is explicitly displayed as 12 radial, 24 angular and 8 vertical cells with radial and vertical stretch 2.2. It is an editable finite-grid experiment, not a spatial-convergence certificate. The automatic selection remains a separate real calculation; its criteria are not declared passed by the original mesh.

For water at 4,392 Hz, the subharmonic capillary–gravity wavelength is approximately 0.456 mm. A 30 mm diameter cup has about 32.9 such wavelengths along its radius. Twelve radial cells do not resolve that pattern. The interface reports the wavelength and mesh resolution so that this limitation is visible when choosing the grid. Recovered appearance and controls do not establish high-frequency spatial accuracy.

High-frequency forcing is integrated with a step no larger than one ninety-sixth of the drive period, also respecting the solver's stability bound. The physical time displayed is the time actually integrated. No render clock or assumed phase advances the fluid state.

The 29 book-derived polynomial methods remain in the production coefficient and operator path described in `BOOK_POLYNOMIAL_METHODS.md`. Static stencil-row addresses are compiled once for an unchanged grid. Evaluations use the same coefficients, numerical operation order and finite-value checks as before; current physical coordinates and velocities are still sampled every step. Exact coefficient arithmetic does not make the evolving binary64 fluid state an exact continuum solution.

## Source and instruction bytes

`BASIN_RESTORATION_INPUTS.json` records the two restoration messages as UTF-8 strings, their byte lengths and SHA-256 hashes, together with the recovered source commit and original source-file hashes. These hashes identify the recorded bytes. They are not signatures, a new Lean proof, or a claim that byte identity proves physical correctness. The pre-existing signed instruction artifacts are unchanged.
