# Cymatica method lineage

## Reviewed code that is not incorporated

The 2015 first-generation CymaScope mobile app was researched because it is a
historical cymatics reference. Public descriptions state that it matched audio
to laboratory CymaScope imagery stored in memory for the first forty piano
notes; no public, suitably licensed source repository was found. It was an
image-lookup visualizer rather than a stone-to-liquid field solver, so no app
code, binary, image bank, or inferred implementation is incorporated here.

The seller-supplied `vedic-cymatic-math` repository was also inspected. Its
`chladni.py` is a 50 by 50 NumPy/SciPy square-grid finite-difference example
that chooses one eigenvector by index. It does not implement the circular
stone plate, centred generalized point load, finite-depth liquid coupling, or
free-surface boundary used by Cymatica 6.8.0. No code from that repository is
incorporated in this release.

## Historical target

The accepted historical request retained by the project specified 4,392 Hz,
one center source, fixed straight boundaries, and a visibly organized cymatic
pattern.

The early executable did not implement its stated physics. Its field was a
radial image of the form

~~~text
r = sqrt[(x-Lx/2)² + (y-Ly/2)²],
u = sin(2πfr) exp(-10r/Lx).
~~~

It omitted a wave speed, did not use the declared material constants, and did
not represent plate or free-surface boundary conditions. A later thin-plate
and acoustic-pressure version was more defensible for Chladni plates but still
was not the water-pool apparatus requested here.

## What release 6.6 carries forward

- the 4,392 Hz acceptance case;
- one exact actuator point;
- one circular stone cup plus all ten inherited straight-wall factors;
- all ten custom boundary records;
- selectable heritage-stone families;
- automatic material refresh after every physical edit;
- high-resolution export.

## What release 6.6 implements

- the metal or stone vibrating plate is replaced by a contained water surface;
- dry grains and particle transport are removed;
- the pressure point cloud, depth cage, rays, and decorative background are
  removed;
- the radial shortcut is removed;
- a fluid-loaded Kirchhoff–Love response is used only for the driven stone
  base and is coupled analytically to the water modes;
- a finite-depth capillary–gravity dispersion solve determines the wavelength;
- a damped-Mathieu estimate reports the Faraday onset;
- the response is correctly subharmonic at half the entered drive frequency;
- the 3D scene is a volumetric stone basin and water body;
- MICRO samples the same solved field at physical micrometre scale as a
  forward-ray-density shadowgraph from the Bessel gradient and Hessian, with
  direct surface relief as secondary context instead of a coloured height map
  or dense curve stack;
- the manual stripe/square/hexagonal/quasipattern menu is removed;
- forty-eight distinct angular/radial basin modes compete through a complete
  disk-integrated cubic overlap matrix; and
- static node descent is replaced by Stokes-layer-dependent node/antinode
  transport using intensity gradient and phase flux;
- the stone-base transfer is applied once inside each fluid coefficient rather
  than multiplied over the assembled water field a second time;
- strongly supercritical inputs retain every positive-growth Mathieu mode
  instead of reusing that near-onset equilibrium.

The default 4,392 Hz water wavelength is about 456 micrometres, not the
dimensionally incomplete 1/f value used by the historical visual. Because
dozens of those wavelengths span the 30 mm cup, the whole-basin view stores
the compensated field, intensity, analytic gradient, and gradient energy in
floating-point mip levels. A pixel-per-wavelength gate resolves supported
slopes and integrates the rest as RMS roughness. The GPU microscope supplies
the resolved micrometre-scale ray field.
Release 6.6 spans sixteen angular orders and three near-resonant radial roots,
retains mode-specific Mathieu growth and phase, couples the one-point elastic
base response, and evolves the nonlinear angular competition instead of
selecting a decorative planform.

## Claim boundary

Release 6.6 is the highest-fidelity method implemented in this browser product,
but it remains a reduced weakly nonlinear Galerkin simulator. It is not full
multiphase CFD, a measured ancient-stone apparatus, or a biological-cell model.
Those distinctions are retained in the interface, documentation, and runtime
diagnostics so the visual quality is not confused with experimental
validation.
