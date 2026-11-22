# Radial moving-grid mass correction

The radial momentum transport already shares the pressure-cell fluxes:
each radial velocity row receives half the flux balance of each adjacent
pressure cell. Previously its kinetic mass used a different quadrature,
`rho * rf * drf * Hr * dtheta * dsigma`. Here `Hr` is the depth interpolated
to the radial face by distance. Its time derivative did not equal the
transferred mesh-flux balance. Reducing the time step cannot remove this
spatial inconsistency on a fixed grid.

For equal radial widths `dr`, the difference between the transferred mesh
rate and the former kinetic-volume derivative is
`dr^2 * dtheta * dsigma * (Ht_right - Ht_left) / 4`.
For unequal adjacent widths `a,b`, even a spatially constant height rate
leaves `(b^2-a^2) * dtheta * dsigma * Ht / 4`. Thus a graded-grid test with
constant height rate is also necessary; a uniform-grid vertical-rise test
alone cannot detect the radial defect.

The corrected radial kinetic volume is

```
V_u(i,k,j) = 1/2 [ V_p(i-1,k,j) + V_p(i,k,j) ],
V_p(i,k,j) = rc[i] drc[i] dtheta H[i,k] dsc[j].
```

This is a transferred, lumped mass quadrature. It is not asserted to equal
the exact geometric integral between the two neighboring pressure centers.
The geometrical face depth `Hr` and the coordinate metric remain unchanged.

For arbitrary height rates, the mass quadrature satisfies

```
dV_u/dt = 1/2 [ A_left Ht_left + A_right Ht_right ] dsc[j]
         = Qmesh_top - Qmesh_bottom,
A_i = rc[i] drc[i] dtheta.
```

The same identity holds for the azimuthal and vertical transferred masses,
including the top vertical half cell. The differences are bounded by
floating-point rounding of the fluxes, rather than a mesh truncation term.
This is a semi-discrete geometric conservation identity; it does not make
the existing first-order time integrator exact.

The change must be applied to every operator using radial kinetic mass:

* Pressure: the integrated divergence `D` is unchanged and the gradient is
  still `G = -M^-1 D^T`. The pressure-work identity and the symmetric pressure
  operator therefore retain the same algebra. The physical slope transpose
  is retained before mass division, as is the native auxiliary-axis transpose.
* Advection: shared transport fluxes and centered transported velocities are
  unchanged. The radial acceleration divides by the transferred mass, whose
  derivative is exactly the mesh-flux term already present in transport.
* Cylindrical advective curvature: the pressure-cell integrated force is
  unchanged and divides by the new radial mass. Its radial/azimuthal kinetic
  power cancellation is retained.
* Viscosity: shared integrated scalar fluxes divide by the new radial mass.
  The integrated cylindrical vector source retains its old geometric
  quadrature, so its acceleration is multiplied by `M_old/M_new`. This keeps
  the scalar and vector contributions consistent with each other.
* Diagnostics and grain coupling: kinetic energy and exported native fluid
  masses use that same transferred mass.

The explicit time-step estimates include
`beta = max(1, max_radial_rows(M_old/M_new))`. Viscous and advective estimates
are divided by `beta`; capillary and gravity estimates are divided by
`sqrt(beta)`. This accounts for the increased acceleration where the new
inertial mass is smaller. It retains the existing stencil estimates and their
empirical stability tests; it is not a new spectral proof for every nonlinear
deformed configuration. No fluid field, force, material coefficient or grain
speed is clipped by this time-step adjustment.

`check-cell3d-gcl.mjs` differentiates the public kinetic energy along moving
geometry, compares it with independently constructed mesh fluxes, and checks
the transport-energy, pressure-work and cylindrical-curvature identities on
uniform and graded grids. It fails against the previous kernel: the first
case has a relative mass-rate error of approximately `9.27e-3`. The repaired
kernel gives approximately `3.11e-13` or less in these mass-rate checks.
The existing analytical physical-gradient, viscous, advection, dispersion
and step-stability tests remain required, alongside JavaScript/C++ parity.

The auxiliary axis velocity has no independent kinetic mass. The existing
canonical advection scheme's nonzero-axis boundary remainder is separate and
is still measured in `check-cell3d.mjs`. The new exact energy-transport checks
use a zero axis value; they must not be reported as an exact whole-vessel
energy theorem for arbitrary auxiliary-axis motion. Spatial convergence near
the axis also needs its own checks: on graded grids the mass ratio need not
converge at the same pointwise order there as at a fixed positive radius.
