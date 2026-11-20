# Independent reference for the pinned circular cylinder

The small-amplitude reference uses a rigid impermeable cylindrical wall and a surface pinned at its rim. These conditions act on different fields. The velocity potential has a Neumann wall condition; the surface displacement has a zero boundary trace. A pinned mode is therefore a coupled series, not a single Bessel function selected by either kind of zero.

R. Kidambi's [*Inviscid Faraday waves in a brimful circular cylinder*](https://doi.org/10.1017/jfm.2013.178), *Journal of Fluid Mechanics* 724 (2013), 671–694, identifies the coupling of Bessel modes and the resulting coupled Mathieu system. The [publisher's abstract](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/inviscid-faraday-waves-in-a-brimful-circular-cylinder/3577F7DDB647754E9AD5C1F786F82AA8) was consulted directly. The variational equations and implementation formulas below were independently derived; they are not quotations of inaccessible article equations. The Bessel product integrals and zero conventions are documented in [NIST DLMF 10.22](https://dlmf.nist.gov/10.22.E4) and [10.21](https://dlmf.nist.gov/10.21).

## 1. Kinetic mass determines the coupled spectrum

Fix azimuthal order \(m>0\), radius \(R\), depth \(h\), gravitational acceleration \(g\), density \(\rho\) and surface tension \(\gamma\). Let \(\alpha_j\) be the positive zeros of \(J'_m\), and put

\[
q_j(r)=J_m(\alpha_jr/R),\quad k_j=\alpha_j/R,\quad
d_j=k_j\tanh(k_jh),\quad b_j=J_m(\alpha_j).
\]

The radial norm is

\[
N_j=\int_0^R q_j(r)^2r\,dr
=\frac{R^2}{2}\left(1-\frac{m^2}{\alpha_j^2}\right)b_j^2.
\]

The harmonic extension satisfying no flow through the bottom and side wall maps potential trace to surface normal velocity with multiplier \(d_j\). If

\[
\eta(r,\theta,t)=\sum_j a_j(t)q_j(r)\cos(m\theta),
\]

its quadratic energies are

\[
\mathcal T=\frac{\rho\pi}{2}\sum_j\frac{N_j}{d_j}\dot a_j^2,
\qquad
\mathcal V=\frac{\rho\pi}{2}\sum_jN_j
\left(g+\frac{\gamma k_j^2}{\rho}\right)a_j^2.
\]

The pinned trace is \(b^Ta=0\). With

\[
M_j=N_j/d_j,\qquad K_j=N_j(g+\gamma k_j^2/\rho),
\]

stationarity gives

\[
(K-\omega^2M)a=\lambda b,\qquad b^Ta=0.
\]

Consequently the eigenvalues satisfy

\[
\sum_j\frac{b_j^2}{K_j-\omega^2M_j}
=\sum_j\frac{c_jd_j}{\Omega_j^2-\omega^2}=0,
\quad
c_j=\frac{2}{R^2(1-m^2/\alpha_j^2)},
\quad
\Omega_j^2=d_j(g+\gamma k_j^2/\rho).
\]

The numerator is \(c_jd_j\). Omitting \(d_j\) changes the kinetic mass and solves a different eigenproblem. `pinnedEdgeSpectrum` now retains this factor in its secular equation, modal coefficients and effective gravity-response coefficient.

Every secular weight is positive. Between consecutive free poles the secular function increases from negative infinity to positive infinity, so each interval has exactly one root. Bisection uses those intervals without skipping or renumbering modes.

The coefficient vector is

\[
a_j\propto\frac{b_j}{K_j-\omega^2M_j}
=\frac{c_jd_j}{b_j(\Omega_j^2-\omega^2)}.
\]

For \(m=0\), the zero-wavenumber constant surface mode is removed by the fixed-volume constraint. Positive derivative zeros describe the remaining volume-preserving radial modes; the angular normalization factor changes but cancels from the eigenproblem.

## 2. Extrapolation and retained finite eigenvectors

For positive capillary stiffness, the large-index secular summand is proportional to \(j^{-2}\); its omitted tail is proportional to \(1/N\). The former \(1/N^2\) correction was incompatible with the corrected kinetic mass.

`pinnedEdgeExtrapolated` uses up to four dyadic basis sizes and Richardson interpolation in \(1/N\). With frequencies from \(N/8,N/4,N/2,N\), the extrapolated frequency is

\[
f_\infty\approx\frac{64f_N-56f_{N/2}+14f_{N/4}-f_{N/8}}{21}.
\]

This cancels the first three powers of inverse basis size when that asymptotic expansion applies. A requested mode uses only levels that contain it. The output records the actual basis sequence and number of levels. The raw frequency and squared frequency are retained as `hzTruncated`, `omegaTruncated` and `w2Truncated`.

The coefficients remain the eigenvector of the finite problem. An extrapolated frequency is never inserted into its finite-basis coefficient denominators. In particular, `pinnedModalDamping` forms its weights from the retained coefficients:

\[
w_j=M_ja_j^2=\frac{N_j}{d_j}a_j^2,
\qquad
d_{\rm eff}=\frac{\sum_jw_jd_j}{\sum_jw_j}.
\]

The latter also equals the gravity sensitivity \(d(\omega^2)/dg\) of the constrained finite eigenproblem. The checks compare it with an independently perturbed eigensolve. The damping diagnostic averages its component-rate model using these same kinetic weights. That component-rate approximation is distinct from the viscous time-dependent solvers used for the evolution comparisons.

For the eight cases retained in `faraday/reference.json`, the independent SciPy calculation compares the 16/32/64/128 extrapolant with a 512/1024/2048/4096 extrapolant. The largest relative error among the first three modes is \(4.870\times10^{-6}\); the existing \(10^{-5}\) check is retained. Direct 128-term errors are approximately \(0.24\%\) to \(0.40\%\). The JSON stores case-specific values and generation provenance. Other reference tables were not regenerated from the implementation.

## 3. A surface representation suitable for initial data

Finite Neumann sums have zero radial derivative at the rim. The pinned continuum surface need not. This makes high derivatives of a truncated Neumann surface series converge nonuniformly near the corner. A nearly converged frequency alone is insufficient to establish that its sampled profile is accurate initial data for a fine-grid acceleration test.

The physical fixtures therefore use a second, independently assembled representation. Let \(\beta_a\) be positive zeros of \(J_m\), and write

\[
\eta(r)=\sum_a u_a\psi_a(r),\qquad
\psi_a(r)=J_m(\beta_ar/R).
\]

These surface functions satisfy the pinned trace. The velocity potential remains expanded in Neumann modes. The exact overlap follows from the radial Sturm–Liouville identity:

\[
I_{aj}=\int_0^R\psi_a(r)q_j(r)r\,dr
=\frac{R^2\beta_aJ'_m(\beta_a)J_m(\alpha_j)}{\alpha_j^2-\beta_a^2}.
\]

The coupled surface-basis matrices are

\[
\widehat M_{ab}=\sum_j\frac{I_{aj}I_{bj}}{N_jd_j},
\]

\[
\widehat K_{ab}=\delta_{ab}\frac{R^2J'_m(\beta_a)^2}{2}
\left(g+\frac{\gamma\beta_a^2}{\rho R^2}\right).
\]

The independent reference solves \(\widehat Ku=\omega^2\widehat Mu\). It is a coupled Dirichlet **surface** series with the correct Neumann **potential** mass. Replacing the problem by one Dirichlet Bessel zero would not produce this matrix or its eigenvalues.

For the physical test cell, the parameters are

\[
R=0.012125\ {\rm m},\quad h=0.003\ {\rm m},\quad
g=9.80665\ {\rm m/s^2},\quad
\rho=998.2041322005837\ {\rm kg/m^3},\quad
\gamma=0.07273614042160757\ {\rm N/m}.
\]

| Mode | Independent secular limit, rad/s | Surface basis 128, rad/s | Surface basis 256, rad/s | Surface basis 512, rad/s |
|---|---:|---:|---:|---:|
| \(m=3\), first | 99.7879663724 | 99.7879885553 | 99.7879691982 | 99.7879667442 |
| \(m=2\), first | 68.1933106246 | 68.1933238493 | 68.1933123103 | 68.1933108826 |
| \(m=1\), second | 115.2244025259 | 115.2244299903 | 115.2244059933 | 115.2244029997 |

The final fixtures use 512 surface terms and 8192 potential terms. They store actual coupled coefficients, independent roots, the secular-limit frequency and basis-convergence values. Eighteen independently selected overlaps were also compared with direct numerical quadrature, agreeing within \(2.6\times10^{-15}\) relative error. The normalized generalized-eigen residuals in an independent 128-term reconstruction were below \(2.4\times10^{-11}\). Increasing the potential basis from 4096 to 16384 changed those Rayleigh frequencies by less than \(3\times10^{-13}\) rad/s.

Finite Dirichlet sums still impose a particular high-derivative endpoint trace. This representation improves the resolved initial surface; it does not eliminate the need to check profile-basis sensitivity near the corner.

### Resolved initial acceleration

At a fixed 32-by-20 meridional grid, Neumann surface truncation alone gave acceleration/theory ratios 1.7133 at 512 terms, 1.2632 at 1024, 1.1407 at 2048 and 1.0757 at 4096. Raising the fluid-grid resolution while holding that initial-profile error fixed did not measure solver convergence.

With the coupled surface representation, the four original three-dimensional grids give:

| Grid \(n_r\times n_\theta\times n_z\) | 256 surface terms | 512 surface terms |
|---|---:|---:|
| 12 × 16 × 10 | 0.89180 | 0.89212 |
| 16 × 24 × 12 | 0.95482 | 0.95568 |
| 24 × 32 × 16 | 0.97962 | 0.97859 |
| 32 × 48 × 20 | 0.99811 | 0.99439 |

Both profile refinements retain the original requirement of decreasing error on every fluid-grid refinement and a final error below five per cent. The final 512-term result is within 0.57 per cent; the 256-to-512 profile sensitivity at that grid is 0.38 percentage points. No acceleration threshold was enlarged to accept the original divergent reference sequence.

## 4. Evolution comparisons

`dns/check-cell3d.mjs` seeds the actual pinned coupled profiles in the nonlinear three-dimensional solver and the independently written fixed-grid azimuthal solver. It retains checks of restoring acceleration, inviscid potential-to-kinetic energy exchange, amplitude scaling, viscous damping, the second radial \(m=1\) mode, generated nonlinear harmonics, and resonant versus detuned forcing.

The periodic-growth reference is the discrete pinned period map for the stated meridional grid, evaluated with Arnoldi iteration and checked under Krylov enlargement and timestep halving. It is not the former free-contact multiplier relabelled as a pinned result. A single-wavenumber surface-Laplacian admission test is inapplicable to a coupled pinned eigenprofile; the period-map reference instead records its actual grid, temporal resolution and convergence.

For the 12 × 8 meridional grid, azimuthal order 3, acceleration amplitude 20 m/s² and drive angular frequency \(2\times99.78796637240848\) rad/s, the independently evolved pinned period map gives:

| Steps per drive period | Krylov dimension 12 | Dimension 20 | Dimension 28 |
|---|---:|---:|---:|
| 867 | 1.5627980812 | 1.5628057703 | 1.5628057701 |
| 1734 | 1.5628001543 | 1.5628079429 | 1.5628079427 |

The dimension-20 to dimension-28 change is below \(2.4\times10^{-10}\); halving the timestep changes the multiplier by \(2.18\times10^{-6}\). The recorded reference is 1.5628079426765598. In the actual transient resonance comparison, the three-dimensional solver's third-period amplification differs from the independently written disc solver by 2.98%, with a 6.67% difference in total amplification over three periods. The existing 5% and 10% accuracy checks are retained. Both solvers grow on resonance and decay when the drive frequency is increased by 35%.

The second radial mode at azimuthal order 1 supplies a separate axis-sensitive viscous comparison. Peak-envelope fits give decay rates 0.97087 s⁻¹ for the disc and 0.97770 s⁻¹ for the three-dimensional solver on their stated coarse meshes, a 0.70% difference against the existing 5% target. This mode also uses its coupled pinned profile, not a free-wall Bessel root.

For a consistent discrete capillary-energy diagnostic, the radial part must be the quadratic form of the actual surface-Laplacian operator. With radial faces \(r_i\), cell-centre separations \(\Delta r_i\), and a pinned outer trace, that radial form contains

\[
\frac{\gamma}{2}\sum_{i=1}^{n_r-1}
\frac{r_i}{\Delta r_i}(\eta_i-\eta_{i-1})^2
+\frac{\gamma}{2}\frac{R}{\Delta r_{n_r}}\eta_{n_r-1}^2.
\]

The endpoint term is required. Using a forward-slope square at cell-centre weights describes a different quadratic form and can report artificial energy loss even at negligible viscosity. Physical damping comparisons use the energy paired with each solver's actual surface operator.

The top vertical diffusion cell also uses its physical half-cell thickness. Its upper flux is the prescribed surface derivative itself. Doubling that derivative and subtracting the lower flux before dividing by the half-cell thickness gives an incorrect factor of two. The manufactured divergence-free field \(u_r=-rz,\ w=z^2\) has \(\partial_{zz}w=2\), including at the top cell, and identifies this error without an oscillation-frequency fit.

### Resolution of the pinned damping comparison

After those independent corrections, the original coarse meshes still do not meet all of the damping accuracy targets for the pinned coupled profile:

| Meridional disc grid | Three-dimensional grid | Disc decay, s⁻¹ | Three-dimensional decay, s⁻¹ | Relative difference |
|---|---|---:|---:|---:|
| 12 × 8 | 12 × 20 × 8 | 1.022821 | 1.141992 | 11.65% |
| 14 × 10 | 14 × 24 × 10 | 1.046021 | 1.116753 | 6.76% |
| 16 × 10 | 16 × 24 × 10 | 1.053197 | 1.108827 | 5.28% |

These values are retained as evidence of the resolution requirement. Angular refinement alone increases the first discrepancy: at fixed 12-by-8 meridional resolution, 20, 24 and 32 angular cells give differences 11.65%, 12.76% and 13.91%. The angular error was partially cancelling the meridional error. A finer angular count is therefore not presented as a sufficient repair.

The physical accuracy checks use the systematically refined grid family 14 × 28 × 10, 21 × 42 × 15 and 28 × 56 × 20. All three counts scale together, with unchanged grading parameters. The original accuracy targets of nine, six and four per cent are retained, together with decreasing differences and an extrapolated difference below one per cent. Extrapolation uses the actual squared spacing parameters \(1,4/9,1/4\); an assumed geometric ratio between equally spaced grid counts is not used.

The measured refined results are:

| Meridional disc grid | Three-dimensional grid | Disc decay, s⁻¹ | Three-dimensional decay, s⁻¹ | Relative difference | Retained accuracy target |
|---|---|---:|---:|---:|---:|
| 14 × 10 | 14 × 28 × 10 | 1.046021 | 1.124016 | 7.456% | 9% |
| 21 × 15 | 21 × 42 × 15 | 1.068487 | 1.097791 | 2.743% | 6% |
| 28 × 20 | 28 × 56 × 20 | 1.076923 | 1.091741 | 1.376% | 4% |

The extrapolated relative difference is −0.1654%, within the retained one-per-cent target. The decay stays positive in both solvers and the finest three-dimensional rate exceeds three times the weighted inviscid bulk estimate, retaining sensitivity to the physical boundary layers. The earlier coarse errors remain in the table above because those meshes do not supply the same pinned-profile accuracy.

The direct JavaScript near-inviscid evolution on a 10 × 16 × 8 grid conserves total energy to −0.0495% over a quarter period and converts 99.08% of the initial potential energy to kinetic energy. This preserves the original 0.5% drift and 90% exchange bounds. Separate seeded-noise checks decay at the returned stable timestep and diverge at ten times that step, retaining the stability test on both sides of its limit.

The costly damping and resonance trajectories explicitly use the production three-dimensional Wasm integrator and pull its evolved fields before evaluating energy or modal amplitude. Restoring acceleration and near-inviscid energy exchange are also checked directly with the JavaScript integrator. Backend equality checks are additional evidence; they do not replace either comparison with the independent physical reference.

### Arithmetic-order and manufactured-boundary checks

The column reconstruction now evaluates normalized cardinal polynomials whose coefficients were constructed exactly, then rounded once for the selected numerical backend. Its floating-point operation order differs from the former direct product of Lagrange factors. The optimization check retains a bitwise comparison with a separately coded normalized Horner evaluation: that reference independently chooses the reflected column, computes its depth, searches the vertical bracket, locates the exact coefficient record by linear search and evaluates the anchored polynomial. It never calls the production column evaluator or its row-application routine.

A second comparison retains the direct Lagrange product formula. For each sampled reconstruction it checks a conditioning-dependent rounding bound rather than treating different floating-point expressions as bit-identical. If \(a\) is the chosen anchor, \(c_{ik}\) are the normalized cardinal coefficients and \(u\) is the normalized coordinate, the magnitude used is

\[
B=|a|+\sum_i|y_i-a|\sum_k|c_{ik}|\,|u|^k
  +\sum_i|y_iL_i(x)|.
\]

With binary64 unit roundoff \(\epsilon=2^{-53}\), the comparison uses \(\gamma_{8n+8}B\), where \(\gamma_q=q\epsilon/(1-q\epsilon)\) and \(n\le4\). The operation budget covers normalization, coefficient rounding, Horner products and sums, anchoring and the independent direct-product evaluation. These sampled inputs are finite normal-range values. The separate exact low-degree reproduction checks continue to establish the coefficient identities. This is a correction to an arithmetic-order reference, not a change to a physical accuracy threshold.

Manufactured zero-slope surface profiles explicitly select the free contact condition. Manufactured pinned profiles select a pinned trace. The two-condition curvature test retains both conditions independently, including its original nine-per-cent rim bound; a trailing free-condition override was removed from the loop's pinned case. All oscillation, damping, nonlinear and resonance fixtures described above use the pinned coupled modes. The separate static grid-sizing checks may use a single Dirichlet Bessel function as an analytically known curvature profile; those checks do not identify it as the coupled inviscid eigenmode and do not use its wavenumber to set a physical oscillation frequency.

## 5. Reproducing the independent surface eigenproblem

The following Python construction reproduces the mixed-basis reference. It requires NumPy and SciPy. The reference generation used SciPy 1.17.0. Its Bessel roots and generalized eigensolver are independent of the JavaScript production kernel.

```python
import numpy as np
from scipy.special import jnp_zeros, jn_zeros, jv, jvp
from scipy.linalg import eigh

m, radial_index = 3, 1
R, h, g = 0.012125, 0.003, 9.80665
rho, gamma = 998.2041322005837, 0.07273614042160757
alpha = jnp_zeros(m, 8192)
k = alpha / R
d = k * np.tanh(k * h)
b = jv(m, alpha)
N = R**2 * (1 - (m / alpha)**2) * b**2 / 2

for count in (128, 256, 512):
    beta = jn_zeros(m, count)
    edge = jvp(m, beta)
    overlap = (
        R**2 * beta[:, None] * edge[:, None] * b[None, :]
        / (alpha[None, :]**2 - beta[:, None]**2)
    )
    mass = (overlap / (N * d)[None, :]) @ overlap.T
    surface_norm = R**2 * edge**2 / 2
    stiffness = surface_norm * (g + gamma / rho * (beta / R)**2)
    eigenvalues, eigenvectors = eigh(
        np.diag(stiffness), mass,
        subset_by_index=(radial_index - 1, radial_index - 1)
    )
    coefficients = eigenvectors[:, 0]
    coefficients /= coefficients[radial_index - 1]
    print(count, np.sqrt(eigenvalues[0]))
```

Changing `m, radial_index` to `(2,1)` or `(1,2)` produces the other physical fixture profiles. For the independent secular calculation, solve `sum(b*b/(K - lambda*M)) = 0` between adjacent free poles, with `M=N/d` and `K=N*(g+gamma*k*k/rho)`. A dense constrained solve using a null-space basis for `b[None,:]` supplies a separate finite-dimensional verification of those roots.
