# Book-derived polynomial mathematics in the simulator

The implementation uses Bharati Krishna Tirthaji's *Vedic Mathematics* as the source for the named methods and the user's authorization to devise their polynomial extensions. The exact algebra lives in `dns/vedic-polynomial.js`; `dns/vedic-stencils.js` constructs interpolation and derivative tables; `dns/vedic-operators.js` constructs moment tensors, positive quadratic forms, reciprocal stencil symbols and retained-plane quadratics. The numerical solver consumes these constructions.

The distinction between a procedure printed in the book and a derived polynomial identity is explicit below. The book does not contain a Faraday-fluid discretization, a pressure preconditioner or a convergence proof for this simulator. Those are applications developed here from stated identities and ordinary numerical mathematics. Some source methods overlap: the two deficiency-square entries share an identity, and the coefficient-sum entries are parts of one construction.

## Sources

1. Bharati Krishna Tirthaji, [*Vedic Mathematics*, primary book text](https://archive.org/stream/vedic-mathematics-bharati-krishna-tirth-ji-maharaj/Vedic%20Mathematics%20-%20Bharati%20Krishna%20Tirth%20Ji%20Maharaj_djvu.txt). Page references below are printed pages in the scanned edition. The text supplies the named arithmetic and algebraic procedures. The equations and integration design in this document also include explicitly identified derivations.
2. Bengt Fornberg, [Generation of Finite Difference Formulas on Arbitrarily Spaced Grids](https://people.sc.fsu.edu/~jburkardt/m_src/differ_arbitrary/fornberg.pdf), *Mathematics of Computation* 51 (1988), 699–706. This supplies the numerical setting of interpolation and derivative weights on distinct nodes.
3. Burhan Sadiq and Divakar Viswanath, [Finite Difference Weights, Spectral Differentiation, and Superconvergence](https://arxiv.org/abs/1102.3203). The polynomial moment equations provide an independent characterization of derivative weights.
4. Jean-Paul Berrut and Lloyd N. Trefethen, [Barycentric Lagrange Interpolation](https://doi.org/10.1137/S0036144502417715), *SIAM Review* 46 (2004), 501–517. This supplies the cardinal-polynomial interpolation setting.
5. Eitan Tadmor, [Fourier differentiation matrix](https://math.umd.edu/~tadmor/lecture-notes/hyperbolic/node17.html). This supplies the Fourier-symbol setting for periodic differentiation operators.
6. NIST Digital Library of Mathematical Functions, [Table 18.3.1](https://dlmf.nist.gov/18.3#T1) and [18.14.1](https://dlmf.nist.gov/18.14.E1). The shifted-Legendre norm and the Legendre bound are used to derive the polynomial speed bound below.
7. Seth C. Spiegel, H. T. Huynh and James R. DeBonis, [De-Aliasing through Over-Integration Applied to the Flux Reconstruction and Discontinuous Galerkin Methods](https://ntrs.nasa.gov/api/citations/20150018404/downloads/20150018404.pdf), NASA Glenn Research Center. Sections III and III.A describe polynomial flux projection and its mass-matrix construction. The shared-face finite-volume construction below is derived here; it is not identified as the paper's complete discontinuous Galerkin scheme. The paper also explains why exact flux integration alone does not establish stability for an under-resolved flow.
8. R. W. C. P. Verstappen and A. E. P. Veldman, [Symmetry-preserving discretization of turbulent flow](https://research.rug.nl/en/publications/symmetry-preserving-discretization-of-turbulent-flow/), *Journal of Computational Physics* 187 (2003), 343–368, and Oleg V. Vasilyev, [On the construction of high order finite difference schemes on non-uniform meshes with good conservation properties](https://web.stanford.edu/group/ctr/ResBriefs98/vasilyev.pdf), Stanford Center for Turbulence Research (1998). These supply the numerical principle of matching discrete adjoint symmetry to momentum and kinetic-energy conservation. The particular projected-face split and moving-volume identity below are derived explicitly for this implementation.

## Exact algebra and numerical state

An exact scalar is a reduced rational `Q(n,d)` with `BigInt` numerator, positive `BigInt` denominator and no rounding during rational operations. A polynomial is an immutable ascending vector `[c0,c1,...]`. Its represented value is

\[
P(x)=\sum_{k=0}^m c_kx^k.
\]

Zero denominators, repeated interpolation nodes and unmet method domains raise errors. A finite JavaScript number is converted to its exact represented dyadic rational by `fromNumber`; a decimal string can instead be interpreted exactly by `fromDecimal`. These are different input conventions. In particular, a binary approximation to one third does not silently become the rational `1/3`.

Coefficient generation remains rational through polynomial multiplication, division, differentiation, integration, matrix construction and positive factorization. `toNumber` performs the explicit final conversion to binary64 for the numerical solver. The evolving velocity, surface and pressure arrays remain numerical floating-point state. Exact coefficient construction does not turn time integration, trigonometric evaluation or the finite-grid solution into exact continuum arithmetic.

## Cardinal polynomials and physical derivatives

For distinct rational nodes \(x_0,\ldots,x_{n-1}\), define

\[
W(x)=\prod_{j=0}^{n-1}(x-x_j),\qquad
L_i(x)=\frac{W(x)/(x-x_i)}{\prod_{j\ne i}(x_i-x_j)}.
\]

The production construction uses primitive integer affine factors: a reduced node \(x_i=n_i/d_i\) contributes \(d_ix-n_i\). A common scaling cancels when the quotient is divided by its value at the node. One nodal-product coefficient is deliberately omitted from convolution and then constructed from the coefficient-sum identity. The quotient for a nonzero-constant affine factor is recovered from its endpoint coefficients and an interior recurrence. A nonmonic factor is normalized with the quotient scaling retained. The zero-constant factor uses polynomial division directly.

The final cardinal polynomial is constructed from

\[
L_{n-1}=1-\sum_{i=0}^{n-2}L_i.
\]

Its coefficients are not independently computed first. For derivatives, the last cardinal derivative follows from the zero aggregate of all cardinal derivatives. Distinct-node uniqueness justifies both constructions.

The generated tables supply the actual interpolation and differentiation operations

\[
f_h(x)=\sum_i f_iL_i(x),\qquad
f_h^{(r)}(x)=\sum_i f_iL_i^{(r)}(x).
\]

The solver's column interpolation and polynomial derivative paths read these tables. Derivatives of a polynomial of degree below the node count are exact before conversion to numerical state. General smooth fields still have interpolation and discretization error. The independent reference calculation solves the moment system

\[
\sum_i w_i x_i^k=\left.\frac{d^r}{dx^r}x^k\right|_{x=x_*},\qquad 0\le k<n.
\]

At an existing node, interpolation has a right-hand side equal to one Vandermonde column. `nodalWeights` uses the resulting unit vector after checking node distinctness. This is the higher-dimensional extension of the proportional-right-hand-side method; it is not a determinant-zero shortcut.

## All 29 source identities and their implemented roles

The S labels denote the sixteen principal methods. US1–US13 denote the thirteen subsidiary methods; their numeric API indices are 17–29. Formula extensions below are statements about rational polynomials. A source name does not remove a function's domain conditions.

| ID | Name and book location | Implementation and consumed result |
|---|---|---|
| S1 | Ekadhikena Purvena, pp2–4 | `ekadhikena` constructs the exact repeating block for a denominator ending in nine. `unitMoment` converts the block back to a rational; the ninth denominator occurs in the cubic triple products consumed by the projected momentum-transport flux. The half-shift square helper is also available, but is not counted as a separate production consumer. |
| S2 | Nikhilam, pp12–13 | `nikhilam(P,Q,B)` computes \(PQ=B(P+Q-B)+(B-P)(B-Q)\). `compileMoments` consumes these products for off-diagonal mass entries, with an actual leading monomial as the base. |
| S3 | Urdhva Tiryagbhyam, pp33–36 | `urdhva` forms each product coefficient by the crosswise sum \(\sum_{i+j=k}p_iq_j\). Its products construct nodal factors, derivatives and moment integrands. |
| S4 | Paravartya Yojayet, pp53–58 | `paravartya` performs exact leading-coefficient elimination and returns quotient and remainder. The cardinal construction consumes the quotient of a zero-constant affine factor and rejects a nonzero remainder. |
| S5 | Sunyam Samyasamuccaye, pp98–113 | The derived `shunyamCoefficient` aggregate construction supplies the final cardinal polynomial and its derivatives from partition of unity. Equal aggregates follow from distinct-node interpolation, not from an arbitrary numerical tolerance. |
| S6 | Anurupye Sunyamanyat, pp130–132 | The derived `nodalWeights` specialization returns a unit solution when the interpolation right-hand side is an existing Vandermonde column. The momentum-transport compiler uses that vector to evaluate the projected flux at an actual flux node. It supplies the right-hand side of the exact mass solve, so it determines the consumed bilinear face coefficients. Non-nodal field reconstruction uses actual cardinal polynomial values. |
| S7 | Sankalana Vyavakalanabhyam, p132 | `symmetricTranslate` forms the sum and difference of the two centered evaluations. Their even and odd pieces both determine endpoint mass forms; the sum also determines the integral. This parity extension is separate from the book's two-equation solver. |
| S8 | Puranapuranabhyam, pp155–156 | `translate` groups a polynomial into quadratic blocks, calls `purana` for each nondegenerate block and combines the shifted blocks with powers of \((x+c)^3\). Completed-square outputs directly construct translated integrands. |
| S9 | Calana Kalanabhyam, pp145,175–178 | `calana` differentiates coefficients exactly. Factor differentiation builds the consumed first and second derivative tables. |
| S10 | Yavadunam, pp24–27 | `yavadunam` forms \(P^2=B(P-d)+d^2\), \(d=B-P\). The diagonal mass integrands consume these squares. |
| S11 | Vyastisamastih, pp160–161 | `symmetricTranslate` omits the actual quartic coefficient from generic translation and constructs its even contribution through `vyashti`: \((A+h)^4+(A-h)^4=2A^4+12A^2h^2+2h^4\). Its odd contribution is constructed separately. Both feed the consumed moment and endpoint profiles; other powers use the stated parity extension. |
| S12 | Sesanyankena Caramena, pp195–197 | `sesanyankena` generates a complete recurring block from exact remainders. `unitMoment` first separates factors of two and five and consumes the exact recovered fraction for other antiderivative denominators. No decimal block is truncated. |
| S13 | Sopantyadvayamantyam, pp136–137 | `sopantyaProductDerivative` uses the AP identity derived below to construct the first derivative of an endpoint cubic basis. The specialization requires four exactly equally spaced nodes. The angular derivative averages two four-node integer AP stencils, so this method supplies an actual solver derivative even when radial or vertical nodes are graded. |
| S14 | Ekanyunena Purvena, pp29–32 | `ekanyunena` computes \(P(B-1)=PB-P\). `kevalaih` consumes the resulting all-nines block to construct the seventh denominator used in the moment tensors. |
| S15 | Gunitasamuccayah, pp76,81–85 | `gunitasamuccaya` computes the product of factor coefficient sums. That value supplies the actual missing nodal-product coefficient in US13. |
| S16 | Gunakasamuccayah, pp175–178 | `gunakasamuccaya` constructs product derivatives by differentiating factors and summing the resulting products. Its output is the general cardinal derivative numerator. |
| US1 /17 | Anurupyena, pp17–22 | `anurupyena` uses a real scaled working base in the product identity. Distinct-index triple products consume the output in the cubic flux tensor, which determines the projected momentum-transport coefficients. |
| US2 /18 | Sisyate Sesasamjnah, pp19,57 | `shishyateKnownLinear` normalizes an actual nonmonic divisor, obtains its quotient and scales that quotient back. The cardinal basis consumes it; the original divisor is not replaced by a different mathematical problem. |
| US3 /19 | Adyamadyenantyamantyena, pp74–75 | `adyamDivide` derives the first and last quotient coefficients from the corresponding dividend and factor coefficients, fills the interior by recurrence and rejects a residual mismatch. Nonzero-constant cardinal factors use this general polynomial extension. |
| US4 /20 | Kevalaih Saptakam Gunyat, p202 | `kevalaih` uses the exact seventh repetend construction \(1/7=143\cdot999/999999\). `unitMoment(6)` consumes it for a degree-six antiderivative coefficient. This method is specifically about sevenths. |
| US5 /21 | Vestanam, pp259–266 | `osculateEvaluate` uses the reverse recurrence and restores its power-of-node scale. Cardinal quotient normalization and derivative evaluation consume the restored polynomial value. Zero-node evaluation returns the constant coefficient. |
| US6 /22 | Yavadunam Tavadunam, pp288–289 | `tavadunamCube` computes \(P^3=B^2(P-2d)+3Bd^2-d^3\). It supplies equal-index cubic flux integrands and the cube needed to combine translated quadratic blocks. |
| US7 /23 | Yavadunam Tavadunikrtya Varganca Yojayet, pp24–27 | `extendedSquare` implements the shared deficiency-square identity. The repeated-pair branch of the cubic flux tensor consumes the square before multiplying by the third basis function. The projected momentum flux consumes these tensor entries. |
| US8 /24 | Antyayordasakepi, pp28–29 | `product` pairs actual affine factors with equal slopes and nonzero sum of constants. `antyayordasakepi` constructs their product by the complementary-parts identity below. Remaining products use convolution. |
| US9 /25 | Antyayoreva, pp138–143 | `antyayorevaProductSecond` supplies \((BCD)''=6C\) for three consecutive monic factors. The second-derivative cardinal basis consumes it in the bending Gram matrix where applicable and in the exact integer second-difference stencil used for the radial and vertical timestep bounds. |
| US10 /26 | Samuccayagunitah, p76 | `coefficientSum` evaluates scaled antiderivatives at the integration endpoints. The fundamental theorem of calculus then supplies the mass and triple-product entries used by momentum transport, along with the other exact integrals. The same method determines the endpoint energy forms from the scaled even and odd polynomials. |
| US11 /27 | Lopanasthapanabhyam, pp77–79 | `retainedQuadratic` and `retainedQuadraticNumber` apply the derived retained-coordinate-plane identity to the actual strain quadratic. General irreducible forms are supported by reconstruction; the literal rational factorizer remains a distinct restricted operation. |
| US12 /28 | Vilokanam, pp147–148 | `reciprocalPolynomial` reduces a symmetric stencil to a polynomial in \(z+z^{-1}\). `compileSymmetricSymbol` and `evaluateSymbol` supply its pressure-preconditioner symbol. Asymmetric inputs are rejected. |
| US13 /29 | Gunitasamuccayah Samuccayagunitah, pp76,82 | `gunitasamuccayaSamuccayagunitah` omits one coefficient during convolution and obtains it from the product of factor sums minus the remaining coefficient sum. The result is the nodal polynomial used by the cardinal construction. |

Conditional methods remain conditional. A near-uniform floating grid is not reclassified as an exact arithmetic progression. A zero constant is not divided by to activate an endpoint quotient formula. The specialized results and general results are both necessary mathematical branches of the same compiler.

## Derivations used in coefficient construction

### Affine complementary parts

Suppose two needed factors have the same slope \(a\) and constants \(b,c\), with \(B=b+c\ne0\). Set \(A=(a/B)x\). Then

\[
(ax+b)(ax+c)=(BA+b)(BA+c)=B^2A(A+1)+bc.
\]

`product` uses this result directly for the selected pair. It does not first compute the pair by convolution and then discard an alternative result. Negative bases and polynomial prefixes are valid in this ring extension; the decimal digit interpretation is not a required domain.

### Endpoint quotient coefficients

For \(P=(a+bx)Q\), with \(a,b\ne0\), the endpoint coefficients give

\[
q_0=p_0/a,\qquad q_{m-1}=p_m/b.
\]

The descending interior recurrence is \(q_k=(p_{k+1}-a q_{k+1})/b\). The remaining coefficient equation is checked exactly. This is a generalization of first-and-last complementary factor recovery to an arbitrary polynomial dividend. For a nonmonic factor, normalizing by its leading coefficient requires scaling the quotient by the reciprocal leading coefficient afterward.

### AP cubic derivatives without poles

Let \(A=x-a\), \(B=A+d\), \(C=A+2d\), \(D=A+3d\), with \(d\ne0\). Expansion gives

\[
CD+BD-BC-AD=d(2C+D).
\]

Therefore

\[
(BCD)'=2BC+AD+d(2C+D),\qquad (BCD)''=6C.
\]

The first formula uses the ultimate-and-twice-penultimate AP relation. The second follows from the adjacent-pair telescoping relation after multiplication removes its denominators. These are polynomial identities valid at the nodes themselves. The implementation does not evaluate singular reciprocals at the factor roots.

For the angular first derivative, the four-point stencils on `[-2,-1,0,1]` and `[-1,0,1,2]`, both evaluated at zero, are averaged. Their resulting five-point weights are `[1,-8,0,8,-1]/12`, followed by division by the actual angular spacing. The integer offsets are exact; no rounded absolute-angle arithmetic is used to infer equal spacing. The second derivative on `[-1,0,1,2]` at zero gives `[1,-2,1,0]`, whose coefficient absolute sum is four. `cell3dGridLimits` scales this dimensionless bound by the inverse square of the minimum radial and vertical spacing; its angular bound is computed separately.

### Osculation with the original value restored

For \(a\ne0\), start with \(v=p_0\) and update \(v\leftarrow p_k+v/a\) for \(k=1,\ldots,m\). The invariant is

\[
a^k v=\sum_{i=0}^k p_i a^i.
\]

The required result is \(a^m v=P(a)\). Omitting that final scale would preserve a related divisibility property while returning the wrong polynomial value. The implementation returns the restored value.

### Completed quadratic blocks and centered parity

For a nonzero quadratic leading coefficient,

\[
ax^2+bx+c=a(x-h)^2+k,\quad h=-b/(2a),\quad k=c-b^2/(4a).
\]

`translate` groups coefficients into quadratic blocks, translates each completed square and combines the blocks with powers of the translated cube. A degenerate block is handled as a linear or constant polynomial without dividing by zero.

For \(R(s)=P(1/2+s)\), define

\[
E(s)=\frac{R(s)+R(-s)}2,\qquad O(s)=\frac{R(s)-R(-s)}2.
\]

The moment and endpoints are then

\[
\int_0^1P(t)\,dt=\int_0^{1/2}2E(s)\,ds,\quad
P(0)=E(1/2)-O(1/2),\quad P(1)=E(1/2)+O(1/2).
\]

`momentProfile` consumes all three quantities. The endpoint evaluations use coefficient sums after variable scaling by one half. Neither the odd part nor its subtraction is merely recorded as a check.

For an actual quartic term `c4*x^4`, `symmetricTranslate` removes that term before applying generic completed-block translation. It uses `c4*vyashti(c,-c,0).polynomial` for the even sum and `c4*(8*c^3*s+8*c*s^3)` for the odd difference. These are the required translated quartic contributions, not an additional calculation of a result already available from the general path.

## Exact polynomial energy, flux and vertical curvature

For physical interval \([a,b]\), let \(\ell=b-a>0\) and \(t=(x-a)/\ell\). `compileMoments(nodes,a,b)` normalizes the actual nodes and constructs their cardinal polynomials on \(t\). The integration interval is \([0,1]\); the interpolation nodes may lie outside that interval when a neighboring stencil is used.

The exact coefficient arrays are

\[
M_{ij}=\int_0^1L_iL_j\,dt,\qquad
T_{ijk}=\int_0^1L_iL_jL_k\,dt,\qquad
B_{ij}=\int_0^1L_i''L_j''\,dt.
\]

The compiler also constructs endpoint matrices \(L_i(0)L_j(0)\) and \(L_i(1)L_j(1)\), endpoint interpolation vectors and mean weights. Cubic cardinal polynomials give degree-six mass and degree-nine triple-product integrands. The integration coefficients are exact rational moments; in particular the seventh and ninth denominators arise from the needed polynomial powers.

For component sample vectors \(u,v,w\) and advector samples \(q\), `energyAndFlux` returns

\[
K=\tfrac12(u^TMu+v^TMv+w^TMw),
\]

\[
F=\tfrac12\sum_{ijk}T_{ijk}(u_i u_j+v_i v_j+w_i w_j)q_k,
\]

\[
C=u^TBu+v^TBv+w^TBw.
\]

It also returns the corresponding endpoint kinetic energies and endpoint energy fluxes. These are reconstructions of the live field samples. A diagnostic energy flux is not identified as a momentum flux merely because it is evaluated during simulation.

The returned values are normalized-coordinate integrals. A physical one-dimensional energy integral is \(\ell K\). Physical second derivatives contribute \(\ell^{-4}C\) to the interval average of squared curvature, or \(\ell^{-3}C\) to its unnormalized one-dimensional integral. Radial, angular and density measures belong to the caller's actual volume convention. The viscosity-weighted vertical-curvature measure describes the vertical second derivative, not the entire vector Laplacian.

### Solver interfaces and consumers

`cell.polynomialOperators.entries` stores a compiled plan for each sigma interval. `cell.polynomialBalance(source)` contracts those plans with the current solved samples and returns the reconstructed kinetic energy, volume, advective rate, vertical energy transport and vertical viscous curvature. Vertical energy transport has units of joule-metres per second under the solver's volume-weighted convention; it is not labelled as a boundary power. Vertical viscous curvature has units of metres per second squared. The consumed top-minus-bottom endpoint kinetic-energy flux gives `verticalSurfaceKineticPower` in watts after density and cross-sectional area weighting. The worker's frame carries the balance results to the display, provenance and export paths.

`stepLimits` uses the larger of its existing sampled-velocity rate and the polynomial rate from `VEDIC_FLUID.measure(...,false)`. The mass form therefore influences actual time advancement as well as reported energy. `buildPreconditioner` compiles the actual pressure-band coefficients into the reciprocal symbol. `surfaceNormalStress` uses the retained-plane strain quadratic. `cell.polynomialSource()` serializes the exact stencil and integral coefficients so their rational construction can be examined alongside the numerical output.

### Positive forms without cancellation repairs

The mass matrix is positive definite for distinct nodes on an interval of positive length. The second-derivative and endpoint matrices are positive semidefinite. `gramFactors` computes their \(LDL^T\) decompositions using exact rationals. A negative pivot is an error. A zero pivot is accepted only when every remaining residual in that column is exactly zero. The positive mass matrix cannot accept a zero pivot.

The retained factors express each quadratic form as

\[
v^TAv=\sum_r d_r(c_r^Tv)^2,\qquad d_r>0.
\]

Only then are the factors converted to numerical coefficients. `massSquared`, `bendingSquared` and endpoint energies contract these positive weighted squares with compensated summation. They do not compute a negative rounded dense form and clip it to zero. Nonfinite contractions and unrepresentable positive weights raise errors.

### A conservative speed bound for the reconstructed polynomial

Let the reconstructed vector velocity have degree below \(n\). The shifted Legendre basis

\[
\phi_j(t)=\sqrt{2j+1}\,P_j(2t-1)
\]

is orthonormal on \([0,1]\). NIST's norm and bound give \(|P_j|\le1\), so Cauchy–Schwarz implies

\[
|p(t)|^2\le\left(\int_0^1p^2\,dt\right)\sum_{j=0}^{n-1}|\phi_j(t)|^2
\le n^2\int_0^1p^2\,dt.
\]

Adding the component inequalities gives

\[
\sup_{0\le t\le1}|\boldsymbol u_h(t)|\le n\sqrt{2K}.
\]

Thus the exact-derived mass form supplies a conservative polynomial speed estimate for the advective timestep. For four-node cubic reconstruction the factor is four. This bound applies inside the integration interval. It does not justify arbitrary extrapolation and is not a theorem of stability for every coupled pressure, boundary and time-integration operation.

## Polynomial momentum transport through shared faces

The momentum update consumes the cubic product tensor through an exact-coefficient local projection and its conservative, energy-compatible split. This replaces the arithmetic product interpolation in the transport operator. It uses the actual volume fluxes and current velocity samples in each coordinate direction. The independent energy-reporting contraction remains available, but it is not the consumer that establishes momentum integration.

Let the normal volume-flux samples be \(q_j\) at four distinct coordinates \(x_j\), and let the transported velocity samples be \(f_k\) at four distinct, possibly staggered coordinates \(y_k\). The target face \(x_*\) must be one of the \(x_j\). Normalize the positive integration interval \([a,b]\) to \(t=(x-a)/(b-a)\), and denote the corresponding cardinal bases by \(L_j\) and \(F_k\). Define

\[
q_h=\sum_jq_jL_j,\qquad f_h=\sum_kf_kF_k,\qquad
C_{lk}=F_k(x_l).
\]

Since both bases span the cubic polynomials, \(f_h=\sum_l(Cf)_lL_l\) exactly. The coefficients \(p_i\) of the \(L^2\) projection \(p_h=\Pi_3(q_hf_h)\) satisfy

\[
\sum_iM_{ri}p_i=\sum_{jl}T_{rjl}q_j(Cf)_l,
\qquad M_{ri}=\int_0^1L_rL_i\,dt,
\qquad T_{rjl}=\int_0^1L_rL_jL_l\,dt.
\]

Let \(e_i=L_i(x_*)\). `nodalWeights` constructs this vector from the exact existing-node specialization. Solve \(M\lambda=e\) with the exact positive \(LDL^T\) factors and define

\[
W_{jk}=\sum_{r=0}^3\sum_{l=0}^3\lambda_rT_{rjl}C_{lk}.
\]

Then the projected face flux is

\[
F_*=p_h(x_*)=\sum_{jk}W_{jk}q_jf_k.
\]

The sixteen coefficients are computed as reduced rationals before conversion to the numerical transport table. The runtime reads the real volume-flux and velocity arrays and contracts those coefficients. It does not construct a second diagnostic-only result in place of the flux supplied to momentum.

The implementation evaluates this basis-independent projection using a fixed cardinal basis \(\phi_i\) at the exact rational nodes \(0,1/3,2/3,1\). Its mass and triple-product tensors are constructed once per module. Exact interpolation matrices \(A_{mj}=L_j(t_m)\) and \(B_{mk}=F_k(t_m)\) map the current physical-grid samples to that basis. The target vector is formed by composing the actual-node unit weights with \(\phi_i(x_j)\). If \(R_{mn}=\sum_r\lambda_rT_{rmn}\) in this canonical basis, the same face matrix is \(W=A^TRB\). This is an exact change of polynomial basis. The canonical tensors, actual coordinate transformations, target weights and final rational coefficients are all retained for serialization; repeated exact integration on large dyadic node fractions is unnecessary.

### Constant fields and the moving mesh

For constant transported velocity \(f_k=c\), the product \(cq_h\) is already cubic. Its projection is itself, giving \(F_*=c q_h(x_*)=cQ_*\), with \(Q_*\) the actual sampled volume flux at the target face. Equivalently,

\[
\sum_kW_{jk}=e_j.
\]

The compiler checks this rational identity. The runtime evaluates the equivalent anchored form

\[
F_*=Q_*\bar f+\sum_{jk}W_{jk}q_j(f_k-\bar f),
\]

with \(\bar f\) the arithmetic mean of the adjacent transported samples. This expression preserves a constant field numerically because every deviation is exactly zero. The anchor neither limits the field nor selects a different equation. A mismatched target flux and sampled target-node flux is an error.

The sigma-coordinate relative flux remains

\[
Q_\sigma=r\,\Delta r\,\Delta\theta\,(\omega-\sigma H_t),
\]

and the existing moving-volume term \(-f\,\Delta Q_{\mathrm{mesh}}/V\) remains in the update. Replacing that term with a fixed-grid advective derivative would lose the geometric conservation relation. This construction retains the original volume flux and its mesh-motion contribution.

Two neighboring four-node flux windows, beginning two indices and one index below the face, are projected and averaged with equal weight. They use the same transported-field stencil; each projection integrates over its own window from its first flux node to its fourth flux node. These intervals are reflections of one another about the face on a uniform grid. Equal averaging preserves the constant-field identity and restores reflection symmetry for uniform periodic geometry. An interior face is a single geometric entity: both adjacent finite volumes use the same windows, samples and projection order. Its contributions therefore cancel in the volume-weighted conservative sum.

Only physical boundaries that impose zero through-flow have zero boundary flux. A zero instantaneous \(Q_*\) at an ordinary interior face does not skip its polynomial reconstruction. The projection of a higher-degree product can differ at a point from the unprojected pointwise product; that is the explicitly defined finite-dimensional projection, not a reason to conceal a coefficient or suppress an update.

### Coefficient sums in the consumed integrals

For \(A'=P\), coefficient summation after scaling gives

\[
\int_a^bP(t)\,dt
=A(b)-A(a)
=\operatorname{sumcoeff}(A(bt))-\operatorname{sumcoeff}(A(at)).
\]

`integralByCoefficientSum` uses this identity for all centered moment integrals. Thus the coefficient-sum method determines both \(M\) and \(T\). The ninth reciprocal denominator, proportional-base triple products and repeated-factor squares also determine \(T\); changing them changes the momentum flux coefficients. The existing-node method determines \(e\), the right-hand side of the mass solve.

### Conservative split and the discrete kinetic-energy identity

A projected conservative flux by itself does not guarantee the existing discrete kinetic-energy identity. Let \(C(Q)\) be the volume-weighted conservative transport matrix assembled from the projected face fluxes, with the current volume flux \(Q\) held fixed. For a face joining nodes \(L,R\), let \(a\) be its effective field-coefficient vector, including the anchored-form coefficient correction. The face contribution is

\[
C_{\mathrm{face}}=(e_R-e_L)a^T,\qquad a^T\mathbf1=Q_*.
\]

The transpose below is the transpose of this fixed-flux linear transport operator, not the nonlinear Jacobian of \(C(Q(f))f\). With all face endpoints retained in the assembly, \(\mathbf1^TC=0\). Define

\[
d=C\mathbf1,\qquad
B=\tfrac12(C-C^T)+\tfrac12\operatorname{diag}(d).
\]

Direct multiplication gives

\[
B\mathbf1=d,\qquad \mathbf1^TB=0,\qquad
f^TBf=\tfrac12\sum_i f_i^2d_i.
\]

The forward and transpose parts both consume the same projected coefficients. Their paired use removes the spurious symmetric energy contribution of the raw projected operator. It is an algebraic transport discretization, not a correction that clips the solved field or changes the measured energy afterward.

For one face, its action can be evaluated without constructing a dense matrix:

\[
B_{\mathrm{face}}f
=\tfrac12\big[(e_R-e_L)F_*-a(f_R-f_L)\big]
+\tfrac12\operatorname{diag}((e_R-e_L)Q_*)f.
\]

The transpose distributes to every transported-field sample used by that face. Omitting non-adjacent stencil samples would fail to implement \(C^T\). Partitioned execution gathers all contributing faces for each owned output row, so it reads neighboring input data without racing to write a neighboring partition's output.

Let \(N_i\) be the outward grid-relative volume-flux net and \(M_i\) the outward mesh-volume-flux net. Then \(d_i=-N_i\), and the actual transport update is

\[
V_i\dot f_i=(Bf)_i-M_i f_i.
\]

For a constant field this becomes \(V_i\dot f_i=-(N_i+M_i)f_i\), the absolute volume-flux net required by geometric conservation. Its kinetic-energy contraction is

\[
\sum_i f_iV_i\dot f_i
=-\tfrac12\sum_i f_i^2(N_i+2M_i)
=-\tfrac12\sum_i f_i^2(D_i+M_i),
\]

where \(D_i=N_i+M_i\) is the absolute volume-flux net. On a discretely divergence-free field, only the mesh-volume term remains. Cylindrical component-curvature terms retain their separately paired cancellation.

Staggered boundary nodes remain part of the algebra when a momentum control face lies between an active node and a constrained node. Their supplied values are read, while inactive output rows are not advanced. Homogeneous constrained-node values contribute no boundary energy work. The radial axis value can be a nonzero regularity reconstruction; it must not be replaced with zero to manufacture an identity. The full-node conservation identity and the boundary-work terms must be interpreted with these actual constraints.

The identities above establish the semidiscrete transport properties. Finite-grid resolution, floating-point accumulation, explicit time integration, viscous terms, pressure projection and physical boundary conditions still determine the accuracy and stability of the complete evolution. They are not a proof of exact continuum evolution or unconditional stability of the time-discrete simulation.

## Reciprocal stencil symbols

For symmetric coefficients \(a_k=a_{2m-k}\), define \(P(z)=\sum_{k=0}^{2m}a_kz^k\). Then

\[
z^{-m}P(z)=a_m+\sum_{j=1}^m a_{m+j}(z^j+z^{-j}).
\]

With \(t=z+z^{-1}\), construct \(R_0(t)=2\), \(R_1(t)=t\) and \(R_j=tR_{j-1}-R_{j-2}\). These satisfy \(R_j(z+z^{-1})=z^j+z^{-j}\). The reduced polynomial is

\[
Q(t)=a_m+\sum_{j=1}^m a_{m+j}R_j(t).
\]

For a periodic Fourier mode, the stencil symbol is \(Q(2\cos\theta)\). `compileSymmetricSymbol` constructs the rational coefficients; `evaluateSymbol` evaluates them using the numerical cosine supplied by the pressure calculation. Constant and zero stencils have their natural constant symbols. Outer symmetric zero coefficients are removed without changing the centered stencil. A nonsymmetric array is rejected instead of being symmetrized silently.

For example, \([-1,2,-1]\) gives \(Q(t)=2-t\), hence \(2-2\cos\theta\). This example is a reference check; production compiles the actual pressure coefficients.

## Retained-plane surface stress

For a homogeneous quadratic

\[
Q(x,y,z)=ax^2+by^2+cz^2+dxy+eyz+fzx,
\]

the retained-coordinate-plane identity is

\[
Q=Q(x,y,0)+Q(0,y,z)+Q(x,0,z)-Q(x,0,0)-Q(0,y,0)-Q(0,0,z).
\]

Every monomial has net multiplicity one. This works for irreducible quadratics as well as products of linear forms. It is a general reconstruction extension of elimination and retention; it does not assert that every physical strain quadratic factors over the rationals.

`quadraticValue` evaluates this identity with exact rational input conversion and a final binary64 conversion. `retainedQuadraticNumber` uses the same reconstruction on evolving floating-point state, in a fixed operation order shared with the native implementation. The six coefficients are ordered `[xx,yy,zz,xy,yz,zx]`. For the solver's strain array they are `[E0,E1,E2,2E3,2E5,2E4]`; coordinates are the components of the surface normal. The result supplies \(n^TEn\) in the surface normal stress. Finite signed, zero and irreducible inputs stay inside the supported domain.

## Validation and scope of guarantees

`dns/check-vedic-polynomial.mjs` exercises exact algebra, allowed and rejected method domains, product reconstruction and independent Vandermonde reference solutions. `dns/check-vedic-operators.mjs` compares every mass, triple-product, mean and second-derivative coefficient against independently generated symbolic-integration values for both a uniform rational grid and an irregular rational grid. It reconstructs every Gram matrix from its exact positive factors and compares live sample contractions against independently integrated polynomials.

The operator checks also perturb consumed numerical factor and tensor entries and observe the predicted output changes. Symbol checks use independent trigonometric sums. The retained-plane floating evaluator is compared with a direct matrix quadratic on a nonflat polynomial velocity/surface state and with the rational evaluator. These checks concern mathematical results and data consumption, not method-call counters.

The authenticated instruction bytes and admission journal establish the recorded instruction and handoff identity. That cryptographic guarantee has a different scope from a mathematical theorem. Lean admission of an envelope is not a Lean proof of the polynomial engine, native compiler, floating-point fluid solver or continuum solution. The exact identities, independent reference checks, and actual solver consumers are recorded separately so that none is substituted for another.

The physical pinned-boundary evolution references, including their independent coupled-mode derivation and initial-profile convergence, are documented in [PINNED_CYLINDER_REFERENCE.md](PINNED_CYLINDER_REFERENCE.md).
