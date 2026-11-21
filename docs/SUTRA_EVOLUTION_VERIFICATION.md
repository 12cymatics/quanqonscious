# Sutra calculations in fluid evolution

`dns/check-vedic-evolution.mjs` tests the production polynomial compiler and the production JavaScript and WebAssembly fluid solvers. It does not substitute a smaller evolution model. The formula derivations and mathematical domains are recorded in [BOOK_POLYNOMIAL_METHODS.md](BOOK_POLYNOMIAL_METHODS.md).

## Construction and consumption

The transport compiler constructs the exact mass matrix and triple-product tensor for cubic interpolation. It solves the exact mass system and evaluates the resulting polynomial projection at a shared physical face. The resulting coefficients are consumed by the conservative momentum-flux divergence and its transpose in the skew transport form. The interpolation and transport tables are compiled once for the grid and read during evolution; recomputing identical rational constants on every timestep would not change the mathematics.

| ID | Production construction | Consumer affecting evolution |
|---|---|---|
| 1 | `ekadhikena` through `unitMoment` | Reciprocal ninth in the integral of the degree-eight triple-product term used by projected momentum flux |
| 2 | `nikhilam` | Off-diagonal mass integrals in the exact transport projection and velocity bound |
| 3 | `urdhva` | Polynomial products used by interpolation, derivatives and transport coefficients |
| 4 | `paravartya` | Cardinal quotients for zero-constant affine factors |
| 5 | `shunyamCoefficient` | The final cardinal basis polynomial and its derivatives from the aggregate identity |
| 6 | `nodalWeights` | Exact selection of the requested face node in the transport projection |
| 7 | `symmetricTranslate` | Centered parity construction of the mass and triple-product integrals |
| 8 | `purana` | Completed quadratic blocks used by translated integral polynomials |
| 9 | `calana` | Field derivative coefficients used by viscous and surface-stress calculations |
| 10 | `yavadunam` | Diagonal mass-integral polynomials |
| 11 | `vyashti` | Quartic contributions to translated integral polynomials |
| 12 | `sesanyankena` | Recurring-fraction integration coefficients used by mass and transport integrals |
| 13 | `sopantyaProductDerivative` | Exact arithmetic-progression derivative coefficients used on angular stencils |
| 14 | `ekanyunena` | The all-nines construction used to recover the seventh integration denominator |
| 15 | `gunitasamuccaya` | Factor-sum value used to construct a needed nodal-product coefficient |
| 16 | `gunakasamuccaya` | General cardinal derivative numerators |
| 17 | `anurupyena` | Distinct-index triple products used by projected momentum flux |
| 18 | `shishyateKnownLinear` | Quotient scaling for actual nonmonic cardinal factors |
| 19 | `adyamDivide` | Endpoint-derived cardinal quotient coefficients |
| 20 | `kevalaih` | The seventh integration denominator in mass and transport integrals |
| 21 | `osculateEvaluate` | Cardinal quotient normalization and polynomial evaluation with its original scale restored |
| 22 | `tavadunamCube` | Equal-index triple products and translated block combination |
| 23 | `extendedSquare` | Repeated-pair triple products used by projected momentum flux |
| 24 | `antyayordasakepi` | Complementary affine factor products used in the cardinal constructions |
| 25 | `antyayorevaProductSecond` | The second-difference bound used by capillary and viscous timestep limits |
| 26 | `coefficientSum` | Evaluation of the scaled exact antiderivative at interval endpoints, determining mass and transport integrals |
| 27 | `retainedQuadraticNumber` | The normal strain quadratic used by surface normal stress and pressure |
| 28 | `reciprocalPolynomial` | Symmetric pressure-stencil symbols used by the pressure preconditioner |
| 29 | `gunitasamuccayaSamuccayagunitah` | The nodal polynomial used to construct interpolation and transport bases |

Some production functions above are derived polynomial extensions of the corresponding named method. The names do not expand their valid mathematical domains. For example, arithmetic-progression identities require exactly equally spaced rational nodes.

## Independent coefficient reference

The transport-reference fixtures were generated using exact symbolic integration in physical coordinates. For flux interpolation polynomials `L`, field interpolation polynomials `F`, target evaluation vector `e`, and integration interval `[a,b]`, the reference construction was

\[
M_{ij}=\int_a^b L_i(x)L_j(x)\,dx,\qquad
W_{jk}=\sum_i (e^TM^{-1})_i\int_a^b L_i(x)L_j(x)F_k(x)\,dx.
\]

The reference directly integrates expanded physical-coordinate polynomials. It does not invoke the production Vedic arithmetic routines or its canonical-basis compiler. The suite compares every exact rational coefficient for a uniform grid, an irregular rational grid with separately staggered field nodes, and an integer flux grid with half-integer field nodes.

It also checks constant-field transport, every monomial product of total degree at most three, and a product whose degree exceeds the retained cubic space. For `q(t)=f(t)=t³` on `[0,1]`, the independently derived projection is

\[
\Pi_3(t^6)=\frac{10}{3}t^3-\frac{45}{14}t^2+\frac67t-\frac1{21}.
\]

Its values are `5/1134` at `t=1/3` and `47/567` at `t=2/3`. These differ from the unprojected point product, so evaluating only `q(target)*f(target)` cannot pass this case.

## Independent transport conservation reference

For frozen volume fluxes `q`, let `R(q)` map velocity samples `f` to the projected momentum fluxes at faces. Let `B` be the incidence matrix that computes outward integrated divergence. The production momentum operator implements

\[
F(q,f)=-\frac12\left(BR(q)-R(q)^TB^T+\operatorname{diag}(Bq)\right)f.
\]

The transpose uses the same projection coefficients. On a periodic line, `1ᵀB=0` and constant reproduction gives `R(q)1=q`. Consequently,

\[
\mathbf1^TF(q,f)=0,\qquad
f^TF(q,f)=-\frac12\sum_i (Bq)_i f_i^2.
\]

The quadratic work is zero when the frozen flux is divergence-free. These are identities of the momentum update; they do not depend on clipping the resulting velocity.

The suite builds `R`, `B` and the resulting dense reference matrix independently for an angular line. Both angular projection tensors come from direct symbolic integration in physical coordinates. The fixture supplies frozen periodic volume fluxes to the actual production momentum function and checks all three velocity components, with constant and spatially varying flux. It compares every momentum value, total momentum and quadratic work. The original three-dimensional kinetic-energy and moving-surface checks remain separate requirements.

## Corruption checks through the actual solver

The suite reads the actual module source and captures one immutable source snapshot. A test-only module loader changes a selected method's returned mathematical result. It runs the ordinary constructors and the ordinary fluid steps on a nonflat surface with cubic velocity profiles. It does not replace the evolution function.

For each of the 29 mapped constructions, the test requires either a changed evolved velocity, elevation or surface-rate field, or an explicit arithmetic rejection. A method-call count is used only to establish that a corruption was applied; a call count alone cannot pass the test.

An arithmetic rejection is required when corruption violates an exact identity. For example, changing the factor coefficient sum can make a purported cardinal divisor cease to divide the nodal polynomial. Allowing evolution after that failure would require disabling a necessary mathematical condition. The unmodified baseline must construct successfully, advance physical time and change its fields.

Perturbations are chosen to alter the result actually consumed. The aggregate construction is tested by changing its reconstructed coefficient. The coefficient-sum construction is tested by changing one summand's contribution, because adding the same constant to both antiderivative endpoint values cancels. The second-derivative test changes the absolute derivative-weight bound, because redistribution between coefficients with the same sign can preserve that bound.

Separately, the suite changes the live numerical transport coefficients in both JavaScript memory and WebAssembly memory. It requires changed momentum-advection output and changed advanced fluid fields. Both engines must agree for the original coefficients and for the deliberately altered coefficients.

## Error handling and numerical range

The native integration suite requires absent or corrupt polynomial tables and absent, uncommitted or corrupt transport tables to stop execution. A failed native instance cannot erase its error and continue; reinitialization constructs a new state. The allocation-exhaustion case requires an immediate native trap, a retained allocation error and refusal of a subsequent step. Arena exhaustion must not return an aliased substitute allocation.

The pressure-range cases use nonzero right-hand sides of order `1e-300` and `1e200`. They require finite nonzero pressure solutions and independently recompute the relative residual after scaling each residual component. This avoids a test whose own squared norm underflows to zero or overflows. JavaScript and native pressure results must agree.

Physical no-slip boundary values, array-index limits, stable timestep selection and finite resource checks are not amplitude repairs. The evolved velocity and elevation fields are not clipped to make them appear stable. Unsupported inputs and arithmetic failures must remain explicit.

## Scope

These checks establish exact coefficient identities, independent reference agreement, actual consumption of generated values and explicit failure on corrupt arithmetic. Rational construction is converted to binary64 for the numerical evolution. The checks do not establish exact continuum evolution, a mesh-convergence theorem, high-frequency resolution on an arbitrary grid, or a Lean proof of the complete simulator. Cryptographic instruction records establish the recorded bytes and have a separate scope.
