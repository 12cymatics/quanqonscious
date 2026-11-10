# GRVQ exact asset integration

This change makes the assessed arithmetic and geometry assets callable through
`grvq_exact_assets.py`, with exact results and receipts tied to the source bytes
used for each call. The scalar reference in `grvq_canonical.py` implements the
supplied five-factor expression. Full GRVQ evolution still requires the missing
equations listed below.

## Implemented calculations

| Asset | Executable entry point | Preserved definition |
| --- | --- | --- |
| Canonical scalar | `evaluate_canonical_field` | Both shape functions, both reciprocal factors, radial subtraction, both Vedic harmonics, and Turyavrtti modulation |
| Three specified field sutras | `evaluate_field_sutra` | Increment plus field amplitude; deficiency times radial term; complete double sum over explicit indexed components |
| Existing 29 arithmetic applications | `evaluate_book_sutra` | Actual calls to `vedic_book.evaluate`, with explicit operands and every returned coefficient |
| Existing authored graph polynomial | `apply_book_sutra_q4` | Every power and coefficient used by `vedic_field.FieldBinding.apply_graph`, on all 16 Q4 vertices |
| Existing exact edge rotations | `execute_book_q4` | Separate ordered and matched maps, every coefficient, all 64 directed Q4 adjacency occurrences |
| Complete sequential book/Q4 execution | `execute_all_book_q4` | Explicit stages 1 through 29, with the full incoming and outgoing state retained at every stage |
| Algebraic coefficient fields | `divide_k2`, `divide_c4`; field-class `inverse` methods | All rational coordinates in the original Q2, K2, and C4 fields |
| Complete four-cube cochains | `q4_diagnostics`; kernel `d0`, `d1`, `d2`, `d3` | 16 vertices, 32 edges, 24 faces, 8 cubes, and 1 four-cell, with signed boundaries |
| Complete Walsh diagnostics | `q4_diagnostics`; kernel `walsh_transform_exact`, `walsh_inverse_exact` | All 16 characters and exact reconstruction |
| Six-component Euclidean Hodge diagnostics | `evaluate_hodge6`, `evaluate_q4_hodge` | Both chiralities, exact reconstruction, all 24 input faces and all 16 local bivectors, with the source's declared averaging factors |

The integration is a source-checkout interface. Run it from the repository root;
these root modules are not installed as console entry points by the existing
package configuration. The facade accepts rational coordinate tuples for K2/C4
and constructs its own field objects, avoiding ambiguous class identities from
different package import paths.

## Canonical scalar and parameter trace

The retained formulation is

\[
F=\left(1-\frac{1}{|S_1|+\epsilon}\right)
  \left(1-\frac{2}{|S_2|+\epsilon}\right)
  \left(1-\frac{r^2}{r^2+r_0^2}\right)
  \left[\sin(k(r+\theta+\phi))+\frac12\cos(2k(r+\theta+\phi))\right]
  \left[1+\gamma\sin(\pi r\theta\phi)\right],
\]

\[
S_1=\sin\theta\cos\phi\,e^{-a_1r},\qquad
S_2=\cos\theta\sin\phi\,e^{-a_2r^2}.
\]

`grvq_canonical._literal_components` constructs these terms without replacing
the radial subtraction or changing the expression into a different formula.
`CANONICAL_EXPRESSION` retains the unspecialized symbolic expression, and each
computed `CanonicalField.expression` retains its literal factor tree.
`CanonicalField.value` evaluates exact arithmetic and exact special-function
values; other transcendental values remain symbolic. No numerical evaluation is
used to manufacture a rational answer.

| Declared parameter | Exact default | Occurs in supplied scalar expression |
| --- | --- | --- |
| `r0_squared` | 1 | Yes |
| `shape_damping_S1` | 1/10 | Yes |
| `shape_damping_S2` | 1/20 | Yes |
| `k_vedic` | 1 | Yes |
| `omega_vedic` | 1 | No time term was supplied |
| `epsilon` | 1/100000000 | Yes |
| `gamma` | 1/2 | Yes |
| `lambda_maya` | 3/10 | No coupling equation was supplied |
| `lambda_QBR` | 1/10 | No backreaction equation was supplied |
| `kappa_GR` | 1/100 | No gravitational coupling equation was supplied |

The activity report measures occurrence in the original symbolic expression.
It does not confuse that with a nonzero sensitivity at every evaluation point.
The input `t` is retained and reported, but the supplied scalar has no explicit
time dependence. With default parameters, the coordinate tuple `(0,0,0)` gives
the exact value `19999999700000001/2`.

The scalar API takes exact finite real inputs in the spherical chart
`r >= 0`, `0 <= theta <= pi`, `0 <= phi < 2*pi`, with positive `r0_squared`
and `epsilon`. It rejects binary floats, parsed text expressions, nonreal
values, and undecidable input bounds. The formula alone does not establish a
globally smooth, single-valued field at spherical coordinate identifications.

## Executing the assets

The Python interface returns an `AssetExecution` containing the actual computed
`.value` and a detached JSON-compatible `.receipt`.

```python
from fractions import Fraction
from grvq_exact_assets import (
    evaluate_canonical_field, evaluate_book_sutra, evaluate_field_sutra,
    apply_book_sutra_q4, q4_diagnostics, divide_k2, divide_c4,
)

field = evaluate_canonical_field(1, 1, Fraction(1, 2))
arithmetic = evaluate_book_sutra(2, {"a": 98, "b": 97, "base": 100})
coupled = evaluate_field_sutra(
    "nikhilam", {"value": 98, "base": 100}, 1, 1, Fraction(1, 2),
)
assert arithmetic.value == 9506
assert coupled.value == 1

mapped = apply_book_sutra_q4(
    3, {"a": [1, 2], "b": [3, 4]}, [1] + [0] * 15, strength=1,
)
diagnostics = q4_diagnostics(mapped.value)
assert diagnostics.value["walsh_reconstruction"] == mapped.value

quotient = divide_k2((1, 0, 0, 0), (1, 1, 0, 0))
assert quotient.value.coords() == (-1, 1, 0, 0)
```

The existing graph equation is

\[
\psi_{\mathrm{out}}=\psi+s\sum_{m=0}^{M-1}c_mL^{m+1}\psi,
\qquad L=\mathrm{degree}-\mathrm{adjacency}.
\]

It is identified in its receipt as an authored construction. The Q4 diagnostic
Laplacian uses `Delta = adjacency - degree`, so `L = -Delta`. Both definitions
retain their original signs. No example operand preset is selected implicitly.

For the field Urdhva operation, pass every component explicitly:

```python
crosswise = evaluate_field_sutra(
    "urdhva_tiryak", {"a": [2, 3], "b": [5, 7]}, 0, 0, 0,
    components=[[11, 13], [17, 19]],
)
assert crosswise.value == 946
```

This executes all four products. The scalar does not define the indexed
component law, so omitting the component matrix raises an error. Missing or
invalid entries raise even when the corresponding operand is zero.

The JSON interface reads a request from stdin or a file:

```sh
python grvq_exact_assets.py - <<'JSON'
{"operation":"canonical_field","arguments":{"r":1,"theta":1,"phi":{"rational":[1,2]}}}
JSON
```

Supported operation names are `canonical_field`, `field_sutra`, `book_sutra`,
`book_q4`, `book_q4_transition`, `book_q4_chain`, `hodge6`, `q4_hodge`, `q4`,
`k2_divide`, and `c4_divide`. Integers remain integers;
`{"rational":[numerator,denominator]}` denotes an exact rational and
`{"constant":"pi"}` denotes exact pi. A complete cochain mapping is encoded
as `{"cochain":[[cell_key,value],...]}`, where each cell key is an integer list.
Duplicate cells, missing cells, floats, and invalid domains cause a nonzero exit
with an error; they do not yield substitute results.

## Complete authored book/Q4 transitions

The continuation makes the other two existing `FieldBinding` maps callable
without changing their equations. `execute_book_q4` requires a named mapping:

| Mapping | Existing calculation and order |
| --- | --- |
| `laplacian_polynomial` | The full polynomial above, every power and coefficient |
| `ordered_edge_rotations` | Every coefficient, then vertex order, then adjacency order |
| `matched_edge_rotations` | Every coefficient, then the existing greedy first-fit matching order |

Both rotation maps apply the source equation

\[
(u,v)\mapsto\left(\frac{(1-t^2)u-2tv}{1+t^2},
                  \frac{2tu+(1-t^2)v}{1+t^2}\right),
\qquad t=s\,c_m.
\]

The lower-index vertex is the first component of each pair. There are 64
adjacency occurrences per coefficient: each of Q4's 32 undirected edges occurs
twice in the source. Both occurrences are executed and recorded. The two maps
have different orders on overlapping edges, so selecting matching changes the
declared map. It is not a transparent substitute for ordered rotation.

`execute_all_book_q4(stages, values, mapping=...)` requires exactly 29 stage
mappings in IDs 1 through 29 order. Each contains exactly `sutra_id`, `operands`,
and `strength`. Every operand mapping must be supplied explicitly. Optional
operands already declared by the selected book function are resolved from that
function's signature and recorded, together with the names resolved by default.
No example preset is selected by the API.

The complete input structure is checked before stage execution. Each stage then
executes the actual book arithmetic and selected field map. Its full output
becomes the next stage's input. Heterogeneous arithmetic operands remain those
explicitly supplied for that stage; the system does not infer them from the
previous arithmetic result. There is no stage fusion, early termination,
coefficient omission, or state compression.

The returned `.value` is the final sixteen-component rational state. Its receipt
retains every full incoming and outgoing state, actual arithmetic value, complete
coefficient tuple, strength, mapping identity, edge order, and squared norm. A
`BookQ4StageError` identifies a failed stage and chains the original exception;
it carries the completed stage records and the current field input. The JSON
command reports these as an error and exits nonzero. It never reports partial
execution as a successful 29-stage result.

For rational states the reported norm is `sum(value**2 for value in state)`.
The rotation paths check exact equality of this norm before and after the
transition. The polynomial path records its actual change and makes no
conservation claim. A norm equality alone cannot establish execution: identity
would also preserve it, so tests additionally check nontrivial expected outputs,
complete edge multiplicity, stage order, and the full sequence of states.

The source of these maps is `vedic_field.py`, as retained at candidate
`0f70360be1880ad91d9aad7e6cbbd5745a2ca0cd`. They remain authored book-to-field
constructions. This complete 29-book sequence does not supply the separate
canonical GRVQ geodesic evolution or the missing 26 field-coupled formulas.

## Exact Hodge calculation recovered from the existing source

The Hodge split is the one the simulator's kernel uses, the tables
`HODGE_SD` and `HODGE_ASD` of `vedic_v18.51.1_exact_phi.html`: the self-dual and
anti-self-dual triples are `(F + *F)/2` and `(F - *F)/2`.
`tests/test_grvq_q4_hodge.py` reads those tables from the page and checks them
against this module's split. The module implements the Hodge equations over
exact rational inputs; the six-face mean and the normalized pressure are its
stated conventions.

The signature is Euclidean `(+,+,+,+)`, orientation `(0,1,2,3)`, and component
order is `(01,02,03,12,13,23)`. For `F=(a,b,c,d,e,f)`:

\[
\star F=(f,-e,d,c,-b,a),\qquad
F_+=\frac{F+\star F}{2},\qquad F_-=\frac{F-\star F}{2}.
\]

Both six-component halves and the original input are retained. The associated
triples are

\[
P=\left(\frac{a+f}{2},\frac{b-e}{2},\frac{c+d}{2}\right),\quad
M=\left(\frac{a-f}{2},\frac{b+e}{2},\frac{c-d}{2}\right).
\]

Reconstruction uses all six retained values:

\[
F=(P_1+M_1,\ P_2+M_2,\ P_3+M_3,\ P_3-M_3,\ M_2-P_2,\ P_1-M_1).
\]

For complete Q4 face input, each of the sixteen vertices obtains each incident
face by clearing its two active axis bits. Both source triples are divided by
six, exactly as the archived source specifies. Their squared combined strength
is `sum(F_ab**2)/72`; strength itself is retained as an exact symbolic square
root. The source mean pressure is `sum(all_24_face_values**2)/48`, using its
explicit simulation permeability of one. Neither this scale nor the Euclidean
signature is substituted for the missing GRVQ spacetime metric or a physical
permeability.

`q4_diagnostics(..., plaquette_cochain=faces)` now also includes this full Hodge
record, alongside the existing coboundaries. The independent entry points
`evaluate_hodge6` and `evaluate_q4_hodge` return the same source-defined
calculations without requiring an unrelated scalar vertex field.

The continuation checks every basis bivector, nonzero anchors, both dualities,
orthogonality, reconstruction, all vertex/face incidences, and the declared
normalizations. These are executable checks, not a new formal proof or a
validation of the physical interpretation.

## Norms, topology, and full-state diagnostics

For `z = a+b*sqrt(2)+i*(c+d*sqrt(2))`, K2 inversion uses `z†/H(z)` with
`H(z)` retained in Q2. The rational total field norm is reported separately.
For example, `z=1+sqrt(2)` has `H=3+2*sqrt(2)` and total norm 1; interchanging
these would give the wrong inverse. C4 retains the sqrt(5) coordinate and
represents the golden ratio exactly.

The arithmetic constructors now reject float-to-rational coercion. Existing
`to_float` renderers and Torch output adapters remain explicit conversion
boundaries. An already constructed `Fraction` is accepted; its earlier origin
cannot be inferred from its value. General canonical sine and exponential
values remain symbolic and are never projected into a fixed algebraic field.

Q4 diagnostics accept optional independent `edge_cochain`, `plaquette_cochain`,
and `cube_cochain` inputs. The exact identity `d1(d0(f))=0` says something about
a gradient; it does not establish that an arbitrary independently supplied
connection has zero curvature. Nonzero boundary anchors and every basis vector
are covered by the new kernel tests. A complete one-cochain supplies all 32
canonical edges; optional reverse entries must be their negatives. Higher
cochains supply all 24 faces or all 8 cubes, including explicit zero values.

The complete Walsh transform is a diagnostic of all 16 input values. The
existing five-character training loss retains its separate definition. Its
five-dimensional projection is not substituted for the full state.

## Execution receipts and validation

A receipt includes explicit inputs, exact outputs, the named calculation,
domain and mapping details, implementation file hashes, SymPy version, and a
digest of the receipt. Integers and rationals use the shared hexadecimal exact
codec, avoiding decimal digit limits. Symbolic results use structural SymPy
representations with hexadecimal integer literals inside their Integer and
Rational nodes. This also preserves coefficients above the interpreter's
decimal-printing limit inside unevaluated formulas, while retaining literal
argument order and symbol assumptions. Incoming expression strings are never evaluated.
The hashes identify bytes and bind the record. They do not prove a physical
model or certify unexecuted functions.

The added GitHub workflow is `.github/workflows/grvq-exact-assets.yml`. It
checks out the actual candidate head, records that SHA, runs the new tests and
the existing arithmetic, field, trace, invariant, and kernel regressions, and
uploads the JUnit reports and dependency inventory. Existing aggregate Python,
JavaScript, and package workflows remain active.

`tools/check_grvq_exact_asset_gates.py` separately injects eleven defects, one per
fresh disposable copy: omitted modulation, omitted final graph coefficient,
reversed second coboundary, a lost Walsh mode, wrong norm in K2 division, and
silent float coercion, decimal-printing failure for a valid symbolic
coefficient, omitted anti-self-dual strength, omitted source normalization,
an omitted final book stage, and restarting each stage from the initial field.
Each must produce an assertion failure, with no
collection errors or skipped tests. Unchanged copies must pass before and
afterward, and every tracked working-source hash must remain identical. These
eleven checks establish the rejection behavior of their named gates; other test
cases are not claimed to have individual mutation evidence.

Tests for this task run in GitHub. Local activity is restricted to source
inspection, editing, source-byte checks, and diff review. The package's existing
test-count coverage map includes the new kernel test file; documented totals
must be taken from GitHub's measured full suite.

## Precise remaining model definitions

`grvq_canonical.require_complete_model` raises `UnresolvedGRVQModelError` with
the missing definitions. Completion requires:

1. Equations assigning roles to omega, Maya coupling, quantum backreaction at
   all required orders, and gravitational coupling.
2. The full 4-by-4 metric, its signature, its relation to the scalar, and its
   evolution law. A scalar value does not determine these.
3. Geodesic initial data and a compatible exact/symplectic evolution
   specification, plus a mathematical absorbing boundary condition tied to the
   actual evolution equation.
4. The other 26 field-coupled sutra formulas, their operand domains, and the
   stage-to-stage field and operand bindings. The existing 29 arithmetic
   applications do not supply those missing formulas.
5. The equation-to-coordinate map for deformulation and the rule defining
   indexed field components.

Existing paths outside this integration still contain different formulations.
`core/operators/grvq_ansatz.py` uses polynomial substitutes for several named
wave shapes and a capped radial factor. `grvq_stack_integration.py` uses a
different radial suppression, finite PDE evolutions, and normalized telemetry.
Those paths have not become implementations of this canonical scalar through
the addition of the reference module. Their calculations are not relabeled as
repaired or fully canonical here.

Task ownership, the pinned source commit, instruction scope, and the unavailable
external collaboration-contract verification are recorded in
`records/grvq-exact-assets-7af632a7.json`. Historical adoption records retain
their own ownership and status.
