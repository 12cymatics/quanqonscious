# Exact 29-sutra formulation: sources, domains and remaining decisions

## Scope and evidence status

This is a source inventory for the **29 vertex-field `SUTRAS[].evolve`
formulas**, not a replacement formulation. Its frozen baseline is PR169 head
`2ab11b4d33b455c37f866e39f056fdef0f88d9e5` (2026-10-29), based on
`66e3383cc716328a234ab35492a08ca75ffaaf23`. Line numbers below refer to that
baseline, before the accompanying exactness corrections. The preserved
`vedic_v18.24_full_kernel.html` supplies the formula bodies; the active
`vedic_v18.51.1_exact_phi.html` supplies the current port and assembly.

**Source inspection establishes that computations exist. It does not establish
that the assembled program implements one complete, unaltered original model.**
This inventory does not report test results. No local tests were run to prepare
it. Any later execution evidence must identify the commit tested on GitHub.

The following labels distinguish evidence:

- **Source fact:** an explicit expression, branch, constant, read or write in
  the cited source.
- **Consequence:** a mathematical or data-flow conclusion from those facts.
- **Unresolved:** a choice not settled consistently by the inspected sources.

Pinned primary sources:

- [Preserved v18.24 vertex source][legacy].
- [Active page at PR169][active].
- [VedicFormulas.lean][vedic-lean], [Vertex.lean][vertex-lean] and
  [Contracts.lean][contracts-lean].
- [PR169 kernel check][kernel-check].

## 1. Which collection of 29 is being implemented?

The vertex target can be identified without substituting a different library:
`SutraWS/VedicFormulas.lean`, lines 8–13, explicitly names the 29
`SUTRAS[].evolve` formulas in the preserved v18.24 file. The active page's
`vedicScopeNote`, line 2500, names the same target.

Other similarly named sources are not interchangeable authorities:

| Source | What it actually specifies | Limitation for this task |
| --- | --- | --- |
| `vedic_v18.24_full_kernel.html`, `SUTRAS` at 5558–6513 | The 29 individual real vertex-field evolve bodies | Contains float conversions, rounding, gates and additional coupled execution paths; the bodies alone are not the complete step. |
| `SutraWS/VedicFormulas.lean` | Identities and properties associated with those bodies and selected couplings | Does not define and prove equivalence of the complete JavaScript step. The page calls correspondence “by transcription” at 2497. |
| `SutraWS/Contracts.lean`, lines 32–35 and 117–148 | Zero-strength identity for abstract per-operator maps | The maps are deliberately left abstract. This is not a proof of the 29 concrete implementations or all their domains. |
| `vedic_trainer/vedic/kernel/sutras_canonical.py`, opening documentation | 29 weighted IDs across nine distinct maps, derived from the old strict path | Substituting these maps would replace the 29 vertex formulas. |
| `vedic_trainer/docs/SUTRA_CATALOGUE.md`, opening section and interpretation notes | 29 different unweighted primitives from `z2_primitives.py`; some truncated briefings were completed by interpretation | Different numbering and formulas; not the present vertex registry. |
| `ALL_29_VEDIC_SUTRAS.md` | The `core/operators/sutra_ops.py` collection | For example, its additive S1 and coordinate-swap S4 differ from the vertex formulas. |
| `VEDIC_SUTRAS_AUTHENTIC_COMPLETE.md` | Classical arithmetic algorithms, with a different sub-sutra numbering | Not a definition of the current 16-vertex evolution. |
| `primarysutra.py`, lines 404–412 | Sixteen main sutras | Its own documentation corrects the earlier claim that it contained 29. The root `AGENTS.md` pointer does not make it a complete 29-vertex specification. |
| `COMPLETE_SUTRA_DEFINITIONS_SUPERIOR.md`, MSTVQ section | A stress/tension pipeline for `core/operators/mstvq.py` | Different state and equations: stress multiplication, phase rotation, suppression and weight adjustment. It does not resolve the vertex page's λ feedback. |
| `GRVQ_STACK_INTEGRATION.md` | A proposed PDE/telemetry integration using other modules | Does not specify the active vertex step or resolve its source conflicts. |

The repository also cites an *Exact Kernel Evolution Blueprint*, 31 July 2026,
in `vedic_trainer/vedic/kernel/k2_field.py` and `sutras_canonical.py`. A standalone
copy was not found in the inspected text/Markdown inventory. The citations are
evidence for those modules' stated coefficient-field policy, not a recovered
complete vertex-coupling specification.

## 2. All 29 vertex operators: preserved identities and source locations

Names below reproduce the preserved `SUTRAS` entries. Operation descriptions
identify the bodies; they do not replace their expressions, ordered loops,
branches or operands. All original locations are in [v18.24][legacy]; all port
locations are in [the PR169 active page][active].

| ID | Preserved name | Body identified by | Original entry starts | Port case starts |
| --- | --- | --- | ---: | ---: |
| S1 | Ekādhikena Pūrveṇa | In-place previous-neighbor multiplicative factor; GRVQ λ0 side effect | 5565 | 789 |
| S2 | Nikhilam | Radix-based affine reflection; MSTVQ B side effect | 5596 | 798 |
| S3 | Ūrdhva-Tiryagbhyām | XOR convolution relaxation; MSTVQ B side effect | 5629 | 805 |
| S4 | Parāvartya Yojayet | Hamming-layer coefficients evaluated by Horner | 5659 | 810 |
| S5 | Śūnyam Sāmyasamuccaye | Complementary-pair relaxation toward negative partners; GRVQ λ1 | 5692 | 822 |
| S6 | Ānurūpye Śūnyamanyat | Ordered edge-ratio correction | 5723 | 829 |
| S7 | Saṅkalana-Vyavakalanābhyām | Ordered symmetric/antisymmetric edge update | 5758 | 844 |
| S8 | Pūraṇāpūraṇābhyām | In-place neighbor-square completion | 5795 | 858 |
| S9 | Calana-Kalanābhyām | Four-neighbor graph Laplacian; GRVQ λ2 | 5825 | 866 |
| S10 | Yāvadūnam | Radix-base deficiency squared in a multiplicative factor | 5855 | 871 |
| S11 | Vyaṣṭisamaṣṭiḥ | Hamming-layer and whole-field mean relaxation | 5878 | 880 |
| S12 | Veṣṭanaṃ | Signed digital-root relaxation | 5909 | 890 |
| S13 | Sopāntyadvayamantyam | Complementary-pair numerator continued fraction and ratio correction | 5931 | 898 |
| S14 | Ekanyūnena Pūrveṇa | In-place inverse-magnitude neighbor factor; GRVQ λ3 | 5974 | 789 |
| S15 | Guṇitasamuccayaḥ | Axis-wise product-of-sums minus sum-of-products residual | 5998 | 915 |
| S16 | Guṇakasamuccayaḥ | Extended Euclid on pair numerators; gcd/Bézout branches | 6036 | 932 |
| S17 | Ānurūpyeṇa | In-place neighbor gradient weighted by the declared φ operand | 6088 | 950 |
| S18 | Śiṣyate Śeṣasaṃjñaḥ | Hamming-index residue checks and residue shift | 6113 | 958 |
| S19 | Kevalaih Saptakam Guṇyāt | Complementary-root discriminant and magnitude-ratio update | 6141 | 966 |
| S20 | Vilokanam | Integer residue correction toward a base multiple | 6189 | 983 |
| S21 | Guṇitasamuccayah Samuccayaguṇitaḥ | Last-digit osculation and modular correction | 6211 | 991 |
| S22 | Dvandvayogaḥ | In-place complementary averaging | 6238 | 1000 |
| S23 | Antyayordaśake'pi | Complementary terminal-digit sum correction | 6258 | 1004 |
| S24 | Lopanasthāpanābhyām | Tail gcd branch or fractional-part/digital-root correction | 6282 | 1012 |
| S25 | Samuccayaguṇitaḥ | Products of axis half-sums; logarithmic comparison and push | 6329 | 1023 |
| S26 | Dhvajāṅka | Maximum-magnitude flag and portion relaxation | 6378 | 1048 |
| S27 | Gunitasamuccayah | Ordered path interpolation, then untouched-neighbor interpolation | 6405 | 1055 |
| S28 | Antyayoḥ Eva | Ordered edge transfer from difference of squares | 6456 | 1078 |
| S29 | Conservation | Mean relaxation with the additional 29/145 weight; MSTVQ B | 6489 | 1086 |

S1 and S14 share a port code block but select different rational factors. Their
sharing a block is not by itself replacement by one generic operator. Conversely,
counting 29 IDs alone is not evidence that 29 different formulas execute.

## 3. What the sources settle sufficiently for exact implementation

### 3.1 Vertex arithmetic and operands

`SutraWS/Vertex.lean` defines Ψ as16 rationals, λ as four rationals, and

\[
\Psi_i=\sum_{k=0}^{3}s_{ik}\lambda_k,\qquad
\lambda_k=\frac1{16}\sum_{i=0}^{15}s_{ik}\Psi_i,
\quad s_{ik}=\begin{cases}1&i\text{ has bit }k,\\-1&\text{otherwise.}\end{cases}
\]

The literal arithmetic, integer floors/remainders, gcd, fixed loop order and
decimal rational operands can be represented without floating-point state or
precision truncation. Preserve in-place reads where the original uses them;
using the entry-state snapshot everywhere changes S1/S6/S7/S8/S14/S15/S17/S22/
S25/S27/S28 behavior.

The normalized strength convention is `alpha = id/435 * strength`, with
strength1 corresponding to the old percentage100. S29 additionally multiplies
by 29/145 = 1/5. This is its defined operand, not a storage approximation.
`Contracts.lean` proves the 435 weight sum; it does not authorize replacing each
individual map by an abstract family map.

### 3.2 S13: the zero-denominator refusal is explicit

**Correction to the earlier review:** original S13 does have an explicit
zero-denominator branch. At [v18.24 line 5941][s13-original] it says:

```js
if (den === 0n) throw new RangeError('S13 continued fraction denominator is zero');
```

PR169's active line 902 replaces this with `continue`. The source-backed
restoration is an explicit failure naming S13, the complementary pair and the
denominator, with no committed replacement state. No new mathematical extension
or user selection is needed to restore this original behavior.

S13 computes the continued fraction of the **absolute normalized numerators**
of the two stored rationals, for at most eight terms, then compares the resulting
convergent to the actual signed rational field ratio (5938–5962). The eight-term
limit and numerator operands are part of this source definition. Including the
rational denominators in the continued-fraction input or extending its term
count would change that definition; it is not merely removing arithmetic
roundoff.

### 3.3 Original conditional branches must remain distinguishable from new skips

| Case | Source-defined behavior | Exact implementation consequence |
| --- | --- | --- |
| S6, 5737 | No correction when the base vertex is zero | Evaluate and record this piecewise branch; do not borrow it to redefine S13. |
| S19, 6153–6177 | Applies the ratio correction only when both magnitudes are positive | A zero-root pair takes the original no-correction branch. For real rational roots the discriminant is `(x-y)^2`; its negative branch cannot be reached by exact valid inputs. |
| S25, 6345–6350 | A zero low half-sum stops its product; zero high half-sums are omitted; the defined `logRatio` becomes zero when the product condition fails | Preserve the stated piecewise expression if this original body is selected. It is not the unrestricted logarithm of the product of all half-sums. |
| GRVQ walk, original `walkMeasure`; active 663–668 | If both neighbor weights are zero, split the remaining mass equally | The fallback is an explicit definition, not an inferred epsilon. `walk_measure_total_split` proves its mass identity only. |
| S15, 6015; S24, 6311; S27, 6431 | Gates at `1e-6`, `0.01` and `1e-10` respectively | These can be exact rational comparisons. Replacing them with `!=0` changes the piecewise maps. |
| S25, 6350 | Requires `abs(logRatio) > 0.001` | Merely comparing the two product magnitudes preserves the sign for nonzero products, but discards this nonzero deadband. |

A formula may evaluate to its input on a valid branch or fixed point. Coverage
must record evaluation and branch selection, not demand a nonzero displacement
for every formula on every state. An undefined required expression is a failure,
not a valid no-op branch.

### 3.4 Literal constants versus exact named mathematical constants

Original `C` at 932–987 defines the sutra φ operand as
`12586269025/7778742049`, and its purported inverse as that value minus1. The
active `PHI_Q`/`PHI_INV_Q` at 645–646 preserve those literal rationals. In contrast,
active `A4.PHI` at 583 is exactly `(1+sqrt(5))/2` and is used in other components.

**Consequence:** exact evaluation of the rational representative does not make
it the algebraic golden ratio. Its square is not exactly itself plus1, and its
product with the declared `PHI_INV_Q` is not exactly 1. Substituting `A4.PHI` into
the sutra body changes the literal model and requires extending every dependent
operation's domain, including ordering, floors and numerator-based operations.

The original also declares `sqrt2=577/408`, `sqrt3=97/56`, and `pi=355/113`.
These are exact rational operands but not the named irrational constants.
`sutras_canonical.py`, 139–148, explicitly distinguishes rational representatives
from algebraic constants. A coefficient manifest must make this distinction
visible rather than claiming both meanings simultaneously.

### 3.5 Which coefficient table actually enters the runtime?

The user's follow-up asks to find and implement the correct coefficients.
For the selected vertex target, code binding resolves the common 29 weights:

\[
w_n=\frac{n}{435},\qquad
\alpha_n=w_n s_n g,\qquad
\sum_{n=1}^{29}w_n=1.
\]

Here `s_n` is normalized strength and `g` is the explicitly selected execution
gain (for example, the half-pass gain in the symmetric mode). The source chain
is `ALPHA.computeQ`, 732–734, to `SUPPRESS.alpha`, 5533–5536, to
`SUPPRESS.alphaPhi`, 5538–5539, to the individual evolve body. `alphaPhi` returns
`alpha` unchanged. The port's `alphaOf`, 1094, implements that chain directly.
S29 then multiplies by 29/145 = 1/5, making its full-strength, full-gain field
weight exactly 1/75.

Two other tables must not be mistaken for these bound operands:

- Original `SUTRA_COEFF`, 3629–3660, is a different φ-based table of 29 values. The only
  other occurrence of that identifier in the old HTML is a comment on S5;
  there is no executable read of the table. Its S11 rational is
  `12586269025/10000000000`, although its comment names `sqrt(phi)` and gives a
  different decimal. Repairing that comment/table would not change this runtime.
- Trainer `COEFFICIENT`, 149–179, is copied into `Sutra.coefficient` metadata at 197.
  The actual trainer `alpha`, 244–252, and `apply_sutra`, 364–379, do not read it.
  Its entries also differ from the original HTML table. There is no import of
  either table into the active page. Correcting trainer metadata cannot be
  reported as correcting the active 29 evolution.

The concrete coefficients below are already present in the PR169 selected
expressions. They can be pinned and checked as exact source operands; no new
physical derivation has been established for their numerical choices.

| Bound operand | Exact value | Source expression |
| --- | --- | --- |
| Common sutra weight | `n/435` | Original 732–734, 3553–3557; port 1094 |
| S29 additional mean-mixing weight | `1/5` | Original 6492; port 1087 |
| GRVQ lazy-walk idleness | `1/2` | Original 1143; port 659 |
| GRVQ neighbor-mean Laplacian multiplier | `1/20` | Original 1363; port 698 |
| GRVQ Ricci multiplier | `1/40` | Original 1366; port 698 |
| GRVQ deficiency term multiplier | `3/10` | Original 1288; port 686 |
| GRVQ phase-cycle multiplier | `7/10` | Original 1308; port 691 |
| GRVQ opposite-neighbor decrement numerator | `2/5` | Original 1323; port 693 |
| GRVQ neighbor-product residual multiplier | `4/5` | Original 1343; port 694 |
| GRVQ neighbor-square difference multiplier | `1/1000` | Original 1371; port 700 |
| GRVQ mean-restoring multiplier | `(9/10)*(1/20)=9/200` | Original 1356, 1372; port 695, 700 |
| MSTVQ isotropic tensor subtraction | `1/2` | Original 1915–1926; port 706–707 |
| S2 B strength | `alpha_2/10` | Original 5616; port 803 |
| S3 B strength | `alpha_3/100` | Original 5647; port 808 |
| S29 B strength | `abs(alpha_29/5)` | Original 6492, 6506; port 1087–1089 |
| MSTVQ λ feedback coefficient after removing numeric rounding | `1/200000000` | Original 1960–1962; port 708, 1141 |
| MSTVQ trace to TGCR κ | `1/500` | Original 1951; port 708, 1649 |

The λ feedback value follows the **complete** original expression:
`Tkk * 0.000005 * 1000 / 1000000 = Tkk / 200000000` before its integer
rounding. Using only the temporary variable named `contribution` would yield
`Tkk/200000`, a thousand-fold different update. The inspected source does not
give an independent physical derivation authorizing that change.

The original GRVQ object also contains floating members such as `g`, `r`, `q`
and `beta`. The implemented terms use the explicit rational multipliers above;
in particular the source itself says `r` is a legacy display coefficient at 1139.
MSTVQ's physical `mu0`/`B_tensor`/`h_m` values are explicitly not used by
`computeExact` (1860–1895). Adding those constants to the tensor would select a
different model, not fix a read operand.

One unresolved old coefficient conflict is precise: the USO headers at 3568–3573
and 5204–5208 state `C_n = 1 + (n/435)(P-1)/S`, while `USO.init`, 3584, sets
`gammas[n]=1/n` and `concurrent`, 3607–3610, uses those gammas. The active vertex
field does not consume this USO. Selecting the header equation for a restored
USO has the exact implementable coefficient `n/435`; claiming the old executable
already used it would be false.

The older `GRVQ FORMULA ALGO (1).TXT`, 434–468, instead introduces ansatz
coefficients refined recursively and normalized with an epsilon. It supplies no
fixed 29-entry coefficient table for the present vertex bodies. These historical
ansatz coefficients must not be substituted for the bound vertex weights.

## 4. Couplings: baseline data flow and a recoverable legacy pipeline

### 4.1 Source-proved behavior of PR169

| Computation | Baseline path | Consequence / remaining question |
| --- | --- | --- |
| GRVQ special contributions | `applySutra`, 1095–1103, adds to local λ for S1/S5/S9/S14; calls `vedicField` without λ | These contributions affect the λ register and later register calculations, not Ψ. |
| Other GRVQ shares | `distributedGRVQ`, 1127–1135, adds a 1/N share for other active IDs | Same local-register destination. |
| λ persistence | `stagesFrom`, 1593, reconstructs λ from input Ψ; `lambdaReg` is returned only through the last-step report at 1657 | Updated λ is not an independent state input of the next step. |
| S2/S3/S29 B | Their cases return B; `applySutra` records it at 1102 | `distributedMSTVQ` computes a fresh B and does not read these recorded B values. |
| Distributed stress | 1137–1143 uses only S1 strength, computes `T_ij=B_i B_j-delta_ij B_squared/2`, adds `T_kk/200000000` to local λ | This stress is zero when S1 is absent, even if a B-setting sutra is active. This selectivity also appears in the old distributed call; changing it requires a defined coupling rule. |
| Separate MSTVQ field stage | 1437–1441 uses Wheeler `magneticTraceTotal` in a complementary rotation | It is not a consumer of the above sutra-derived B or T. |
| TGCR κ | 1649 retains `κ + trace(T)/500`;1659 uses κ in geometry/energy diagnostics | `tgcrTau`, 1445, does not use κ. This alone does not prove an error: the old `computeEnergy`, 2682–2687, likewise explicitly uses κ in energy. No inspected source mandates inserting κ into the Ψ rotation. |

The source gives exact rational expressions for the selected GRVQ terms
(`evolveExact`, 1359–1373; port `grvqTerms`, 696–702), the MSTVQ tensor and these
feedback coefficients. Computing those expressions exactly is feasible on
valid rational inputs. **Which state consumes them and when is a separate
equation/assembly requirement.** A diagnostic need not force Ψ to change in
order to be a legitimate calculation.

### 4.2 Original non-strict execution order: concrete candidate source

The following pipeline is present in the original file and can be transcribed
and reviewed as a complete candidate. It is **not currently the enabled old
runtime**: `STRICT_SUTRA_ONLY=true` at 466 selects the other branch at 7670–7688.

| Order | Original operation and location | State effects / exactness work still needed |
| --- | --- | --- |
| 1 |`LeanState.syncFromLambda`; `VTX.forward`, 7663–7668 |Persistent λ is projected to Ψ before the branch. The projection retains only four axis modes, not an arbitrary previous 16-component field.|
| 2 |USO injection, 7692–7705 |Adds to Ψ using USO.P/S, float rounding and a magnitude gate.|
| 3 |Temporal sutra dispatch, 7714–7751 |Calls each selected original `evolve` only on its firing frames; may scale the displacement. `TEMPORAL`, 3404–3446, defines the phase schedule and fractional-power amplitude.|
| 4 |Distributed GRVQ, 7770–7783 |Updates λ sequentially with 1/N shares.|
| 5 |Distributed MSTVQ/Wheeler/TGCR, 7785–7796 |Recomputes B/T using S1 strength; feeds λ and κ; changes `vortexOmega` from Wheeler.|
| 6 |MAGFIELD, 7802–7806 |Updates magnetic memory, applies Lorentz changes to Ψ, computes reconnection and calls its MSTVQ feed.|
| 7 |ENGINE, 7810–7814 |Syncs from Ψ, accumulates sutra displacement, evolves 8-component state, syncs back; rebuilds Hamiltonian every 15 cycles.|
| 8 |JET and WORMHOLE_PROTO, 7816–7823 |Additional defined subsystems; their state and activation rules belong to this candidate pipeline.|
| 9 |`VTX.inverse`, 7827 |Reconstructs λ from the resulting Ψ.|
| 10 |Oscillatory λ driver, 7829–7849 |Changes λ after inverse, so these changes can enter the next frame's forward transform.|
| 11 |CASCADE, time, R4 and USO, 7852–7861 |Updates further state and derived quantities on their specified schedule.|
| 12 |TGCR, 7866–7876 |Updates bond-current vortex/energy every five cycles.|
| 13 |MSTVQ, 7880–7889 |A second stress/κ/λ feedback pass every eight cycles, after inverse.|
| 14 |ANSATZ, 7892–7898 |Energy/parameter/λ feedback on the 30/60-cycle schedule.|

Two details prevent treating λ as automatically disposable merely because an
inverse appears later in the frame:

- `ENGINE.rebuild`, 1843–1845, reads modified λ through `computeGvals`, 1562–1578,
  before inverse. The resulting Hamiltonian is retained and is consumed by
  `evolve`/`syncToVTX`, 1760–1790.
- Driver, MSTVQ and Ansatz changes after inverse can survive into the next
  frame's `VTX.forward`.

This candidate is not a ready-made exact physical specification. It contains
rounding, finite matrix-exponential series, normalization, thresholds and
trigonometric/transcendental operations. Its GRVQ header at 1126–1136 also states
that earlier internals were replaced by catalogue terms. Selecting this source
must therefore identify whether those literal discrete algorithms, or an earlier
named continuous formulation, are required.

For example, old `ricciTerm` returns `snapKappa(...)` on a `10^-9` grid at 1237–1271.
The active port computes the unsnapped rational curvature. Removing that
representation approximation preserves the displayed curvature equation, but
cannot be described as bit-for-bit agreement with the old rounded program.

There is also a concrete old write/read mismatch to resolve explicitly:
`MAGFIELD.feedToMSTVQ`, 2344–2355, assigns `MSTVQ.B`, whereas `MSTVQ.compute`,
1932–1935, calls `computeExact`, 1916–1929, which reads `B_exact`. Copying the
caller and assuming its comments prove the coupling would retain this mismatch.

## 5. Remaining requirements that need an explicit source selection

These are bounded decisions, not permission to supply new equations. Existing
user instructions selecting a source take precedence; do not ask again where
that selection is already established.

| Requirement | Available source choices / evidence | What must be recorded before claiming complete fidelity |
| --- | --- | --- |
| Field formula target | The 29 vertex bodies are the current declared target; other registries of 29 differ | Pin these bodies and their operand definitions. Do not substitute the strict generic families or unweighted primitives. |
| Meaning of “exact φ” and other constants | Literal rational `C`/`PHI_Q` versus algebraic `A4.PHI`; different modules already use different domains | State which mathematical constant each dependency requires. Do not silently replace a rational representative or label it algebraically exact. |
| Piecewise thresholds | Original finite thresholds versus the active page's exact-nonzero replacements | Retain the original thresholds for a literal port, or cite an instruction explicitly selecting a changed map. S25 needs its full logarithmic deadband, not sign alone. |
| λ/Ψ state ownership | Old persistent λ with forward/inverse projection versus active persistent Ψ with derived λ and local feedback register | Select the complete update equations and timing. `inverse(forward(λ))=λ` does not imply `forward(inverse(Ψ))=Ψ` for arbitrary Ψ. |
| Coupled assembly | Old non-strict pipeline above versus PR169 assembly | Identify required components, equations, read/write destinations, ordering and cadence. Neither switching the old flag nor inserting a guessed λ→Ψ term proves equivalence. |
| MSTVQ source/overwrite semantics | Per-sutra B, distributed S1 B, MAGFIELD B/B_exact, second periodic feedback | Define which B feeds which T and which updates persist. Preserve source coefficients; do not sum/replace channels merely because their names match. |
| Initial condition | Old λ=(3, 4, 5, 6) forwarded to Ψ; active `initState` adds `1/(1+(hw-2)^2)` | Record the selected initial condition. The added dielectric component changes Ψ while leaving the four inverse axis modes unchanged; agreement of λ alone cannot verify the seed. |
| Schedule and all 29 | Old temporal firing; active SERIES/PARALLEL/SYMMETRIC_CONCURRENT plus strength glide | Require the selected mode's full 29 evaluation schedule. Presence in a registry or a zero-strength entry is not execution evidence. Do not require nonzero output at valid fixed points. |
| Transcendental and continuous semantics | Active Maya note at 2498 specifies five Euler substeps and finite rational enclosures; legacy engine has a finite exponential series | Exact arithmetic for a discrete update is not an exact solution of a continuous differential equation. Keep the selected exact expression; do not call an enclosure midpoint or truncated series the original exact value. |
| Singular/unsupported inputs | S13 has a source-defined error; other original piecewise branches are explicit | Report the exact domain failure. Do not invent epsilon offsets, borrowed skips, identity substitutions or replacement seeds. |

No inspected document combines these choices into one consistent, complete
original 29-sutra coupled specification. This is a bounded statement about the
sources listed here, not a claim that no such specification exists elsewhere.

## 6. Smallest implementation that can make honest progress now

1. **Pin a formula manifest.** Record all 29 IDs, original entry locations,
   source commit, literal constants, branch conditions, operand types, in-place
   order and the chosen execution mode. The vertex identity itself is already
   established by `VedicFormulas.lean`; it need not be replaced while resolving
   the surrounding assembly.
2. **Use exact values throughout the calculation.** Remove value-changing
   `storeQ`/128-bit quantization from state and intermediate corrections.
   Rational reduction by gcd is exact; bounded-precision rounding is not.
   Preserve exact algebraic or symbolic values where the selected expression
   requires them. An unsupported operation must identify itself rather than
   return a different value. A symbolic expression alone is not evidence that
   required numerical decisions were evaluated.
3. **Make each step transactional and traceable.** Keep the input checkpoint;
   commit output only after all required formula and coupling nodes succeed.
   Record formula ID, exact operands, selected branch and exact result. On S13
   zero division, retain the checkpoint and report the pair/denominator.
4. **Require every selected calculation.** A complete 29 step must account for
   all 29 formula invocations and all required coupling nodes in the declared
   order. Track `evaluated`, `undefined` and `blocked-by-dependency` separately;
   do not award a completed step when a required calculation was skipped. Valid
   no-correction branches remain evaluated formulas.
5. **Keep unresolved dependencies visible.** Compute defined independent
   expressions exactly where possible, and identify their actual consumers.
   An unresolved λ persistence or B-source choice prevents a claim of complete
   coupled fidelity. It does not authorize disabling that requirement or
   inventing a force to make a diagnostic alter Ψ.
6. **Respect finite resources without changing the equations.** Exact integer
   growth can require a worker, checkpointing and continued computation. If a
   calculation cannot finish, report it as incomplete with its exact checkpoint;
   do not round, cap, drop a term, lower a strength or substitute a simpler map.
7. **Verify the selected target on GitHub.** Expected values and domain results
   must come from an independently specified exact reference, with coupled
   read/write behavior included. Source parsing checks may support this but do
   not replace arithmetic equivalence. Existing assertions must not be weakened
   to make the new behavior pass.

## 7. What the existing comparison does and does not establish

`tools/check_exact_phi_kernel.mjs`, 231–294, extracts old source but rewrites
rounding expressions, rewrites S15, fixes `R4val`/USO to constants and stubs
GRVQ/MSTVQ at 258–259. Its acceptance condition at 292 permits a relative
displacement difference of `2^-32`. It is not an unmodified full-coupling oracle.
The separate coupling checks at 322 onward explicitly assert the PR169
register-only destination, including 128-bit storage at 341.

`VedicFormulas.lean` proves useful local identities. For example,
`log_ratio_sign` requires both products to be nonzero, `inverse_inject` proves
what a particular axis injection would do, and `stress_trace` proves the tensor
trace. None of those theorems selects whether that injection belongs in the
original dynamics, supplies S25's discarded threshold, or establishes full-step
JavaScript equivalence.

The completion claim must therefore remain narrower than “all original
formulas and everything coupled are exact” until the selected formulas,
domains, constants, initial conditions and assembly have all been satisfied
and verified at the same commit.

[legacy]: https://github.com/12cymatics/quanqonscious/blob/2ab11b4d33b455c37f866e39f056fdef0f88d9e5/vedic_v18.24_full_kernel.html
[active]: https://github.com/12cymatics/quanqonscious/blob/2ab11b4d33b455c37f866e39f056fdef0f88d9e5/vedic_v18.51.1_exact_phi.html
[vedic-lean]: https://github.com/12cymatics/quanqonscious/blob/2ab11b4d33b455c37f866e39f056fdef0f88d9e5/SutraWS/VedicFormulas.lean
[vertex-lean]: https://github.com/12cymatics/quanqonscious/blob/2ab11b4d33b455c37f866e39f056fdef0f88d9e5/SutraWS/Vertex.lean
[contracts-lean]: https://github.com/12cymatics/quanqonscious/blob/2ab11b4d33b455c37f866e39f056fdef0f88d9e5/SutraWS/Contracts.lean
[kernel-check]: https://github.com/12cymatics/quanqonscious/blob/2ab11b4d33b455c37f866e39f056fdef0f88d9e5/tools/check_exact_phi_kernel.mjs
[s13-original]: https://github.com/12cymatics/quanqonscious/blob/2ab11b4d33b455c37f866e39f056fdef0f88d9e5/vedic_v18.24_full_kernel.html#L5941
