# Exact 29-sutra simulator: coefficient provenance audit

Date: 2026-10-29.

## Scope and result

This is a static source and history audit of the simulator at PR #169 head **2ab11b4d33b455c37f866e39f056fdef0f88d9e5**, with a separate repair-status review of PR #170 at **8e8356e1aa29d1aee207ad292baee0ed8c80cddd** in section 8. It distinguishes mathematical definitions, normalization identities, inherited model parameters, rational representatives, numerical approximation settings, and interaction defaults. It does not certify a physical model, a complete coupling implementation, or a successful run. No tests were executed for this report.

**Snapshot convention:** sections 1–7 and their A/O/C line references describe the PR169 baseline. The words “active” and “current” in those sections refer to that audited baseline, not to the subsequent repair. In particular, the Damper startup, finite storage precision, incomplete magnetic projection, and operand-slider findings are historical findings; section 8 states what the repair changes.

The source supports several precise corrections, especially the magnetic Hodge coefficients and misleading constant names. It does **not** supply an independently derived replacement for every free parameter. Replacing an unproven number with another plausible number would not resolve that gap.

The main findings are:

1. The exact algebraic golden ratio already exists in A4, but the sutra and GRVQ channels use a different rational number. The value named PHI_INV_Q is not the reciprocal of that rational number.
2. Most GRVQ and magnetic-memory weights, the four lambda seeds, and the nominal simulation clock can be traced to the preserved source, including its earlier upload. Their presence supplies model provenance, not physical calibration or a mathematical derivation of their numerical values.
3. The baseline magnetic diagnostic uses a six-to-three projection that the preserved source explicitly replaced with a complete self-dual/anti-self-dual Hodge decomposition. Correct coefficient matrices and their derivation are available in the repository; the PR170 snapshot restores them.
4. The separate gammaW value 424923/10000, the powers-of-two coupling defaults, and the dielectric mixing rule first appear in the repository with the addition of the active page. Earlier saved versions also contain them, so this is a first repository occurrence, not a finding about who originally chose them. No units or derivation were established for gammaW.
5. The S17_STENCIL constants are left over from an earlier, different S17 implementation. The current page exports them but does not use them to evolve the field. The current S17 applies a nearest-neighbor gradient with a PHI_Q operand.
6. Several coefficient tables elsewhere in the repository have incorrect mathematical labels. They are not the table that weights the current browser updates. Changing their labels or values alone cannot correct the live simulator.

This report covers all materially used scalar constants in the active numerical model, its operator operands, numerical approximation settings, and user-facing controls. Render scales that affect interpretation are included separately. CSS colors, font sizes, array capacity allocations, and ordinary loop indices are not model coefficients. Faraday-cell material properties and fluid parameters belong to a different model and are outside this audit.

## 1. Source register and authority

All references of the form A:644 below mean the line at the immutable revision linked here. They do not refer to line numbers in a later working tree.

| Ref | Source | What it can establish |
| --- | --- | --- |
| A | [Active vedic_v18.51.1_exact_phi.html at PR #169 head](https://github.com/12cymatics/quanqonscious/blob/2ab11b4d33b455c37f866e39f056fdef0f88d9e5/vedic_v18.51.1_exact_phi.html) | Actual baseline declarations, consumers, UI defaults, and update order. |
| O | [Preserved vedic_v18.24_full_kernel.html at the same head](https://github.com/12cymatics/quanqonscious/blob/2ab11b4d33b455c37f866e39f056fdef0f88d9e5/vedic_v18.24_full_kernel.html) | The preserved comparison source. It contains multiple kernels and later changes despite its filename. |
| U | [Original v18.24 upload, a8339dc69bf9dd1e99571f97a89728ceeeadefe1](https://github.com/12cymatics/quanqonscious/blob/a8339dc69bf9dd1e99571f97a89728ceeeadefe1/vedic_v18.24_full_kernel.html) | Earlier literal values, uploaded through the 12cymatics account. Upload provenance does not prove original authorship or independent derivation. |
| I | [First repository version of the active file, c4e7f9e0c1321f3d0354f8612864f9138c36e091](https://github.com/12cymatics/quanqonscious/blob/c4e7f9e0c1321f3d0354f8612864f9138c36e091/vedic_v18.51.1_exact_phi.html) | The already-present gammaW, numerical gain defaults, dielectric rule, and former stencil consumer. |
| H | [Hodge correction commit b24792f9d7f4b35e0c3ada988ec8fc51356914e6](https://github.com/12cymatics/quanqonscious/commit/b24792f9d7f4b35e0c3ada988ec8fc51356914e6) | Explicit rejection of the old guessed projection and replacement by both Hodge chiralities. |
| M | [Maya and magnetic-memory introduction, 6a44a7418f931cad64213ca91d3e162603d7b3cd](https://github.com/12cymatics/quanqonscious/commit/6a44a7418f931cad64213ca91d3e162603d7b3cd) | Introduction of phase seed 1/16, five substeps, polarisation, memory, and the old projection in the active page. Its message is an implementation account, not an independent equation specification. |
| W | [SutraWS/WheelerGeometry.lean](https://github.com/12cymatics/quanqonscious/blob/2ab11b4d33b455c37f866e39f056fdef0f88d9e5/SutraWS/WheelerGeometry.lean) | Conditional identities for the declared geometry. Its precession-linearity theorem holds for arbitrary gammaW. |
| R | [SutraWS/RenderCertificates.lean](https://github.com/12cymatics/quanqonscious/blob/2ab11b4d33b455c37f866e39f056fdef0f88d9e5/SutraWS/RenderCertificates.lean) | Positivity/count facts about the literal stencil, not a derivation of those values or evidence of current consumption. |
| C | [vedic_trainer/vedic/kernel/sutras_canonical.py](https://github.com/12cymatics/quanqonscious/blob/2ab11b4d33b455c37f866e39f056fdef0f88d9e5/vedic_trainer/vedic/kernel/sutras_canonical.py) | A separate rational operator implementation, with a metadata coefficient table and an explicit rational-proxy caveat. |
| B | [vedic_trainer/docs/BIT_EXACT_PROTOCOL.md](https://github.com/12cymatics/quanqonscious/blob/2ab11b4d33b455c37f866e39f056fdef0f88d9e5/vedic_trainer/docs/BIT_EXACT_PROTOCOL.md) | The trainer's reference contract selects STRICT_SUTRA_KERNEL rather than SUTRAS[].evolve. It therefore cannot silently settle which different operator family the browser should implement. |
| P | [PR170 repair snapshot, 8e8356e1aa29d1aee207ad292baee0ed8c80cddd](https://github.com/12cymatics/quanqonscious/blob/8e8356e1aa29d1aee207ad292baee0ed8c80cddd/vedic_v18.51.1_exact_phi.html) | Static evidence for the specific repairs in section 8. This is a different snapshot from A; it does not change the historical line references or establish execution success. |

The repository's initial active-file commit was authored by Claude and added the file wholesale. Its parent tree has no earlier version of that file. The original v18.24 upload already has many of the model weights; later July commits change its curvature, Hodge, and Maya formulas. A comment saying “canonical,” “exact,” “derived,” or “decided with the user” is not, by itself, the underlying derivation or an accessible record of approval. The audit did not locate a complete authoritative equation-and-units specification that derives all of the active coefficient choices.

Additional saved source/protocol artifacts were inspected privately to extend the search. They were not copied into the repository. They corroborate earlier presence of some values but do not provide the missing gammaW calibration or a complete derivation of the active coupling defaults.

## 2. Independently derivable identities

### 2.1 Two different golden-ratio objects

Let p = 12586269025, q = 7778742049, and r = p/q. These are F50 and F49 under F0 = 0, F1 = 1. A:645–646 defines PHI_Q = r and PHI_INV_Q = r − 1. O:933 calls the ratio F51/F50; C:136–148 correctly identifies F50/F49 and explicitly says it is not the exact golden ratio.

The integer identity is:

    p² − pq − q² = −1.

Consequently:

    r² − r − 1 = −1/q²,
    r(r − 1) = 1 − 1/q²,
    1/r − (r − 1) = 1/(pq).

Thus PHI_INV_Q is the inherited companion r − 1, not the exact reciprocal q/p. This is a mathematical discrepancy even though it is small.

A:583 instead defines exact phi = (1 + sqrt(5))/2 in A4. It satisfies phi² = phi + 1 and phi⁻¹ = (sqrt(5) − 1)/2 exactly. The exact KX constants at A:2548–2577 give:

| Object | Exact expression |
| --- | --- |
| phi | (1 + sqrt(5))/2 |
| phi² | (3 + sqrt(5))/2 |
| phi³ | 2 + sqrt(5) |
| phi⁻¹ | (sqrt(5) − 1)/2 |
| phi⁻² | (3 − sqrt(5))/2 |
| phi⁻³ | sqrt(5) − 2 |
| Golden angle in degrees | 360 phi⁻² = 540 − 180 sqrt(5) |
| Complementary angle in degrees | 180 − 360 phi⁻² = −360 + 180 sqrt(5) |
| Half the golden angle | 270 − 90 sqrt(5) |
| Declared dielectric rescaling | 2 phi = 1 + sqrt(5) |

The identities validate the expressions once those quantities are defined that way. They do not derive a dielectric material law or a gyromagnetic ratio from phi.

**Correction depends on the declared operand.** If an operator requires literal r, preserve r and name r − 1 accurately; if it requires the reciprocal of r, use q/p; if it requires mathematical phi and its inverse, use the exact A4 pair and propagate that field through every consumer. These are different formulas. A rational-storage state cannot simply accept an A4 constant at one declaration: S13/S16 use rational numerators, and floor/remainder/sign paths need an explicit exact-domain definition.

### 2.2 Counts and normalized weights

| Quantity | Derivation | Limit of the conclusion |
| --- | --- | --- |
| 435 | 1 + … + 29 = 29 × 30 / 2 | Arithmetic identity. Choosing operator weight id/435 remains a model convention. |
| 16 vertices | 2⁴ | Follows from the declared four-bit hypercube. |
| 32 edges | 4 × 2³ | Follows from that graph. |
| 24 faces | choose(4,2) × 2² | Follows from that cell complex. |
| 8 cubes | choose(4,3) × 2 | Follows from that cell complex. |
| 1/16, 1/4, 1/3, 1/6 in averages | Reciprocal of the relevant vertex, neighbor, face-per-edge, or face-per-vertex count | Derives an arithmetic mean once averaging is the specified operation; it does not select the averaging model. |
| Vertex energy shares 16 rho_v/E | Sum over vertices is 16 | Requires E ≠ 0; ensures unit mean share. |
| Edge energy shares 8(rho_tail + rho_head)/E | Every vertex occurs in four edges, so the sum is 8 × 4E/E = 32 | Requires E ≠ 0; normalization does not derive the vortex-source law. |
| Symmetric/antisymmetric factors 1/2 | (x+y)/2 and (x−y)/2 reconstruct x,y | Exact decomposition. |
| Cayley pair coefficients (1−t²)/(1+t²), 2t/(1+t²) | Their squared sum is 1 | Exact rotation for parameter t. It does not identify t with an exact continuous-time rotation angle. |
| S21 osculator 5 for modulus 7 in base 10 | 10 × 5 ≡ 1 mod 7 | For n = 10a+b, divisibility by 7 is equivalent to a+5b ≡ 0. O:6215's explanation “(7+1)/2 rounded” is incorrect; the value 5 has this modular derivation. |

### 2.3 Source-backed Hodge correction

A:1300 declares the old HODGE matrix used by bvecAt at A:1334 and reconnectionsOf at A:1336–1343. It maps six face components into three components with guessed half weights. O:2008–2018 explicitly states that there is no canonical lossless map of this six-dimensional bivector space into one three-vector. Commit H replaces that projection.

For the ordered pairs (01,02,03,12,13,23), the correct Euclidean Hodge split in O:2062–2076 is:

| Face component | Self-dual contribution | Anti-self-dual contribution |
| --- | --- | --- |
| F01 | (1/2,0,0) | (1/2,0,0) |
| F02 | (0,1/2,0) | (0,1/2,0) |
| F03 | (0,0,1/2) | (0,0,1/2) |
| F12 | (0,0,1/2) | (0,0,−1/2) |
| F13 | (0,−1/2,0) | (0,1/2,0) |
| F23 | (1/2,0,0) | (−1/2,0,0) |

They implement F+ = (F + star F)/2 and F− = (F − star F)/2. A:1442–1443 already contains these matrices for TGCR. The magnetic diagnostic should consume both triples, as O:2195–2214 and O:2293–2307 do. Its inherited per-vertex division by six is a separate averaging convention and should not be dropped accidentally.

The reconnection threshold −7/10 has the exact square-root-free equivalent dot < 0 and dot² > (49/100) norm²_a norm²_b. The threshold itself is selected by the source. For the magnitude floor, O:2296/2302 skips magnitude strictly less than 1/1000, whereas A:1340 skips squared magnitude less than or equal to 1/1000000. The equality boundary therefore also differs. A faithful port should preserve the specified strict boundary.

## 3. Numerical-model coefficient matrix

Status vocabulary: **identity** means derivable from the stated definition; **source parameter** means an inherited literal with no independent numerical derivation found; **unresolved parameter** means the active choice has no established originating equation/units; **approximation** means a finite representative or numerical precision choice; **inactive** means no current evolution consumer was found.

### 3.1 Time, initialization, and global weights

| Constant or rule | Baseline value | Consumer and source | Classification / finding |
| --- | --- | --- | --- |
| SUTRA_SUM | 435 | A:643,1094; O:5518–5539 | Identity for the sum; choice of proportional id weighting is a source parameter. |
| alphaOf | (id/435) × strength × gain | A:1094; S1–S29 | Actual live operator weighting. The unrelated COEFFICIENT/SUTRA_COEFF tables do not enter it. |
| Symmetric composition gain | 1/2 on forward and reverse passes | A:1113–1117 | Declared composition rule. For nonlinear operators, two half-gain calls need not equal one full-gain call. |
| Distributed GRVQ share | 1/N active operators | A:1127–1133; O distributed-update loop | Source scheduling convention, distinct from id/435. The selected axis is (id−1) mod 4. |
| Dedicated GRVQ axes | S1→0, S5→1, S9→2, S14→3 | A:712,1097–1100; O corresponding evolve bodies | Inherited routing map; no independent derivation of the assignment found. |
| SIM_DT | 16667/50000 | A:644,657,689; O:3390; U:3118 | Exact transcription of 16.667/50, commented “50× slower than realtime.” A nominal UI/frame-time choice, not a derived physical timestep. |
| GRVQ time bucket | floor(t/1000), four-cycle | A:689–690; O:1294 onward | Inherited deterministic phase schedule. Do not infer a calibrated physical frequency. |
| Pair-rotation dt | 1/64 | A:2966; MAG/MSTVQ/TGCR/ZPE | Present at I:3913. No equation relating this to SIM_DT, wall time, or Maya log-scale time was found. |
| Lambda seed | (3,4,5,6) | A:647; O:3381; U:3109 | Inherited initial condition, not a derived constant. |
| Dielectric seed | 1/[1+(hw(v)−2)²] | A:1145–1148; I:836–839 | Chosen profile. The shift 2 centers the four-dimensional Hamming layers; the profile form remains a model choice. |
| Initial psi | forwardL(lambda seed) + dielectric seed | A:1571 | This addition is an active-model initial condition. The lambda seed alone does not justify the added profile. |
| Initial memory and phase | A=0 on 32 edges, chi=0 on 16 vertices; kappa=1; Maya=1/16 | A:1603–1607 | Declared initial conditions. Maya seed differs from both uploaded and preserved v18.24. |

### 3.2 GRVQ

| Term | Baseline coefficient / expression | Source | Finding |
| --- | --- | --- | --- |
| Lazy-walk idleness | 1/2 | A:659–667; O:1140–1143 | Later source parameter. A nonzero idleness may be needed to avoid the described degeneracy; that argument does not uniquely force 1/2. |
| Zero-neighbor split | 1/2 to each neighboring share before multiplying by 1−idleness | A:665; preserved walkMeasure | Source extension at a zero total weight. It is a selected symmetric extension, not the value of an otherwise undefined ratio. |
| Neighbor Laplacian mean | (previous+next)/2 − current | A:660; O:1164 onward | 1/2 is a two-neighbor average for the selected four-cycle. |
| Ricci averaging | Mean of the two edge curvatures | A:676–681 | The 1/2 follows from averaging two neighbors. Unit edge distance is part of the chosen graph metric. |
| Vacuum term | (3/10) lambda × [(B−abs(lambda))/B]²; B=max(10,2abs(lambda)+1) | A:683–687; O:1277 onward | All of 3/10, base floor 10, and affine base rule are inherited model choices. |
| Quantum term | 7/10 times lambda times cycle [r,r−1,−r,−(r−1)] | A:688–692; O:1294 onward | Source coefficient plus rational representative. The inverse label is inaccurate. |
| Beta term | −lambda × (2/5)/(opposite²+1) | A:693; O:1316 onward | Inherited 2/5 and denominator offset 1; no numerical derivation found. |
| Gamma coupling | (4/5)(previous×next − current²) | A:694; O:1334 onward | Inherited model coefficient. |
| Regularizer | (9/10)(mean(lambda)−lambda) | A:695; O:1350–1356 | Inherited model coefficient; mean factor 1/4 is structural. |
| Outer term weights | Laplacian 1/20; Ricci 1/40; difference of neighbor squares 1/1000; regularizer 1/20 | A:696–700; O:1359–1373 | Source parameters. O explicitly explains 1/40 as keeping the new curvature contribution in the previous range, not as a unique physical derivation. |

The old GRVQ object also declares g=1, r=1/2, lambda=1/5, alpha=3/5, delta=1/10, and scale_100=1/100. They are not all consumed by its current evolveExact formula. The operative coefficients are the ones above, not every name in O:1137–1154.

At the audited PR head, GRVQ updates a temporary lambda register reconstructed from psi at the start of each step. The result is exported, but A:1592–1601 passes afterS directly into subsequent field stages, and the next step reconstructs lambda again. Establishing coefficients does not resolve this missing causal feedback into psi.

### 3.3 Stress, magnetic memory, TGCR, and ZPE

| Constant or rule | Baseline value | Source / consumer | Finding |
| --- | --- | --- | --- |
| Stress isotropic coefficient | 1/2 in Bij−deltaij abs(B)²/2 | A:706–707; O:1914–1928 | Defined normalized tensor. No physical permeability factor appears in this computation. |
| S2 local stress-field scale | alpha/10 | A:803; O:5616 | Inherited strength argument of setB. |
| S3 local stress-field scale | alpha/100 | A:808; O:5647 | Inherited strength argument. |
| S29 local stress-field scale | abs(alpha × 1/5) by default | A:1087–1089; O:6506 | Inherited weighting choice. |
| Distributed MSTVQ field strength | abs(alpha of active S1), otherwise zero | A:1137–1143 | Specific active routing rule, not a universal stress normalization. |
| MSTVQ_FEED | 1/200000000 | A:708,1141; O:1960–1962; U:1868 onward | Exact value of the unrounded source expression: (5/1000000) × 1000/1000000 = 1/200000000. It is not 5/1000000. Units/intent of this double scaling remain unestablished. |
| TGCR_KAPPA_FEED | 1/500 | A:708,1649; O:1951; U:1859 | Exact transcription of 0.002. In this baseline kappa changes TGCR diagnostics, not tgcrTau. |
| TGCR initial kappa, eta | 1 and 1/2 | A:1606,1302; O:2576–2577 | Inherited initial stiffness and model coupling. |
| Magnetic retention | 499/500 | A:1302,1324,1328; O:2170 | Exact transcription of 0.998; source comment says “longer B memory.” This is selected discrete damping per step. |
| Faraday coefficient | 1/10 | A:1302,1307; O induction update | Exact transcription of 0.1 in the source's graph induction rule. It is not a Faraday-cell fluid parameter. |
| Toroidal / poloidal source strengths | 6/100 and eta × 3/100 | A:1302,1327; O:2166–2167 | Inherited source choices; with eta=1/2 the latter is 3/200. |
| PHI_CYCLE | [1,13/10,1,7/10] | A:1301,1326; O:2164 | Inherited four-point modulation, with no mathematical phi in its entries. The “golden walk” wording does not derive these values. |
| Vertex/edge energy-share multipliers | 16 and 8 | A:1311–1320 | Exact unit-mean normalization for the selected graph, as derived above. |
| Helicity face average | 1/3 | A:1332–1333; O:2222–2240 | Three incident faces per edge; the definition of this helicity functional remains a model choice. |
| Pressure and vector face average | 1/6 | A:1308,1334; O:2208–2209 | Six incident faces per vertex. The pressureAt output is the mean square flux; fluxEnergy separately includes 1/2 at A:1406. |
| Reconnection floor | squared norm 1/1000000 | A:1335–1340; O:2296,2302 | Squared version of source magnitude floor 1/1000; baseline equality boundary differs. Diagnostic threshold, not a continuous physical reconnection law. |
| Reconnection angle threshold | −7/10, squared form 49/100 | A:1342; O:2308 | Source threshold; the square is an identity. |
| Magnetic Hodge weights | Old 6→3 matrix | A:1300,1334 | Source-backed correction available: use both exact chiralities in section 2.3. |
| TGCR Hodge weights | +/-1/2 | A:1442–1455; O Hodge split | Derived from (F +/- star F)/2 for the chosen orientation. |
| TGCR energy coefficients | 1/2, kappa/2, eta | A:1460; O:2682–2687 | Inherited stated functional; eta remains a parameter. |
| Pair-mixing MSTVQ gain | 1/128 | A:2971,1437–1441 | Separate coupling from MSTVQ_FEED. Present in I. No derivation found. |
| TGCR field-rotation gain | 1/256 | A:2972,1444–1446 | Present in I. No derivation found. |
| ZPE pair-rotation gain | 1/512 | A:2973,1463–1465 | Present in I. It is not the old ENGINE_8x8 trace-removal factor 1/8 in O:1662. These are different models. |
| ZPE source support | Added only at hw(v)=2 | A:1467–1470 | Selected source geometry, six vertices; strength is dt × drive. |

### 3.4 Dielectric, Wheeler, and geometric gains

| Constant or rule | Baseline value | Source / consumer | Finding |
| --- | --- | --- | --- |
| Dielectric mixing | old field 1/2; complement mean 1/4; neighbor mean 1/8; plane source 1/8 | A:1149–1160; I:840–850 | The four weights sum to 1. This proves a convex mixture under the stated inputs, not why this mixture was selected. |
| Dielectric source gain | 1 + activeLoad/435 | A:1151–1152 | Current activeLoad is sum of strengths, A:1472. At all 29 unit strengths this gives 1+29/435=16/15, not 2. Any old proof assuming weighted load sum(id×strength)=435 needs reconciliation. |
| Polarisation gain | chi/8 times normalized vertex energy share | A:1160,1320,2975 | Added in commit M. Source definition; no independent physical susceptibility calibration found. |
| Dielectric scale | slider/10 × 1/8 | A:2960,2967 | UI-controlled model gain, present in I. |
| Dielectric algebraic scale | 2phi | A:1174 | Exact algebraic expression. Choosing this rescaling is a model definition. |
| Magnetic layer weight | k²/(1+k²), k=hw−2 | A:1170–1175 | Defined to vanish on the inertial plane. The vanishing theorem does not independently validate the assumed law. |
| Applied H | slider/10 × 1/64 | A:2961,2970 | UI-controlled dimensionless model gain; no calibrated field unit found. |
| gammaW | 424923/10000 = 42.4923 | A:2969,1176; I:3916 | Unresolved origin/units. W's linearity theorem accepts arbitrary gammaW. Neither the decimal nor that theorem identifies a material gyromagnetic ratio. |
| coherence | 1/4 | A:2968 | Inactive in the current dynamics; only parameter-object declarations were found. |
| Geometry compression gain geoD | 1/16 | A:2977,1483; rendering at A:3772,3803 | Chosen geometry scale, present in I; no feedback into psi found. |
| Geometry radial gain geoM | 1/8 | A:2978,1485; A:3807 | Same classification. |
| Geometry rotation gain geoP | 1/128 | A:2979,1487; A:3811 | Same classification; a Cayley parameter scale, not a calibrated angular frequency. |
| Four-to-three geometry embedding | +/-1 coordinates and +/-1/sqrt(2) offsets; z=hw−2 | A:1473–1489 | Exact embedding construction; its geometric interpretation is a chosen convention. |

There is a numerical clue, not a derivation: the separate exact complementary angle −360 + 180 sqrt(5) is about 42.49223595 degrees, close to gammaW. The code never establishes that gammaW is this angle, nor how an angle becomes the multiplier of its magnetic field. It would be unjustified to replace gammaW with that expression on proximity alone. It would likewise be unjustified to insert a measured proton gyromagnetic ratio without a declared species, field unit, time unit, and angular-frequency/frequency convention.

The formula text displayed at A:5108 also writes M in terms of D, while computeWheeler uses dielectricBaseQ and excludes the 2phi factor from magneticTrace. The source must specify whether the trace or the full phi-scaled dielectric belongs in that coupling before a factor can be corrected.

### 3.5 Maya and numerical approximation settings

| Constant or setting | Baseline value | Source | Finding |
| --- | --- | --- | --- |
| Maya beta coefficients | A=phi/e; B=e/pi | A:1250–1258; O:2972–2987 | Inherited named model definitions. Current implementation computes intervals, not exact rational point values of these transcendental expressions. |
| Maya phase seed | 1/16 | A:1253; introduced by M | New chosen initial condition relative to O:3041's snapped stable root and U:2831's zero seed. |
| Maya substeps / h | 5 and 1/500 | A:1254 | Total log-scale increment 1/100. U:2879 defaults to five substeps; O:3144 defaults to one. Calling this unchanged v18.24 without a revision is ambiguous. |
| Maya observation factor | 1 + beta/[20(1+abs(beta))] | A:1260; O:3182–3185 | 1/20 exactly transcribes source modulation 0.05. It changes the displayed field in this baseline, not psi. |
| Maya Lipschitz bound | upper(A)+3 upper(B) on abs(phase)≤1 | A:1261–1265 | Derived bound from beta'=A cos(phase)−3B phase² and the stated domain. Domain is essential. |
| Root-search bracket | [3/4,1], up to 24 bisections | A:1280–1288 | Numerical enclosure choice with sign checks; not a physical coefficient or the exact root itself. |
| Enclosure precision | ENC_BITS=160; root brackets use 168 | A:1191,1210 | Numerical approximation budget. |
| e series length | 48 terms with tail bound | A:1236–1240 | Certified enclosure construction, not exact e stored as Q. |
| pi bracket seed | k=137; N=547342635041262974904156351058877841049928; endpoints (N−1)/2^137 and (N+2)/2^137 | A:1242–1248 | Candidate bracket validated using trig bounds. Its width is 3/2^137, not 2^-160. |
| Trig truncation lengths | 44 for pi sign checks; 40 for Maya sine | A:1245,1256 | Numerical series truncations with enclosures. |
| Maya error-bound grid | 256 fractional bits | A:1272 | Outward-rounded error bound. The bound is relative to the exact Euler map, not the continuous ODE solution. |
| State storage precision | STORE_BITS=128 | A:454–465,1587–1589 | Baseline storeQ rounds rational results, including internal S6/S7/S15/S28 updates. Exact rational arithmetic before rounding does not make the complete update unrounded. |

## 4. All 29 active operand defaults and fixed factors

All bounds in this table are active API restrictions from A:713–750, not derived maximal mathematical domains. Defaults are mostly exposed literals from the corresponding preserved evolve bodies. Values called “proportion” or “ratio” use r=PHI_Q, not A4.PHI. Where no operand is listed, the global alpha still applies.

| S | Default operands and accepted range | Additional fixed factors / structure | Preserved source anchor |
| --- | --- | --- | --- |
| 1 | axis=0, integers0–3; one=1, [1/16,8] | Adds one in the neighbor-square denominator; dedicated GRVQ axis0 | O:5560–5584 |
| 2 | radix=10, integers2–16 | Base is first radix power above max(1,max abs(psi)); local MSTVQ alpha/10 | O:5586–5621 |
| 3 | divisor=16, [1,64] | XOR convolution of 16 vertices; local MSTVQ alpha/100 | O:5623–5652 |
| 4 | quarters=4, [1,16] | Horner polynomial of five Hamming-layer means at r/(1+r) | O:5654–5685 |
| 5 | None | Eight complement pairs; negative reflection; GRVQ axis1 | O:5687–5716 |
| 6 | ratio=r, [1,4] | Correction split includes 1/2 | O:5718–5751 |
| 7 | None | Symmetric/antisymmetric halves (x+y)/2,(x−y)/2 | O:5753–5788 |
| 8 | pull=1/4, [1/16,1] | Pair minimizers use 1/2 and neighbor mean uses 1/4 | O:5790–5818 |
| 9 | None | Degree-four graph Laplacian; GRVQ axis2 | O:5820–5848 |
| 10 | radix=10, integers2–16 | First radix power above max(1,abs(mean)); squared deficiency | O:5850–5871 |
| 11 | None | Half of layer mean plus whole mean | O:5873–5903 |
| 12 | modulus=9, integers2–16 | Decimal digital-root interpretation is specific to default9 | O:5905–5925 |
| 13 | terms=8, integers1–16 | Continued-fraction convergent of rational numerators; update weights alpha/2 and alpha/4 | O:5927–5968 |
| 14 | axis=3, integers0–3; one=1, [1/16,8] | Reciprocal abs-neighbor-plus-one; GRVQ axis3 | O:5970–5992 |
| 15 | None | Opposite axis-side multiplier includes residual/2 | O:5994–6030 |
| 16 | None | Integer gcd/Bézout on rational numerators; no numeric coefficient table | O:6032–6081 |
| 17 | proportion=r, [1/16,4] | In-place nearest-neighbor gradient divided by4; no S17_STENCIL consumer | O:6083–6107 |
| 18 | modA=5 and modB=7, each integers2–16 | The two defaults are coprime; the implementation's correction uses the first residue while the second participates in its condition | O:6109–6134 |
| 19 | ratio=r, [1,4]; rate=1/10, [1/128,1] | Root-ratio correction; negative-discriminant branch has alpha/2 | O:6136–6183 |
| 20 | base=7, integers2–16 | Nearest-multiple correction; half-base tie convention is part of integer logic | O:6185–6204 |
| 21 | osculator=5, integers1–9; modulus=7, integers2–16 | Decimal last digit uses base10; default5 has modular derivation, but arbitrary independently changed operands need not retain that divisibility meaning | O:6206–6231 |
| 22 | None | In-place complement average1/2 | O:6233–6252 |
| 23 | tens=10, integers2–16 | Decimal last digits remain base10; adjustment split1/2 | O:6254–6276 |
| 24 | digits=2, integers1–4 | Tail modulus10^digits; digital-root divisor9 | O:6278–6322 |
| 25 | layer=1/4, [1/16,1]; gradient=1/20, [1/128,1] | Hamming center2; separate inertial-plane gradient rule | O:6324–6372 |
| 26 | portion=16, [1,64] | Flag value divided by selected portion; 16 is source choice tied to vertex count | O:6374–6398 |
| 27 | None | Four prescribed five-vertex Hamming paths; mean of four increments; untouched-neighbor pull alpha/2 | O:6400–6449 |
| 28 | rate=1/20, [1/128,1] | Paired transfer of a difference of squares | O:6451–6482 |
| 29 | weight=1/5, [1/16,1] | Exact simplification of source29/145; additional multiplier beyond id/435 | O:6484–6511 |

Fixed values such as 1/4 and 1/20 in S8/S25/S28/S29 are not implied merely by their sutra names. They are coefficients of the selected source operator. Finite continued-fraction depth8 in S13 is an algorithmic operand: an exact eight-term convergent does not imply equality to the full ratio for all inputs.

The S27 paths are [0,1,3,7,15], [0,2,6,14,15], [0,4,12,13,15], and [0,8,10,11,15], in that order. The first-path-to-modify rule affects behavior. Those choices should not be replaced by an arbitrary equivalent-looking covering of the graph.

## 5. Interaction and UI defaults that affect the computation

### 5.1 Controls

| Control / rule | Raw range and default | Exact conversion / consequence | Baseline source |
| --- | --- | --- | --- |
| Strength | Integer0–100; per-preset values below | raw/100, so 1% increments | A:2914–2926,4644–4645 |
| Toggle on | 50 | Sets strength1/2 | A:4690 |
| ALL ON | 50 for every sutra | Activates all29 at strength1/2 | A:5266 |
| ALL OFF | 0 | Removes all29 from activeSutras | A:5271 |
| Noninteger sutra operand slider | Integer slider coordinate, intended scale16 | raw/16 | A:2932–2943,4670–4672 |
| Dielectric | Raw1–50, default10 | s=raw/80; default1/8 | A:226,2960,2967 |
| B field | Raw0–50, default10 | H=raw/640; default1/64 | A:227,2961,2970 |
| Vortex omega | Raw−60–60, markup default0 | omega=raw/10; preset overrides it | A:232,2962,2976 |
| Drive | Raw0–40, default0 | drive=raw/4; ZPE source per step=raw/256 | A:233,2963,2974,1467 |
| Polarisation chi | Raw0–30, default10 | chi=raw/10; default1 | A:234,2964,2975 |
| Glide | Integers1–120, default10 | Linear ramp over this many computed steps, not ten integrations of each coefficient per step | A:264,2997,1609–1630 |
| Wall-clock pace | 1000ms per scheduled step | Separate from SIM_DT and dt | A:2996 |
| Initial composition before preset | SERIES | Startup preset overrides it | A:2982 |

The operand-slider scale16 cannot represent every declared default: r, 1/10, and 1/20 are examples. The display position is computed by integer division q.n×16/q.d, whereas the kernel initially retains the exact default. In addition, a minimum such as1/128 maps to slider minimum1/8 while the input parser accepts integer strings only. This is a source-level control/grid inconsistency, not evidence that the declared coefficient itself should be rounded to a sixteenth. An exact text/rational control or a domain-compatible grid is required to preserve exact operands when edited.

### 5.2 Presets

The following are exact **configuration choices**, with no independent derivation found. Percentages below are raw strength percentages; omega, drive, and chi below are their effective values after slider conversion. Every preset resets to the seed and resets operand defaults. A:5302–5346 is the source.

| Preset | Composition | Sutras at percent strength | omega | drive | chi |
| --- | --- | --- | --- | --- | --- |
| Damper | Parallel | S9=100, S11=100, S29=100 | 1 | 0 | 1 |
| Inductor | Symmetric concurrent | S3=30, S9=30, S27=50 | 3 | 1 | 1 |
| Dynamo | Symmetric concurrent | S3=30, S9=30, S11=30, S25=30, S29=30 | 2 | 1 | 1 |
| Containment | Series | S9=30, S28=30 | 4 | 0 | 1 |
| Capacitor | Parallel | S4=50, S5=50, S26=50 | 1 | 0 | 3 |
| Balancer | Symmetric concurrent | S2=30, S9=30 | 1 | 0 | 1 |
| Checksum | Parallel | S12=100, S24=100, S18=50, S9=30 | 1 | 0 | 1 |
| Lock | Symmetric concurrent | S20=100, S9=30 | 1 | 0 | 1 |
| Rejector | Parallel | S3=50, S9=30, S11=30 | 1 | 0 | 1 |
| Mixer | Symmetric concurrent | S22=50, S9=30 | 1 | 0 | 1 |
| Amplifier | Series | S1=50 | 1 | 0 | 1 |
| Correlator | Symmetric concurrent | S3=50, S9=30 | 1 | 2 | 1 |

Baseline startup matters: A:2946 first initializes S9/S22/S29 at50%, but A:5874 applies DEFAULT_PRESET=Damper when workers start successfully. The effective successful startup is therefore Parallel S9/S11/S29 at100%, with a ten-step ramp from zero. That baseline is not an all29 run. A passing individual-operator gate does not change that fact. The PR170 startup and preset behavior are different, as documented in section 8.

## 6. Rendering constants and inactive tables

| Item | Values / source | Interpretation |
| --- | --- | --- |
| S17_STENCIL | [9/128,7/256,3/128,7/512,9/640], A:633–634 | Current definition and export only. In I:693–696 it drove a different S17, with factor3 on Hamming-distance-one edges and factor1 on Hamming-distance-three edges. R:28–58 proves basic literal facts, not why these coefficients belong to current S17. |
| Exact octant root | (sqrt(2)/2)(1+i), powers0–7 | A:2536–2544; exact eighth-root-of-unity geometry. |
| Fine-structure-named render coefficient | C.alpha=1000000/137035999 | A:2613,2622,2666; inherited rounded rational from O:970–973. Used in golden-vortex/counterspace rendering, not the audited state step. Do not call the displayed decimal an exact measured physical constant. |
| Wheeler render multiplier | 8 × Math.PI × exact phi³ projected to float | A:2615–2616; physical interpretation unestablished, pi is a rendering float here. |
| Radial width | eps=1/10 | A:2644; inherited render-profile choice, O radialFactor call0.1. |
| Render modulation | pressure0.15; vortex twist0.3; phase factors3,1/2,1/4 | A:2695,2701,2707 | Presentation scales; no state feedback found. |
| Drawing-gain defaults | field60, tube35, y45, twist0.08, ripple8, bulge5, vortex1, Wheeler1; flux bulge22, ring18, twist1/2 | A:3644; matching controls A:270 onward | Selected drawing gains, separate from dynamics coefficients. |
| Surface and field resolutions | field160×80; torus304×152; field segments4, rings64, spokes144; haze grid40 | A:2714,3463,3661 | Numerical rendering/field-sampling resolution, not exact continuum fields. |
| Display shape and projection | base torus radii180/70; camera distance700; fill0.9; frame share0.98 | A:3633–3661 | Presentation geometry. |
| Display timing | nominal frame16.667ms; Y rotation0.0022; X rotation0.0004 at frequency0.0007; fades250ms; framing600/10000ms | A:332–336,3585,3637 | Wall-clock presentation values; separate from the simulation clocks. |
| Surrounding-field exclusion scale | EPS=10^-9 × displayed R | A:3468 | Floating field-evaluation tolerance. It is outside the exact Q state, but matters to any claim that the surrounding fields are exact. |

### Three different “coefficient” systems must remain distinguishable

1. **Active browser:** alphaOf=id/435×strength×gain, plus the operands and couplings tabulated above. This is the live path.
2. **Preserved HTML SUTRA_COEFF:** O:3629–3660 is a phi-based table. Its comments claim derivations, but O:5538–5539's alphaPhi simply returns alpha. The table is not multiplied into those evolve strengths. One entry, S11=12586269025/10000000000, is about1.2586 while its comment says sqrt(phi) about1.2720; source presence does not validate that label.
3. **Trainer COEFFICIENT:** C:149–179 stores rational representatives in the Sutra metadata object at C:197. C:244–252's alpha uses id/435 and percent strength; apply_sutra does not use the metadata coefficient. The explicit proxy caveat at C:136–148 is correct, but several specific labels below it are not.

For example, silver ratio is exactly1+sqrt(2), whereas the S9 metadata literal is2.665144. S28's claimed1−1/256 is exactly255/256, not9961578/10000000; S29's claimed1−1/1024 is exactly1023/1024, not9990234/10000000. Other named integral and special-function entries need the same definition-versus-literal treatment. Correcting these metadata errors is useful; it is not evidence that the active browser has adopted the named constants or that those named constants should multiply its current formulas.

## 7. Corrections supported by evidence and remaining dependencies

| Action | Evidence / precise target | Status and dependency |
| --- | --- | --- |
| Restore complete magnetic Hodge coefficients | Use both matrices in section2.3; preserve the inherited /6 convention; evaluate reconnection dot/norm over all six coordinates | Direct source and algebra support. A concrete code repair is possible. |
| Correct reconnection threshold boundary | Skip norm² strictly below10^-6 if preserving O's magnitude<10^-3 rule | Direct source support; clarify any intentional change separately. |
| Distinguish exact phi from rational r | Exact A4 phi/inverse are available; r's companion r−1 is not1/r | Algebra supports correction. Requires selecting the declared operator domain and propagating it, not one replacement literal. |
| Correct Fibonacci indexing / misleading inverse name | F50/F49; r−1 vs q/p | No numerical model change is necessary to correct the description. |
| Correct misleading named-constant metadata | Record literal rational separately from the mathematical defining expression | Does not wire these constants into the active operator path. Do not change live equations by importing an unrelated table. |
| Preserve source-exact scalar conversions | 0.998→499/500, 0.002→1/500, 0.000005×1000/1000000→1/200000000 | These current rational conversions are algebraically faithful to the unrounded source expression. No invented replacement is warranted. |
| Make edited operands exactly expressible | Repair the scale16 control mismatch without rounding the default to the slider | Source-level inconsistency is identified; UI implementation choice can preserve the exact parameter. |
| Establish gammaW | Need originating equation, physical/dimensionless interpretation, units, and allowed algebraic domain | No justified replacement value found. Nearby golden-angle values and unrelated measured gyromagnetic values are insufficient. |
| Establish dt, initial profile, and scale gains | Need selected discretization, relation among the three clocks, units, initial conditions, and source for the active power-of-two gains | Source literals are reproducible, but no unique derived replacements were established. |
| Establish dielectric load semantics | Is load=sum(strength) or sum(id×strength), and why divide by435? | Current value and earlier weighted interpretation differ by a factor15 at all29 unit strengths. A proof about one does not certify the other. |
| Establish all required couplings | Specify how lambda register, stress, kappa, psi, dielectric, and memory feed one another | Baseline values can be computed correctly while their outputs do not causally enter the next field state. Constants alone cannot repair this. |
| Establish exact Maya semantics | Define whether target is the continuous ODE or a specified discrete Euler map, with exact transcendental values or certified enclosures | Current finite rational enclosures and stored midpoints are not an unrounded exact solution of the continuous ODE. |

## 8. Repair status, separate from the baseline

### 8.1 Published PR170 snapshot

The following are static observations of P, commit **8e8356e1aa29d1aee207ad292baee0ed8c80cddd**. They establish the stated implementation changes, not that a complete nonlinear step finishes within available time or memory, and not that the complete physical formulation has been recovered. Execution results must identify their own tested commit.

| Baseline finding | Change present in P | What remains unestablished |
| --- | --- | --- |
| `storeQ` reduces rational storage to a finite precision budget | P:454–464 sets `STORE_BITS=null` and canonicalizes the full BigInt numerator/denominator without rounding. The storage-error ledger returns zero, P:1602–1605. | Rational arithmetic exactness does not validate the selected coefficients, finite Maya enclosures, time discretization, or coupling assembly. Resource noncompletion is not a successful exact calculation. |
| Damper startup selects three sutras; controls permit disabling operators | P:3015 selects all 29 at 50%; P:5954 sends that configuration without applying Damper. P:2984–2989 refuses zero strength; P:5351 disables ALL OFF. P:5416 adjusts only preset-listed strengths and keeps the remaining sutras active. | This defines the requested live configuration. It does not establish that all 29 finish executing. Preset values remain explicit choices, not independently derived coefficients. |
| Selecting 29 operators alone is not evidence of completed evaluation | P:1693–1721 requires each ID 1–29 with positive strength and checks the completed record: 29 applications for Series/Parallel and 58 in the specified forward/reverse order for Symmetric Concurrent. The live worker calls this entry point at P:3122. | A successful record would establish this execution condition only; it would not settle missing source correspondence or downstream coupling. A refused or unfinished step must not be reported as completed. |
| The magnetic reconnection diagnostic discards half of the two-form information | P:1324–1328 restores the source-derived self-dual and anti-self-dual half matrices; P:1362–1374 consumes all six coordinates and preserves the source's strict lower norm threshold. | The Hodge half factors follow from `(F±*F)/2`. The inherited vertex-average factor 1/6 and reconnection selection thresholds are still declared model choices, not a derived physical reconnection law. |
| S13 silently skips a zero continued-fraction denominator | P:921 restores the preserved source's explicit RangeError and identifies the pair. | This restores a defined refusal; it does not extend the mathematical domain or guarantee that every all29 trajectory avoids the refusal. |
| The scale16 operand slider cannot express some exact defaults | P:2996–3013 displays exact rationals and accepts an integer, fraction, or finite-decimal string, then applies the declared domain check. | Operand editability is repaired. The choice of an operand and its domain must still match the intended equation. |
| The historical phi names conflate an exact rational with the algebraic constant and its inverse | P:644–657 explicitly identifies `PHI_Q` as the preserved literal and `PHI_INV_Q` as its difference from 1, and records the nonzero identity residuals. | The numbers are intentionally not substituted. Selecting literal r, reciprocal 1/r, or algebraic phi still requires the source-domain decision described in section 2.1. |

P:649–665 explicitly records `FORMULATION_STATUS.complete=false`. Its scope notes at P:2560–2563 distinguish exact rational storage from model validity and finite Maya approximation. The trainer metadata table is not corrected by these browser changes, and the inactive S17 stencil is not made a live coefficient by documenting it.

### 8.2 Source thresholds restored in this revision

This revision restores four additional preserved piecewise conditions after P: S15 requires `abs(residual)>1/1000000`; S24's digital-root branch requires `abs(psi_i)>1/100` (the condition is on the vertex value, not on its correction error); S27's path update requires `abs(path_target-path_value)>1/10000000000`; and S25 requires `abs(log(abs(prodS))-log(abs(prodT)))>1/1000` on its source-defined nonzero-product branch. Their original locations are O:6015, O:6311, O:6431, and O:6350 respectively. S27's final interpolation of vertices not modified by the path passes retains its separately defined, ungated update.

The active page's `SUTRA_THRESHOLDS` declaration and the corresponding `vedicField` cases now consume these exact rational thresholds. S25's `logRatioExceeds` decides the equivalent strict comparison of `max(abs(prodS),abs(prodT))/min(abs(prodS),abs(prodT))` against `exp(1/1000)` using refining certified rational Taylor bounds, without substituting a rounded logarithm. These thresholds have explicit source provenance; their restoration does not derive a physical justification for their selected magnitudes. **GitHub runtime evidence for this subsequent revision is pending.** These changes and their eventual execution result must not be attributed to the earlier P snapshot.

### 8.3 Definitions still needed for the complete formulation

The remaining questions are narrower than “none of the coefficients is known.” The scalar values and their current consumers are inventoried above; the unresolved requirement is what fully specified model those values must implement.

1. **Operand field and named constants:** select the inherited rational r, its actual reciprocal, or exact algebraic phi, and define how every numerator, floor, remainder, sign and ordering operation extends to that chosen domain. Supply gammaW's originating equation and units or declare it a dimensionless parameter. Neither a nearby angle nor a measured constant with unestablished units supplies that definition.
2. **Time and initial state:** specify the relationship among pair-rotation `dt=1/64`, `SIM_DT=16667/50000`, Maya's five `1/500` substeps, and scheduled wall time. Confirm whether the field begins with the preserved lambda seed alone or the active dielectric addition, and define the chosen dielectric and Maya initial profiles.
3. **Dielectric load and stage gains:** settle whether load is the unweighted sum of strengths or the ID-weighted sum, why 435 normalizes it, and how the source-present powers-of-two stage gains and dielectric mixing coefficients enter the intended equations. Existence of an earlier literal does not identify a unique physical calibration.
4. **Persistent coupling assembly:** specify how GRVQ/MSTVQ lambda changes, stress, TGCR kappa, magnetic memory, dielectric, and psi causally enter the next state. The baseline records temporary register updates that are not reconciled with the persistent field assembly; computing and displaying those register values does not supply the missing dependency.
5. **Selected source family and complete domains:** retain the identified `SUTRAS[].evolve` collection unless a different specification is explicitly selected. Preserve its ordered loops and piecewise conditions, and resolve each remaining source/port discrepancy. A different 29-entry library, the old S17 stencil, or a metadata table cannot silently define the missing formula.
6. **Maya's exact target:** choose the continuous transcendental ODE or a specified discrete map, state how transcendental quantities are represented, and define what certified approximation is acceptable if any. Unrounded rational storage alone cannot make finite enclosures and midpoint Euler updates an exact continuous solution.

No calculation should be declared complete on the strength of a coefficient's type, a copied literal, a passing equality to a reference that shares the same value, or a theorem parameterized by that value. The evidence needed is the specified equation, the coefficient's definition or authorized parameter choice, its actual consumer, its domain/units, and an independent check of the resulting calculation.
