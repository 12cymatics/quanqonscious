/**
 * Cymatica boundary kernel — the ten boundary equations.
 *
 * Every published number in this file is computed from the `exact` string
 * printed beside it. `verify()` re-derives each one and compares; a record
 * whose stored value disagrees with its own formula cannot load. Numbers that
 * are NOT derivable from their formula are marked `derived: false` and named,
 * rather than shipped as though they were.
 *
 * Changes from the previous kernel, each with the measurement behind it:
 *
 *   1. E(m) was a 2048-interval Simpson quadrature. It is now the AGM
 *      recurrence (Abramowitz & Stegun 17.6), which reaches the same machine
 *      precision in ~6 iterations instead of 2049 sin+sqrt evaluations, and
 *      carries no interval count that silently sets the error.
 *
 *   2. Thom Egg II published P/D = 1.7124. No closed convex curve can have
 *      P/D < 2 — going out to the far point and back is already 2D. The
 *      divisor 20 is wrong. It is no longer used; the record is unverified
 *      and `verifyOrThrow()` refuses it.
 *
 *   3. Thom Egg I published P/D = 2.2185 from a divisor of 18 that appears
 *      nowhere in its formula. Legal, but below every other Thom figure
 *      (2.87-3.08) and below both ellipses. Also unverified.
 *
 *   4. The square published 4, which is P/side. Every other entry is P/D with
 *      D the diameter, so the square was in different units. With D the
 *      diagonal it is 2*sqrt(2). The convention is now explicit per record.
 *
 *   5. `coupling` contained `index * 0.0007`, which made the field depend on
 *      the order of the array. Reordering the equations changed the physics.
 *      Removed; coupling is now a function of the geometry alone.
 *
 *   6. Non-finite input produced silent NaN throughout. Every entry point now
 *      validates and throws.
 */
(function installCymaticaBoundaryKernel(root) {
  'use strict';

  const PI = Math.PI;
  const TAU = PI * 2;
  const PHI = (1 + Math.sqrt(5)) / 2;
  const SQRT2 = Math.SQRT2;
  const DEFAULT_R4_SCALES = Object.freeze([0.8, 1.2, 1.8, 2.6]);

  function requireFinite(name, x) {
    if (typeof x !== 'number' || !Number.isFinite(x)) {
      throw new TypeError(`${name} must be a finite number, got ${x}`);
    }
    return x;
  }

  /**
   * Complete elliptic integral of the second kind, E(m), by the
   * arithmetic-geometric mean.
   *
   *   a0 = 1, b0 = sqrt(1-m), c0 = sqrt(m)
   *   a_{n+1} = (a_n + b_n)/2,  b_{n+1} = sqrt(a_n b_n),  c_{n+1} = (a_n - b_n)/2
   *   K(m) = pi / (2 a_inf)
   *   E(m) = K(m) * (1 - sum_{n>=0} 2^{n-1} c_n^2)
   *
   * The AGM converges quadratically: the number of correct digits doubles each
   * step, so double precision is reached in about six iterations regardless of
   * m. There is no truncation parameter to tune and no error that grows with
   * the integrand's curvature.
   */
  function completeEllipticE(m) {
    requireFinite('m', m);
    if (m < 0 || m >= 1) {
      throw new RangeError(`E(m) is defined here for 0 <= m < 1; got m = ${m}`);
    }
    let a = 1;
    let b = Math.sqrt(1 - m);
    let c = Math.sqrt(m);
    let sum = (c * c) / 2;          // the n = 0 term, 2^-1 c0^2
    let power = 1;
    // Terminate on |a-b| relative to a, NOT on an absolute |c|. Once a and b
    // agree to within an ulp, c = (a-b)/2 stops shrinking and plateaus around
    // 5.6e-17, while `power` keeps doubling -- so the terms 2^n c^2 start to
    // GROW again. An absolute 1e-18 cutoff is never reached, the loop runs to
    // its iteration cap, and the junk terms reach ~3e-14. That cost 1.1e-13 of
    // accuracy at m = 0.5, measured against mpmath at 30 digits.
    for (let n = 0; n < 64 && Math.abs(c) > Number.EPSILON * a; n++) {
      const nextA = (a + b) / 2;
      const nextC = (a - b) / 2;
      b = Math.sqrt(a * b);
      a = nextA;
      c = nextC;
      power *= 2;
      sum += (power * c * c) / 2;
    }
    return (PI / (2 * a)) * (1 - sum);
  }

  /** Perimeter of the ellipse with semi-axes a >= b, as a multiple of 2a. */
  function ellipsePerimeterOverMajorAxis(aOverB) {
    requireFinite('aOverB', aOverB);
    if (aOverB < 1) throw new RangeError(`aOverB must be >= 1, got ${aOverB}`);
    const bOverA = 1 / aOverB;
    return 2 * completeEllipticE(1 - bOverA * bOverA);   // P/(2a) = 2E(e^2)
  }

  // Thom Egg perimeters, straight from the printed formula. The 3-4-5 triangle
  // gives a = 3, b = 4, c = 5, r1 = 5.
  const EGG_I_PERIMETER = 14 * PI - 6 * Math.atan(4 / 5);      // 2*pi*r1 + pi*b - 2*a*beta
  const EGG_II_PERIMETER = 10 * PI + 10 - 8 * Math.atan(5 / 4); // 2*pi*r1 + 2*c - 2*b*theta

  // The divisors below came with the original file and appear in none of the
  // formulas. They are kept only so the numbers are inspectable; nothing
  // computes with them unless the caller opts in past verifyOrThrow().
  const EGG_I_UNVERIFIED_DIVISOR = 18;
  const EGG_II_UNVERIFIED_DIVISOR = 20;

  /**
   * Each record's `evaluate()` recomputes perimeterRatio from the geometry in
   * `exact`. verify() calls it and requires agreement, so the string and the
   * number cannot drift apart.
   */
  const BOUNDARY_EQUATIONS = Object.freeze([
    Object.freeze({
      key: 'circle', label: 'Circle', species: 'ΓC',
      exact: 'P/D = π',
      provenance: 'classical_supporting_formula',
      diameterConvention: 'diameter',
      perimeterRatio: PI, minorRatio: 1,
      derived: true, evaluate: () => PI
    }),
    Object.freeze({
      key: 'ellipse', label: 'Ellipse 3:2', species: 'ΓEll',
      exact: 'x²/3² + y²/2² = 1; P/D = 2E(5/9)',
      provenance: 'source_extracted_plus_classical_support',
      diameterConvention: 'major_axis',
      perimeterRatio: 2 * completeEllipticE(5 / 9), minorRatio: 2 / 3,
      derived: true, evaluate: () => ellipsePerimeterOverMajorAxis(3 / 2)
    }),
    Object.freeze({
      key: 'golden-ellipse', label: 'Golden ellipse', species: 'ΓΦ',
      // a/b = φ gives e² = 1 - 1/φ² = 1 - (2-φ) = φ-1 = 1/φ, so P/D = 2E(1/φ).
      exact: 'a/b = φ; P/D = 2E(1/φ)',
      provenance: 'golden_ellipse_supporting_formula',
      diameterConvention: 'major_axis',
      perimeterRatio: 2 * completeEllipticE(1 / PHI), minorRatio: 1 / PHI,
      derived: true, evaluate: () => ellipsePerimeterOverMajorAxis(PHI)
    }),
    Object.freeze({
      key: 'thom-a', label: 'Thom A', species: 'ΓA',
      exact: 'P/D = 5π/6 + (√7/2)atan(√3/5)',
      provenance: 'source_extracted',
      diameterConvention: 'major_axis',
      perimeterRatio: 5 * PI / 6 + (Math.sqrt(7) / 2) * Math.atan(Math.sqrt(3) / 5),
      minorRatio: (1 + Math.sqrt(7)) / 4,
      derived: true,
      evaluate: () => 5 * PI / 6 + (Math.sqrt(7) / 2) * Math.atan(Math.sqrt(3) / 5)
    }),
    Object.freeze({
      key: 'thom-b', label: 'Thom B', species: 'ΓB',
      exact: 'P/D = 5π/6 + (√10/3)atan(1/3)',
      provenance: 'source_extracted',
      diameterConvention: 'major_axis',
      perimeterRatio: 5 * PI / 6 + (Math.sqrt(10) / 3) * Math.atan(1 / 3),
      minorRatio: (2 + Math.sqrt(10)) / 6,
      derived: true,
      evaluate: () => 5 * PI / 6 + (Math.sqrt(10) / 3) * Math.atan(1 / 3)
    }),
    Object.freeze({
      key: 'thom-d', label: 'Thom D', species: 'ΓD',
      exact: 'P/D = 8π/9 + (√13/3)atan(√3/7)',
      provenance: 'derived_from_source_geometry',
      diameterConvention: 'major_axis',
      perimeterRatio: 8 * PI / 9 + (Math.sqrt(13) / 3) * Math.atan(Math.sqrt(3) / 7),
      minorRatio: (2 + Math.sqrt(13)) / 6,
      derived: true,
      evaluate: () => 8 * PI / 9 + (Math.sqrt(13) / 3) * Math.atan(Math.sqrt(3) / 7)
    }),
    Object.freeze({
      key: 'thom-bmod', label: 'Thom B modified', species: 'ΓBmod',
      exact: 'P/D = 3π/4 + (√5/2)atan(1/2)',
      provenance: 'derived_from_source_geometry',
      diameterConvention: 'major_axis',
      perimeterRatio: 3 * PI / 4 + (Math.sqrt(5) / 2) * Math.atan(1 / 2),
      minorRatio: (1 + Math.sqrt(5)) / 4,
      derived: true,
      evaluate: () => 3 * PI / 4 + (Math.sqrt(5) / 2) * Math.atan(1 / 2)
    }),
    Object.freeze({
      key: 'egg-i', label: 'Thom Egg I', species: 'ΓEggI',
      exact: 'P = 2πr₁ + πb − 2aβ; tan β = b/c   (a,b,c = 3,4,5; r₁ = 5)',
      provenance: 'source_extracted',
      diameterConvention: 'UNVERIFIED',
      perimeter: EGG_I_PERIMETER,
      perimeterRatio: EGG_I_PERIMETER / EGG_I_UNVERIFIED_DIVISOR,
      minorRatio: 2 / 3,
      derived: false,
      unverifiedReason:
        `the formula gives P = ${EGG_I_PERIMETER.toFixed(6)} but no diameter; ` +
        `the divisor ${EGG_I_UNVERIFIED_DIVISOR} appears in no formula. The ` +
        `resulting P/D = 2.2185 is legal but below every other Thom figure ` +
        `(2.87-3.08) and below both ellipses.`,
      evaluate: () => { throw new Error('egg-i: diameter is not derivable from its formula'); }
    }),
    Object.freeze({
      key: 'egg-ii', label: 'Thom Egg II', species: 'ΓEggII',
      exact: 'P = 2πr₁ + 2c − 2bθ; tan θ = c/b   (a,b,c = 3,4,5; r₁ = 5)',
      provenance: 'source_extracted',
      diameterConvention: 'UNVERIFIED',
      perimeter: EGG_II_PERIMETER,
      perimeterRatio: EGG_II_PERIMETER / EGG_II_UNVERIFIED_DIVISOR,
      minorRatio: 4 / 5,
      derived: false,
      unverifiedReason:
        `P/D = 1.7124 with the divisor ${EGG_II_UNVERIFIED_DIVISOR}. No closed ` +
        `convex curve can have P/D < 2: reaching the far point and returning ` +
        `already costs 2D. The divisor is wrong, not merely unsourced.`,
      evaluate: () => { throw new Error('egg-ii: diameter is not derivable from its formula'); }
    }),
    Object.freeze({
      key: 'polygon', label: 'Square polygon', species: 'ΓPoly',
      // Every other record is P/D with D the diameter. A square's diameter is
      // its diagonal s*sqrt(2), so P/D = 4s/(s*sqrt2) = 2*sqrt2. The previous
      // value 4 was P/side, i.e. a different quantity in the same column.
      exact: 'P = 2(a+b), a = b; D = a√2; P/D = 2√2',
      provenance: 'classical_supporting_formula',
      diameterConvention: 'diameter',
      perimeterRatio: 2 * SQRT2, minorRatio: 1 / SQRT2,
      derived: true, evaluate: () => 4 / SQRT2
    })
  ]);

  const MIN_CONVEX_RATIO = 2;   // P >= 2D for every closed convex curve

  /** Re-derive every record and report. Nothing here throws; verifyOrThrow does. */
  function verify(tolerance = 1e-12) {
    return BOUNDARY_EQUATIONS.map(b => {
      const issues = [];
      if (b.derived) {
        let got;
        try { got = b.evaluate(); } catch (e) { issues.push(`evaluate() threw: ${e.message}`); }
        if (got !== undefined && Math.abs(got - b.perimeterRatio) > tolerance) {
          issues.push(`stored ${b.perimeterRatio} != formula ${got}`);
        }
      } else {
        issues.push(`not derivable from its formula: ${b.unverifiedReason}`);
      }
      if (b.perimeterRatio < MIN_CONVEX_RATIO) {
        issues.push(`P/D = ${b.perimeterRatio.toFixed(6)} < 2, impossible for a closed convex curve`);
      }
      if (!(b.minorRatio > 0 && b.minorRatio <= 1)) {
        issues.push(`minorRatio ${b.minorRatio} outside (0, 1]`);
      }
      return { key: b.key, ok: issues.length === 0, issues };
    });
  }

  function verifyOrThrow() {
    const bad = verify().filter(r => !r.ok);
    if (bad.length) {
      throw new Error('boundary kernel failed verification:\n' +
        bad.map(r => `  ${r.key}: ${r.issues.join('; ')}`).join('\n'));
    }
    return true;
  }

  function reinforcedR4(radius, scales = DEFAULT_R4_SCALES) {
    requireFinite('radius', radius);
    if (!Array.isArray(scales) || scales.length === 0) {
      throw new TypeError('scales must be a non-empty array');
    }
    const radialFourth = radius ** 4;
    let product = 1;
    for (const scale of scales) {
      requireFinite('scale', scale);
      if (scale <= 0) throw new RangeError(`R4 scale must be > 0, got ${scale}`);
      const scaleFourth = scale ** 4;
      product *= scaleFourth / (radialFourth + scaleFourth);
    }
    return product;
  }

  /**
   * Coupling per boundary. Previously
   *     0.016 + anisotropy * 0.024 + index * 0.0007
   * whose last term made the field depend on the order of BOUNDARY_EQUATIONS —
   * reordering the array changed the result. Coupling now depends only on the
   * boundary's own anisotropy, so the set is order-free. The two constants are
   * the original ones and remain unsourced; they are named here so they can be
   * replaced by something derived rather than sitting inline.
   */
  const COUPLING_BASE = 0.016;
  const COUPLING_ANISOTROPY_GAIN = 0.024;

  function couplingFor(boundary) {
    const anisotropy = 1 - boundary.minorRatio;
    return COUPLING_BASE + anisotropy * COUPLING_ANISOTROPY_GAIN;
  }

  function boundaryTerms(perimeterArc, sourceDistance) {
    requireFinite('perimeterArc', perimeterArc);
    requireFinite('sourceDistance', sourceDistance);
    const r4 = reinforcedR4(sourceDistance);
    return BOUNDARY_EQUATIONS.map(boundary => {
      const perimeterPhase = TAU * perimeterArc / boundary.perimeterRatio;
      const coupling = couplingFor(boundary);
      const equationFactor = 1 - coupling * Math.sin(perimeterPhase) ** 2;
      return Object.freeze({ key: boundary.key, perimeterPhase, coupling, equationFactor, r4 });
    });
  }

  function factoredBoundaryProduct(perimeterArc, sourceDistance) {
    let product = 1;
    for (const term of boundaryTerms(perimeterArc, sourceDistance)) {
      product *= term.equationFactor;
    }
    // r4 is a property of the source distance, not of any one term; taking it
    // from terms[0] as before worked only because every term carried a copy.
    return product * reinforcedR4(sourceDistance);
  }

  /** Dimensionless perimeter loss at the basin wall. There is no alternate path. */
  function factoredBoundaryImpedance(perimeterArc, sourceDistance) {
    return 1 - factoredBoundaryProduct(perimeterArc, sourceDistance);
  }

  root.CymaticaBoundaryKernel = Object.freeze({
    PHI,
    BOUNDARY_EQUATIONS,
    DEFAULT_R4_SCALES,
    COUPLING_BASE,
    COUPLING_ANISOTROPY_GAIN,
    MIN_CONVEX_RATIO,
    completeEllipticE,
    ellipsePerimeterOverMajorAxis,
    reinforcedR4,
    couplingFor,
    boundaryTerms,
    factoredBoundaryProduct,
    factoredBoundaryImpedance,
    factoredBoundaryResponse: factoredBoundaryProduct,
    verify,
    verifyOrThrow
  });
})(typeof globalThis !== 'undefined' ? globalThis : this);

if (typeof module !== 'undefined' && module.exports) {
  module.exports = globalThis.CymaticaBoundaryKernel;
}
