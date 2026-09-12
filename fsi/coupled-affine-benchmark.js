'use strict';
/**
 * Cymatica coupled affine benchmark — an EXACT solution of the incompressible
 * Navier–Stokes equations coupled to a finite-strain elastic vessel and a
 * pinned liquid–air surface, on the stated finite time interval.
 *
 * WHAT THIS IS NOT. OpenAI's September 2026 Navier–Stokes result is a
 * finite-time breakdown construction: for each positive viscosity it produces
 * a smooth compactly supported force, chosen existentially from the
 * constructed flow's own momentum residual, under which an initially resting
 * whole-space flow has bounded kinetic energy and unbounded velocity as t → 1.
 * That force is not this apparatus's actuator, the theorem supplies no
 * elastic-vessel or pinned free-surface solution, and it does not say this cup
 * blows up. It is not wired into the renderer and nothing here reproduces it.
 *
 * What IS implemented is the coupled benchmark the same release supplies: a
 * manufactured problem with prescribed distributed body forces and exterior
 * tractions, for which every field is written down in closed form and every
 * local residual is EXACTLY zero over the rationals -- not small, zero.
 *
 * Its loads and flat free surface differ from the centre-driven Faraday
 * problem the renderer shows, so it is exposed alongside that scenario and
 * never replaces it.
 *
 * Geometry (nondimensional). Liquid  X²+Y² < 1, 0 < Z < 1  (reference volume π).
 * Solid = wall 1 < X²+Y² < 4, 0 < Z < 1  plus  bottom X²+Y² < 4, −1 < Z < 0
 * (reference volume 7π).
 */

/* ── exact rationals over BigInt ─────────────────────────────────────────── */
function bgcd(a, b){ a = a < 0n ? -a : a; b = b < 0n ? -b : b;
  while (b){ const t = a % b; a = b; b = t; } return a; }
class Q {
  constructor(n, d = 1n){
    if (d === 0n) throw new RangeError('Q: zero denominator');
    if (d < 0n){ n = -n; d = -d; }
    const g = bgcd(n, d) || 1n;
    this.n = n/g; this.d = d/g;
  }
  // Accepts an optional denominator. Without it Q.of(8n, 3n) silently returned
  // 8 -- the second argument was dropped -- and every fractional constant in
  // the energy ledger was wrong while all residuals still read zero, because
  // nothing depended on them until the kinetic-power check did.
  static of(v, den){
    if (den !== undefined){
      const num = Q.of(v), d = Q.of(den);
      return num.div(d);
    }
    if (v instanceof Q) return v;
    if (typeof v === 'bigint') return new Q(v, 1n);
    if (typeof v === 'number')
      throw new TypeError(
        `physical inputs must be exact: got the binary64 value ${v}. ` +
        `Pass a string ("5/4", "1.25") or a BigInt.`);
    if (typeof v === 'string'){
      const s = v.trim();
      let m = /^([+-]?\d+)\s*\/\s*(\d+)$/.exec(s);
      if (m) return new Q(BigInt(m[1]), BigInt(m[2]));
      m = /^([+-]?)(\d*)\.(\d+)$/.exec(s);
      if (m){
        const sign = m[1] === '-' ? -1n : 1n;
        const scale = 10n ** BigInt(m[3].length);
        return new Q(sign*(BigInt(m[2] || '0')*scale + BigInt(m[3])), scale);
      }
      if (/^[+-]?\d+$/.test(s)) return new Q(BigInt(s), 1n);
    }
    throw new TypeError(`Q.of: cannot read ${JSON.stringify(v)} exactly`);
  }
  add(o){ o = Q.of(o); return new Q(this.n*o.d + o.n*this.d, this.d*o.d); }
  sub(o){ o = Q.of(o); return new Q(this.n*o.d - o.n*this.d, this.d*o.d); }
  mul(o){ o = Q.of(o); return new Q(this.n*o.n, this.d*o.d); }
  div(o){ o = Q.of(o); if (o.n === 0n) throw new RangeError('Q: division by zero');
          return new Q(this.n*o.d, this.d*o.n); }
  neg(){ return new Q(-this.n, this.d); }
  pow(k){                                   // integer exponent, negatives allowed
    if (k < 0) return Q.ONE.div(this.pow(-k));
    let r = Q.ONE, b = this;
    for (let i = 0; i < k; i++) r = r.mul(b);
    return r;
  }
  get isZero(){ return this.n === 0n; }
  cmp(o){ o = Q.of(o); const l = this.n*o.d, r = o.n*this.d;
          return l < r ? -1 : l > r ? 1 : 0; }
  toString(){ return this.d === 1n ? `${this.n}` : `${this.n}/${this.d}`; }
  toNumber(){ return Number(this.n)/Number(this.d); }   // display only
}
Q.ZERO = new Q(0n, 1n); Q.ONE = new Q(1n, 1n);

/* ── declared geometry, integrated exactly ───────────────────────────────
   Second moments and volumes are COMPUTED from the stated radii and heights
   rather than written in as 31/2 and 7/3. Per the common factor π, for an
   annulus r0<R<r1 between z0 and z1:
       volume   = (r1²−r0²)(z1−z0)
       radial   = (r1⁴−r0⁴)/2 · (z1−z0)      ∫(X²+Y²)
       vertical = (r1²−r0²)(z1³−z0³)/3       ∫Z²

   These figures do NOT appear in any residual, and no local identity can
   check them. d/dt K = P_body is Reynolds transport: it holds for whatever
   domain is supplied, so a wrong moment moves both sides together and the
   residual stays zero. The geometry sets the reported energy magnitudes and
   nothing else constrains it, which is why it is derived here rather than
   asserted and why no gate below claims to cover it. */
function regionMoments(r0, r1, z0, z1){
  r0 = Q.of(r0); r1 = Q.of(r1); z0 = Q.of(z0); z1 = Q.of(z1);
  const dr2 = r1.pow(2).sub(r0.pow(2)), dz = z1.sub(z0);
  return {
    volume:   dr2.mul(dz),
    radial:   r1.pow(4).sub(r0.pow(4)).div(Q.of(2n)).mul(dz),
    vertical: dr2.mul(z1.pow(3).sub(z0.pow(3))).div(Q.of(3n))
  };
}
const GEOMETRY = {
  liquid: regionMoments(0n, 1n, 0n, 1n),                    // X²+Y²<1, 0<Z<1
  wall:   regionMoments(1n, 2n, 0n, 1n),                    // 1<X²+Y²<4, 0<Z<1
  bottom: regionMoments(0n, 2n, '-1', 0n)                   // X²+Y²<4, −1<Z<0
};
GEOMETRY.solid = {
  volume:   GEOMETRY.wall.volume.add(GEOMETRY.bottom.volume),
  radial:   GEOMETRY.wall.radial.add(GEOMETRY.bottom.radial),
  vertical: GEOMETRY.wall.vertical.add(GEOMETRY.bottom.vertical)
};

/* ── the construction ────────────────────────────────────────────────────── */
const DEFAULTS = { rhoF: '1', mu: '1', rhoS: '1', G: '1', gamma: '1' };
const S_MIN = '5/4', S_MAX = '3/2';

function evaluate(opts = {}){
  const o = { ...DEFAULTS, ...opts };
  const s     = Q.of(o.stretch ?? S_MIN);
  const rhoF  = Q.of(o.rhoF), mu = Q.of(o.mu);
  const rhoS  = Q.of(o.rhoS), G  = Q.of(o.G), gamma = Q.of(o.gamma);
  for (const [k, v] of [['rhoF',rhoF],['mu',mu],['rhoS',rhoS],['G',G],['gamma',gamma]])
    if (v.cmp(Q.ZERO) <= 0) throw new RangeError(`${k} must be positive, got ${v}`);
  const inRange = s.cmp(Q.of(S_MIN)) >= 0 && s.cmp(Q.of(S_MAX)) <= 0;

  // ---- material response at this stretch -------------------------------
  const sigR = G.mul(s.pow(4).sub(s.pow(2)));               // G(s⁴ − s²)
  const sigZ = G.mul(s.pow(-8).sub(s.pow(-4)));             // G(s⁻⁸ − s⁻⁴)
  const d    = sigR.sub(sigZ);
  const six  = Q.of(6n);
  const f    = s.mul(d).div(six.mul(mu));                   // ṡ = f(s)
  const a    = d.div(six.mul(mu));                          // a = ṡ/s
  // d′(s) = G(4s³ − 2s − 4s⁻⁵ + 8s⁻⁹)
  const dP = G.mul(Q.of(4n).mul(s.pow(3))
                   .sub(Q.of(2n).mul(s))
                   .sub(Q.of(4n).mul(s.pow(-5)))
                   .add(Q.of(8n).mul(s.pow(-9))));
  const fP   = d.add(s.mul(dP)).div(six.mul(mu));           // f′(s)
  const sDot = f, sDDot = fP.mul(f);                        // ṡ, s̈
  const aDot = dP.div(six.mul(mu)).mul(f);                  // ȧ = a′(s)·ṡ

  // ---- kinematics -------------------------------------------------------
  const F      = [s, s, s.pow(-2)];                         // diag
  const detF   = F[0].mul(F[1]).mul(F[2]);
  const W      = G.div(Q.of(4n)).mul(
                   Q.of(2n).mul(s.pow(2).sub(Q.ONE).pow(2))
                   .add(s.pow(-4).sub(Q.ONE).pow(2)));
  const P      = [G.mul(s.pow(3).sub(s)), G.mul(s.pow(3).sub(s)),
                  G.mul(s.pow(-6).sub(s.pow(-2)))];
  const sigmaS = [P[0].mul(F[0]), P[1].mul(F[1]), P[2].mul(F[2])];   // P Fᵀ / J, J = 1

  // ---- liquid -----------------------------------------------------------
  const divU   = a.add(a).sub(Q.of(2n).mul(a));             // a + a − 2a
  const p      = Q.of(2n).mul(sigR).add(sigZ).div(Q.of(3n)).neg();
  const sigmaF = [p.neg().add(Q.of(2n).mul(mu).mul(a)),
                  p.neg().add(Q.of(2n).mul(mu).mul(a)),
                  p.neg().sub(Q.of(4n).mul(mu).mul(a))];
  // b_f = ((ȧ+a²)x, (ȧ+a²)y, (−2ȧ+4a²)z); the momentum residual is
  //   ρ_f(∂ₜu + (u·∇)u − b_f) − ∇·σ_f, and ∇·σ_f = 0 because σ_f is uniform.
  const bfXY   = aDot.add(a.pow(2));
  const bfZ    = Q.of(-2n).mul(aDot).add(Q.of(4n).mul(a.pow(2)));
  const convXY = aDot.add(a.pow(2));                        // ∂ₜu + (u·∇)u, x,y
  const convZ  = Q.of(-2n).mul(aDot).add(Q.of(4n).mul(a.pow(2)));
  const fluidMomentum = [rhoF.mul(convXY.sub(bfXY)), rhoF.mul(convXY.sub(bfXY)),
                         rhoF.mul(convZ.sub(bfZ))];
  // b_s = χ_tt, and Div P = 0 because P is uniform.
  const solidMomentum = Q.ZERO;
  const stressMatch   = [sigmaF[0].sub(sigmaS[0]), sigmaF[1].sub(sigmaS[1]),
                         sigmaF[2].sub(sigmaS[2])];
  const angularMomentum = Q.ZERO;                           // σ_s diagonal ⇒ symmetric

  // ---- free surface z = s⁻², radius s; flat, so curvature vanishes -------
  // S(A,B,t) = (sA, sB, s⁻²) and u at that point are equal componentwise:
  //   S_t = (ṡA, ṡB, −2s⁻³ṡ),  u = (a·sA, a·sB, −2a·s⁻²) and a·s = ṡ.
  const kinZ    = Q.of(-2n).mul(s.pow(-3)).mul(sDot)
                    .sub(Q.of(-2n).mul(a).mul(s.pow(-2)));
  const pAir    = sigZ.neg();
  const tractionZ = sigmaF[2].sub(pAir.neg());              // σ_f·n − σ_ext·n
  const curvature = Q.ZERO;

  // ---- exact energy ledger, all divided by the common factor π ----------
  const half = Q.of(1n, 2n);
  const s6   = s.pow(6);
  const gf = GEOMETRY.liquid, gs = GEOMETRY.solid;
  const totalVolume = gf.volume.add(gs.volume);
  // K/π = (ρ ṡ²/2)(m_radial + 4 m_vertical / s⁶)
  const kinetic = (rho, g) => rho.mul(sDot.pow(2)).mul(half)
      .mul(g.radial.add(Q.of(4n).mul(g.vertical).div(s6)));
  const Kf = kinetic(rhoF, gf), Ks = kinetic(rhoS, gs);
  const Usolid   = gs.volume.mul(W);
  const Usurface = gamma.mul(s.pow(2));
  const dissip   = Q.of(12n).mul(mu).mul(a.pow(2)).mul(gf.volume);
  const pExt     = Q.of(2n).mul(a).mul(d).mul(totalVolume);
  const pSupport = Q.of(2n).mul(gamma).mul(s).mul(sDot);
  const Wdot     = Q.of(2n).mul(a).mul(d);                  // Ẇ = 2ad
  // Body-force power equals d/dt(K_f+K_s) by construction, so the balance
  // reduces to the stored, surface, dissipated and external terms.
  const energyResidual = gs.volume.mul(Wdot)
                          .add(Q.of(2n).mul(gamma).mul(s).mul(sDot))
                          .add(dissip)
                          .sub(pExt).sub(pSupport);
  // Ẇ from the energy itself, to check the 2ad shortcut rather than assume it
  const dWds = Q.of(2n).mul(G).mul(
                 s.pow(3).sub(s).sub(s.pow(-9)).add(s.pow(-5)));
  const WdotResidual = dWds.mul(sDot).sub(Wdot);
  // the closure the whole construction rests on
  const closureResidual = six.mul(mu).mul(a).sub(d);

  // Body-force power against kinetic-energy rate, computed independently on
  // both sides. Without this the domain's second moments (31/2 radial, 7/3
  // vertical for the solid; 1/2 and 1/3 for the liquid) carry no weight: the
  // energy residual above is the reduced form, in which the kinetic terms have
  // already cancelled, so a wrong moment would pass unnoticed.
  const s7 = s.pow(7), sd3 = sDot.pow(3);
  // d/dt[(ρ/2)ṡ²(mR + 4 mZ s⁻⁶)]
  const kineticRate = (rho, g) => rho.mul(half).mul(
      Q.of(2n).mul(sDot).mul(sDDot).mul(g.radial)
      .add(Q.of(8n).mul(sDot).mul(sDDot).mul(g.vertical).div(s6))
      .sub(Q.of(24n).mul(g.vertical).mul(sd3).div(s7)));
  // ∫ρ b·v over the current domain: (x²+y²) picks up s², z² picks up s⁻⁴
  const bodyPower = (rho, g) => rho.mul(
      a.mul(aDot.add(a.pow(2))).mul(s.pow(2)).mul(g.radial)
      .add(Q.of(4n).mul(a).mul(aDot).sub(Q.of(8n).mul(a.pow(3)))
           .mul(s.pow(-4)).mul(g.vertical)));
  const dKf = kineticRate(rhoF, gf), pBodyF = bodyPower(rhoF, gf);
  const dKs = kineticRate(rhoS, gs), pBodyS = bodyPower(rhoS, gs);
  const fluidKineticPower = dKf.sub(pBodyF);
  const solidKineticPower = dKs.sub(pBodyS);

  // ---- clock ------------------------------------------------------------
  const fAt = (x) => { const q = Q.of(x);
    const sr = G.mul(q.pow(4).sub(q.pow(2))), sz = G.mul(q.pow(-8).sub(q.pow(-4)));
    return q.mul(sr.sub(sz)).div(six.mul(mu)); };
  const tStarLower = Q.ONE.div(Q.of(4n).mul(fAt(S_MAX)));
  const tStarUpper = Q.ONE.div(Q.of(4n).mul(fAt(S_MIN)));

  return {
    scope: {
      // what this object does and does not establish
      exactLocalResidualsOverQ: true,
      constructedCoupledSolution: true,
      interval: `${S_MIN} ≤ s ≤ ${S_MAX}`,
      stretchInRange: inRange,
      reproducesOpenAIBlowup: false,
      solvesCentreDrivenFaradayProblem: false,
      materialLawCalibratedToStone: false,
      globalSpaceTimeProof: false
    },
    state: { s, sDot, sDDot, a, aDot, f, fPrime: fP, detF, W, p, pAir },
    stress: { sigmaR: sigR, sigmaZ: sigZ, d, P, sigmaS, sigmaF },
    residuals: {
      incompressibility: divU,
      fluidMomentum, solidMomentum,
      fluidSolidStressMatch: stressMatch,
      angularMomentum,
      freeSurfaceKinematic: kinZ,
      freeSurfaceTraction: tractionZ,
      freeSurfaceCurvature: curvature,
      energyBalance: energyResidual,
      storedEnergyRate: WdotResidual,
      viscousElasticClosure: closureResidual,
      fluidKineticPower, solidKineticPower
    },
    geometry: { ...GEOMETRY, totalVolume,
      note: 'moments and volumes integrated from the declared radii and heights; '
          + 'no local residual constrains them' },
    energy: { Kf, Ks, Usolid, Usurface, dissipation: dissip,
              externalStressPower: pExt, supportPower: pSupport, Wdot,
              bodyPowerFluid: pBodyF, bodyPowerSolid: pBodyS,
              kineticRateFluid: dKf, kineticRateSolid: dKs,
              note: 'every term divided by the common symbolic factor π' },
    clock: { dtByDs: Q.ONE.div(f), tStarLower, tStarUpper,
             note: 't = ∫ dξ/f(ξ) from 5/4; T* is bounded, not evaluated in closed form here' }
  };
}

// Walk Q values, arrays AND plain objects. The first version handled only the
// first two, so it collected nothing from the residuals object and reported
// allZero over an empty list -- a check that could not fail. It now refuses an
// empty collection for the same reason.
function allResidualsZero(r){
  const flat = [];
  const walk = (v, path) => {
    if (v instanceof Q){ flat.push([path, v]); return; }
    if (Array.isArray(v)){ v.forEach((x, i) => walk(x, `${path}[${i}]`)); return; }
    if (v && typeof v === 'object'){
      for (const k of Object.keys(v)) walk(v[k], path ? `${path}.${k}` : k);
    }
  };
  walk(r.residuals, '');
  if (!flat.length) throw new Error('allResidualsZero: collected no residuals');
  const bad = flat.filter(([, q]) => !q.isZero);
  return { count: flat.length, allZero: bad.length === 0,
           nonzero: bad.map(([k, q]) => `${k}=${q}`) };
}

const API = { Q, evaluate, allResidualsZero, DEFAULTS, S_MIN, S_MAX };
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.CymaticaCoupledBenchmark = API;
