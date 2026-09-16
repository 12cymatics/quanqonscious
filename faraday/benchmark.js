function bgcd(a, b){ a = a < 0n ? -a : a; b = b < 0n ? -b : b;
  while (b){ const t = a % b; a = b; b = t; } return a; }
class Q {
  constructor(n, d = 1n){
    if (d === 0n) throw new RangeError('Q: zero denominator');
    if (d < 0n){ n = -n; d = -d; }
    const g = bgcd(n, d) || 1n;
    this.n = n/g; this.d = d/g;
  }

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
  pow(k){
    if (k < 0) return Q.ONE.div(this.pow(-k));
    let r = Q.ONE, b = this;
    for (let i = 0; i < k; i++) r = r.mul(b);
    return r;
  }
  get isZero(){ return this.n === 0n; }
  cmp(o){ o = Q.of(o); const l = this.n*o.d, r = o.n*this.d;
          return l < r ? -1 : l > r ? 1 : 0; }
  toString(){ return this.d === 1n ? `${this.n}` : `${this.n}/${this.d}`; }
  toNumber(){ return Number(this.n)/Number(this.d); }
}
Q.ZERO = new Q(0n, 1n); Q.ONE = new Q(1n, 1n);

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
  liquid: regionMoments(0n, 1n, 0n, 1n),
  wall:   regionMoments(1n, 2n, 0n, 1n),
  bottom: regionMoments(0n, 2n, '-1', 0n)
};
GEOMETRY.solid = {
  volume:   GEOMETRY.wall.volume.add(GEOMETRY.bottom.volume),
  radial:   GEOMETRY.wall.radial.add(GEOMETRY.bottom.radial),
  vertical: GEOMETRY.wall.vertical.add(GEOMETRY.bottom.vertical)
};

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

  const sigR = G.mul(s.pow(4).sub(s.pow(2)));
  const sigZ = G.mul(s.pow(-8).sub(s.pow(-4)));
  const d    = sigR.sub(sigZ);
  const six  = Q.of(6n);
  const f    = s.mul(d).div(six.mul(mu));
  const a    = d.div(six.mul(mu));

  const dP = G.mul(Q.of(4n).mul(s.pow(3))
                   .sub(Q.of(2n).mul(s))
                   .sub(Q.of(4n).mul(s.pow(-5)))
                   .add(Q.of(8n).mul(s.pow(-9))));
  const fP   = d.add(s.mul(dP)).div(six.mul(mu));
  const sDot = f, sDDot = fP.mul(f);
  const aDot = dP.div(six.mul(mu)).mul(f);

  const F      = [s, s, s.pow(-2)];
  const detF   = F[0].mul(F[1]).mul(F[2]);
  const W      = G.div(Q.of(4n)).mul(
                   Q.of(2n).mul(s.pow(2).sub(Q.ONE).pow(2))
                   .add(s.pow(-4).sub(Q.ONE).pow(2)));
  const P      = [G.mul(s.pow(3).sub(s)), G.mul(s.pow(3).sub(s)),
                  G.mul(s.pow(-6).sub(s.pow(-2)))];
  const sigmaS = [P[0].mul(F[0]), P[1].mul(F[1]), P[2].mul(F[2])];

  const gradU  = [a, a, Q.of(-2n).mul(a)];
  const gradUdot = [aDot, aDot, Q.of(-2n).mul(aDot)];
  const divU   = gradU[0].add(gradU[1]).add(gradU[2]);
  const p      = Q.of(2n).mul(sigR).add(sigZ).div(Q.of(3n)).neg();
  const sigmaF = gradU.map(L => p.neg().add(Q.of(2n).mul(mu).mul(L)));

  const bf     = [aDot.add(a.pow(2)), aDot.add(a.pow(2)),
                  Q.of(-2n).mul(aDot).add(Q.of(4n).mul(a.pow(2)))];
  const conv   = gradU.map((L, i) => gradUdot[i].add(L.pow(2)));
  const fluidMomentum = conv.map((c, i) => rhoF.mul(c.sub(bf[i])));

  const solidMomentum = Q.ZERO;
  const stressMatch   = [sigmaF[0].sub(sigmaS[0]), sigmaF[1].sub(sigmaS[1]),
                         sigmaF[2].sub(sigmaS[2])];
  const sigmaSFull = [[Q.ZERO,Q.ZERO,Q.ZERO],[Q.ZERO,Q.ZERO,Q.ZERO],
                      [Q.ZERO,Q.ZERO,Q.ZERO]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++)
        sigmaSFull[i][j] = sigmaSFull[i][j].add(
          (i === k ? P[i] : Q.ZERO).mul(j === k ? F[j] : Q.ZERO));
  const angularMomentum = [sigmaSFull[0][1].sub(sigmaSFull[1][0]),
                           sigmaSFull[0][2].sub(sigmaSFull[2][0]),
                           sigmaSFull[1][2].sub(sigmaSFull[2][1])];

  const kinZ    = Q.of(-2n).mul(s.pow(-3)).mul(sDot)
                    .sub(Q.of(-2n).mul(a).mul(s.pow(-2)));
  const pAir    = sigZ.neg();
  const tractionZ = sigmaF[2].sub(pAir.neg());
  const curvature = Q.ZERO;

  const half = Q.of(1n, 2n);
  const s6   = s.pow(6);
  const gf = GEOMETRY.liquid, gs = GEOMETRY.solid;
  const totalVolume = gf.volume.add(gs.volume);

  const kinetic = (rho, g) => rho.mul(sDot.pow(2)).mul(half)
      .mul(g.radial.add(Q.of(4n).mul(g.vertical).div(s6)));
  const Kf = kinetic(rhoF, gf), Ks = kinetic(rhoS, gs);
  const Usolid   = gs.volume.mul(W);
  const Usurface = gamma.mul(s.pow(2));
  const dissip   = Q.of(12n).mul(mu).mul(a.pow(2)).mul(gf.volume);
  const pExt     = Q.of(2n).mul(a).mul(d).mul(totalVolume);
  const pSupport = Q.of(2n).mul(gamma).mul(s).mul(sDot);
  const Wdot     = Q.of(2n).mul(a).mul(d);

  const energyResidual = gs.volume.mul(Wdot)
                          .add(Q.of(2n).mul(gamma).mul(s).mul(sDot))
                          .add(dissip)
                          .sub(pExt).sub(pSupport);

  const dWds = Q.of(2n).mul(G).mul(
                 s.pow(3).sub(s).sub(s.pow(-9)).add(s.pow(-5)));
  const WdotResidual = dWds.mul(sDot).sub(Wdot);

  const aFromStress = sigmaS[0].add(p).div(Q.of(2n).mul(mu));
  const closureResidual = six.mul(mu).mul(aFromStress).sub(d);

  const s7 = s.pow(7), sd3 = sDot.pow(3);

  const kineticRate = (rho, g) => rho.mul(half).mul(
      Q.of(2n).mul(sDot).mul(sDDot).mul(g.radial)
      .add(Q.of(8n).mul(sDot).mul(sDDot).mul(g.vertical).div(s6))
      .sub(Q.of(24n).mul(g.vertical).mul(sd3).div(s7)));

  const bodyPower = (rho, g) => rho.mul(
      a.mul(aDot.add(a.pow(2))).mul(s.pow(2)).mul(g.radial)
      .add(Q.of(4n).mul(a).mul(aDot).sub(Q.of(8n).mul(a.pow(3)))
           .mul(s.pow(-4)).mul(g.vertical)));
  const dKf = kineticRate(rhoF, gf), pBodyF = bodyPower(rhoF, gf);
  const dKs = kineticRate(rhoS, gs), pBodyS = bodyPower(rhoS, gs);
  const fluidKineticPower = dKf.sub(pBodyF);
  const solidKineticPower = dKs.sub(pBodyS);

  const fAt = (x) => { const q = Q.of(x);
    const sr = G.mul(q.pow(4).sub(q.pow(2))), sz = G.mul(q.pow(-8).sub(q.pow(-4)));
    return q.mul(sr.sub(sz)).div(six.mul(mu)); };
  const tStarLower = Q.ONE.div(Q.of(4n).mul(fAt(S_MAX)));
  const tStarUpper = Q.ONE.div(Q.of(4n).mul(fAt(S_MIN)));

  return {
    scope: {

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
