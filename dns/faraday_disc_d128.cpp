#include "dyadic128.hpp"

extern "C" {

void* memset(void* dst, int c, unsigned long n){
  unsigned char* q = (unsigned char*)dst;
  for (unsigned long i = 0; i < n; i++) q[i] = (unsigned char)c;
  return dst;
}
void* memcpy(void* dst, const void* src, unsigned long n){
  unsigned char* d = (unsigned char*)dst;
  const unsigned char* q = (const unsigned char*)src;
  for (unsigned long i = 0; i < n; i++) d[i] = q[i];
  return dst;
}

static const int ARENA = 1500000;
static D128 arena[ARENA];
static int arenaUsed = 0;

static int nr, nz, mMode, steps, contactPinned, kMax, nVec, prepared, lastError;
static D128 scal[8];
static D128 *rf, *zf, *rc, *zc, *drc, *dzc, *drf, *dzf, *drive;
static D128 *u, *v, *w, *p, *eta;
static D128 *us, *vs, *ws, *lu, *lv, *lw, *dv, *gu, *gv, *gw, *ps, *dudz, *dvdz;
static D128 *vecIn, *vecOut, *basis, *hess, *coef;
static D128 *band, *dinv, *probe, *probeOut, *upJ, *upI, *tbuf;
static D128 *idrf, *idzf, *idzc, *sOutS, *sInS, *sM, *aOut, *aIn, *cA, *cB;
static D128 *bOut, *bIn, *faW, *fbW, *mOverRf, *gvA, *gvB, *dAu, *dBu, *dWc, *dVc, *topCoef;
static D128 rho, nu, gam, dt, href, uref, dtnu, dtOverRho, rhoOverDt, twoRhoNu, wzA, wzB, wzC, axisR, gAxis, idzfTop;
static D128 pressureAsymmetry;

static D128* take(int n){
  if (arenaUsed + n > ARENA){ lastError = 1; return arena; }
  D128* q = arena + arenaUsed; arenaUsed += n;
  for (int i = 0; i < n; i++) q[i] = dZero();
  return q;
}

static inline int iu(int i, int j){ return i*nz + j; }
static inline int iw(int i, int j){ return i*(nz + 1) + j; }
static inline int ip(int i, int j){ return i*nz + j; }

int setup(int NR, int NZ, int STEPS, int M, int PINNED, int KMAX){
  if (NR < 6 || NZ < 4 || STEPS < 1 || M < 0 || KMAX < 2) return 5;
  nr = NR; nz = NZ; steps = STEPS; mMode = M; contactPinned = PINNED ? 1 : 0; kMax = KMAX;
  arenaUsed = 0; lastError = 0; prepared = 0;
  const int nu_ = (nr + 1)*nz, nw_ = nr*(nz + 1), np_ = nr*nz;
  nVec = 2*(nr - 1)*nz + nr*nz + nr + (mMode == 1 ? nz : 0);
  rf = take(nr + 1); zf = take(nz + 1); rc = take(nr); zc = take(nz);
  drc = take(nr); dzc = take(nz); drf = take(nr + 1); dzf = take(nz + 1);
  drive = take(steps);
  u = take(nu_); v = take(nu_); w = take(nw_); p = take(np_); eta = take(nr);
  us = take(nu_); vs = take(nu_); ws = take(nw_);
  lu = take(nu_); lv = take(nu_); lw = take(nw_);
  dv = take(np_); gu = take(nu_); gv = take(nu_); gw = take(nw_);
  ps = take(nr); dudz = take(nr + 1); dvdz = take(nr + 1);
  vecIn = take(nVec); vecOut = take(nVec);
  basis = take((kMax + 1)*nVec); hess = take((kMax + 1)*kMax); coef = take(kMax + 1);
  band = take(np_*(nz + 1)); dinv = take(np_); probe = take(np_); probeOut = take(np_);
  upJ = take(np_); upI = take(np_); tbuf = take(nz + 1);
  idrf = take(nr + 1); idzf = take(nz + 1); idzc = take(nz);
  sOutS = take(nr); sInS = take(nr); sM = take(nr);
  aOut = take(nr + 1); aIn = take(nr + 1); cA = take(nr + 1); cB = take(nr + 1);
  bOut = take(nr); bIn = take(nr); faW = take(nr + 1); fbW = take(nr + 1); mOverRf = take(nr + 1);
  gvA = take(nr + 1); gvB = take(nr + 1);
  dAu = take(np_); dBu = take(np_); dWc = take(nr); dVc = take(np_); topCoef = take(nr);
  return lastError;
}

D128* ptrScal(){ return scal; }
D128* ptrRf(){ return rf; }   D128* ptrZf(){ return zf; }
D128* ptrRc(){ return rc; }   D128* ptrZc(){ return zc; }
D128* ptrDrc(){ return drc; } D128* ptrDzc(){ return dzc; }
D128* ptrDrf(){ return drf; } D128* ptrDzf(){ return dzf; }
D128* ptrDrive(){ return drive; }
D128* ptrVecIn(){ return vecIn; }
D128* ptrVecOut(){ return vecOut; }
D128* ptrU(){ return u; } D128* ptrV(){ return v; } D128* ptrW(){ return w; }
D128* ptrP(){ return p; } D128* ptrEta(){ return eta; }
D128* ptrBasis(){ return basis; } D128* ptrHess(){ return hess; } D128* ptrCoef(){ return coef; }
D128* ptrProbe(){ return probe; } D128* ptrProbeOut(){ return probeOut; }
D128* ptrPressureAsymmetry(){ return &pressureAsymmetry; }
int getNVec(){ return nVec; }
int getLastError(){ return lastError; }
int getArenaUsed(){ return arenaUsed; }

static D128 wzSurface(int i){
  return dAdd(dSub(dMul(wzA, w[iw(i, nz)]), dMul(wzB, w[iw(i, nz - 1)])), dMul(wzC, w[iw(i, nz - 2)]));
}

static D128 surfaceLaplacian(int i){
  const D128 e = eta[i];
  D128 s = dNeg(dMul(sM[i], e));
  if (i < nr - 1) s = dAdd(s, dMul(sOutS[i], dSub(eta[i + 1], e)));
  else if (contactPinned) s = dSub(s, dMul(sOutS[i], e));
  if (i > 0) s = dSub(s, dMul(sInS[i], dSub(e, eta[i - 1])));
  return s;
}

static void surfacePressure(const D128 &gNow){
  for (int i = 0; i < nr; i++)
    ps[i] = dAdd(dSub(dMul(dMul(rho, gNow), eta[i]), dMul(gam, surfaceLaplacian(i))), dMul(twoRhoNu, wzSurface(i)));
}

static void surfaceSlopes(){
  for (int i = 0; i <= nr; i++){
    if (i == 0 && mMode == 1){ dudz[0] = dNeg(dMul(w[iw(0, nz)], idrf[0])); dvdz[0] = dNeg(dudz[0]); continue; }
    if (i == 0 || i == nr){ dudz[i] = dZero(); dvdz[i] = dZero(); continue; }
    const D128 wl = w[iw(i - 1, nz)], wr = w[iw(i, nz)];
    dudz[i] = dMul(dSub(wl, wr), idrf[i]);
    dvdz[i] = dMul(mOverRf[i], dAdd(dMul(faW[i], wr), dMul(fbW[i], wl)));
  }
}

static void lapUV(const D128* f, const D128* other, const D128* dfdzTop, D128* out){
  for (int i = 1; i < nr; i++){
    for (int j = 0; j < nz; j++){
      const int c = iu(i, j);
      const D128 fc = f[c];
      const D128 radial = dSub(dMul(aOut[i], dSub(f[iu(i + 1, j)], fc)), dMul(aIn[i], dSub(fc, f[iu(i - 1, j)])));
      const D128 below = j == 0 ? dMul(fc, idzf[0]) : dMul(dSub(fc, f[iu(i, j - 1)]), idzf[j]);
      const D128 above = j == nz - 1 ? dfdzTop[i] : dMul(dSub(f[iu(i, j + 1)], fc), idzf[j + 1]);
      const D128 vertical = dMul(dSub(above, below), idzc[j]);
      const D128 coupling = dNeg(dAdd(dMul(cA[i], fc), dMul(cB[i], other[c])));
      out[c] = dAdd(dAdd(radial, vertical), coupling);
    }
  }
}

static void lapAxis(const D128* dfdzTop){
  for (int j = 0; j < nz; j++){
    const int c = iu(0, j), b = iu(1, j);
    const D128 a = u[c];
    const D128 radial = dMul(dSub(dSub(u[b], v[b]), dScale2(a, 1)), axisR);
    const D128 below = j == 0 ? dMul(a, idzf[0]) : dMul(dSub(a, u[iu(0, j - 1)]), idzf[j]);
    const D128 above = j == nz - 1 ? dfdzTop[0] : dMul(dSub(u[iu(0, j + 1)], a), idzf[j + 1]);
    lu[c] = dAdd(radial, dMul(dSub(above, below), idzc[j]));
    lv[c] = dNeg(lu[c]);
  }
}

static void lapW(){
  for (int i = 0; i < nr; i++){
    const D128 wz = wzSurface(i);
    for (int j = 1; j <= nz; j++){
      const int c = iw(i, j);
      const D128 wc = w[c];
      D128 radial = i == nr - 1 ? dNeg(dMul(bOut[i], wc)) : dMul(bOut[i], dSub(w[iw(i + 1, j)], wc));
      if (i > 0) radial = dSub(radial, dMul(bIn[i], dSub(wc, w[iw(i - 1, j)])));
      const D128 below = j == 1 ? dMul(wc, idzc[0]) : dMul(dSub(wc, w[iw(i, j - 1)]), idzc[j - 1]);
      const D128 vertical = j == nz ? dMul(dScale2(dSub(wz, below), 1), idzf[j])
                                    : dMul(dSub(dMul(dSub(w[iw(i, j + 1)], wc), idzc[j]), below), idzf[j]);
      lw[c] = dSub(dAdd(radial, vertical), dMul(sM[i], wc));
    }
  }
}

static void divergence(const D128* U, const D128* V, const D128* W, D128* out){
  for (int i = 0; i < nr; i++)
    for (int j = 0; j < nz; j++){
      const int c = ip(i, j);
      D128 s = dMul(dAu[c], U[iu(i + 1, j)]);
      if (i > 0) s = dSub(s, dMul(dBu[c], U[iu(i, j)]));
      s = dAdd(s, dMul(dWc[i], dSub(W[iw(i, j + 1)], W[iw(i, j)])));
      if (mMode != 0) s = dAdd(s, dMul(dVc[c], dAdd(V[iu(i, j)], V[iu(i + 1, j)])));
      out[c] = s;
    }
}

static void gradient(const D128* q){
  for (int j = 0; j < nz; j++){
    if (mMode == 1){ gu[iu(0, j)] = dMul(gAxis, q[ip(0, j)]); gv[iu(0, j)] = dNeg(gu[iu(0, j)]); }
    else { gu[iu(0, j)] = dZero(); gv[iu(0, j)] = dZero(); }
    gu[iu(nr, j)] = dZero(); gv[iu(nr, j)] = dZero();
  }
  for (int i = 1; i < nr; i++)
    for (int j = 0; j < nz; j++){
      const D128 qa = q[ip(i - 1, j)], qb = q[ip(i, j)];
      gu[iu(i, j)] = dMul(dSub(qb, qa), idrf[i]);
      gv[iu(i, j)] = dAdd(dMul(gvA[i], qa), dMul(gvB[i], qb));
    }
  for (int i = 0; i < nr; i++){
    gw[iw(i, 0)] = dZero();
    for (int j = 1; j < nz; j++) gw[iw(i, j)] = dMul(dSub(q[ip(i, j)], q[ip(i, j - 1)]), idzf[j]);
    gw[iw(i, nz)] = dNeg(dMul(q[ip(i, nz - 1)], idzfTop));
  }
}

static void applyL(const D128* q, D128* out){
  gradient(q);
  divergence(gu, gv, gw, out);
}

void applyLtoProbe(){ applyL(probe, probeOut); }

static D128 recip(const D128 &x){ return dDiv(dFromI64(1), x); }

static int factorPressure(){
  const int n = nr*nz, b = nz, bw = nz + 1;
  for (int color = 0; color < 5; color++){
    for (int k = 0; k < n; k++) probe[k] = dZero();
    for (int i = 0; i < nr; i++)
      for (int j = 0; j < nz; j++)
        if ((i + 2*j) % 5 == color) probe[ip(i, j)] = dFromI64(1);
    applyL(probe, probeOut);
    for (int i = 0; i < nr; i++)
      for (int j = 0; j < nz; j++){
        if ((i + 2*j) % 5 != color) continue;
        const int c = ip(i, j);
        band[c*bw] = dNeg(probeOut[c]);
        if (j + 1 < nz) band[(c + 1)*bw + 1] = dNeg(probeOut[c + 1]);
        if (i + 1 < nr) band[(c + nz)*bw + b] = dNeg(probeOut[c + nz]);
        upJ[c] = j > 0 ? dNeg(probeOut[c - 1]) : dZero();
        upI[c] = i > 0 ? dNeg(probeOut[c - nz]) : dZero();
      }
  }
  pressureAsymmetry = dZero();
  for (int i = 0; i < nr; i++)
    for (int j = 0; j < nz; j++){
      const int c = ip(i, j);
      if (j > 0){
        const D128 rel = dDiv(dAbs(dSub(upJ[c], band[c*bw + 1])), dAbs(band[c*bw + 1]));
        if (dCmp(rel, pressureAsymmetry) > 0) pressureAsymmetry = rel;
      }
      if (i > 0){
        const D128 rel = dDiv(dAbs(dSub(upI[c], band[c*bw + b])), dAbs(band[c*bw + b]));
        if (dCmp(rel, pressureAsymmetry) > 0) pressureAsymmetry = rel;
      }
    }
  D128 *t = tbuf;
  for (int r = 0; r < n; r++){
    const int c0 = r - b < 0 ? 0 : r - b;
    for (int c = c0; c < r; c++){
      D128 s = band[r*bw + (r - c)];
      const int k0 = (c - b > c0) ? c - b : c0;
      for (int k = k0; k < c; k++) s = dSub(s, dMul(t[k - c0], band[c*bw + (c - k)]));
      t[c - c0] = s;
    }
    D128 d = band[r*bw];
    for (int c = c0; c < r; c++){
      const D128 l = dMul(t[c - c0], dinv[c]);
      d = dSub(d, dMul(t[c - c0], l));
      band[r*bw + (r - c)] = l;
    }
    if (dSign(d) <= 0) return 3;
    band[r*bw] = d;
    dinv[r] = recip(d);
  }
  return 0;
}

static void solvePressure(const D128* rhs){
  const int n = nr*nz, b = nz, bw = nz + 1;
  for (int r = 0; r < n; r++){
    D128 s = dNeg(rhs[r]);
    const int c0 = r - b < 0 ? 0 : r - b;
    for (int c = c0; c < r; c++) s = dSub(s, dMul(band[r*bw + (r - c)], p[c]));
    p[r] = s;
  }
  for (int r = 0; r < n; r++) p[r] = dMul(p[r], dinv[r]);
  for (int r = n - 1; r >= 0; r--){
    D128 s = p[r];
    const int k1 = r + b >= n ? n - 1 : r + b;
    for (int k = r + 1; k <= k1; k++) s = dSub(s, dMul(band[k*bw + (k - r)], p[k]));
    p[r] = s;
  }
}

int prepare(){
  lastError = 0;
  rho = scal[0]; nu = scal[1]; gam = scal[2]; dt = scal[3]; href = scal[4]; uref = scal[5];
  const D128 one = dFromI64(1), two = dFromI64(2), mD = dFromI64(mMode), m2 = dFromI64((i64)mMode*mMode);
  dtnu = dMul(dt, nu); dtOverRho = dDiv(dt, rho); rhoOverDt = dDiv(rho, dt);
  twoRhoNu = dMul(dMul(two, rho), nu);
  {
    const D128 a = dSub(zf[nz], zf[nz - 1]), b = dSub(zf[nz], zf[nz - 2]), ba = dSub(b, a);
    wzA = dDiv(dAdd(a, b), dMul(a, b));
    wzB = dDiv(b, dMul(a, ba));
    wzC = dDiv(a, dMul(b, ba));
  }
  for (int i = 0; i <= nr; i++) idrf[i] = recip(drf[i]);
  for (int j = 0; j <= nz; j++) idzf[j] = recip(dzf[j]);
  for (int j = 0; j < nz; j++) idzc[j] = recip(dzc[j]);
  idzfTop = idzf[nz];
  for (int i = 0; i < nr; i++){
    const D128 vol = dMul(rc[i], drc[i]);
    sOutS[i] = dDiv(rf[i + 1], dMul(drf[i + 1], vol));
    sInS[i] = i == 0 ? dZero() : dDiv(rf[i], dMul(drf[i], vol));
    sM[i] = dDiv(m2, dMul(rc[i], rc[i]));
    bOut[i] = sOutS[i];
    bIn[i] = sInS[i];
    dWc[i] = vol;
    topCoef[i] = dMul(vol, idzf[nz]);
  }
  for (int i = 1; i < nr; i++){
    const D128 r = rf[i], sumW = dAdd(drc[i - 1], drc[i]);
    const D128 vol = dMul(r, dScale2(sumW, -1));
    aOut[i] = dDiv(dDiv(rc[i], drc[i]), vol);
    aIn[i] = dDiv(dDiv(rc[i - 1], drc[i - 1]), vol);
    const D128 r2 = dMul(r, r);
    cA[i] = dDiv(dAdd(m2, one), r2);
    cB[i] = dDiv(dMul(two, mD), r2);
    faW[i] = dDiv(drc[i - 1], sumW);
    fbW[i] = dDiv(drc[i], sumW);
    mOverRf[i] = dDiv(mD, r);
    gvA[i] = dNeg(dDiv(dMul(mD, drc[i - 1]), dMul(r, sumW)));
    gvB[i] = dNeg(dDiv(dMul(mD, drc[i]), dMul(r, sumW)));
  }
  axisR = recip(dMul(drc[0], rc[0]));
  gAxis = dDiv(dScale2(drc[0], -1), dMul(rc[0], rc[0]));
  for (int i = 0; i < nr; i++)
    for (int j = 0; j < nz; j++){
      const int c = ip(i, j);
      dAu[c] = dMul(dzc[j], rf[i + 1]);
      dBu[c] = dMul(dzc[j], rf[i]);
      dVc[c] = dScale2(dMul(dMul(mD, drc[i]), dzc[j]), -1);
    }
  const int err = factorPressure();
  if (err){ lastError = err; return err; }
  prepared = 1;
  return lastError;
}

static void stepOnce(const D128 &gNow){
  const int nu_ = (nr + 1)*nz, nw_ = nr*(nz + 1);
  surfaceSlopes();
  lapUV(u, v, dudz, lu);
  lapUV(v, u, dvdz, lv);
  lapW();
  const int i0 = mMode == 1 ? 0 : 1;
  if (i0 == 0) lapAxis(dudz);
  for (int k = 0; k < nu_; k++){ us[k] = u[k]; vs[k] = v[k]; }
  for (int k = 0; k < nw_; k++) ws[k] = w[k];
  for (int i = i0; i < nr; i++)
    for (int j = 0; j < nz; j++){
      const int c = iu(i, j);
      us[c] = dAdd(u[c], dMul(dtnu, lu[c]));
      vs[c] = dAdd(v[c], dMul(dtnu, lv[c]));
    }
  for (int i = 0; i < nr; i++)
    for (int j = 1; j <= nz; j++){
      const int c = iw(i, j);
      ws[c] = dAdd(w[c], dMul(dtnu, lw[c]));
    }
  surfacePressure(gNow);
  divergence(us, vs, ws, dv);
  const int np_ = nr*nz;
  for (int c = 0; c < np_; c++) dv[c] = dMul(dv[c], rhoOverDt);
  for (int i = 0; i < nr; i++) dv[ip(i, nz - 1)] = dSub(dv[ip(i, nz - 1)], dMul(topCoef[i], ps[i]));
  solvePressure(dv);
  gradient(p);
  for (int i = 0; i < nr; i++) gw[iw(i, nz)] = dAdd(gw[iw(i, nz)], dMul(ps[i], idzfTop));
  for (int i = i0; i < nr; i++)
    for (int j = 0; j < nz; j++){
      const int c = iu(i, j);
      u[c] = dSub(us[c], dMul(dtOverRho, gu[c]));
      v[c] = dSub(vs[c], dMul(dtOverRho, gv[c]));
    }
  for (int i = 0; i < nr; i++)
    for (int j = 1; j <= nz; j++){
      const int c = iw(i, j);
      w[c] = dSub(ws[c], dMul(dtOverRho, gw[c]));
    }
  for (int i = 0; i < nr; i++) w[iw(i, 0)] = dZero();
  for (int i = 0; i < nr; i++) eta[i] = dAdd(eta[i], dMul(dt, w[iw(i, nz)]));
}

int loadState(const D128* vec){
  const int nu_ = (nr + 1)*nz, nw_ = nr*(nz + 1), np_ = nr*nz;
  for (int k = 0; k < nu_; k++){ u[k] = dZero(); v[k] = dZero(); }
  for (int k = 0; k < nw_; k++) w[k] = dZero();
  for (int k = 0; k < np_; k++) p[k] = dZero();
  int q = 0;
  for (int i = 1; i < nr; i++) for (int j = 0; j < nz; j++) u[iu(i, j)] = dMul(vec[q++], uref);
  for (int i = 1; i < nr; i++) for (int j = 0; j < nz; j++) v[iu(i, j)] = dMul(vec[q++], uref);
  for (int i = 0; i < nr; i++) for (int j = 1; j <= nz; j++) w[iw(i, j)] = dMul(vec[q++], uref);
  for (int i = 0; i < nr; i++) eta[i] = dMul(vec[q++], href);
  if (mMode == 1)
    for (int j = 0; j < nz; j++){ u[iu(0, j)] = dMul(vec[q++], uref); v[iu(0, j)] = dNeg(u[iu(0, j)]); }
  return q;
}

int storeState(D128* vec){
  int q = 0;
  for (int i = 1; i < nr; i++) for (int j = 0; j < nz; j++) vec[q++] = dDiv(u[iu(i, j)], uref);
  for (int i = 1; i < nr; i++) for (int j = 0; j < nz; j++) vec[q++] = dDiv(v[iu(i, j)], uref);
  for (int i = 0; i < nr; i++) for (int j = 1; j <= nz; j++) vec[q++] = dDiv(w[iw(i, j)], uref);
  for (int i = 0; i < nr; i++) vec[q++] = dDiv(eta[i], href);
  if (mMode == 1) for (int j = 0; j < nz; j++) vec[q++] = dDiv(u[iu(0, j)], uref);
  return q;
}

int loadVecIn(){ if (!prepared) return 4; loadState(vecIn); return 0; }
int storeVecOut(){ storeState(vecOut); return 0; }

int runSteps(int from, int to){
  if (!prepared) return 4;
  if (from < 0 || to > steps || from > to) return 5;
  for (int n = from; n < to; n++) stepOnce(drive[n]);
  return lastError;
}

int applyPeriodMap(){
  if (!prepared) return 4;
  loadState(vecIn);
  for (int n = 0; n < steps; n++) stepOnce(drive[n]);
  storeState(vecOut);
  return lastError;
}

static D128 dot(const D128* a, const D128* b, int n){
  D128 s = dZero();
  for (int k = 0; k < n; k++) s = dAdd(s, dMul(a[k], b[k]));
  return s;
}

int arnoldiStart(){
  const D128 nrm = dSqrt(dot(vecIn, vecIn, nVec));
  if (dIsZero(nrm)) return 6;
  for (int k = 0; k < nVec; k++) basis[k] = dDiv(vecIn[k], nrm);
  return 0;
}

int arnoldiExtend(int j){
  if (!prepared) return 4;
  if (j < 0 || j >= kMax) return 5;
  const D128* vj = basis + j*nVec;
  loadState(vj);
  for (int n = 0; n < steps; n++) stepOnce(drive[n]);
  storeState(vecOut);
  for (int i = 0; i <= kMax; i++) hess[i*kMax + j] = dZero();
  for (int pass = 0; pass < 2; pass++)
    for (int i = 0; i <= j; i++){
      const D128* vi = basis + i*nVec;
      const D128 d = dot(vi, vecOut, nVec);
      for (int k = 0; k < nVec; k++) vecOut[k] = dSub(vecOut[k], dMul(d, vi[k]));
      hess[i*kMax + j] = dAdd(hess[i*kMax + j], d);
    }
  const D128 hn = dSqrt(dot(vecOut, vecOut, nVec));
  hess[(j + 1)*kMax + j] = hn;
  if (dIsZero(hn)) return 0;
  D128* vn = basis + (j + 1)*nVec;
  for (int k = 0; k < nVec; k++) vn[k] = dDiv(vecOut[k], hn);
  return lastError;
}

int combineBasis(int used){
  if (used < 1 || used > kMax + 1) return 5;
  for (int k = 0; k < nVec; k++) vecOut[k] = dZero();
  for (int j = 0; j < used; j++){
    const D128 c = coef[j];
    if (dIsZero(c)) continue;
    const D128* vj = basis + j*nVec;
    for (int k = 0; k < nVec; k++) vecOut[k] = dAdd(vecOut[k], dMul(c, vj[k]));
  }
  const D128 nrm = dSqrt(dot(vecOut, vecOut, nVec));
  if (dIsZero(nrm)) return 6;
  for (int k = 0; k < nVec; k++) vecIn[k] = dDiv(vecOut[k], nrm);
  return 0;
}

}
