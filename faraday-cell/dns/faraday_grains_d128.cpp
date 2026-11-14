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

static const int ARENA = 1800000;
static D128 arena[ARENA];
static int arenaUsed = 0, lastError = 0;
static int nGrains, faces, phases, mMode, ready;
static D128 scal[16];
static D128 *rf, *invDr, *Ut, *Vt, *dUt, *dVt, *normal, *sPr, *sPs, *sPt;
static D128 *gx, *gy, *gvx, *gvy;
static i32 stuckArena[200000];
static i32 *stuck;
static i32 slidSteps;
static D128 C, M, mfAdded, muS, muK, h, rMax, rMax2, denomInv, one;

static D128* take(int n){
  if (arenaUsed + n > ARENA){ lastError = 1; return arena; }
  D128* q = arena + arenaUsed; arenaUsed += n;
  for (int i = 0; i < n; i++) q[i] = dZero();
  return q;
}

int setupGrains(int N, int FACES, int PHASES, int M_){
  if (N < 1 || N > 200000 || FACES < 3 || PHASES < 1 || M_ < 0) return 5;
  arenaUsed = 0; lastError = 0; ready = 0; slidSteps = 0;
  nGrains = N; faces = FACES; phases = PHASES; mMode = M_;
  rf = take(faces); invDr = take(faces);
  Ut = take(phases*faces); Vt = take(phases*faces); dUt = take(phases*faces); dVt = take(phases*faces);
  normal = take(phases);
  sPr = take(faces); sPs = take(faces); sPt = take(faces);
  gx = take(N); gy = take(N); gvx = take(N); gvy = take(N);
  stuck = stuckArena;
  for (int n = 0; n < N; n++) stuck[n] = 1;
  return lastError;
}

D128* ptrScal(){ return scal; }
D128* ptrRf(){ return rf; }
D128* ptrUt(){ return Ut; } D128* ptrVt(){ return Vt; } D128* ptrDUt(){ return dUt; } D128* ptrDVt(){ return dVt; }
D128* ptrNormal(){ return normal; }
D128* ptrStreamR(){ return sPr; } D128* ptrStreamS(){ return sPs; } D128* ptrStreamT(){ return sPt; }
D128* ptrX(){ return gx; } D128* ptrY(){ return gy; } D128* ptrVX(){ return gvx; } D128* ptrVY(){ return gvy; }
i32* ptrStuck(){ return stuck; }
int getSlidSteps(){ return slidSteps; }
int getLastError(){ return lastError; }

int prepareGrains(){
  C = scal[0]; M = scal[1]; mfAdded = scal[2]; muS = scal[3]; muK = scal[4]; h = scal[5]; rMax = scal[6];
  rMax2 = dMul(rMax, rMax);
  one = dFromI64(1);
  for (int i = 0; i + 1 < faces; i++){
    const D128 d = dSub(rf[i + 1], rf[i]);
    if (dSign(d) <= 0) return 5;
    invDr[i] = dDiv(one, d);
  }
  denomInv = dDiv(one, dAdd(M, dMul(h, C)));
  ready = 1;
  return 0;
}

static int faceBelow(const D128 &r){
  int lo = 0, hi = faces - 2;
  while (lo < hi){
    const int mid = (lo + hi + 1) >> 1;
    if (dCmp(rf[mid], r) <= 0) lo = mid; else hi = mid - 1;
  }
  return lo;
}

static int floorSmall(const D128 &x){
  if (dIsZero(x) || x.s) return 0;
  if (x.e >= 0) return 0x7FFFFFFF;
  const int sh = -x.e;
  if (sh >= 128) return 0;
  u64 hi = x.hi, lo = x.lo;
  if (sh >= 64){ lo = hi >> (sh - 64); hi = 0; }
  else { lo = (lo >> sh) | (sh ? (hi << (64 - sh)) : 0); hi = sh ? (hi >> sh) : hi; }
  if (hi || lo > 0x7FFFFFFFULL) return 0x7FFFFFFF;
  return (int)lo;
}

static void angular(const D128 &c, const D128 &s, int power, D128 &cm, D128 &sm){
  D128 rr = one, ri = dZero(), br = c, bi = s;
  int e = power;
  while (e > 0){
    if (e & 1){ const D128 t = dSub(dMul(rr, br), dMul(ri, bi)); ri = dAdd(dMul(rr, bi), dMul(ri, br)); rr = t; }
    e >>= 1;
    if (e){ const D128 t = dSub(dMul(br, br), dMul(bi, bi)); bi = dScale2(dMul(br, bi), 1); br = t; }
  }
  cm = rr; sm = ri;
}

static int nSand, driftNr, driftNth, driftM, sandReady;
static D128 *sx, *sy, *driftUR, *driftG, *driftC, *driftInvDc;
static D128 driftDr, driftInvDr, sandRMax, sandRMax2;
static int sandArenaStart;

int setupSand(int N, int NRD, int NTHD, int MD){
  if (N < 1 || NRD < 2 || NTHD < 1 || MD < 0) return 5;
  if (sandReady) arenaUsed = sandArenaStart;
  sandArenaStart = arenaUsed;
  nSand = N; driftNr = NRD; driftNth = NTHD; driftM = MD; sandReady = 0;
  sx = take(N); sy = take(N);
  driftUR = take(NRD*NTHD); driftG = take(NRD*NTHD); driftC = take(NTHD); driftInvDc = take(NTHD);
  return lastError;
}

D128* ptrSandX(){ return sx; } D128* ptrSandY(){ return sy; }
D128* ptrDriftUR(){ return driftUR; } D128* ptrDriftG(){ return driftG; } D128* ptrDriftC(){ return driftC; }

int prepareSand(){
  driftDr = scal[8]; sandRMax = scal[9];
  if (dSign(driftDr) <= 0 || dSign(sandRMax) <= 0) return 5;
  one = dFromI64(1);
  driftInvDr = dDiv(one, driftDr);
  sandRMax2 = dMul(sandRMax, sandRMax);
  for (int j = 0; j + 1 < driftNth; j++){
    const D128 d = dSub(driftC[j], driftC[j + 1]);
    if (dSign(d) <= 0) return 5;
    driftInvDc[j] = dDiv(one, d);
  }
  sandReady = 1;
  return 0;
}

static void driftAtPoint(const D128 &x, const D128 &y, D128 &vx, D128 &vy){
  const D128 r2 = dAdd(dMul(x, x), dMul(y, y));
  if (dIsZero(r2)){ vx = dZero(); vy = dZero(); return; }
  const D128 r = dSqrt(r2), ir = dDiv(one, r);
  const D128 c0 = dMul(x, ir), s0 = dMul(y, ir);
  D128 wc = one, ws = dZero();
  if (driftNth > 1) angular(c0, s0, driftM, wc, ws);
  const D128 fr = dMul(r, driftInvDr);
  int i = floorSmall(fr);
  if (i > driftNr - 2) i = driftNr - 2;
  const D128 wr = dSub(fr, dFromI64(i)), wr0 = dSub(one, wr);
  D128 ur, g = dZero();
  if (driftNth > 1){
    int lo = 0, hi = driftNth - 2;
    while (lo < hi){ const int mid = (lo + hi + 1) >> 1; if (dCmp(driftC[mid], wc) >= 0) lo = mid; else hi = mid - 1; }
    const int j = lo;
    const D128 wt = dMul(dSub(driftC[j], wc), driftInvDc[j]), wt0 = dSub(one, wt);
    const D128 *A = driftUR, *B = driftG;
    const int a00 = i*driftNth + j, a01 = a00 + 1, a10 = a00 + driftNth, a11 = a10 + 1;
    ur = dAdd(dMul(wr0, dAdd(dMul(wt0, A[a00]), dMul(wt, A[a01]))), dMul(wr, dAdd(dMul(wt0, A[a10]), dMul(wt, A[a11]))));
    g = dAdd(dMul(wr0, dAdd(dMul(wt0, B[a00]), dMul(wt, B[a01]))), dMul(wr, dAdd(dMul(wt0, B[a10]), dMul(wt, B[a11]))));
  } else {
    ur = dAdd(dMul(wr0, driftUR[i]), dMul(wr, driftUR[i + 1]));
  }
  const D128 ut = dMul(ws, g);
  vx = dSub(dMul(ur, c0), dMul(ut, s0));
  vy = dAdd(dMul(ur, s0), dMul(ut, c0));
}

static D128 driftDtSlot, driftOut[2];
D128* ptrDriftDt(){ return &driftDtSlot; }
D128* ptrDriftOut(){ return driftOut; }

int advectSand(int nSteps){
  if (!sandReady) return 4;
  const D128 dt = driftDtSlot;
  for (int step = 0; step < nSteps; step++)
    for (int n = 0; n < nSand; n++){
      D128 vx, vy;
      driftAtPoint(sx[n], sy[n], vx, vy);
      sx[n] = dAdd(sx[n], dMul(vx, dt)); sy[n] = dAdd(sy[n], dMul(vy, dt));
      const D128 rr2 = dAdd(dMul(sx[n], sx[n]), dMul(sy[n], sy[n]));
      if (dCmp(rr2, sandRMax2) > 0){
        const D128 k = dDiv(sandRMax, dSqrt(rr2));
        sx[n] = dMul(sx[n], k); sy[n] = dMul(sy[n], k);
      }
    }
  return lastError;
}

int driftAt(int n){
  if (!sandReady) return 4;
  if (n < 0 || n >= nSand) return 5;
  driftAtPoint(sx[n], sy[n], driftOut[0], driftOut[1]);
  return 0;
}

static const int RASTER_MAX = 512;
static unsigned short occ[RASTER_MAX*RASTER_MAX];
static unsigned char cover[RASTER_MAX*RASTER_MAX];
static int rasterN, rasterC, rasterR;
static u32 packState = 0x9E3779B9u;
static D128 rasterScale, rasterInvScale, rasterCD;

static u32 packRand(){ u32 a = packState; a ^= a << 13; a ^= a >> 17; a ^= a << 5; packState = a; return a; }

int setupRaster(int n, int center, int radius, u32 seed){
  if (n < 8 || n > RASTER_MAX || center < 1 || radius < 1 || center + radius >= n) return 5;
  rasterN = n; rasterC = center; rasterR = radius; packState = seed ? seed : 0x9E3779B9u;
  const i64 lim = (2*(i64)radius + 1)*(2*(i64)radius + 1);
  for (int y = 0; y < n; y++)
    for (int x = 0; x < n; x++){
      const i64 dx = 2*(i64)(x - center) + 1, dy = 2*(i64)(y - center) + 1;
      cover[y*n + x] = dx*dx + dy*dy <= lim ? 1 : 0;
    }
  rasterScale = scal[10];
  if (dSign(rasterScale) <= 0) return 5;
  rasterInvScale = dDiv(dFromI64(1), rasterScale);
  rasterCD = dFromI64(center);
  return 0;
}

unsigned short* ptrOcc(){ return occ; }

static int pixelOf(const D128 &v){ return floorSmall(dAdd(rasterCD, dMul(v, rasterScale))); }

int binSand(){
  if (!sandReady) return 4;
  const int n = rasterN;
  for (int k = 0; k < n*n; k++) occ[k] = 0;
  for (int g = 0; g < nSand; g++){
    const int x = pixelOf(sx[g]), y = pixelOf(sy[g]);
    if (x >= 0 && y >= 0 && x < n && y < n) occ[y*n + x]++;
  }
  return 0;
}

int packSand(int budget){
  if (!sandReady) return -4;
  const int n = rasterN;
  binSand();
  int moved = 0;
  for (int g = 0; g < nSand && budget > 0; g++){
    const int xi = pixelOf(sx[g]), yi = pixelOf(sy[g]);
    if (xi < 0 || yi < 0 || xi >= n || yi >= n) continue;
    const int c = yi*n + xi;
    if (occ[c] <= 1) continue;
    budget--;
    for (int ring = 1; ring <= 24; ring++){
      int seen = 0, pickX = 0, pickY = 0, pickC = -1;
      for (int dy = -ring; dy <= ring; dy++){
        const bool edge = dy == ring || dy == -ring;
        for (int dx = -ring; dx <= ring; dx++){
          if (!edge && dx != ring && dx != -ring) continue;
          const int nx = xi + dx, ny = yi + dy;
          if (nx < 1 || ny < 1 || nx >= n - 1 || ny >= n - 1) continue;
          const int nc = ny*n + nx;
          if (occ[nc] + 1 > 1 || !cover[nc]) continue;
          seen++;
          if (seen == 1 || (packRand() % (u32)seen) == 0){ pickX = nx; pickY = ny; pickC = nc; }
        }
      }
      if (seen > 0){
        occ[c]--; occ[pickC]++;
        const D128 fx = dAdd(dScale2(one, -2), dScale2(dFromI64((i64)(packRand() >> 1)), -32));
        const D128 fy = dAdd(dScale2(one, -2), dScale2(dFromI64((i64)(packRand() >> 1)), -32));
        sx[g] = dMul(dSub(dAdd(dFromI64(pickX), fx), rasterCD), rasterInvScale);
        sy[g] = dMul(dSub(dAdd(dFromI64(pickY), fy), rasterCD), rasterInvScale);
        moved++;
        break;
      }
    }
  }
  return moved;
}

int advanceGrains(int nSteps, int phase0){
  if (!ready) return 4;
  if (phase0 < 0 || phase0 >= phases) return 5;
  for (int step = 0; step < nSteps; step++){
    const int ph = (phase0 + step) % phases;
    const D128 *U = Ut + ph*faces, *V = Vt + ph*faces, *dU = dUt + ph*faces, *dV = dVt + ph*faces;
    const D128 Nf = normal[ph];
    const D128 stickLimit = dMul(muS, Nf), slide = dMul(muK, Nf), stick2 = dMul(stickLimit, stickLimit);
    for (int n = 0; n < nGrains; n++){
      const D128 px = gx[n], py = gy[n];
      const D128 r2 = dAdd(dMul(px, px), dMul(py, py));
      D128 c = one, s = dZero(), r = dZero();
      if (!dIsZero(r2)){ r = dSqrt(r2); const D128 ir = dDiv(one, r); c = dMul(px, ir); s = dMul(py, ir); }
      D128 cm, sm;
      angular(c, s, mMode, cm, sm);
      const int i = faceBelow(r);
      const D128 wR = dMul(dSub(r, rf[i]), invDr[i]);
      const D128 wL = dSub(one, wR);
      const D128 ur = dAdd(dMul(wL, U[i]), dMul(wR, U[i + 1])), ut = dAdd(dMul(wL, V[i]), dMul(wR, V[i + 1]));
      const D128 ar = dAdd(dMul(wL, dU[i]), dMul(wR, dU[i + 1])), at = dAdd(dMul(wL, dV[i]), dMul(wR, dV[i + 1]));
      D128 vr = dMul(ur, cm), vt = dMul(ut, sm);
      const D128 qr = dMul(ar, cm), qt = dMul(at, sm);
      const D128 spr = dAdd(dMul(wL, sPr[i]), dMul(wR, sPr[i + 1]));
      const D128 sps = dAdd(dMul(wL, sPs[i]), dMul(wR, sPs[i + 1]));
      const D128 spt = dAdd(dMul(wL, sPt[i]), dMul(wR, sPt[i + 1]));
      vr = dAdd(vr, dAdd(dMul(spr, dMul(cm, cm)), dMul(sps, dMul(sm, sm))));
      vt = dAdd(vt, dMul(spt, dMul(sm, cm)));
      const D128 ufx = dSub(dMul(vr, c), dMul(vt, s)), ufy = dAdd(dMul(vr, s), dMul(vt, c));
      const D128 afx = dSub(dMul(qr, c), dMul(qt, s)), afy = dAdd(dMul(qr, s), dMul(qt, c));
      const D128 fx = dAdd(dMul(C, ufx), dMul(mfAdded, afx)), fy = dAdd(dMul(C, ufy), dMul(mfAdded, afy));
      if (stuck[n]){
        const D128 f2 = dAdd(dMul(fx, fx), dMul(fy, fy));
        if (dCmp(f2, stick2) <= 0) continue;
        stuck[n] = 0;
        const D128 fs = dSqrt(f2), k = dSub(one, dDiv(slide, fs));
        const D128 nx = dMul(dMul(h, dMul(fx, k)), denomInv), ny = dMul(dMul(h, dMul(fy, k)), denomInv);
        gvx[n] = nx; gvy[n] = ny;
        gx[n] = dAdd(px, dMul(nx, h)); gy[n] = dAdd(py, dMul(ny, h));
      } else {
        const D128 vx = gvx[n], vy = gvy[n];
        const D128 sp2 = dAdd(dMul(vx, vx), dMul(vy, vy));
        D128 dx = dZero(), dy = dZero();
        if (!dIsZero(sp2)){ const D128 isp = dDiv(one, dSqrt(sp2)); dx = dMul(vx, isp); dy = dMul(vy, isp); }
        const D128 nvx = dMul(dAdd(dMul(M, vx), dMul(h, dSub(fx, dMul(slide, dx)))), denomInv);
        const D128 nvy = dMul(dAdd(dMul(M, vy), dMul(h, dSub(fy, dMul(slide, dy)))), denomInv);
        if (!dIsZero(sp2) && dSign(dAdd(dMul(nvx, vx), dMul(nvy, vy))) <= 0){
          gvx[n] = dZero(); gvy[n] = dZero();
          const D128 f2 = dAdd(dMul(fx, fx), dMul(fy, fy));
          stuck[n] = dCmp(f2, stick2) <= 0 ? 1 : 0;
        } else { gvx[n] = nvx; gvy[n] = nvy; }
        gx[n] = dAdd(px, dMul(gvx[n], h)); gy[n] = dAdd(py, dMul(gvy[n], h));
        slidSteps++;
      }
      const D128 rr2 = dAdd(dMul(gx[n], gx[n]), dMul(gy[n], gy[n]));
      if (dCmp(rr2, rMax2) > 0){
        const D128 inv = dDiv(one, dSqrt(rr2));
        const D128 ex = dMul(gx[n], inv), ey = dMul(gy[n], inv);
        gx[n] = dMul(ex, rMax); gy[n] = dMul(ey, rMax);
        const D128 out = dAdd(dMul(gvx[n], ex), dMul(gvy[n], ey));
        if (dSign(out) > 0){ gvx[n] = dSub(gvx[n], dMul(out, ex)); gvy[n] = dSub(gvy[n], dMul(out, ey)); }
      }
    }
  }
  return lastError;
}

}
