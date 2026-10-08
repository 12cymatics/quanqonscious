// The nonlinear three-dimensional cell solver, in C++, compiled to WebAssembly.
//
// WHY. dns/faraday-cell3d.js is the reference and it is correct, and it is also
// what the page draws, which means its step cost is the frame rate. Measured on
// this container: 85.6 ms a step at 16x24x10 and 64.3 ms at 10x24x8, against a
// capillary-limited step of ~5e-5 s -- 1278x slower than real time. Nothing in
// the discretisation can be cheapened without changing the answer, so what is
// left is to run the same arithmetic faster.
//
// HOW, AND WHY IT CAN BE TRUSTED. This is a transcription of the JavaScript, not
// a reimplementation: same discretisation, same flux forms, same conjugate
// gradient with the same tolerance and cap, same order of operations down to the
// grouping of each sum. It is compiled freestanding for wasm32 by clang -- no
// Emscripten, no libc -- and uses only + - * / and sqrt. sqrt is an IEEE-754
// exact operation and a single wasm instruction, so it agrees with JavaScript's
// Math.sqrt to the last bit.
//
// The transcendentals are NOT computed here. The drive needs cos(omega t) and
// the stability limit needs tanh; JavaScript evaluates both and passes the
// numbers in. That is what makes bit-for-bit parity achievable rather than
// approximate, and dns/check-cell3d-wasm.mjs asserts it on every array.
//
// The grid is not built here either, for the same reason it is not built in
// faraday_disc.cpp: JavaScript builds it and writes the node arrays into this
// module's memory, so the graded-grid code has exactly one implementation and
// the two cannot diverge on it.
//
// Nothing is approximated for speed. No float, no fast-math, no reassociation:
// -ffp-contract=off keeps the compiler from fusing a multiply and an add,
// because an FMA rounds once where JavaScript rounds twice and the two would
// drift apart.
//
// WHAT IS NOT HERE. The renderer's own chains -- `etaAt`, `etaDotAt`,
// `etaSlopeAt`, `etaDotSlopeAt` and the extended arrays they read -- stay in
// JavaScript, because the renderer is JavaScript and they are called once per
// pixel per frame, not once per step. `refreshMetric` below fills every array
// `step` reads AND those extended arrays, so the gate can compare all of them
// and the two implementations cannot drift on any of it.

extern "C" {

/* Freestanding means no libc, and -O3 turns the zero-fill loops below into calls
   to memset. Supplying it here rather than importing it keeps the module with no
   imports at all, so the browser and node both instantiate it with an empty
   import object and there is nothing to get wrong. */
void* memset(void* dst, int c, unsigned long n){
  unsigned char* q = (unsigned char*)dst;
  for (unsigned long i = 0; i < n; i++) q[i] = (unsigned char)c;
  return dst;
}

/* And memcpy, for the same reason: -O3 emits a call to it for the small
   fixed-size copies in the family descriptors and the stencil buffers. Supplying
   it keeps the module with NO IMPORTS AT ALL, so node and a browser both
   instantiate it with an empty import object and there is nothing to get wrong
   between them. The gate asserts the import list is empty, because an import that
   reappeared would be a libc call -- and a libm call is exactly what cannot be
   bit-identical to JavaScript. */
void* memcpy(void* dst, const void* src, unsigned long n){
  unsigned char* d = (unsigned char*)dst;
  const unsigned char* q = (const unsigned char*)src;
  for (unsigned long i = 0; i < n; i++) d[i] = q[i];
  return dst;
}

/* A value no effective gravity can be, so that a step run without JavaScript
   having set the drive's cos for this instant REFUSES instead of silently reusing
   the previous instant's. */
#define CELL3D_GRAVITY_UNSET (-1.0e308)

static const int ARENA = 12000000;         // doubles; 96 MB
static double arena[ARENA];
static int   arenaUsed = 0;

/* 0 ok. 1 the arena is exhausted. 2 the conjugate gradient did not converge.
   3 the free surface has reached the floor, which is the one refusal the
   JavaScript raises from refreshMetric: z = sigma*(h + eta) requires a positive
   depth everywhere, and a non-positive one means there is no single-valued
   surface left to follow. 5 to 7 are named where they are raised. (4 was the
   diagonal preconditioner's, which the flat-cell one replaced; that one is built in
   JavaScript and refuses there.) The caller turns each of these into the same Error
   the JavaScript throws -- this module never continues past one. */
static int lastError = 0;
static int errI = -1, errK = -1;           // where case 3 was found

static int nr, nth, nz, nw;
static int NU, NV, NW, NP, NE, NX;
static double R, h, rho, nu, gamma_, gBase, dth;
static int pinned;                         // the contact line: 0 free, 1 pinned

static double *rf, *sf, *rc, *sc, *drc, *dsc, *drf, *dsf, *rx;
static double *u, *v, *w, *om, *p, *eta;
static double *H, *Hr, *Hth, *Hdr, *Hdth, *Ht;
static double *Hx, *Hxr, *Hxt;
static double *Ex, *Er, *Eth, *Edr, *Edth, *Exr, *Ext;
static double *Tx, *Txr, *Txt, *Tr, *Tth, *Tdr, *Tdth;
static double *gu, *gv, *gw, *gom, *divw;
static double *cgr, *cgd, *cgq, *cgz;
/* The pressure solve's preconditioner: the azimuthal Fourier basis and the per-mode band
   Cholesky factors, both written in by JavaScript (dftc, dfts, pcband), and the two working
   rows of an application. buildPreconditioner in the JavaScript says what it is. */
static double *dftc, *dfts, *pcband, *pchatc, *pchats;
static int cgIters = 0; static double cgResidual = 0.0;
static double *lapU, *lapV, *lapW, *advU, *advV, *advW;
static double *fsr, *fst, *fsz, *kap, *psurf;
static double *rbU, *sbW, *axEx;
/* H at every node of each family, one table per family, filled at the end of the
   metric; and famLaplacian's face terms, each formed once and read by both cells it
   separates. The JavaScript's FAM[*].Hcol and _fRlo/_fRhi/_fT/_fS. */
static double *hcolP, *hcolU, *hcolV, *hcolW;
static double *fRlo, *fRhi, *fT, *fS;

/* The integer bracket tables. A separate arena because the doubles' one holds
   doubles: a reinterpreted slice would be a different alignment on a different
   target, and this costs four kilobytes. */
static const int IARENA = 400000;
static int iarena[IARENA];
static int iarenaUsed = 0;
static int* itake(int n){
  if (iarenaUsed + n > IARENA){ lastError = 1; return iarena; }
  int* q = iarena + iarenaUsed; iarenaUsed += n;
  for (int i = 0; i < n; i++) q[i] = 0;
  return q;
}

/* Node geometry, one descriptor per staggered family -- the C++ face of
   FaradayCell3D's FAM. Every family shares the same periodic uniform theta and
   the same flux algebra; they differ only in where their nodes sit in r and
   sigma, by how much their theta nodes are offset from the cell centres, and in
   what happens at each boundary. So the viscous operator is written once and
   instantiated four times. */
struct Fam {
  int axisSign;        // +1 for a scalar, -1 for a horizontal vector component
  double thOff;        // theta offset of the nodes, in cells
  int stride;          // the sigma stride of the family's index map
  const double* rn; const double* rb; int nI;
  const double* sn; const double* sb; int nJ;
  int rLo, rHi, sLo, sHi;
  const int* hBr;      // the radial bracket of each node in the extended list
  double* Hcol;        // H at each node, (a*nth + k), refreshed with the metric
};
static Fam FAMP, FAMU, FAMV, FAMW;

static double* take(int n){
  if (arenaUsed + n > ARENA){ lastError = 1; return arena; }
  double* q = arena + arenaUsed; arenaUsed += n;
  for (int i = 0; i < n; i++) q[i] = 0.0;
  return q;
}

/* theta is periodic, so k wraps. Exactly the JavaScript's `kw`: the double
   modulo is what makes a negative k land on the right column instead of reading
   a neighbouring radius and looking like a physical asymmetry. */
static inline int kw(int k){ return ((k % nth) + nth) % nth; }
static inline int iu(int i, int k, int j){ return (i*nth + kw(k))*nz + j; }
static inline int iv(int i, int k, int j){ return (i*nth + kw(k))*nz + j; }
static inline int iw(int i, int k, int j){ return (i*nth + kw(k))*nw + j; }
static inline int ip(int i, int k, int j){ return (i*nth + kw(k))*nz + j; }
static inline int ie(int i, int k){ return i*nth + kw(k); }
/* Every family's index map is the same expression with its own sigma stride:
   nz for p, u and v, nz + 1 for w, whose nodes are the sigma faces. */
static inline int famIdx(const Fam& f, int a, int k, int b){
  return (a*nth + kw(k))*f.stride + b;
}

int cell3d_init(int nr_, int nth_, int nz_, int pinned_,
                double R_, double h_, double rho_, double nu_,
                double gamma__, double g_, double dth_){
  arenaUsed = 0; lastError = 0; errI = -1; errK = -1;
  nr = nr_; nth = nth_; nz = nz_; nw = nz_ + 1; pinned = pinned_;
  R = R_; h = h_; rho = rho_; nu = nu_; gamma_ = gamma__; gBase = g_; dth = dth_;

  NU = (nr + 1)*nth*nz; NV = nr*nth*nz; NW = nr*nth*nw;
  NP = nr*nth*nz;       NE = nr*nth;    NX = (nr + 2)*nth;

  rf = take(nr + 1); sf = take(nz + 1); rc = take(nr); sc = take(nz);
  drc = take(nr); dsc = take(nz); drf = take(nr + 1); dsf = take(nz + 1);
  rx = take(nr + 2);

  u = take(NU); v = take(NV); w = take(NW); om = take(NW);
  p = take(NP); eta = take(NE);

  H = take(NE); Hr = take((nr + 1)*nth); Hth = take(NE);
  Hdr = take(NE); Hdth = take(NE); Ht = take(NE);
  Hx = take(NX); Hxr = take(NX); Hxt = take(NX);

  Ex = take(NX); Er = take((nr + 1)*nth); Eth = take(NE);
  Edr = take(NE); Edth = take(NE); Exr = take(NX); Ext = take(NX);

  Tx = take(NX); Txr = take(NX); Txt = take(NX);
  Tr = take((nr + 1)*nth); Tth = take(NE); Tdr = take(NE); Tdth = take(NE);

  gu = take(NU); gv = take(NV); gw = take(NW); gom = take(NW);
  divw = take(NP);
  cgr = take(NP); cgd = take(NP); cgq = take(NP); cgz = take(NP);
  {
    const int M = (nth >> 1) + 1, n2 = nr*nz, W = nz + 1;
    dftc = take(M*nth); dfts = take(M*nth);
    pcband = take(M*n2*W); pchatc = take(M*n2); pchats = take(M*n2);
  }

  lapU = take(NU); lapV = take(NV); lapW = take(NW);
  advU = take(NU); advV = take(NV); advW = take(NW);
  fsr = take(NE); fst = take(NE); fsz = take(NE);
  kap = take(NE); psurf = take(NE);
  rbU = take(nr + 2); sbW = take(nz + 2);
  /* axisU's one row of extrapolated values. From the arena rather than the stack:
     wasm-ld gives this module a 64 KB stack by default, and a fixed-size local
     array sized for the largest plausible nth would be most of it. */
  axEx = take(nth);
  hcolP = take(NE); hcolU = take((nr + 1)*nth); hcolV = take(NE); hcolW = take(NE);
  fRlo = take(nth*(nz + 1)); fRhi = take(nth*(nz + 1)); fT = take(nth*(nz + 1));
  fS = take(nz + 2);

  cgIters = 0; cgResidual = 0.0;
  return lastError;
}

int    cell3d_error(){ return lastError; }
int    cell3d_errorI(){ return errI; }
int    cell3d_errorK(){ return errK; }
void   cell3d_clearError(){ lastError = 0; errI = -1; errK = -1; }
int    cell3d_arenaUsed(){ return arenaUsed; }

/* The pointers JavaScript writes the grid and the state through, and reads the
   metric back out of. A wasm32 `double*` IS a byte offset into this module's
   linear memory, which is exactly what a Float64Array view needs, so these are
   returned as pointers and used as offsets with no arithmetic in between -- the
   same thing faraday_disc.cpp's ptrRf() does.

   One keyed function rather than thirty-seven named ones: the key is checked
   against the list in dns/faraday-cell3d-wasm.js by the gate, so a renumbering
   cannot silently point JavaScript at the wrong array. An unknown key returns
   null, and the loader refuses on it rather than reading from offset zero --
   which is `rf`, and would have looked like a grid that had gone wrong. */
double* cell3d_ptr(int which){
  switch (which){
    case  0: return rf;
    case  1: return sf;
    case  2: return rc;
    case  3: return sc;
    case  4: return drc;
    case  5: return dsc;
    case  6: return drf;
    case  7: return dsf;
    case  8: return rx;
    case  9: return u;
    case 10: return v;
    case 11: return w;
    case 12: return om;
    case 13: return p;
    case 14: return eta;
    case 15: return H;
    case 16: return Hr;
    case 17: return Hth;
    case 18: return Hdr;
    case 19: return Hdth;
    case 20: return Ht;
    case 21: return Hx;
    case 22: return Hxr;
    case 23: return Hxt;
    case 24: return Ex;
    case 25: return Er;
    case 26: return Eth;
    case 27: return Edr;
    case 28: return Edth;
    case 29: return Exr;
    case 30: return Ext;
    case 31: return Tx;
    case 32: return Txr;
    case 33: return Txt;
    case 34: return Tr;
    case 35: return Tth;
    case 36: return Tdr;
    case 37: return Tdth;
    case 38: return gu;
    case 39: return gv;
    case 40: return gw;
    case 41: return gom;
    case 42: return divw;
    case 43: return cgr;
    case 44: return cgd;
    case 45: return cgq;
    case 46: return cgz;
    case 47: return lapU;
    case 48: return lapV;
    case 49: return lapW;
    case 50: return advU;
    case 51: return advV;
    case 52: return advW;
    case 53: return fsr;
    case 54: return fst;
    case 55: return fsz;
    case 56: return kap;
    case 57: return psurf;
    case 58: return hcolP;
    case 59: return hcolU;
    case 60: return hcolV;
    case 61: return hcolW;
    case 62: return dftc;
    case 63: return dfts;
    case 64: return pcband;
    case 65: return pchatc;
    case 66: return pchats;
    default: return 0;
  }
}

/* The sizes JavaScript needs to build its views, by the same keys. A view built
   one element short would read a neighbouring array's first value as its own
   last, which is a defect no comparison of the overlapping part can see. */
int cell3d_len(int which){
  switch (which){
    case  0: return nr + 1;
    case  1: return nz + 1;
    case  2: return nr;
    case  3: return nz;
    case  4: return nr;
    case  5: return nz;
    case  6: return nr + 1;
    case  7: return nz + 1;
    case  8: return nr + 2;
    case  9: return NU;
    case 10: return NV;
    case 11: return NW;
    case 12: return NW;
    case 13: return NP;
    case 14: return NE;
    case 15: return NE;
    case 16: return (nr + 1)*nth;
    case 17: return NE;
    case 18: return NE;
    case 19: return NE;
    case 20: return NE;
    case 21: return NX;
    case 22: return NX;
    case 23: return NX;
    case 24: return NX;
    case 25: return (nr + 1)*nth;
    case 26: return NE;
    case 27: return NE;
    case 28: return NE;
    case 29: return NX;
    case 30: return NX;
    case 31: return NX;
    case 32: return NX;
    case 33: return NX;
    case 34: return (nr + 1)*nth;
    case 35: return NE;
    case 36: return NE;
    case 37: return NE;
    case 38: return NU;
    case 39: return NV;
    case 40: return NW;
    case 41: return NW;
    case 42: return NP;
    case 43: return NP;
    case 44: return NP;
    case 45: return NP;
    case 46: return NP;
    case 47: return NU;
    case 48: return NV;
    case 49: return NW;
    case 50: return NU;
    case 51: return NV;
    case 52: return NW;
    case 53: return NE;
    case 54: return NE;
    case 55: return NE;
    case 56: return NE;
    case 57: return NE;
    case 58: return NE;
    case 59: return (nr + 1)*nth;
    case 60: return NE;
    case 61: return NE;
    case 62: return ((nth >> 1) + 1)*nth;
    case 63: return ((nth >> 1) + 1)*nth;
    case 64: return ((nth >> 1) + 1)*nr*nz*(nz + 1);
    case 65: return ((nth >> 1) + 1)*nr*nz;
    case 66: return ((nth >> 1) + 1)*nr*nz;
    default: return -1;
  }
}

/* The family descriptors, and the radial bracket of every family node in the
   extended node list. Built AFTER JavaScript has written the grid in, because
   they are made of it -- which is why this is not part of cell3d_init.

   The bracket is the one HatH's linear search would find; finding it once per
   node here instead of on every reconstruction is what the JavaScript does too,
   and the search is integer comparison on the same doubles, so both sides find
   the same bracket. colValueAtZ was 38.5 per cent of a step before this table
   existed, and at the rim node that search walked the whole list. */
void cell3d_buildFamilies(){
  iarenaUsed = 0;
  rbU[0] = 0.0;
  for (int i = 0; i < nr; i++) rbU[i+1] = rc[i];
  rbU[nr+1] = R;
  sbW[0] = 0.0;
  for (int j = 0; j < nz; j++) sbW[j+1] = sc[j];
  sbW[nz+1] = 1.0;

  FAMP.axisSign = +1; FAMP.thOff = 0.5; FAMP.stride = nz;
  FAMP.rn = rc; FAMP.rb = rf;  FAMP.nI = nr; FAMP.rLo = 0; FAMP.rHi = nr - 1;
  FAMP.sn = sc; FAMP.sb = sf;  FAMP.nJ = nz; FAMP.sLo = 0; FAMP.sHi = nz - 1;
  FAMP.Hcol = hcolP;

  FAMU.axisSign = -1; FAMU.thOff = 0.5; FAMU.stride = nz;
  FAMU.rn = rf; FAMU.rb = rbU; FAMU.nI = nr + 1; FAMU.rLo = 1; FAMU.rHi = nr - 1;
  FAMU.sn = sc; FAMU.sb = sf;  FAMU.nJ = nz; FAMU.sLo = 0; FAMU.sHi = nz - 1;
  FAMU.Hcol = hcolU;

  FAMV.axisSign = -1; FAMV.thOff = 0.0; FAMV.stride = nz;
  FAMV.rn = rc; FAMV.rb = rf;  FAMV.nI = nr; FAMV.rLo = 0; FAMV.rHi = nr - 1;
  FAMV.sn = sc; FAMV.sb = sf;  FAMV.nJ = nz; FAMV.sLo = 0; FAMV.sHi = nz - 1;
  FAMV.Hcol = hcolV;

  /* w's nodes run 0..nz and its viscous term is solved on 1..nz, the surface
     node included: that node is a solved advection unknown and the traction
     closes its viscous side, over the same half control volume the advection
     uses, because it is that shared volume which makes the discrete energy
     identity exact. */
  FAMW.axisSign = +1; FAMW.thOff = 0.5; FAMW.stride = nz + 1;
  FAMW.rn = rc; FAMW.rb = rf;  FAMW.nI = nr; FAMW.rLo = 0; FAMW.rHi = nr - 1;
  FAMW.sn = sf; FAMW.sb = sbW; FAMW.nJ = nz + 1; FAMW.sLo = 1; FAMW.sHi = nz;
  FAMW.Hcol = hcolW;

  Fam* fams[4] = { &FAMP, &FAMU, &FAMV, &FAMW };
  for (int t = 0; t < 4; t++){
    Fam& fam = *fams[t];
    int* br = itake(fam.nI);
    for (int a = 0; a < fam.nI; a++){
      int b = 0;
      while (b < nr && rx[b+1] < fam.rn[a]) b++;
      br[a] = b > nr ? nr : b;
    }
    fam.hBr = br;
  }
}

/* ---- the metric -------------------------------------------------------- */

/* H and its derivatives from the current eta, and the extended columns every
   reconstruction interpolates in. A transcription of `refreshMetric`, in its
   order, including the two groupings that are load bearing there:

   - the radial face value is written as an increment from one end,
     lo + (a/(a+b))*(hi-lo), and not as the weighted sum (b*lo + a*hi)/(a+b).
     For lo == hi the weighted form rounds twice and lands within an ulp, and the
     centred slope then differences two such values over drc, amplifying that ulp
     by h/drc. A flat surface would carry slopes of 1e-13 instead of zero.
   - eta's own faces and slopes are differenced from eta and NEVER from H. H's
     face values are (h + eta_lo) + f*((h + eta_hi) - (h + eta_lo)), and that
     inner difference loses the low bits: with h = 3e-3 the ulp of h + eta is
     4.3e-19. The solver does not care, the renderer does, and the two chains are
     kept separate so H's stays untouched. */
static double HatHBr(int a, double r, double th);

int cell3d_refreshMetric(){
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++){
      const int e = ie(i, k);
      H[e] = h + eta[e];
      if (!(H[e] > 0)){ lastError = 3; errI = i; errK = k; return lastError; }
    }

  for (int k = 0; k < nth; k++){
    Hr[0*nth + k] = H[ie(0, k)];
    for (int i = 1; i < nr; i++){
      const double a = drc[i-1], b = drc[i];
      const double lo = H[ie(i-1, k)], hi = H[ie(i, k)];
      Hr[i*nth + k] = lo + (a/(a + b))*(hi - lo);
    }
    Hr[nr*nth + k] = pinned ? h : H[ie(nr-1, k)];
  }
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++)
      Hth[i*nth + k] = 0.5*(H[ie(i, k-1)] + H[ie(i, k)]);
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++){
      const int e = ie(i, k);
      Hdr[e]  = (Hr[(i+1)*nth + kw(k)] - Hr[i*nth + kw(k)])/drc[i];
      Hdth[e] = (Hth[i*nth + kw(k+1)] - Hth[i*nth + kw(k)])/dth;
    }

  const int half = nth >> 1;
  const int freeCL = pinned ? 0 : 1;
  for (int k = 0; k < nth; k++){
    const int ka = kw(k + half);
    Hx [0*nth + k] =  H[ie(0, ka)];
    Hxr[0*nth + k] = -Hdr[ie(0, ka)];
    Hxt[0*nth + k] =  Hdth[ie(0, ka)];
    for (int i = 0; i < nr; i++){
      Hx [(i+1)*nth + k] = H[ie(i, k)];
      Hxr[(i+1)*nth + k] = Hdr[ie(i, k)];
      Hxt[(i+1)*nth + k] = Hdth[ie(i, k)];
    }
    Hx [(nr+1)*nth + k] = freeCL ? H[ie(nr-1, k)] : h;
    Hxr[(nr+1)*nth + k] = freeCL ? 0.0 : (h - H[ie(nr-1, k)])/drf[nr];
    Hxt[(nr+1)*nth + k] = freeCL ? Hdth[ie(nr-1, k)] : 0.0;
  }

  for (int k = 0; k < nth; k++){
    const int ka = kw(k + half);
    Ex[0*nth + k] = eta[ie(0, ka)];
    for (int i = 0; i < nr; i++) Ex[(i+1)*nth + k] = eta[ie(i, k)];
    Ex[(nr+1)*nth + k] = freeCL ? eta[ie(nr-1, k)] : 0.0;
  }

  for (int k = 0; k < nth; k++){
    Er[0*nth + k] = eta[ie(0, k)];
    for (int i = 1; i < nr; i++){
      const double a = drc[i-1], b = drc[i];
      const double lo = eta[ie(i-1, k)], hi = eta[ie(i, k)];
      Er[i*nth + k] = lo + (a/(a + b))*(hi - lo);
    }
    Er[nr*nth + k] = freeCL ? eta[ie(nr-1, k)] : 0.0;
  }
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++)
      Eth[i*nth + k] = 0.5*(eta[ie(i, k-1)] + eta[ie(i, k)]);
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++){
      const int e = ie(i, k);
      Edr[e]  = (Er[(i+1)*nth + kw(k)] - Er[i*nth + kw(k)])/drc[i];
      Edth[e] = (Eth[i*nth + kw(k+1)] - Eth[i*nth + kw(k)])/dth;
    }
  for (int k = 0; k < nth; k++){
    const int ka = kw(k + half);
    Exr[0*nth + k] = -Edr[ie(0, ka)];
    Ext[0*nth + k] =  Edth[ie(0, ka)];
    for (int i = 0; i < nr; i++){
      Exr[(i+1)*nth + k] = Edr[ie(i, k)];
      Ext[(i+1)*nth + k] = Edth[ie(i, k)];
    }
    Exr[(nr+1)*nth + k] = freeCL ? 0.0 : (0.0 - eta[ie(nr-1, k)])/drf[nr];
    Ext[(nr+1)*nth + k] = freeCL ? Edth[ie(nr-1, k)] : 0.0;
  }

  for (int k = 0; k < nth; k++){
    Tr[0*nth + k] = Ht[ie(0, k)];
    for (int i = 1; i < nr; i++){
      const double a = drc[i-1], b = drc[i];
      const double lo = Ht[ie(i-1, k)], hi = Ht[ie(i, k)];
      Tr[i*nth + k] = lo + (a/(a + b))*(hi - lo);
    }
    Tr[nr*nth + k] = freeCL ? Ht[ie(nr-1, k)] : 0.0;
  }
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++)
      Tth[i*nth + k] = 0.5*(Ht[ie(i, k-1)] + Ht[ie(i, k)]);
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++){
      const int e = ie(i, k);
      Tdr[e]  = (Tr[(i+1)*nth + kw(k)] - Tr[i*nth + kw(k)])/drc[i];
      Tdth[e] = (Tth[i*nth + kw(k+1)] - Tth[i*nth + kw(k)])/dth;
    }
  for (int k = 0; k < nth; k++){
    const int ka = kw(k + half);
    Tx [0*nth + k] =  Ht[ie(0, ka)];
    Txr[0*nth + k] = -Tdr[ie(0, ka)];
    Txt[0*nth + k] =  Tdth[ie(0, ka)];
    for (int i = 0; i < nr; i++){
      Tx [(i+1)*nth + k] = Ht[ie(i, k)];
      Txr[(i+1)*nth + k] = Tdr[ie(i, k)];
      Txt[(i+1)*nth + k] = Tdth[ie(i, k)];
    }
    Tx [(nr+1)*nth + k] = freeCL ? Ht[ie(nr-1, k)] : 0.0;
    Txr[(nr+1)*nth + k] = freeCL ? 0.0 : (0.0 - Ht[ie(nr-1, k)])/drf[nr];
    Txt[(nr+1)*nth + k] = freeCL ? Tdth[ie(nr-1, k)] : 0.0;
  }

  /* H at every family node, once per metric rather than once per reconstruction,
     with the column's azimuth taken in [0, nth) so that a column has one depth
     whatever index it is reached by. refreshMetric says why that matters. */
  Fam* fams[4] = { &FAMP, &FAMU, &FAMV, &FAMW };
  for (int t = 0; t < 4; t++){
    Fam& fam = *fams[t];
    for (int a = 0; a < fam.nI; a++)
      for (int k = 0; k < nth; k++)
        fam.Hcol[a*nth + k] = HatHBr(fam.hBr[a], fam.rn[a], (k + fam.thOff)*dth);
  }
  return lastError;
}

/* ---- the projection ---------------------------------------------------- */

/* Net flux out of each cell, in the transformed coordinates:
      d_r(r H u) + d_theta(H v) + d_sigma(r Omega)
   integrated over the cell. Unnormalised, as in the JavaScript: the volume
   division belongs to the gradient, which must be this operator's exact
   transpose over those volumes. */
void cell3d_divergence(const double* uu, const double* vv, const double* omm,
                       double* out){
  for (int i = 0; i < nr; i++){
    const double rci = rc[i], dr = drc[i], rIn = rf[i], rOut = rf[i+1];
    for (int k = 0; k < nth; k++){
      const int kp = k + 1 == nth ? 0 : k + 1;
      const double HrIn = Hr[i*nth + k], HrOut = Hr[(i+1)*nth + k];
      const double HthIn = Hth[i*nth + k], HthOut = Hth[i*nth + kp];
      const int bIn = (i*nth + k)*nz, bOut = ((i+1)*nth + k)*nz, bK = (i*nth + kp)*nz;
      const int bW = (i*nth + k)*nw;
      for (int j = 0; j < nz; j++){
        const double radial = dth*dsc[j]*(
            rOut*HrOut*uu[bOut + j]
          - rIn *HrIn *uu[bIn + j]);
        const double azim = dr*dsc[j]*(
            HthOut*vv[bK + j]
          - HthIn *vv[bIn + j]);
        const double vert = rci*dr*dth*(
            omm[bW + j + 1] - omm[bW + j]);
        out[bIn + j] = radial + azim + vert;
      }
    }
  }
}

/* The exact transpose of the divergence, divided by minus each face's own
   control volume -- accumulated in the same order the divergence reads its
   arguments, which is what makes the transpose exact rather than approximately
   so. Exactness is the whole point: it is what makes divergence(gradient(.))
   symmetric, and a symmetric negative definite operator is the only kind
   conjugate gradients is entitled to converge on.

   The second loop is the slope operator's transpose, which is what makes this
   the PHYSICAL pressure gradient rather than the covariant one. The divergence
   reads Omega; in the physical w that is w - sigma(u H_r + (v/r) H_theta), so u
   and v enter it through that term too and the transpose must return part of
   every sigma face's contribution to the four r faces and the four theta faces
   that meet there. Without it, measured on p = A r + B z over a surface at
   eta/h = 0.4, the radial component returns 86.99 where the physical gradient is
   137.00. */
void cell3d_gradient(const double* q, double* ou, double* ov, double* ow){
  for (int i = 0; i < NU; i++) ou[i] = 0.0;
  for (int i = 0; i < NV; i++) ov[i] = 0.0;
  for (int i = 0; i < NW; i++) ow[i] = 0.0;
  for (int i = 0; i < nr; i++){
    const double rci = rc[i], dr = drc[i], rIn = rf[i], rOut = rf[i+1];
    for (int k = 0; k < nth; k++){
      const int kp = k + 1 == nth ? 0 : k + 1;
      const double HrIn = Hr[i*nth + k], HrOut = Hr[(i+1)*nth + k];
      const double HthIn = Hth[i*nth + k], HthOut = Hth[i*nth + kp];
      const int bIn = (i*nth + k)*nz, bOut = ((i+1)*nth + k)*nz, bK = (i*nth + kp)*nz;
      const int bW = (i*nth + k)*nw;
      for (int j = 0; j < nz; j++){
        const double qc = q[bIn + j];
        ou[bOut + j] += qc*dth*dsc[j]*rOut*HrOut;
        ou[bIn  + j] -= qc*dth*dsc[j]*rIn *HrIn;
        ov[bK  + j] += qc*dr*dsc[j]*HthOut;
        ov[bIn + j] -= qc*dr*dsc[j]*HthIn;
        ow[bW + j + 1] += qc*rci*dr*dth;
        ow[bW + j    ] -= qc*rci*dr*dth;
      }
    }
  }
  for (int i = 0; i < nr; i++){
    const double ri = rc[i];
    for (int k = 0; k < nth; k++){
      const int e = i*nth + k;
      const double cr = -0.25*Hdr[e], ct = -0.25*Hdth[e]/ri;
      const int kp = k + 1 == nth ? 0 : k + 1;
      const int bIn = (i*nth + k)*nz, bOut = ((i+1)*nth + k)*nz, bK = (i*nth + kp)*nz;
      const int bW = (i*nth + k)*nw;
      for (int j = 1; j <= nz; j++){
        const double raw = ow[bW + j], sv = sf[j];
        const int jm = j - 1, jp = j == nz ? nz - 1 : j;
        const double du = cr*sv*raw, dv = ct*sv*raw;
        ou[bIn + jm] += du; ou[bOut + jm] += du;
        ou[bIn + jp] += du; ou[bOut + jp] += du;
        ov[bIn + jm] += dv; ov[bK + jm] += dv;
        ov[bIn + jp] += dv; ov[bK + jp] += dv;
      }
    }
  }
  /* u at the axis and the rim is prescribed, so its gradient there is not solved
     for and is zeroed; the same for Omega on the floor. */
  for (int k = 0; k < nth; k++)
    for (int j = 0; j < nz; j++){
      ou[k*nz + j] = 0.0;
      ou[(nr*nth + k)*nz + j] = 0.0;
    }
  for (int i = 1; i < nr; i++)
    for (int k = 0; k < nth; k++){
      const double Hf = Hr[i*nth + k];
      const int b = (i*nth + k)*nz;
      for (int j = 0; j < nz; j++)
        ou[b + j] /= -(rf[i]*drf[i]*dth*Hf*dsc[j]);
    }
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++){
      const double Hf = Hth[i*nth + k];
      const int b = (i*nth + k)*nz;
      for (int j = 0; j < nz; j++)
        ov[b + j] /= -(rc[i]*drc[i]*dth*Hf*dsc[j]);
    }
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++){
      const double Hc = H[i*nth + k];
      const int b = (i*nth + k)*nw;
      ow[b] = 0.0;                              // impermeable floor
      for (int j = 1; j <= nz; j++)
        ow[b + j] /= -(rc[i]*drc[i]*dth*Hc*dsf[j]);
    }
}

/* Omega from a given (u, v, w) triple -- the same definition omegaFromW applies
   to the state, with the index arithmetic hoisted out of the innermost loop
   exactly as the JavaScript hoists it. This is the second half of the
   conjugate-gradient matvec and runs once per iteration, a hundred and more
   times a step. */
void cell3d_omegaOf(const double* uu, const double* vv, const double* ww,
                    double* out){
  for (int i = 0; i < nr; i++){
    const double rci = rc[i];
    for (int k = 0; k < nth; k++){
      const int e = i*nth + k;
      const double hr = Hdr[e], hth = Hdth[e];
      const int kp = k + 1 == nth ? 0 : k + 1;
      const int bIn = e*nz, bOut = ((i+1)*nth + k)*nz, bK = (i*nth + kp)*nz;
      const int bW = e*nw;
      for (int j = 0; j <= nz; j++){
        const int c = bW + j;
        const double sv = sf[j];
        if (sv == 0.0){ out[c] = ww[c]; continue; }
        const int jm = j == 0 ? 0 : j - 1, jp = j == nz ? nz - 1 : j;
        const double ua = 0.25*(uu[bIn + jm] + uu[bOut + jm]
                              + uu[bIn + jp] + uu[bOut + jp]);
        const double va = 0.25*(vv[bIn + jm] + vv[bK + jm]
                              + vv[bIn + jp] + vv[bK + jp]);
        out[c] = ww[c] - sv*(ua*hr + (va/rci)*hth);
      }
    }
  }
}

/* The pressure operator, D W^-1 D^T, with D read in the PHYSICAL velocity. */
void cell3d_applyL(const double* q, double* out){
  cell3d_gradient(q, gu, gv, gw);
  cell3d_omegaOf(gu, gv, gw, gom);
  cell3d_divergence(gu, gv, gom, out);
}

/* THE PRESSURE SOLVE'S PRECONDITIONER: THE FLAT CELL, INVERTED EXACTLY -- the application
   of it, transcribed from applyPreconditioner in the JavaScript, which says what it is and
   why, in its order down to the grouping of every sum. Flat, the operator is block circulant
   in theta and couples a column only to itself and its two neighbours, symmetrically, so the
   azimuthal Fourier basis diagonalises it into one real symmetric band matrix per mode, of
   half-width nz, and -L_m has an exact band Cholesky factor.

   THE FACTOR IS NOT BUILT HERE. JavaScript builds it -- buildPreconditioner, from a flat twin
   of the cell, checking the structure the decomposition rests on -- and copies it in with
   the grid, as it copies the grid itself, so the factorisation has one implementation. So
   are the cos and sin tables: this module evaluates no transcendental. */
static void applyPreconditioner(const double* r, double* z){
  const int n2 = nr*nz, bw = nz, W = bw + 1, M = (nth >> 1) + 1;
  for (int x = 0; x < M*n2; x++){ pchatc[x] = 0.0; pchats[x] = 0.0; }
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++){
      const int rb = (i*nth + k)*nz;
      for (int m = 0; m < M; m++){
        const double cm = dftc[m*nth + k], sm = dfts[m*nth + k];
        const int hb = m*n2 + i*nz;
        for (int j = 0; j < nz; j++){
          const double x = r[rb + j];
          pchatc[hb + j] += x*cm;
          pchats[hb + j] += x*sm;
        }
      }
    }
  for (int m = 0; m < M; m++){
    const double* B = pcband + m*n2*W;
    const int hb = m*n2;
    const int both = !(m == 0 || 2*m == nth);
    for (int comp = 0; comp < (both ? 2 : 1); comp++){
      double* h = (comp == 0 ? pchatc : pchats) + hb;
      for (int a = 0; a < n2; a++){
        double sum = h[a];
        for (int pp = (a - bw > 0 ? a - bw : 0); pp < a; pp++) sum -= B[a*W + (a - pp)]*h[pp];
        h[a] = sum/B[a*W];
      }
      for (int a = n2 - 1; a >= 0; a--){
        double sum = h[a];
        const int top = n2 - 1 < a + bw ? n2 - 1 : a + bw;
        for (int pp = a + 1; pp <= top; pp++) sum -= B[pp*W + (pp - a)]*h[pp];
        h[a] = sum/B[a*W];
      }
    }
  }
  const double w0 = 1.0/nth, w1 = 2.0/nth;
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++){
      const int rb = (i*nth + k)*nz;
      for (int j = 0; j < nz; j++){
        double acc = 0.0;
        for (int m = 0; m < M; m++){
          const int a = m*n2 + i*nz + j;
          const double wm = (m == 0 || 2*m == nth) ? w0 : w1;
          acc += wm*(pchatc[a]*dftc[m*nth + k] + pchats[a]*dfts[m*nth + k]);
        }
        z[rb + j] = -acc;
      }
    }
}

/* The same, exported so the gate can compare it with the JavaScript's on any vector. */
void cell3d_applyPreconditioner(const double* r, double* z){ applyPreconditioner(r, z); }

/* Conjugate gradients, preconditioned by the flat cell's exact inverse. It stops on the TRUE
   operator's residual against the caller's tolerance, so the preconditioner changes how many
   iterations that takes and not what it converges to. */
double cell3d_solveP(const double* rhs, double tol, int maxIt){
  const int n = NP;
  cell3d_applyL(p, cgq);
  double rr = 0.0;
  for (int i = 0; i < n; i++){ cgr[i] = rhs[i] - cgq[i]; rr += cgr[i]*cgr[i]; }
  const double rr0 = rr;
  if (rr0 == 0.0){ cgIters = 0; cgResidual = 0.0; return 0.0; }
  applyPreconditioner(cgr, cgz);
  double rz = 0.0;
  for (int i = 0; i < n; i++){ cgd[i] = cgz[i]; rz += cgr[i]*cgz[i]; }
  int it = 0;
  for (; it < maxIt; it++){
    cell3d_applyL(cgd, cgq);
    double dq = 0.0;
    for (int i = 0; i < n; i++) dq += cgd[i]*cgq[i];
    if (dq == 0.0) break;
    const double alpha = rz/dq;
    double rr2 = 0.0;
    for (int i = 0; i < n; i++){
      p[i] += alpha*cgd[i]; cgr[i] -= alpha*cgq[i]; rr2 += cgr[i]*cgr[i]; }
    if (__builtin_sqrt(rr2/rr0) < tol){ rr = rr2; it++; break; }
    applyPreconditioner(cgr, cgz);
    double rz2 = 0.0;
    for (int i = 0; i < n; i++) rz2 += cgr[i]*cgz[i];
    const double beta = rz2/rz; rz = rz2; rr = rr2;
    for (int i = 0; i < n; i++) cgd[i] = cgz[i] + beta*cgd[i];
  }
  cgIters = it; cgResidual = __builtin_sqrt(rr/rr0);
  if (!(cgResidual < tol)) lastError = 2;
  return cgResidual;
}

int    cell3d_cgIters(){ return cgIters; }
double cell3d_cgResidual(){ return cgResidual; }

/* sigma (u dH/dr + (v/r) dH/dtheta) at a sigma face, with u and v averaged from
   the four faces of the cell that meet there. */
static double slopeTermOf(const double* uu, const double* vv, int i, int k, int j){
  const int e = ie(i, k);
  const double sv = sf[j];
  if (sv == 0.0) return 0.0;
  const int jm = j == 0 ? 0 : j - 1, jp = j == nz ? nz - 1 : j;
  const double ua = 0.25*(uu[iu(i, k, jm)] + uu[iu(i+1, k, jm)]
                        + uu[iu(i, k, jp)] + uu[iu(i+1, k, jp)]);
  const double va = 0.25*(vv[iv(i, k, jm)] + vv[iv(i, k+1, jm)]
                        + vv[iv(i, k, jp)] + vv[iv(i, k+1, jp)]);
  return sv*(ua*Hdr[e] + (va/rc[i])*Hdth[e]);
}

void cell3d_omegaFromW(){
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++)
      for (int j = 0; j <= nz; j++){
        const int c = iw(i, k, j);
        om[c] = w[c] - slopeTermOf(u, v, i, k, j);
      }
}

void cell3d_wFromOmega(){
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++)
      for (int j = 0; j <= nz; j++){
        const int c = iw(i, k, j);
        w[c] = om[c] + slopeTermOf(u, v, i, k, j);
      }
}

/* Largest absolute divergence, per unit volume, over the cells. */
double cell3d_maxDivergence(){
  cell3d_divergence(u, v, om, divw);
  double worst = 0.0;
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++){
      const double vol = rc[i]*drc[i]*dth*H[ie(i, k)];
      for (int j = 0; j < nz; j++){
        double d = divw[ip(i, k, j)]/(vol*dsc[j]);
        if (d < 0.0) d = -d;
        if (d > worst) worst = d;
      }
    }
  return worst;
}

/* ---- the reconstruction primitives ------------------------------------- */

/* The derivative at x of the polynomial through the first n of (xs, ys). Written
   as the sum over j of ys[j] times the derivative of the j-th Lagrange basis
   polynomial, and that derivative as the sum over i != j of the product over
   m != i, j of (x - xs[m]) -- which has no removable singularity, so x may be a
   node as well as a point between nodes.

   A cubic is needed, not merely convenient. A finite-volume row's Laplacian is a
   difference of two face fluxes divided by the row's own thickness; in the
   interior the two carry the same truncation error and it cancels, but at a
   boundary row one face IS the boundary and there is nothing for the interior
   face's error to cancel against, so it is divided by the thickness undiminished.
   With a two-point difference the sigma = 0 row read 1.748e-1, 7.679e-2,
   3.598e-2, 1.741e-2 over nz = 16/32/64/128; with four points, 6.98e-5, 6.38e-6,
   9.32e-7, 1.76e-7. */
static double polyDerivAt(const double* xs, const double* ys, int n, double x){
  double d = 0.0;
  for (int j = 0; j < n; j++){
    double den = 1.0;
    for (int m = 0; m < n; m++) if (m != j) den *= xs[j] - xs[m];
    double num = 0.0;
    for (int i = 0; i < n; i++){
      if (i == j) continue;
      double prod = 1.0;
      for (int m = 0; m < n; m++) if (m != j && m != i) prod *= x - xs[m];
      num += prod;
    }
    d += ys[j]*num/den;
  }
  return d;
}

/* The four columns that straddle a theta node symmetrically with the node itself
   left out, as offsets in k -- on a uniform grid the classical fourth-order
   central difference. */
static const int TH_NODE[4] = { -2, -1, 1, 2 };

/* H at an arbitrary position, bilinear on the extended grid, with the radial
   bracket already known -- which it is at every family node, from the table
   cell3d_buildFamilies builds. */
static double HatHBr(int a, double r, double th){
  const double r0 = rx[a], r1 = rx[a+1];
  const double fr = (r - r0)/(r1 - r0);
  const double tt = th/dth - 0.5;
  const double kbd = __builtin_floor(tt);
  const int kb = (int)kbd;
  const double ft = tt - kbd;
  const int k0 = kw(kb), k1 = kw(kb + 1);
  const double h00 = Hx[a*nth + k0], h01 = Hx[a*nth + k1];
  const double h10 = Hx[(a+1)*nth + k0], h11 = Hx[(a+1)*nth + k1];
  return (1 - fr)*((1 - ft)*h00 + ft*h01) + fr*((1 - ft)*h10 + ft*h11);
}
static int hatBracket(double r){
  int a = 0;
  while (a < nr && rx[a+1] < r) a++;
  if (a > nr) a = nr;
  return a;
}
static double HatH(double r, double th){ return HatHBr(hatBracket(r), r, th); }

/* H and its two horizontal slopes, into a caller-owned triple. */
static void HatInto(double r, double th, double* o){
  const int a = hatBracket(r);
  const double r0 = rx[a], r1 = rx[a+1];
  const double fr = (r - r0)/(r1 - r0);
  const double tt = th/dth - 0.5;
  const double kbd = __builtin_floor(tt);
  const int kb = (int)kbd;
  const double ft = tt - kbd;
  const int k0 = kw(kb), k1 = kw(kb + 1);
  const double h00 = Hx[a*nth + k0], h01 = Hx[a*nth + k1];
  const double h10 = Hx[(a+1)*nth + k0], h11 = Hx[(a+1)*nth + k1];
  o[0] = (1 - fr)*((1 - ft)*h00 + ft*h01) + fr*((1 - ft)*h10 + ft*h11);
  o[1] = (((1 - ft)*h10 + ft*h11) - ((1 - ft)*h00 + ft*h01))/(r1 - r0);
  o[2] = ((1 - fr)*(h01 - h00) + fr*(h11 - h10))/dth;
}

/* The CENTRED slopes of H at an arbitrary position, bilinear on the extended
   slope grids. Distinct from HatInto's slopes, which are the bracket's own and so
   are centred only at a face midpoint: these are what the sigma-face cross terms
   need, because those are evaluated at nodes. Interpolating the precomputed
   centred slopes rather than taking the bracket's own cost the whole operator an
   order and a half when it was got wrong -- every family read 0.5 instead of 2. */
static void Hslope(double r, double th, double* o){
  const int a = hatBracket(r);
  const double r0 = rx[a], r1 = rx[a+1];
  const double fr = (r - r0)/(r1 - r0);
  const double tt = th/dth - 0.5;
  const double kbd = __builtin_floor(tt);
  const int kb = (int)kbd;
  const double ft = tt - kbd;
  const int k0 = kw(kb), k1 = kw(kb + 1);
  o[0] = (1 - fr)*((1 - ft)*Hxr[a*nth + k0] + ft*Hxr[a*nth + k1])
       + fr*((1 - ft)*Hxr[(a+1)*nth + k0] + ft*Hxr[(a+1)*nth + k1]);
  o[1] = (1 - fr)*((1 - ft)*Hxt[a*nth + k0] + ft*Hxt[a*nth + k1])
       + fr*((1 - ft)*Hxt[(a+1)*nth + k0] + ft*Hxt[(a+1)*nth + k1]);
}

/* A field's value in one column at an arbitrary physical height.
 *
 * This is the primitive the whole operator is built on. Anything that compares
 * values from two different columns must do it at a COMMON PHYSICAL HEIGHT:
 * under z = sigma H two columns' sigma levels are at different heights whenever
 * the surface is deformed, so comparing them at equal sigma carries an O(dH)
 * error that vanishes when flat and does not converge when not. That error made
 * the coupling terms read order 2.00 flat and -0.52 at eta/h = 0.4.
 *
 * CUBIC in sigma, anchored on the LEVEL and not chosen by bracketing the target,
 * so that two columns compared at one height use the same node positions and
 * their reconstruction errors cancel instead of jumping as a target crosses a
 * node. `bracket` overrides that for the sigma-face cross terms, whose target is
 * far from the level -- famLaplacian says why, with the numbers.
 *
 * Radial index a < 0 is the antipodal column reflected through the axis, carrying
 * the family's own sign. */
static double colValueAtZ(const double* f, const Fam& fam, int a, int k,
                          double z, int lev, int hasLev, int bracket){
  const double* sn = fam.sn; const int nJ = fam.nJ; const int half = nth >> 1;
  int aa = a, kk = k; double sign = 1.0;
  if (a < 0){ aa = -1 - a; kk = k + half; sign = (double)fam.axisSign; }
  const int col = aa*nth + kw(kk);
  const double Hc = fam.Hcol[col];           // the column's own depth: refreshMetric
  const double ss = z/Hc;
  const int base = col*fam.stride;
  if (nJ == 1) return sign*f[base];
  const int n = nJ < 4 ? nJ : 4;
  int j0;
  if (bracket){
    int lo = 0, hi = nJ - 1;
    while (hi - lo > 1){ const int m = (lo + hi) >> 1; if (sn[m] <= ss) lo = m; else hi = m; }
    j0 = lo - 1;
  } else j0 = (hasLev ? lev : 1) - 1;
  if (j0 + n > nJ) j0 = nJ - n;
  if (j0 < 0) j0 = 0;
  double v = 0.0;
  for (int i = 0; i < n; i++){
    double L = 1.0;
    for (int m = 0; m < n; m++)
      if (m != i) L *= (ss - sn[j0 + m])/(sn[j0 + i] - sn[j0 + m]);
    v += f[base + j0 + i]*L;
  }
  return sign*v;
}

/* The same stencil, differentiated instead of evaluated: df/dz in one column at
   one physical height. z = sigma H at fixed (r, theta), so d/dz is (1/H) d/dsigma.
   It shares colValueAtZ's conventions because the two are read at the same points
   and a difference in stencil between them would be one nothing would catch. */
static double colDerivAtZ(const double* f, const Fam& fam, int a, int k,
                          double z, int lev, int hasLev){
  const double* sn = fam.sn; const int nJ = fam.nJ; const int half = nth >> 1;
  int aa = a, kk = k; double sign = 1.0;
  if (a < 0){ aa = -1 - a; kk = k + half; sign = (double)fam.axisSign; }
  const double Hc = fam.Hcol[aa*nth + kw(kk)];
  const double ss = z/Hc;
  if (nJ == 1) return 0.0;
  const int n = nJ < 4 ? nJ : 4;
  int j0 = (hasLev ? lev : 1) - 1;
  if (j0 + n > nJ) j0 = nJ - n;
  if (j0 < 0) j0 = 0;
  double sx[4], sy[4];
  const int base = (aa*nth + kw(kk))*fam.stride;
  for (int m = 0; m < n; m++){ sx[m] = sn[j0 + m]; sy[m] = f[base + j0 + m]; }
  return sign*polyDerivAt(sx, sy, n, ss)/Hc;
}

/* ---- the axis ---------------------------------------------------------- */

/* The radial velocity at the axis. u_r is not stored there as an independent
   value: a single-valued vector field requires u_r(0, theta) = -u_r(0, theta+pi),
   so the axis row is the antisymmetric part of an extrapolation from the two
   nodes outside it. Second order, and it satisfies the constraint by construction
   rather than by an assertion afterwards. Setting it to zero instead is exact for
   m = 0 and every m >= 2, and wrong for exactly the one mode that is non-zero at
   the axis. */
void cell3d_axisU(){
  const int half = nth >> 1;
  const double r1 = rf[1], r2 = rf[2];
  double* ex = axEx;
  for (int j = 0; j < nz; j++){
    for (int k = 0; k < nth; k++){
      const double a = u[iu(1, k, j)], b = u[iu(2, k, j)];
      ex[k] = a + (0 - r1)*(b - a)/(r2 - r1);
    }
    for (int k = 0; k < nth; k++)
      u[iu(0, k, j)] = 0.5*(ex[k] - ex[kw(k + half)]);
  }
}

/* ---- the surface's own quantities ------------------------------------- */

/* The outward normal of the free surface at an arbitrary position, from the
   surface's own slopes: N = (-H_r, -H_theta/r, 1). Returned unnormalised as well
   as normalised, because the tangent vectors are orthogonal to N without
   normalising and the traction algebra is cleaner in those terms. Filled as
   [sr, st, len, nr, nth, nz]. */
static void surfaceNormal(double r, double th, double* o){
  double g[2];
  Hslope(r, th, g);
  const double sr = g[0], st = g[1]/r;
  const double len = __builtin_sqrt(1 + sr*sr + st*st);
  o[0] = sr; o[1] = st; o[2] = len;
  o[3] = -sr/len; o[4] = -st/len; o[5] = 1/len;
}

/* The nine covariant derivatives of the velocity at the free surface above one
   pressure cell, in the order
       [u_r,r  u_r,th  u_r,z   u_th,r  u_th,th  u_th,z   u_z,r  u_z,th  u_z,z]
   so the two that carry the rotating basis are
       u_r,theta     = (1/r) du_r/dtheta - u_theta/r
       u_theta,theta = (1/r) du_theta/dtheta + u_r/r
   The strain tensor and the surface flux are both built from these, so they are
   formed once: two functions each forming their own nine would be two chances for
   them to disagree about one.

   Every horizontal derivative is taken BETWEEN COLUMNS AT ONE PHYSICAL HEIGHT --
   this cell's own surface height. Measured with the neighbour read at its own
   sigma = 1 instead: E_rr at order -0.007 while the flat case still passed. */
static void surfaceGradient(int i, int k, double* out){
  const double r = rc[i];
  const double z = H[ie(i, k)];
  const int lu_ = FAMU.nJ - 2, lw_ = FAMW.nJ - 2;
  #define UV(a, kk) colValueAtZ(u, FAMU, (a), (kk), z, lu_, 1, 0)
  #define UD(a, kk) colDerivAtZ(u, FAMU, (a), (kk), z, lu_, 1)
  #define VV(a, kk) ((a) > nr - 1 ? 0.0 : colValueAtZ(v, FAMV, (a), (kk), z, lu_, 1, 0))
  #define VD(a, kk) ((a) > nr - 1 ? 0.0 : colDerivAtZ(v, FAMV, (a), (kk), z, lu_, 1))
  #define WV(a, kk) ((a) > nr - 1 ? 0.0 : colValueAtZ(w, FAMW, (a), (kk), z, lw_, 1, 0))
  #define WD(a, kk) ((a) > nr - 1 ? 0.0 : colDerivAtZ(w, FAMW, (a), (kk), z, lw_, 1))
  #define RCOF(a) ((a) < 0 ? -rc[-1 - (a)] : ((a) > nr - 1 ? R : rc[(a)]))
  const double span = RCOF(i + 1) - RCOF(i - 1);

  const double uIn = UV(i, k), uOut = UV(i + 1, k);
  const double ur = 0.5*(uIn + uOut);
  const double vLo = VV(i, k), vHi = VV(i, k + 1);
  const double ut = 0.5*(vLo + vHi);

  out[0] = (uOut - uIn)/drc[i];
  out[1] = 0.25*(UV(i, k+1) - UV(i, k-1) + UV(i+1, k+1) - UV(i+1, k-1))/(dth*r) - ut/r;
  out[2] = 0.5*(UD(i, k) + UD(i + 1, k));
  out[3] = 0.5*(VV(i+1, k) - VV(i-1, k) + VV(i+1, k+1) - VV(i-1, k+1))/span;
  out[4] = (vHi - vLo)/(dth*r) + ur/r;
  out[5] = 0.5*(VD(i, k) + VD(i, k+1));
  out[6] = (WV(i+1, k) - WV(i-1, k))/span;
  out[7] = 0.5*(WV(i, k+1) - WV(i, k-1))/(dth*r);
  out[8] = WD(i, k);
  #undef UV
  #undef UD
  #undef VV
  #undef VD
  #undef WV
  #undef WD
  #undef RCOF
}

/* The rate-of-strain tensor at the free surface above one pressure cell, as
   [E_rr, E_thth, E_zz, E_rth, E_rz, E_thz] -- the symmetric part of the nine
   derivatives above, and nothing more. */
static void surfaceStrain(int i, int k, double* out){
  double g[9];
  surfaceGradient(i, k, g);
  out[0] = g[0];
  out[1] = g[4];
  out[2] = g[8];
  out[3] = 0.5*(g[1] + g[3]);
  out[4] = 0.5*(g[2] + g[6]);
  out[5] = 0.5*(g[5] + g[7]);
}

/* The flux the vector Laplacian's sigma = 1 face must carry, per component, once
   the free-surface stress conditions are imposed: grad u_i . N.
   .
   It is not just the traction. famLaplacian's sigma-face flux is proj*(grad f.N),
   and the free surface gives a traction 2 rho nu E.n. For incompressible flow with
   constant viscosity 2 nu div E and nu grad^2 u are the same VOLUME operator, but
   their face fluxes differ by nu u_{j,i} n_j -- a term that integrates to zero over
   a closed surface and does not vanish face by face. The conversion is an identity:
   from E_ij = (u_{i,j} + u_{j,i})/2,
       grad u_i . n = 2 (E.n)_i - u_{j,i} n_j
   and the tangential condition is imposed by PROJECTION -- E.n replaced by
   (n.E.n) n, exact at any slope -- so with the unnormalised normal
       grad u_i . N = 2 lambda N_i - u_{j,i} N_j.
   Nothing divides by 1 - |grad eta|^2, so the forty-five degree degeneracy never
   appears. */
static void surfaceLapFluxes(int i, int k, double* out){
  double g[9];
  surfaceGradient(i, k, g);
  double n[6];
  surfaceNormal(rc[i], (k + 0.5)*dth, n);
  const double Nr = -n[0], Nt = -n[1], Nz = 1.0;
  const double E0 = g[0], E1 = g[4], E2 = g[8];
  const double E3 = 0.5*(g[1] + g[3]), E4 = 0.5*(g[2] + g[6]), E5 = 0.5*(g[5] + g[7]);
  const double NEN = E0*Nr*Nr + E1*Nt*Nt + E2*Nz*Nz
                   + 2*(E3*Nr*Nt + E4*Nr*Nz + E5*Nt*Nz);
  const double lam = NEN/(n[2]*n[2]);
  for (int c = 0; c < 3; c++)
    out[c] = 2*lam*(c == 0 ? Nr : c == 1 ? Nt : Nz)
           - (g[c]*Nr + g[3 + c]*Nt + g[6 + c]*Nz);
}

/* The viscous normal stress at the free surface above one pressure cell,
   2 rho nu n.E.n, contracted with the FULL normal rather than with z-hat. The
   flat-normal form 2 rho nu dw/dz is what the two-dimensional solver next door is
   entitled to, because it linearises about a flat surface; this one is not. */
static double surfaceNormalStress(int i, int k){
  double n[6];
  surfaceNormal(rc[i], (k + 0.5)*dth, n);
  double E[6];
  surfaceStrain(i, k, E);
  const double a = n[3], b = n[4], c = n[5];
  const double nEn = E[0]*a*a + E[1]*b*b + E[2]*c*c
                   + 2*(E[3]*a*b + E[4]*a*c + E[5]*b*c);
  return 2*rho*nu*nEn;
}

/* The free surface's own flux, per component, at every pressure cell. Formed once
   per viscous evaluation because all three components come from one rate-of-strain
   tensor, and forming it three times would be three chances for them to disagree
   about one surface. */
void cell3d_refreshSurfaceFluxes(){
  double t3[3];
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++){
      surfaceLapFluxes(i, k, t3);
      const int e = ie(i, k);
      fsr[e] = t3[0]; fst[e] = t3[1]; fsz[e] = t3[2];
    }
}

/* and that flux where one family's own sigma = 1 face sits, which is not where the
   pressure cells are. u's faces are at r faces, v's at theta faces, and w's
   already at the pressure cell's own position.
   .
   The radial one is a weighted interpolation rather than a mean: an r face is the
   midpoint of its two cell centres only when the two cells are equally wide, and
   this radius is graded towards the rim. Both forms are second order at the face
   -- measured 5.12e-2, 1.17e-2, 2.80e-3 at order 2.12 then 2.07 for the weights,
   and 6.10e-2, 1.43e-2, 3.46e-3 at 2.09 then 2.05 for the mean -- and the weights
   are kept because a surface-flux error is divided by the top row's thickness in
   the surface row, so a nineteen per cent smaller constant is worth two arithmetic
   operations. Azimuthally the mean is right rather than merely close: theta is
   uniform, so a theta face IS the midpoint of its two cells, exactly. */
static double surfaceFluxFace(int which, int a, int k){
  if (which == 2) return 0.5*(fst[ie(a, k - 1)] + fst[ie(a, k)]);
  if (which == 3) return fsz[ie(a, k)];
  const double lo = fsr[ie(a - 1, k)], hi = fsr[ie(a, k)];
  const double wa = drc[a-1], wb = drc[a];
  return lo + (wa/(wa + wb))*(hi - lo);
}

/* ---- the viscous operator ---------------------------------------------- */

/* The metric Laplacian at any of the four node families.
 *
 * TANGENTIAL DERIVATIVES ARE TAKEN AT A COMMON PHYSICAL HEIGHT. This is the whole
 * design and not an optimisation; the obvious discretisation does not converge.
 * Under z = sigma H every physical derivative is a difference of two terms,
 *     df/dr|_z = d_r f - (sigma H_r / H) d_sigma f
 * individually O(1) for a field of z alone, cancelling EXACTLY in the continuum
 * and only to O(dtheta^2) discretely; the Laplacian then divides a difference of
 * face fluxes by dtheta, and the 1/r^2 factor near the axis amplifies that by
 * 1/dr^2, so refining makes it worse. Measured with f = sin(kz) over a surface
 * varying only in theta at eta/h = 0.3: family p read 1.77e-1 then 1.72e-1, order
 * 0.04, and family v read 8.63e-1 then 3.42e+0, order -1.99. The cure is to remove
 * the subtraction, not to compute it more carefully: each column is reconstructed
 * as a function of physical height and the two are differenced at the SAME height.
 *
 * `hasBc` says a boundary VALUE is available at the walls; every such value in
 * this solver is zero, because the wall does not move, so the value itself is
 * written as a literal 0 and what `hasBc` changes is which branches exist.
 * `sFluxKind` (0 none, 1 u, 2 v, 3 w) overrides the sigma = 1 face with
 * grad f . N directly -- which is how the free surface is closed, because its
 * condition is a traction and a traction is a flux, not a value. The axis needs no
 * entry: at r = 0 the face area is exactly zero, and inward stencils use the
 * antipodal column with the family's reflection sign. */
static void famLaplacian(const double* f, double* out, const Fam& fam,
                         int hasBc, int sFluxKind){
  const double* rn = fam.rn; const double* rb = fam.rb;
  const double* sn = fam.sn; const double* sb = fam.sb;
  const int nI = fam.nI, nJ = fam.nJ;
  const double thOff = fam.thOff;

  const int aHi = (hasBc && rn[nI-1] < R) ? nI : nI - 1;
  /* How far inward the stencil may reach. A family whose first node is ON the
     axis -- u, whose nodes are the r faces -- has no antipodal continuation to
     offer: column -1 would be that same node reflected, at the same radius zero,
     and two stencil points at one abscissa is a division by zero rather than a
     wide stencil. Such a family stops at its own axis node, which carries the
     reflection already (axisU). */
  const int aLo = rn[0] > 0 ? -nI : 0;

  const double sLoB = sb[0], sHiB = sb[nJ];
  const int jLo = (hasBc && sn[0] > sLoB) ? -1 : 0;
  const int jHi = (hasBc && !sFluxKind && sn[nJ-1] < sHiB) ? nJ : nJ - 1;
  if (jHi - jLo < 3){ lastError = 5; return; }

  double sx[4], sy[4], rx4[4], ry4[4], tx4[4], ty4[4];
  double hA[3], hB[3], midS[2];

  /* EVERY FACE IS FORMED ONCE, and both cells it separates read that one number --
     famLaplacian in the JavaScript says why, and why the r and sigma faces' shared
     value is the old one bit for bit while the theta faces' is a correction. The
     cell then sums its six faces in the order it always did, with the same signs:
     IEEE arithmetic is exactly symmetric under negation, so `flux += -T` is the old
     `flux += side*T` with side = -1. */
  const int sLo = fam.sLo, sHi = fam.sHi, nb = sHi - sLo + 1;
  auto rSkip = [&](int A){ return rb[A] == 0.0 || (!hasBc && A > nI - 1); };
  auto sTop = [&](int B){ return B > nJ - 1 && sFluxKind != 0; };
  auto sSkip = [&](int B){ return !sTop(B) && (B > nJ - 1 || B < 1) && !hasBc; };
  /* The r face at rb[A], for every column and row: df/dr|_z there, third order, from
     the cubic through the four columns that straddle it, each reconstructed at the
     SAME physical height. Four and not two because the radial grid is graded, so a
     two-point difference is centred at the midpoint of its columns rather than at
     the face, and at the rim column there is nothing for that error to cancel
     against: measured on family p over a flat surface the rim column read 2.27e-3 at
     order 1.71 then 1.43 while every interior column ran at 2.04 or better. A column
     past the rim is the wall, so it is an ordinary fourth point. The JavaScript
     asks bc for the wall's value at the wall's own sigma; every wall value in this
     solver is zero -- the wall does not move -- so it is written as 0 here. */
  auto rFaces = [&](int A, double* buf){
    if (rSkip(A)) return;
    const double rface = rb[A];
    for (int k = 0; k < nth; k++){
      const double g0 = HatH(rface, (k + thOff)*dth);
      for (int b = sLo; b <= sHi; b++){
        const double dsb = sb[b+1] - sb[b], sMid = 0.5*(sb[b] + sb[b+1]);
        const double z = sMid*g0;
        int j0 = A - 2;
        if (j0 + 3 > aHi) j0 = aHi - 3;
        if (j0 < aLo) j0 = aLo;
        for (int m = 0; m < 4; m++){
          const int ac = j0 + m;
          rx4[m] = ac > nI - 1 ? R : (ac >= 0 ? rn[ac] : -rn[-1 - ac]);
          ry4[m] = ac > nI - 1 ? 0.0 : colValueAtZ(f, fam, ac, k, z, b, 1, 0);
        }
        const double d = polyDerivAt(rx4, ry4, 4, rface);
        buf[k*nb + b - sLo] = rface*dth*g0*dsb*d;
      }
    }
  };
  double* lo = fRlo; double* hi = fRhi;
  rFaces(fam.rLo, lo);
  for (int a = fam.rLo; a <= fam.rHi; a++){
    const double dra = rb[a+1] - rb[a];
    rFaces(a + 1, hi);
    const bool loR = !rSkip(a), hiR = !rSkip(a + 1);

    /* ---- this column's theta faces: face k lies between columns k - 1 and k ----
       df/dtheta|_z AT A THETA FACE, two points, and two points deliberately.
       theta is uniform and periodic, so the difference over the interval it spans is
       centred exactly at the face and its truncation coefficient is the same at every
       k, which cancels between opposite faces. More than that, THIS STENCIL IS
       LOAD-BEARING for the cylindrical coupling: for a field uniform in Cartesian terms
       the azimuthal part of the scalar Laplacian is a spurious -u/r^2 and it is the
       coupling term (2/r^2) du_theta/dtheta that cancels it, between two DISCRETE
       expressions, so it holds only while the two use matching stencils. Raising this
       one to four points on its own took the vector Laplacian from order 2.10 to 0.67
       on a flat surface. */
    for (int k = 0; k < nth; k++){
      const double g0 = HatH(rn[a], (k + thOff)*dth - 0.5*dth);
      for (int b = sLo; b <= sHi; b++){
        const double dsb = sb[b+1] - sb[b], sMid = 0.5*(sb[b] + sb[b+1]);
        const double z = sMid*g0;
        const double d = (colValueAtZ(f, fam, a, k, z, b, 1, 0)
                        - colValueAtZ(f, fam, a, k - 1, z, b, 1, 0))/((k - (k - 1))*dth);
        fT[k*nb + b - sLo] = dra*g0*dsb*d/rn[a];
      }
    }

    for (int k = 0; k < nth; k++){
      const double th = (k + thOff)*dth;
      HatInto(rn[a], th, hA);
      Hslope(rn[a], th, midS);
      const double proj = rn[a]*dra*dth;

      /* ---- this column's sigma faces: the curved sheets z = sigma H, whose normal
              is proportional to (-sigma H_r, -sigma H_theta / r, 1); face B lies
              between rows B - 1 and B ---- */
      for (int B = sLo; B <= sHi + 1; B++){
        if (sTop(B)){
          /* THE FREE SURFACE, closed by a FLUX rather than by a value. */
          fS[B - sLo] = proj*surfaceFluxFace(sFluxKind, a, k);
          continue;
        }
        if (sSkip(B)) continue;
        const double sface = sb[B];
        /* The four values straddling this sigma face. A boundary value counts as one
           of them, at the boundary's own sigma; where one is unavailable the four run
           one-sided into the interior, which is third order there too. */
        int j0 = B - 2;
        if (j0 < jLo) j0 = jLo;
        if (j0 + 3 > jHi) j0 = jHi - 3;
        for (int m = 0; m < 4; m++){
          const int jj = j0 + m;
          sx[m] = jj < 0 ? sLoB : (jj > nJ - 1 ? sHiB : sn[jj]);
          sy[m] = (jj < 0 || jj > nJ - 1) ? 0.0 : f[famIdx(fam, a, k, jj)];
        }
        const double dsg = polyDerivAt(sx, sy, 4, sface);
        const double z = sface*hA[0];
        /* THE TANGENTIAL GRADIENTS A SIGMA FACE WANTS, and the one place in this
           operator where rule 1's common height must NOT be anchored on the level. On
           a sigma face near the surface the height wanted is off a neighbour's own
           level by H_theta dtheta / H, measured at 18 to 20 times the top row's
           thickness over 16/32/64 -- a ratio refinement does not reduce. Anchored, the
           cubic extrapolated ten stencil widths past its own nodes and the azimuthal
           cross term read 3.05e-3, 1.94e-3, 5.91e-4, order 0.65 then 1.71. Bracketed,
           3.73e-3, 9.06e-4, 3.94e-4, 1.25e-4 over four grids. With the reconstruction
           replaced by its analytic value the same term read 1.95e-3, 3.15e-4,
           4.53e-5, 5.84e-6 at order 2.63, 2.80, 2.96 -- so neither stencil is the
           difficulty, the reconstruction is, and that is recorded as an open item
           rather than as a tolerance. */
        int ja = a - 1;
        if (ja + 3 > aHi) ja = aHi - 3;
        if (ja < aLo) ja = aLo;
        for (int m = 0; m < 4; m++){
          const int a2 = ja + m;
          rx4[m] = a2 > nI - 1 ? R : (a2 >= 0 ? rn[a2] : -rn[-1 - a2]);
          ry4[m] = a2 > nI - 1 ? 0.0 : colValueAtZ(f, fam, a2, k, z, 0, 0, 1);
        }
        const double dSigR = polyDerivAt(rx4, ry4, 4, rn[a]);
        for (int m = 0; m < 4; m++){
          tx4[m] = (k + TH_NODE[m])*dth;
          ty4[m] = colValueAtZ(f, fam, a, k + TH_NODE[m], z, 0, 0, 1);
        }
        const double dSigTh = polyDerivAt(tx4, ty4, 4, k*dth);
        fS[B - sLo] = proj*( dsg/hA[0]
                           - sface*midS[0]*dSigR
                           - (sface*midS[1]/(rn[a]*rn[a]))*dSigTh );
      }

      const int kp = k + 1 == nth ? 0 : k + 1;
      for (int b = sLo; b <= sHi; b++){
        const double dsb = sb[b+1] - sb[b];
        const int c = b - sLo;
        /* The six faces in the order the cell always summed them: r below and
           above, theta below and above, sigma below and above. */
        double flux = 0.0;
        if (loR) flux += -lo[k*nb + c];
        if (hiR) flux += hi[k*nb + c];
        flux += -fT[k*nb + c];
        flux += fT[kp*nb + c];
        if (!sSkip(b)) flux += -fS[c];
        if (!sSkip(b + 1)) flux += fS[c + 1];
        out[famIdx(fam, a, k, b)] = flux/(rn[a]*dra*dth*hA[0]*dsb);
      }
    }
    double* t = lo; lo = hi; hi = t;
  }
}

/* The viscous term for the velocity, as the vector Laplacian in cylindrical
   coordinates:
       (grad^2 u)_r     = grad^2 u_r  - u_r/r^2  - (2/r^2) d u_th / d theta
       (grad^2 u)_theta = grad^2 u_th - u_th/r^2 + (2/r^2) d u_r  / d theta
       (grad^2 u)_z     = grad^2 w
   The scalar part is famLaplacian at each component's own nodes; the rest is the
   curvature of the coordinate system, algebraic and pointwise. The two coupling
   terms are not optional and not small: for a field uniform in Cartesian terms the
   scalar Laplacian of each component carries a spurious -u/r^2, and it is exactly
   the coupling that cancels it.

   `hasBc` closes the three scalar operators at the walls, where no slip makes
   every boundary value zero. */
void cell3d_viscous(int hasBc){
  cell3d_refreshSurfaceFluxes();
  famLaplacian(u, lapU, FAMU, hasBc, 1);
  famLaplacian(v, lapV, FAMV, hasBc, 2);
  famLaplacian(w, lapW, FAMW, hasBc, 3);
  if (lastError) return;

  /* d v / d theta at a u node, and d u / d theta at a v node, both taken at the
     target node's own PHYSICAL HEIGHT. u sits at (rf, theta centre) and v at
     (rc, theta face), so on a deformed surface their sigma levels are at
     different heights, and averaging across at equal sigma carries an O(dH)
     error: order 2.00 at eta/h = 0 for both components and -0.52 at 0.4. */
  for (int i = FAMU.rLo; i <= FAMU.rHi; i++){
    const double r = rf[i], inv = 1/(r*r);
    for (int k = 0; k < nth; k++){
      const double Hu = HatH(r, (k + 0.5)*dth);
      for (int j = 0; j < nz; j++){
        const int c = iu(i, k, j);
        const double z = sc[j]*Hu;
        const double vA = 0.5*(colValueAtZ(v, FAMV, i - 1, k, z, j, 1, 0)
                             + (i <= nr - 1 ? colValueAtZ(v, FAMV, i, k, z, j, 1, 0) : 0.0));
        const double vB = 0.5*(colValueAtZ(v, FAMV, i - 1, k + 1, z, j, 1, 0)
                             + (i <= nr - 1 ? colValueAtZ(v, FAMV, i, k + 1, z, j, 1, 0) : 0.0));
        lapU[c] += -inv*u[c] - 2*inv*(vB - vA)/dth;
      }
    }
  }
  for (int i = FAMV.rLo; i <= FAMV.rHi; i++){
    const double r = rc[i], inv = 1/(r*r);
    for (int k = 0; k < nth; k++){
      const double Hv = HatH(r, k*dth);
      for (int j = 0; j < nz; j++){
        const int c = iv(i, k, j);
        const double z = sc[j]*Hv;
        const double uA = 0.5*(colValueAtZ(u, FAMU, i, k - 1, z, j, 1, 0)
                             + colValueAtZ(u, FAMU, i + 1, k - 1, z, j, 1, 0));
        const double uB = 0.5*(colValueAtZ(u, FAMU, i, k, z, j, 1, 0)
                             + colValueAtZ(u, FAMU, i + 1, k, z, j, 1, 0));
        lapV[c] += -inv*v[c] + 2*inv*(uB - uA)/dth;
      }
    }
  }
}

/* ---- advection --------------------------------------------------------- */

/* The volumetric face fluxes of a PRESSURE cell, in the transformed coordinates.
   The radial and azimuthal ones are exactly the terms the divergence forms. The
   vertical one comes in three pieces because the sigma sheets themselves move
   when eta does: fluxSabs is r Omega, the absolute transport that continuity
   balances; fluxSmesh is r sigma dH/dt, the sheet's own motion; their difference
   is the GRID-RELATIVE transport, which is what carries momentum across a moving
   sheet.

   NOTHING IS CLAMPED TO ZERO AT A BOUNDARY, and that is the point. Each of the
   four boundary fluxes is already exactly zero for a physical reason: rf[0] is
   exactly zero so the axis face has no area; no penetration holds u at the rim at
   zero; no slip holds Omega on the floor at zero, and sigma kills the mesh term
   there in any case; and the surface is material, so Omega there IS dH/dt and the
   two pieces are a floating-point difference of equals. Writing the expressions
   out rather than clamping means a violated boundary condition shows up as a
   divergence or an energy imbalance instead of being masked by a branch that
   answers zero whatever the state says. */
static inline double fluxR(int i, int k, int b){
  return dth*dsc[b]*rf[i]*Hr[i*nth + kw(k)]*u[iu(i, k, b)];
}
static inline double fluxTh(int i, int k, int b){
  return drc[i]*dsc[b]*Hth[i*nth + kw(k)]*v[iv(i, k, b)];
}
static inline double fluxSabs(int i, int k, int b){
  return rc[i]*drc[i]*dth*om[iw(i, k, b)];
}
static inline double fluxSmesh(int i, int k, int b){
  return rc[i]*drc[i]*dth*sf[b]*Ht[ie(i, k)];
}
static inline double fluxS(int i, int k, int b){
  return fluxSabs(i, k, b) - fluxSmesh(i, k, b);
}

/* The transport part of the advective term, in conservative flux form, centred.
 *
 * NO UPWINDING. Upwinding adds numerical dissipation, indistinguishable from
 * viscosity in the answer, and would fake the damping that sets the Faraday
 * threshold. Centred flux-form advection is exactly energy neutral instead.
 *
 * THE FORM IS THE MOVING-MESH ONE. On a mesh that moves, what the finite-volume
 * balance conserves is the momentum CONTENT of a cell, V u, not u:
 *     d(V u)/dt + sum_faces F_rel phi = 0   ==>   V du/dt = -f - u dV/dt
 * so the cell's own volume rate appears, and it must be the net of the SAME mesh
 * fluxes that were subtracted to make F_rel -- the discrete geometric conservation
 * law. Without it the energy residual on a divergence-free field sat at 2.5e-3 of
 * the terms it sums and would not move when the projection tolerance was tightened
 * by five decades, because it was not the projection's error at all.
 *
 * THE ADVECTED QUANTITY IS A PLAIN ARITHMETIC MEAN, not a common-height
 * reconstruction. The common-height rule protects a cancellation inside a
 * DERIVATIVE; an average has none, and the flux form in these coordinates never
 * forms such a difference. What the mean must do instead is telescope, and
 * reconstructing at a common height breaks that and the scheme stops conserving
 * energy.
 *
 * THE TRANSPORT FLUXES ARE THE PRESSURE CELLS' OWN, AVERAGED ONTO THE MOMENTUM
 * CONTROL VOLUMES, and the averaging is what makes the net flux out of a momentum
 * cell exactly half the sum of its two neighbours' divergences, for any field
 * whatever. */
void cell3d_advectTransport(){
  const int half = nth >> 1;
  for (int i = 0; i < NU; i++) advU[i] = 0.0;
  for (int i = 0; i < NV; i++) advV[i] = 0.0;
  for (int i = 0; i < NW; i++) advW[i] = 0.0;

  /* ---- radial momentum, on the u control volumes ----------------------
     rc[i-1] .. rc[i] in r, one pressure cell in theta and in sigma. Its r faces
     sit at the two pressure centres either side, so each carries the mean of the
     fluxes through the two cell faces bracketing it; its theta and sigma faces
     each span half of each of the two cells it straddles. Only the sigma faces
     move. */
  for (int i = 1; i < nr; i++){
    for (int k = 0; k < nth; k++){
      const double Hf = Hr[i*nth + kw(k)];
      for (int b = 0; b < nz; b++){
        const int c = iu(i, k, b);
        const double V = rf[i]*drf[i]*dth*Hf*dsc[b];
        const double QrIn  = 0.5*(fluxR(i-1, k, b) + fluxR(i, k, b));
        const double QrOut = 0.5*(fluxR(i, k, b)   + fluxR(i+1, k, b));
        const double QtLo  = 0.5*(fluxTh(i-1, k, b) + fluxTh(i, k, b));
        const double QtHi  = 0.5*(fluxTh(i-1, k+1, b) + fluxTh(i, k+1, b));
        const double QsLo  = 0.5*(fluxS(i-1, k, b) + fluxS(i, k, b));
        const double QsHi  = 0.5*(fluxS(i-1, k, b+1) + fluxS(i, k, b+1));
        const double QmLo  = 0.5*(fluxSmesh(i-1, k, b) + fluxSmesh(i, k, b));
        const double QmHi  = 0.5*(fluxSmesh(i-1, k, b+1) + fluxSmesh(i, k, b+1));
        /* below the first sigma node the wall value is zero by no slip; above the
           last there is no u node, so the nearest value stands. Both faces carry
           exactly zero flux, so neither choice enters the answer -- they exist
           because a face value has to be a number. */
        const double uDn = b == 0      ? 0.0 : u[iu(i, k, b-1)];
        const double uUp = b == nz - 1 ? u[c] : u[iu(i, k, b+1)];
        const double f = QrOut*0.5*(u[c] + u[iu(i+1, k, b)])
                       - QrIn *0.5*(u[iu(i-1, k, b)] + u[c])
                       + QtHi *0.5*(u[c] + u[iu(i, k+1, b)])
                       - QtLo *0.5*(u[iu(i, k-1, b)] + u[c])
                       + QsHi *0.5*(u[c] + uUp)
                       - QsLo *0.5*(uDn + u[c]);
        advU[c] = -(f + u[c]*(QmHi - QmLo))/V;
      }
    }
  }

  /* ---- azimuthal momentum, on the v control volumes --------------------- */
  for (int i = 0; i < nr; i++){
    for (int k = 0; k < nth; k++){
      const double Hf = Hth[i*nth + kw(k)];
      for (int b = 0; b < nz; b++){
        const int c = iv(i, k, b);
        const double V = rc[i]*drc[i]*dth*Hf*dsc[b];
        const double QrIn  = 0.5*(fluxR(i, k-1, b)   + fluxR(i, k, b));
        const double QrOut = 0.5*(fluxR(i+1, k-1, b) + fluxR(i+1, k, b));
        const double QtLo  = 0.5*(fluxTh(i, k-1, b) + fluxTh(i, k, b));
        const double QtHi  = 0.5*(fluxTh(i, k, b)   + fluxTh(i, k+1, b));
        const double QsLo  = 0.5*(fluxS(i, k-1, b) + fluxS(i, k, b));
        const double QsHi  = 0.5*(fluxS(i, k-1, b+1) + fluxS(i, k, b+1));
        const double QmLo  = 0.5*(fluxSmesh(i, k-1, b) + fluxSmesh(i, k, b));
        const double QmHi  = 0.5*(fluxSmesh(i, k-1, b+1) + fluxSmesh(i, k, b+1));
        /* inward of the first column the continuation is the antipodal one with
           u_theta's own sign; outward of the last the sidewall holds v at zero.
           Both faces carry exactly zero flux. */
        const double vIn  = i == 0      ? -v[iv(0, k + half, b)] : v[iv(i-1, k, b)];
        const double vOut = i == nr - 1 ? 0.0 : v[iv(i+1, k, b)];
        const double vDn  = b == 0      ? 0.0 : v[iv(i, k, b-1)];
        const double vUp  = b == nz - 1 ? v[c] : v[iv(i, k, b+1)];
        const double f = QrOut*0.5*(v[c] + vOut)
                       - QrIn *0.5*(vIn + v[c])
                       + QtHi *0.5*(v[c] + v[iv(i, k+1, b)])
                       - QtLo *0.5*(v[iv(i, k-1, b)] + v[c])
                       + QsHi *0.5*(v[c] + vUp)
                       - QsLo *0.5*(vDn + v[c]);
        advV[c] = -(f + v[c]*(QmHi - QmLo))/V;
      }
    }
  }

  /* ---- vertical momentum, on the w control volumes ---------------------
     sc[b-1] .. sc[b] in sigma for b below nz, and sc[nz-1] .. 1 for the surface
     half cell. There is no pressure cell at b = nz, so that half cell's horizontal
     faces carry half of cell nz-1's fluxes and nothing else; its top face carries
     the full flux at sigma = 1, which the kinematic condition makes zero. Leaving
     that node out would put a face with non-zero flux on the edge of the energy
     sum, and the identity would hold only up to the work done through it. */
  for (int i = 0; i < nr; i++){
    for (int k = 0; k < nth; k++){
      const double Hc = H[ie(i, k)];
      for (int b = 1; b <= nz; b++){
        const int c = iw(i, k, b);
        const double V = rc[i]*drc[i]*dth*Hc*dsf[b];
        const int top = b == nz;
        const double QrIn  = 0.5*(fluxR(i, k, b-1)   + (top ? 0.0 : fluxR(i, k, b)));
        const double QrOut = 0.5*(fluxR(i+1, k, b-1) + (top ? 0.0 : fluxR(i+1, k, b)));
        const double QtLo  = 0.5*(fluxTh(i, k, b-1)   + (top ? 0.0 : fluxTh(i, k, b)));
        const double QtHi  = 0.5*(fluxTh(i, k+1, b-1) + (top ? 0.0 : fluxTh(i, k+1, b)));
        const double QsLo  = 0.5*(fluxS(i, k, b-1) + fluxS(i, k, b));
        const double QsHi  = top ? fluxS(i, k, nz) : 0.5*(fluxS(i, k, b) + fluxS(i, k, b+1));
        const double QmLo  = 0.5*(fluxSmesh(i, k, b-1) + fluxSmesh(i, k, b));
        const double QmHi  = top ? fluxSmesh(i, k, nz)
                                 : 0.5*(fluxSmesh(i, k, b) + fluxSmesh(i, k, b+1));
        /* inward of the first column the continuation is the antipodal one, and w
           is a scalar under that reflection; outward of the last the sidewall holds
           w at zero. Above the surface node there is nothing, so the node itself
           stands -- against a flux the kinematic condition makes zero. */
        const double wIn  = i == 0      ? w[iw(0, k + half, b)] : w[iw(i-1, k, b)];
        const double wOut = i == nr - 1 ? 0.0 : w[iw(i+1, k, b)];
        const double wUp  = top ? w[c] : w[iw(i, k, b+1)];
        const double f = QrOut*0.5*(w[c] + wOut)
                       - QrIn *0.5*(wIn + w[c])
                       + QtHi *0.5*(w[c] + w[iw(i, k+1, b)])
                       - QtLo *0.5*(w[iw(i, k-1, b)] + w[c])
                       + QsHi *0.5*(w[c] + wUp)
                       - QsLo *0.5*(w[iw(i, k, b-1)] + w[c]);
        advW[c] = -(f + w[c]*(QmHi - QmLo))/V;
      }
    }
  }
}

/* The two terms the rotating basis contributes to the cylindrical momentum
 * equations, +u_theta^2/r in the radial one and -u_r u_theta/r in the azimuthal.
 *
 * THEY CANCEL EXACTLY IN THE ENERGY, and getting that cancellation is the whole
 * design. In the continuum it is POINTWISE. On a staggered grid u_r and u_theta
 * live in different places, and interpolating each to the other's node leaves
 * (mean v)^2 on one side against v^2 on the other, which do not cancel: the pair
 * then acts as an energy source of the scheme's own truncation order. What does
 * cancel is to form the product at ONE place, the pressure cell centre, where both
 * components have a single value, and to distribute it as exact adjoints of those
 * cell-centre averages -- so the two sums are the same number twice with opposite
 * signs, cancelling in floating point to the last bit rather than to an order. */
void cell3d_advectCurvature(){
  for (int i = 1; i < nr; i++)
    for (int k = 0; k < nth; k++){
      const double Hf = Hr[i*nth + kw(k)];
      for (int b = 0; b < nz; b++){
        const double vA = 0.5*(v[iv(i-1, k, b)] + v[iv(i-1, k+1, b)]);
        const double GvA = vA*vA/rc[i-1];
        const double vB = 0.5*(v[iv(i, k, b)] + v[iv(i, k+1, b)]);
        const double GvB = vB*vB/rc[i];
        const double VA = rc[i-1]*drc[i-1]*dth*H[ie(i-1, k)]*dsc[b];
        const double VB = rc[i]*drc[i]*dth*H[ie(i, k)]*dsc[b];
        advU[iu(i, k, b)] += 0.5*(VA*GvA + VB*GvB)
                           / (rf[i]*drf[i]*dth*Hf*dsc[b]);
      }
    }
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++){
      const double Hf = Hth[i*nth + kw(k)];
      for (int b = 0; b < nz; b++){
        const double uA = 0.5*(u[iu(i, k-1, b)] + u[iu(i+1, k-1, b)]);
        const double vA = 0.5*(v[iv(i, k-1, b)] + v[iv(i, k, b)]);
        const double PuA = uA*vA/rc[i];
        const double uB = 0.5*(u[iu(i, k, b)] + u[iu(i+1, k, b)]);
        const double vB = 0.5*(v[iv(i, k, b)] + v[iv(i, k+1, b)]);
        const double PuB = uB*vB/rc[i];
        const double VA = rc[i]*drc[i]*dth*H[ie(i, k-1)]*dsc[b];
        const double VB = rc[i]*drc[i]*dth*H[ie(i, k)]*dsc[b];
        advV[iv(i, k, b)] -= 0.5*(VA*PuA + VB*PuB)
                           / (rc[i]*drc[i]*dth*Hf*dsc[b]);
      }
    }
}

void cell3d_advect(){ cell3d_advectTransport(); cell3d_advectCurvature(); }

/* ---- the free surface's slopes, metric, area and curvature ------------- */

/* The four face slopes of eta around one cell, written ONCE because the curvature
 * is the exact adjoint of exactly these and the two must not be able to drift
 * apart. Filled as [inner r, outer r, lower theta, upper theta, then the two
 * radial weights].
 *
 * The axis needs no condition: its face area rf[0] is exactly zero, so the area
 * functional weights that slope by nothing. The rim needs one, and which one is
 * the contact condition.
 *
 * The weights are the faces' own areas, which is second order because the two
 * faces bracket the centre symmetrically, and it is that weighting which makes the
 * curvature's radial face coefficient come out as the width-weighted mean of
 * 1/sqrt(1 + |grad eta|^2). The pinned rim is the exception: a two-point
 * difference against the wall value is the exact slope at the midpoint of rc[nr-1]
 * and R, which sits at rc + drc/4, while the inner face's slope is at rc - drc/2,
 * so the weights that centre the estimate on rc are exactly one third and two
 * thirds on any grid. Weighting by face area instead put the estimate a quarter of
 * a cell off centre: rows nr-1 and nr-2 sat at 1.3% and 0.7% and did not converge
 * while every interior row ran at second order. */
static void etaSlopes(int i, int k, double* out){
  const int e = ie(i, k);
  const int pinnedRim = (i == nr - 1 && pinned) ? 1 : 0;
  out[0] = i == 0 ? 0.0 : (eta[e] - eta[ie(i-1, k)])/drf[i];
  if (i < nr - 1) out[1] = (eta[ie(i+1, k)] - eta[e])/drf[i+1];
  else if (!pinnedRim) out[1] = 0.0;
  else out[1] = (0.0 - eta[e])/drf[nr];
  out[2] = (eta[e] - eta[ie(i, k-1)])/dth;
  out[3] = (eta[ie(i, k+1)] - eta[e])/dth;
  const double wsum = rf[i] + rf[i+1];
  out[4] = rf[i]/wsum; out[5] = rf[i+1]/wsum;
}

/* 1 + |grad eta|^2 at a cell centre, written out rather than as
   1 + surfaceSlopeSquared, because ((1 + A) + B) + C and 1 + ((A + B) + C) are not
   the same double and this quantity feeds the curvature, the surface pressure and
   every measured order in the file. */
static double surfaceMetric(int i, int k){
  double sl[6];
  etaSlopes(i, k, sl);
  const double r = rc[i];
  return 1 + sl[4]*sl[0]*sl[0] + sl[5]*sl[1]*sl[1]
           + 0.5*(sl[2]*sl[2] + sl[3]*sl[3])/(r*r);
}

/* |grad eta|^2 alone, which the excess area needs WITHOUT the one: sqrt(1+q) - 1
   loses every significant digit for small q, and so does surfaceArea() - pi R^2,
   where at eta = 1e-9 m the difference is 1.2e-18 out of operands of 4.6e-4. */
static double surfaceSlopeSquared(int i, int k){
  double sl[6];
  etaSlopes(i, k, sl);
  const double r = rc[i];
  return sl[4]*sl[0]*sl[0] + sl[5]*sl[1]*sl[1]
       + 0.5*(sl[2]*sl[2] + sl[3]*sl[3])/(r*r);
}

double cell3d_surfaceArea(){
  double A = 0.0;
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++)
      A += rc[i]*drc[i]*dth*__builtin_sqrt(surfaceMetric(i, k));
  return A;
}

/* The area the surface has IN EXCESS of flat. Sum rc drc dtheta is exactly pi R^2,
   so the excess is the same sum of sqrt(1+q) - 1 term by term, written as
   q/(1 + sqrt(1+q)) -- the same number in exact arithmetic, keeping every digit
   where the subtraction has none. Measured before this existed: the excess came out
   1.1926e-18 against the 1.1596e-18 its own amplitude scaling demands, 2.8 per cent
   wrong, and the total energy then drifted 25.3 per cent over a quarter period
   where the same run at eta = 1e-7 drifts 0.0356. That was the DIAGNOSTIC failing,
   not the solver. */
double cell3d_surfaceExcessArea(){
  double dA = 0.0;
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++){
      const double q = surfaceSlopeSquared(i, k);
      dA += rc[i]*drc[i]*dth*(q/(1 + __builtin_sqrt(1 + q)));
    }
  return dA;
}

/* The mean curvature of the free surface, div(grad eta / sqrt(1 + |grad eta|^2)) --
   NOT the linearised Laplacian of eta, because at the drives this cell runs
   |grad eta| is of order one.

   IT IS DEFINED AS THE VARIATIONAL DERIVATIVE OF THE DISCRETE AREA,
       kappa_j = -(1/(rc drc dtheta)) dA/d eta_j
   and that is the design rather than a way of writing it down: the expression IS
   the finite-volume divergence form, it is second order because the area is, and
   the work the capillary term does is exactly -gamma dA/dt, so the exchange
   between kinetic and surface energy is an identity in floating point rather than
   a tolerance. Written as a scatter, in the same accumulate-then-divide shape the
   gradient uses, because that is what makes it the exact adjoint of the slopes. */
void cell3d_curvature(double* out){
  for (int c = 0; c < NE; c++) out[c] = 0.0;
  double sl[6];
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++){
      const int e = ie(i, k);
      etaSlopes(i, k, sl);
      const double rr = rc[i]*rc[i];
      const double X = 1 + sl[4]*sl[0]*sl[0] + sl[5]*sl[1]*sl[1]
                         + 0.5*(sl[2]*sl[2] + sl[3]*sl[3])/rr;
      const double Wt = rc[i]*drc[i]*dth/(2*__builtin_sqrt(X));
      const double cIn = Wt*2*sl[4]*sl[0], cOut = Wt*2*sl[5]*sl[1];
      if (i > 0){
        out[e] += cIn/drf[i];
        out[ie(i-1, k)] -= cIn/drf[i];
      }
      if (i < nr - 1){
        out[e] -= cOut/drf[i+1];
        out[ie(i+1, k)] += cOut/drf[i+1];
      } else if (pinned){
        out[e] -= cOut/drf[nr];
      }
      const double cLo = Wt*sl[2]/rr, cHi = Wt*sl[3]/rr;
      out[e] += (cLo - cHi)/dth;
      out[ie(i, k-1)] -= cLo/dth;
      out[ie(i, k+1)] += cHi/dth;
    }
  for (int i = 0; i < nr; i++){
    const double vol = rc[i]*drc[i]*dth;
    for (int k = 0; k < nth; k++) out[ie(i, k)] /= -vol;
  }
}

/* ---- the free surface's pressure, and the step ------------------------- */

/* The effective gravity g + a cos(omega_d t). The cos is the one transcendental
   the physics needs and it is NOT computed here: JavaScript evaluates it and sets
   this before each step, which is what makes bit-for-bit parity achievable rather
   than approximate -- a libm cos and V8's cos need not agree on the last bit, and
   the whole step hangs off this number. */
static double gEff = CELL3D_GRAVITY_UNSET;
void cell3d_setGravity(double g){ gEff = g; }
double cell3d_gravity(){ return gEff; }

/* The pressure the free surface carries, at every pressure column: the hydrostatic
   response of the displaced elevation to the instantaneous gravity, the capillary
   term with the FULL mean curvature rather than its linearisation, and the viscous
   normal stress with the full rate-of-strain contraction against the true normal.

       p_s = rho g_eff eta - gamma kappa + 2 rho nu (n.E.n)

   It is the inhomogeneous DIRICHLET value on the projection, never a force on the
   predictor -- see cell3d_step. */
int cell3d_surfacePressure(double* out){
  if (gEff == CELL3D_GRAVITY_UNSET){ lastError = 6; return lastError; }
  const double g = gEff;
  cell3d_curvature(kap);
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++){
      const int e = ie(i, k);
      out[e] = rho*g*eta[e] - gamma_*kap[e] + surfaceNormalStress(i, k);
    }
  return lastError;
}

/* ONE STEP OF THE NAVIER-STOKES EQUATIONS. Everything above is an operator; this
   is what makes the file a solver. Three parts of the order were established by
   measurement rather than by choice:

   1. THE SURFACE PRESSURE IS AN INHOMOGENEOUS DIRICHLET VALUE INSIDE THE
      PROJECTION, not a force on the predictor. As a predictor force it is
      p_s/(rho H dsigma) -- O(1/dsigma) -- and the projection then cancels almost
      all of it, leaving the physical acceleration as the difference of two large
      numbers and a step limit that collapses with the grid. The two-dimensional
      solver went non-finite 0.57 periods in that way at nr = 48, nz = 20, m = 12.
   2. THE AXIS AND WALL VALUES ARE SET BEFORE Omega IS FORMED. Omega is derived
      from u, v and w, and u at the axis face enters it through the innermost
      cell's slope term, so setting that value after forming Omega leaves the two
      inconsistent: measured, a projection asked for 1e-14 reported a divergence of
      6.74e-2 where the same projection at 1e-9 reported 3.40e-8.
   3. ETA ADVANCES ON THE CORRECTED Omega AT sigma = 1, which IS the kinematic
      condition. Pairing the surface pressure at the old time with the surface
      velocity at the new one is symplectic Euler on the surface oscillator, stable
      for omega dt < 2 rather than merely less unstable than forward Euler.

   The predictor carries the viscous and advective terms in full. Gravity does not
   appear in it: this pressure is the total one, and the whole of gravity's effect
   on the interior is the hydrostatic head the surface value carries. */
int cell3d_step(double dt){
  if (!(dt > 0)){ lastError = 7; return lastError; }

  /* the surface's own state, read BEFORE anything moves: the stresses and the
     curvature belong to the surface the velocity is being advanced over */
  cell3d_omegaFromW();
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++) Ht[ie(i, k)] = om[iw(i, k, nz)];
  if (cell3d_surfacePressure(psurf)) return lastError;

  /* the explicit right-hand side: viscous, with the free surface's own traction on
     the sigma = 1 face, and advective, in the conservative grid-relative form */
  cell3d_viscous(1);
  if (lastError) return lastError;
  cell3d_advect();

  for (int i = 1; i < nr; i++)
    for (int k = 0; k < nth; k++)
      for (int j = 0; j < nz; j++){
        const int c = iu(i, k, j);
        u[c] += dt*(nu*lapU[c] + advU[c]);
      }
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++)
      for (int j = 0; j < nz; j++){
        const int c = iv(i, k, j);
        v[c] += dt*(nu*lapV[c] + advV[c]);
      }
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++)
      for (int j = 1; j <= nz; j++){
        const int c = iw(i, k, j);
        w[c] += dt*(nu*lapW[c] + advW[c]);
      }

  /* the prescribed values, and only then Omega -- reason 2 above */
  for (int k = 0; k < nth; k++)
    for (int j = 0; j < nz; j++) u[iu(nr, k, j)] = 0.0;   // no penetration at the rim
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++) w[iw(i, k, 0)] = 0.0;   // no slip on the floor
  cell3d_axisU();
  cell3d_omegaFromW();

  /* the projection, with the surface pressure as its Dirichlet value -- reason 1 */
  cell3d_divergence(u, v, om, divw);
  const double scale = rho/dt;
  for (int c = 0; c < NP; c++) divw[c] *= scale;
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++)
      divw[ip(i, k, nz - 1)] -= rc[i]*drc[i]*dth
        *psurf[ie(i, k)]/(H[ie(i, k)]*dsf[nz]);
  for (int c = 0; c < NP; c++) p[c] = 0.0;
  cell3d_solveP(divw, 1e-11, 400*(nr + nth + nz));
  if (lastError) return lastError;

  cell3d_gradient(p, gu, gv, gw);
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++){
      const int e = ie(i, k);
      gw[iw(i, k, nz)] += psurf[e]/(H[e]*dsf[nz]);
    }

  const double s2 = dt/rho;
  for (int i = 1; i < nr; i++)
    for (int k = 0; k < nth; k++)
      for (int j = 0; j < nz; j++){
        const int c = iu(i, k, j);
        u[c] -= s2*gu[c];
      }
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++)
      for (int j = 0; j < nz; j++){
        const int c = iv(i, k, j);
        v[c] -= s2*gv[c];
      }
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++)
      for (int j = 1; j <= nz; j++){
        const int c = iw(i, k, j);
        w[c] -= s2*gw[c];
      }
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++) w[iw(i, k, 0)] = 0.0;
  cell3d_axisU();
  cell3d_omegaFromW();

  /* eta on the corrected Omega at the surface, which IS dH/dt -- reason 3 above */
  for (int i = 0; i < nr; i++)
    for (int k = 0; k < nth; k++){
      const int e = ie(i, k);
      eta[e] += dt*om[iw(i, k, nz)];
      Ht[e] = om[iw(i, k, nz)];
    }
  /* The caller owns the clock, because the caller owns the cos that reads it. */
  cell3d_refreshMetric();
  /* A step that has consumed its gravity must not silently reuse it: the next one
     is a different instant of the drive. */
  gEff = CELL3D_GRAVITY_UNSET;
  return lastError;
}

}  // extern "C"
