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
#define CELL3D_GRAVITY_UNSET (-1.0e308)
static const int ARENA = 12000000;
static double arena[ARENA];
static int arenaUsed = 0;
static int lastError = 0;
static int errI = -1, errK = -1;
static int nr, nth, nz, nw;
static int NU, NV, NW, NP, NE, NX;
static double R, h, rho, nu, gamma_, gBase, dth;
static int pinned;
static double *rf, *sf, *rc, *sc, *drc, *dsc, *drf, *dsf, *rx;
static double *u, *v, *w, *om, *p, *eta;
static double *H, *Hr, *Hth, *Hdr, *Hdth, *Ht;
static double *Hx, *Hxr, *Hxt;
static double *Ex, *Er, *Eth, *Edr, *Edth, *Exr, *Ext;
static double *Tx, *Txr, *Txt, *Tr, *Tth, *Tdr, *Tdth;
static double *gu, *gv, *gw, *gom, *divw;
static double *cgr, *cgd, *cgq, *cgz;
static double *twc, *tws, *pcband, *pchatc, *pchats;
static double *pcxr, *pcxi, *pcyr, *pcyi, *pcar, *pcai;
static int fftFac[64]; static int nFac = 0;
static int cgIters = 0; static double cgResidual = 0.0;
static double *lapU, *lapV, *lapW, *advU, *advV, *advW;
static double *fsr, *fst, *fsz, *kap, *psurf;
static double *rbU, *sbW, *axEx;
static double *hcolP, *hcolU, *hcolV, *hcolW;
static double *fRlo, *fRhi, *fT, *fS;
static const int IARENA = 400000;
static int iarena[IARENA];
static int iarenaUsed = 0;
static int* itake(int n){
if (iarenaUsed + n > IARENA){ lastError = 1; return iarena; }
int* q = iarena + iarenaUsed; iarenaUsed += n;
for (int i = 0; i < n; i++) q[i] = 0;
return q;
}
struct Fam {
int axisSign;
double thOff;
int stride;
const double* rn; const double* rb; int nI;
const double* sn; const double* sb; int nJ;
int rLo, rHi, sLo, sHi;
const int* hBr;
double* Hcol;
};
static Fam FAMP, FAMU, FAMV, FAMW;
static double* take(int n){
if (arenaUsed + n > ARENA){ lastError = 1; return arena; }
double* q = arena + arenaUsed; arenaUsed += n;
for (int i = 0; i < n; i++) q[i] = 0.0;
return q;
}
static inline int kw(int k){ return ((k % nth) + nth) % nth; }
static inline int iu(int i, int k, int j){ return (i*nth + kw(k))*nz + j; }
static inline int iv(int i, int k, int j){ return (i*nth + kw(k))*nz + j; }
static inline int iw(int i, int k, int j){ return (i*nth + kw(k))*nw + j; }
static inline int ip(int i, int k, int j){ return (i*nth + kw(k))*nz + j; }
static inline int ie(int i, int k){ return i*nth + kw(k); }
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
NP = nr*nth*nz; NE = nr*nth; NX = (nr + 2)*nth;
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
const int M = (nth >> 1) + 1, n2 = nr*nz, W = nz + 1, L = nth >> 1;
twc = take(nth); tws = take(nth);
pcband = take(M*n2*W); pchatc = take(M*n2); pchats = take(M*n2);
pcxr = take(L*nz); pcxi = take(L*nz); pcyr = take(L*nz); pcyi = take(L*nz);
pcar = take(nz); pcai = take(nz);
nFac = 0;
int n = L;
while (n % 4 == 0){ fftFac[nFac++] = 4; n /= 4; }
while (n % 2 == 0){ fftFac[nFac++] = 2; n /= 2; }
for (int pr = 3; n > 1; pr += 2) while (n % pr == 0){ fftFac[nFac++] = pr; n /= pr; }
}
lapU = take(NU); lapV = take(NV); lapW = take(NW);
advU = take(NU); advV = take(NV); advW = take(NW);
fsr = take(NE); fst = take(NE); fsz = take(NE);
kap = take(NE); psurf = take(NE);
rbU = take(nr + 2); sbW = take(nz + 2);
axEx = take(nth);
hcolP = take(NE); hcolU = take((nr + 1)*nth); hcolV = take(NE); hcolW = take(NE);
fRlo = take(nth*(nz + 1)); fRhi = take(nth*(nz + 1)); fT = take(nth*(nz + 1));
fS = take(nz + 2);
cgIters = 0; cgResidual = 0.0;
return lastError;
}
int cell3d_error(){ return lastError; }
int cell3d_errorI(){ return errI; }
int cell3d_errorK(){ return errK; }
void cell3d_clearError(){ lastError = 0; errI = -1; errK = -1; }
int cell3d_arenaUsed(){ return arenaUsed; }
double* cell3d_ptr(int which){
switch (which){
case 0: return rf;
case 1: return sf;
case 2: return rc;
case 3: return sc;
case 4: return drc;
case 5: return dsc;
case 6: return drf;
case 7: return dsf;
case 8: return rx;
case 9: return u;
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
case 62: return twc;
case 63: return tws;
case 64: return pcband;
case 65: return pchatc;
case 66: return pchats;
default: return 0;
}
}
int cell3d_len(int which){
switch (which){
case 0: return nr + 1;
case 1: return nz + 1;
case 2: return nr;
case 3: return nz;
case 4: return nr;
case 5: return nz;
case 6: return nr + 1;
case 7: return nz + 1;
case 8: return nr + 2;
case 9: return NU;
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
case 62: return nth;
case 63: return nth;
case 64: return ((nth >> 1) + 1)*nr*nz*(nz + 1);
case 65: return ((nth >> 1) + 1)*nr*nz;
case 66: return ((nth >> 1) + 1)*nr*nz;
default: return -1;
}
}
void cell3d_buildFamilies(){
iarenaUsed = 0;
rbU[0] = 0.0;
for (int i = 0; i < nr; i++) rbU[i+1] = rc[i];
rbU[nr+1] = R;
sbW[0] = 0.0;
for (int j = 0; j < nz; j++) sbW[j+1] = sc[j];
sbW[nz+1] = 1.0;
FAMP.axisSign = +1; FAMP.thOff = 0.5; FAMP.stride = nz;
FAMP.rn = rc; FAMP.rb = rf; FAMP.nI = nr; FAMP.rLo = 0; FAMP.rHi = nr - 1;
FAMP.sn = sc; FAMP.sb = sf; FAMP.nJ = nz; FAMP.sLo = 0; FAMP.sHi = nz - 1;
FAMP.Hcol = hcolP;
FAMU.axisSign = -1; FAMU.thOff = 0.5; FAMU.stride = nz;
FAMU.rn = rf; FAMU.rb = rbU; FAMU.nI = nr + 1; FAMU.rLo = 1; FAMU.rHi = nr - 1;
FAMU.sn = sc; FAMU.sb = sf; FAMU.nJ = nz; FAMU.sLo = 0; FAMU.sHi = nz - 1;
FAMU.Hcol = hcolU;
FAMV.axisSign = -1; FAMV.thOff = 0.0; FAMV.stride = nz;
FAMV.rn = rc; FAMV.rb = rf; FAMV.nI = nr; FAMV.rLo = 0; FAMV.rHi = nr - 1;
FAMV.sn = sc; FAMV.sb = sf; FAMV.nJ = nz; FAMV.sLo = 0; FAMV.sHi = nz - 1;
FAMV.Hcol = hcolV;
FAMW.axisSign = +1; FAMW.thOff = 0.5; FAMW.stride = nz + 1;
FAMW.rn = rc; FAMW.rb = rf; FAMW.nI = nr; FAMW.rLo = 0; FAMW.rHi = nr - 1;
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
Hdr[e] = (Hr[(i+1)*nth + kw(k)] - Hr[i*nth + kw(k)])/drc[i];
Hdth[e] = (Hth[i*nth + kw(k+1)] - Hth[i*nth + kw(k)])/dth;
}
const int half = nth >> 1;
const int freeCL = pinned ? 0 : 1;
for (int k = 0; k < nth; k++){
const int ka = kw(k + half);
Hx [0*nth + k] = H[ie(0, ka)];
Hxr[0*nth + k] = -Hdr[ie(0, ka)];
Hxt[0*nth + k] = Hdth[ie(0, ka)];
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
Edr[e] = (Er[(i+1)*nth + kw(k)] - Er[i*nth + kw(k)])/drc[i];
Edth[e] = (Eth[i*nth + kw(k+1)] - Eth[i*nth + kw(k)])/dth;
}
for (int k = 0; k < nth; k++){
const int ka = kw(k + half);
Exr[0*nth + k] = -Edr[ie(0, ka)];
Ext[0*nth + k] = Edth[ie(0, ka)];
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
Tdr[e] = (Tr[(i+1)*nth + kw(k)] - Tr[i*nth + kw(k)])/drc[i];
Tdth[e] = (Tth[i*nth + kw(k+1)] - Tth[i*nth + kw(k)])/dth;
}
for (int k = 0; k < nth; k++){
const int ka = kw(k + half);
Tx [0*nth + k] = Ht[ie(0, ka)];
Txr[0*nth + k] = -Tdr[ie(0, ka)];
Txt[0*nth + k] = Tdth[ie(0, ka)];
for (int i = 0; i < nr; i++){
Tx [(i+1)*nth + k] = Ht[ie(i, k)];
Txr[(i+1)*nth + k] = Tdr[ie(i, k)];
Txt[(i+1)*nth + k] = Tdth[ie(i, k)];
}
Tx [(nr+1)*nth + k] = freeCL ? Ht[ie(nr-1, k)] : 0.0;
Txr[(nr+1)*nth + k] = freeCL ? 0.0 : (0.0 - Ht[ie(nr-1, k)])/drf[nr];
Txt[(nr+1)*nth + k] = freeCL ? Tdth[ie(nr-1, k)] : 0.0;
}
Fam* fams[4] = { &FAMP, &FAMU, &FAMV, &FAMW };
for (int t = 0; t < 4; t++){
Fam& fam = *fams[t];
for (int a = 0; a < fam.nI; a++)
for (int k = 0; k < nth; k++)
fam.Hcol[a*nth + k] = HatHBr(fam.hBr[a], fam.rn[a], (k + fam.thOff)*dth);
}
return lastError;
}
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
static void gradientWithSurface(const double* q, double* ou, double* ov, double* ow,
const double* surfaceP){
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
ou[bIn + j] -= qc*dth*dsc[j]*rIn *HrIn;
ov[bK + j] += qc*dr*dsc[j]*HthOut;
ov[bIn + j] -= qc*dr*dsc[j]*HthIn;
ow[bW + j + 1] += qc*rci*dr*dth;
ow[bW + j ] -= qc*rci*dr*dth;
}
}
}
if (surfaceP)
for (int i = 0; i < nr; i++)
for (int k = 0; k < nth; k++)
ow[(i*nth + k)*nw + nz] -= surfaceP[i*nth + k]*rc[i]*drc[i]*dth;
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
ow[b] = 0.0;
for (int j = 1; j <= nz; j++)
ow[b + j] /= -(rc[i]*drc[i]*dth*Hc*dsf[j]);
}
}
void cell3d_gradient(const double* q, double* ou, double* ov, double* ow){
gradientWithSurface(q, ou, ov, ow, 0);
}
void cell3d_gradientSurface(const double* q, const double* surfaceP,
double* ou, double* ov, double* ow){
gradientWithSurface(q, ou, ov, ow, surfaceP);
}
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
void cell3d_applyL(const double* q, double* out){
cell3d_gradient(q, gu, gv, gw);
cell3d_omegaOf(gu, gv, gw, gom);
cell3d_divergence(gu, gv, gom, out);
}
static void cell3dFft(double* xr, double* xi, double* yr, double* yi, int L, int nb, int sign,
double** outR, double** outI){
const double* C = twc; const double* S = tws;
double* ar = pcar; double* ai = pcai;
int n = L, s = 1;
for (int f = 0; f < nFac; f++){
const int p = fftFac[f], m = n/p, step = L/n, ist = s*m*nb, ost = s*nb, sub = L/p;
for (int pp = 0; pp < m; pp++)
for (int q = 0; q < s; q++){
const int ib = (q + s*pp)*nb, ob = (q + s*p*pp)*nb;
if (p == 2){
const int tix = 2*pp*step;
const double wr = C[tix], wi = sign*S[tix];
for (int b = 0; b < nb; b++){
const double a0r = xr[ib + b], a0i = xi[ib + b];
const double a1r = xr[ib + ist + b], a1i = xi[ib + ist + b];
yr[ob + b] = a0r + a1r; yi[ob + b] = a0i + a1i;
const double dr = a0r - a1r, di = a0i - a1i;
if (pp == 0){ yr[ob + ost + b] = dr; yi[ob + ost + b] = di; }
else { yr[ob + ost + b] = dr*wr - di*wi; yi[ob + ost + b] = dr*wi + di*wr; }
}
} else if (p == 4){
const int t1 = 2*pp*step, t2 = 4*pp*step, t3 = 6*pp*step;
const double w1r = C[t1], w1i = sign*S[t1], w2r = C[t2], w2i = sign*S[t2];
const double w3r = C[t3], w3i = sign*S[t3];
for (int b = 0; b < nb; b++){
const double a0r = xr[ib + b], a0i = xi[ib + b];
const double a1r = xr[ib + ist + b], a1i = xi[ib + ist + b];
const double a2r = xr[ib + 2*ist + b], a2i = xi[ib + 2*ist + b];
const double a3r = xr[ib + 3*ist + b], a3i = xi[ib + 3*ist + b];
const double s0r = a0r + a2r, s0i = a0i + a2i, d0r = a0r - a2r, d0i = a0i - a2i;
const double s1r = a1r + a3r, s1i = a1i + a3i, d1r = a1r - a3r, d1i = a1i - a3i;
const double qr = sign < 0 ? d1i : -d1i, qi = sign < 0 ? -d1r : d1r;
const double y1r = d0r + qr, y1i = d0i + qi;
const double y2r = s0r - s1r, y2i = s0i - s1i;
const double y3r = d0r - qr, y3i = d0i - qi;
yr[ob + b] = s0r + s1r; yi[ob + b] = s0i + s1i;
if (pp == 0){
yr[ob + ost + b] = y1r; yi[ob + ost + b] = y1i;
yr[ob + 2*ost + b] = y2r; yi[ob + 2*ost + b] = y2i;
yr[ob + 3*ost + b] = y3r; yi[ob + 3*ost + b] = y3i;
} else {
yr[ob + ost + b] = y1r*w1r - y1i*w1i; yi[ob + ost + b] = y1r*w1i + y1i*w1r;
yr[ob + 2*ost + b] = y2r*w2r - y2i*w2i; yi[ob + 2*ost + b] = y2r*w2i + y2i*w2r;
yr[ob + 3*ost + b] = y3r*w3r - y3i*w3i; yi[ob + 3*ost + b] = y3r*w3i + y3i*w3r;
}
}
} else {
for (int t = 0; t < p; t++){
for (int b = 0; b < nb; b++){ ar[b] = xr[ib + b]; ai[b] = xi[ib + b]; }
for (int r = 1; r < p; r++){
const int tix = 2*sub*((r*t) % p), rb = ib + r*ist;
const double wr = C[tix], wi = sign*S[tix];
for (int b = 0; b < nb; b++){
const double xr1 = xr[rb + b], xi1 = xi[rb + b];
ar[b] += xr1*wr - xi1*wi; ai[b] += xr1*wi + xi1*wr;
}
}
const int o = ob + t*ost;
if (pp == 0 || t == 0)
for (int b = 0; b < nb; b++){ yr[o + b] = ar[b]; yi[o + b] = ai[b]; }
else {
const int tix = 2*pp*t*step;
const double wr = C[tix], wi = sign*S[tix];
for (int b = 0; b < nb; b++){
yr[o + b] = ar[b]*wr - ai[b]*wi; yi[o + b] = ar[b]*wi + ai[b]*wr;
}
}
}
}
}
n = m; s *= p;
double* tr = xr; xr = yr; yr = tr;
tr = xi; xi = yi; yi = tr;
}
*outR = xr; *outI = xi;
}
static void applyPreconditioner(const double* r, double* z){
const int n2 = nr*nz, bw = nz, W = bw + 1, M = (nth >> 1) + 1, L = nth >> 1;
const double* C = twc; const double* Sn = tws;
double* xr = pcxr; double* xi = pcxi; double* yr = pcyr; double* yi = pcyi;
double *Zr, *Zi;
for (int i = 0; i < nr; i++){
const int rb = i*nth*nz;
for (int n = 0; n < L; n++)
for (int j = 0; j < nz; j++){
xr[n*nz + j] = r[rb + 2*n*nz + j];
xi[n*nz + j] = r[rb + (2*n + 1)*nz + j];
}
cell3dFft(xr, xi, yr, yi, L, nz, -1, &Zr, &Zi);
for (int m = 0; m <= L; m++){
const int a = m == L ? 0 : m, b = m == 0 ? 0 : L - m;
const double c = C[m], sn = Sn[m];
const int hb = m*n2 + i*nz;
const int real = m == 0 || m == L;
for (int j = 0; j < nz; j++){
const double z1r = Zr[a*nz + j], z1i = Zi[a*nz + j];
const double z2r = Zr[b*nz + j], z2i = Zi[b*nz + j];
const double er = (z1r + z2r)/2.0, ei = (z1i - z2i)/2.0;
const double orr = (z1i + z2i)/2.0, oi = (z2r - z1r)/2.0;
pchatc[hb + j] = er + (orr*c + oi*sn);
pchats[hb + j] = real ? 0.0 : -(ei + (oi*c - orr*sn));
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
for (int i = 0; i < nr; i++){
for (int m = 0; m < L; m++){
const int b = L - m, h1 = m*n2 + i*nz, h2 = b*n2 + i*nz;
const double c = C[m], sn = Sn[m];
for (int j = 0; j < nz; j++){
const double x1r = pchatc[h1 + j], x1i = m == 0 ? 0.0 : -pchats[h1 + j];
const double x2r = pchatc[h2 + j], x2i = b == L ? 0.0 : -pchats[h2 + j];
const double er = (x1r + x2r)/2.0, ei = (x1i - x2i)/2.0;
const double dr = (x1r - x2r)/2.0, di = (x1i + x2i)/2.0;
const double orr = dr*c - di*sn, oi = dr*sn + di*c;
xr[m*nz + j] = er - oi;
xi[m*nz + j] = ei + orr;
}
}
cell3dFft(xr, xi, yr, yi, L, nz, 1, &Zr, &Zi);
const int rb = i*nth*nz;
for (int n = 0; n < L; n++)
for (int j = 0; j < nz; j++){
z[rb + 2*n*nz + j] = -(Zr[n*nz + j]/L);
z[rb + (2*n + 1)*nz + j] = -(Zi[n*nz + j]/L);
}
}
}
void cell3d_applyPreconditioner(const double* r, double* z){ applyPreconditioner(r, z); }
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
int cell3d_cgIters(){ return cgIters; }
double cell3d_cgResidual(){ return cgResidual; }
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
double cell3d_maxDivergence(){
const double* fields[] = {u, v, w, om, eta, H, Ht, p};
const int lengths[] = {NU, NV, NW, NW, NE, NE, NE, NP};
for (int f = 0; f < 8; f++)
for (int c = 0; c < lengths[f]; c++)
if (!__builtin_isfinite(fields[f][c]) || (f == 5 && !(fields[f][c] > 0.0))){
lastError = 8; errI = f; errK = c; __builtin_trap();
}
cell3d_omegaOf(u, v, w, gom);
cell3d_divergence(u, v, gom, divw);
double worst = 0.0;
for (int i = 0; i < nr; i++)
for (int k = 0; k < nth; k++){
const double vol = rc[i]*drc[i]*dth*H[ie(i, k)];
for (int j = 0; j < nz; j++){
const double volume = vol*dsc[j];
if (!(__builtin_isfinite(volume) && volume > 0.0)){
lastError = 9; errI = i; errK = k*nz + j; __builtin_trap();
}
double d = divw[ip(i, k, j)]/volume;
if (!__builtin_isfinite(d)){
lastError = 8; errI = 8; errK = ip(i, k, j); __builtin_trap();
}
if (d < 0.0) d = -d;
if (d > worst) worst = d;
}
}
return worst;
}
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
static const int TH_NODE[4] = { -2, -1, 1, 2 };
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
static double colValueAtZ(const double* f, const Fam& fam, int a, int k,
double z, int lev, int hasLev, int bracket){
const double* sn = fam.sn; const int nJ = fam.nJ; const int half = nth >> 1;
int aa = a, kk = k; double sign = 1.0;
if (a < 0){ aa = -1 - a; kk = k + half; sign = (double)fam.axisSign; }
const int col = aa*nth + kw(kk);
const double Hc = fam.Hcol[col];
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
static void surfaceNormal(double r, double th, double* o){
double g[2];
Hslope(r, th, g);
const double sr = g[0], st = g[1]/r;
const double len = __builtin_sqrt(1 + sr*sr + st*st);
o[0] = sr; o[1] = st; o[2] = len;
o[3] = -sr/len; o[4] = -st/len; o[5] = 1/len;
}
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
static void refreshSurfaceFluxRows(int i0, int i1){
double t3[3];
const int iFirst = i0 < 0 ? 0 : i0, iLast = i1 > nr - 1 ? nr - 1 : i1;
for (int i = iFirst; i <= iLast; i++)
for (int k = 0; k < nth; k++){
surfaceLapFluxes(i, k, t3);
const int e = ie(i, k);
fsr[e] = t3[0]; fst[e] = t3[1]; fsz[e] = t3[2];
}
}
void cell3d_refreshSurfaceFluxes(){ refreshSurfaceFluxRows(0, nr - 1); }
static double surfaceFluxFace(int which, int a, int k){
if (which == 2) return 0.5*(fst[ie(a, k - 1)] + fst[ie(a, k)]);
if (which == 3) return fsz[ie(a, k)];
const double lo = fsr[ie(a - 1, k)], hi = fsr[ie(a, k)];
const double wa = drc[a-1], wb = drc[a];
return lo + (wa/(wa + wb))*(hi - lo);
}
static void famLaplacian(const double* f, double* out, const Fam& fam,
int hasBc, int sFluxKind, int i0, int i1){
const double* rn = fam.rn; const double* rb = fam.rb;
const double* sn = fam.sn; const double* sb = fam.sb;
const int nI = fam.nI, nJ = fam.nJ;
const double thOff = fam.thOff;
const int aHi = (hasBc && rn[nI-1] < R) ? nI : nI - 1;
const int aLo = rn[0] > 0 ? -nI : 0;
const double sLoB = sb[0], sHiB = sb[nJ];
const int jLo = (hasBc && sn[0] > sLoB) ? -1 : 0;
const int jHi = (hasBc && !sFluxKind && sn[nJ-1] < sHiB) ? nJ : nJ - 1;
if (jHi - jLo < 3){ lastError = 5; return; }
double sx[4], sy[4], rx4[4], ry4[4], tx4[4], ty4[4];
double hA[3], hB[3], midS[2];
const int sLo = fam.sLo, sHi = fam.sHi, nb = sHi - sLo + 1;
auto rSkip = [&](int A){ return rb[A] == 0.0 || (!hasBc && A > nI - 1); };
auto sTop = [&](int B){ return B > nJ - 1 && sFluxKind != 0; };
auto sSkip = [&](int B){ return !sTop(B) && (B > nJ - 1 || B < 1) && !hasBc; };
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
const int aFirst = i0 > fam.rLo ? i0 : fam.rLo, aLast = i1 < fam.rHi ? i1 : fam.rHi;
if (aFirst > aLast) return;
rFaces(aFirst, lo);
for (int a = aFirst; a <= aLast; a++){
const double dra = rb[a+1] - rb[a];
rFaces(a + 1, hi);
const bool loR = !rSkip(a), hiR = !rSkip(a + 1);
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
for (int B = sLo; B <= sHi + 1; B++){
if (sTop(B)){
fS[B - sLo] = proj*surfaceFluxFace(sFluxKind, a, k);
continue;
}
if (sSkip(B)) continue;
const double sface = sb[B];
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
void cell3d_viscousRows(int hasBc, int i0, int i1){
refreshSurfaceFluxRows(i0 - 1, i1);
famLaplacian(u, lapU, FAMU, hasBc, 1, i0, i1);
famLaplacian(v, lapV, FAMV, hasBc, 2, i0, i1);
famLaplacian(w, lapW, FAMW, hasBc, 3, i0, i1);
if (lastError) return;
const int uLo = i0 > FAMU.rLo ? i0 : FAMU.rLo, uHi = i1 < FAMU.rHi ? i1 : FAMU.rHi;
const int vLo = i0 > FAMV.rLo ? i0 : FAMV.rLo, vHi = i1 < FAMV.rHi ? i1 : FAMV.rHi;
for (int i = uLo; i <= uHi; i++){
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
for (int i = vLo; i <= vHi; i++){
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
void cell3d_viscous(int hasBc){ cell3d_viscousRows(hasBc, 0, nr - 1); }
void cell3d_advectTransportRows(int i0, int i1){
const int half = nth >> 1;
const int lo0 = i0 > 0 ? i0 : 0, lo1 = i0 > 1 ? i0 : 1;
const int hiU = i1 >= nr - 1 ? nr : i1, hiV = i1 < nr - 1 ? i1 : nr - 1;
for (int i = lo0*nth*nz; i < (hiU + 1)*nth*nz; i++) advU[i] = 0.0;
for (int i = lo0*nth*nz; i < (hiV + 1)*nth*nz; i++) advV[i] = 0.0;
for (int i = lo0*nth*nw; i < (hiV + 1)*nth*nw; i++) advW[i] = 0.0;
for (int i = lo1; i <= hiV; i++){
for (int k = 0; k < nth; k++){
const double Hf = Hr[i*nth + kw(k)];
for (int b = 0; b < nz; b++){
const int c = iu(i, k, b);
const double V = rf[i]*drf[i]*dth*Hf*dsc[b];
const double QrIn = 0.5*(fluxR(i-1, k, b) + fluxR(i, k, b));
const double QrOut = 0.5*(fluxR(i, k, b) + fluxR(i+1, k, b));
const double QtLo = 0.5*(fluxTh(i-1, k, b) + fluxTh(i, k, b));
const double QtHi = 0.5*(fluxTh(i-1, k+1, b) + fluxTh(i, k+1, b));
const double QsLo = 0.5*(fluxS(i-1, k, b) + fluxS(i, k, b));
const double QsHi = 0.5*(fluxS(i-1, k, b+1) + fluxS(i, k, b+1));
const double QmLo = 0.5*(fluxSmesh(i-1, k, b) + fluxSmesh(i, k, b));
const double QmHi = 0.5*(fluxSmesh(i-1, k, b+1) + fluxSmesh(i, k, b+1));
const double uDn = b == 0 ? 0.0 : u[iu(i, k, b-1)];
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
for (int i = lo0; i <= hiV; i++){
for (int k = 0; k < nth; k++){
const double Hf = Hth[i*nth + kw(k)];
for (int b = 0; b < nz; b++){
const int c = iv(i, k, b);
const double V = rc[i]*drc[i]*dth*Hf*dsc[b];
const double QrIn = 0.5*(fluxR(i, k-1, b) + fluxR(i, k, b));
const double QrOut = 0.5*(fluxR(i+1, k-1, b) + fluxR(i+1, k, b));
const double QtLo = 0.5*(fluxTh(i, k-1, b) + fluxTh(i, k, b));
const double QtHi = 0.5*(fluxTh(i, k, b) + fluxTh(i, k+1, b));
const double QsLo = 0.5*(fluxS(i, k-1, b) + fluxS(i, k, b));
const double QsHi = 0.5*(fluxS(i, k-1, b+1) + fluxS(i, k, b+1));
const double QmLo = 0.5*(fluxSmesh(i, k-1, b) + fluxSmesh(i, k, b));
const double QmHi = 0.5*(fluxSmesh(i, k-1, b+1) + fluxSmesh(i, k, b+1));
const double vIn = i == 0 ? -v[iv(0, k + half, b)] : v[iv(i-1, k, b)];
const double vOut = i == nr - 1 ? 0.0 : v[iv(i+1, k, b)];
const double vDn = b == 0 ? 0.0 : v[iv(i, k, b-1)];
const double vUp = b == nz - 1 ? v[c] : v[iv(i, k, b+1)];
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
for (int i = lo0; i <= hiV; i++){
for (int k = 0; k < nth; k++){
const double Hc = H[ie(i, k)];
for (int b = 1; b <= nz; b++){
const int c = iw(i, k, b);
const double V = rc[i]*drc[i]*dth*Hc*dsf[b];
const int top = b == nz;
const double QrIn = 0.5*(fluxR(i, k, b-1) + (top ? 0.0 : fluxR(i, k, b)));
const double QrOut = 0.5*(fluxR(i+1, k, b-1) + (top ? 0.0 : fluxR(i+1, k, b)));
const double QtLo = 0.5*(fluxTh(i, k, b-1) + (top ? 0.0 : fluxTh(i, k, b)));
const double QtHi = 0.5*(fluxTh(i, k+1, b-1) + (top ? 0.0 : fluxTh(i, k+1, b)));
const double QsLo = 0.5*(fluxS(i, k, b-1) + fluxS(i, k, b));
const double QsHi = top ? fluxS(i, k, nz) : 0.5*(fluxS(i, k, b) + fluxS(i, k, b+1));
const double QmLo = 0.5*(fluxSmesh(i, k, b-1) + fluxSmesh(i, k, b));
const double QmHi = top ? fluxSmesh(i, k, nz)
: 0.5*(fluxSmesh(i, k, b) + fluxSmesh(i, k, b+1));
const double wIn = i == 0 ? w[iw(0, k + half, b)] : w[iw(i-1, k, b)];
const double wOut = i == nr - 1 ? 0.0 : w[iw(i+1, k, b)];
const double wUp = top ? w[c] : w[iw(i, k, b+1)];
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
void cell3d_advectTransport(){ cell3d_advectTransportRows(0, nr - 1); }
void cell3d_advectCurvatureRows(int i0, int i1){
const int lo0 = i0 > 0 ? i0 : 0, lo1 = i0 > 1 ? i0 : 1, hiV = i1 < nr - 1 ? i1 : nr - 1;
for (int i = lo1; i <= hiV; i++)
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
for (int i = lo0; i <= hiV; i++)
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
void cell3d_advectCurvature(){ cell3d_advectCurvatureRows(0, nr - 1); }
void cell3d_advect(){ cell3d_advectTransport(); cell3d_advectCurvature(); }
void cell3d_advectRows(int i0, int i1){ cell3d_advectTransportRows(i0, i1); cell3d_advectCurvatureRows(i0, i1); }
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
static double surfaceMetric(int i, int k){
double sl[6];
etaSlopes(i, k, sl);
const double r = rc[i];
return 1 + sl[4]*sl[0]*sl[0] + sl[5]*sl[1]*sl[1]
+ 0.5*(sl[2]*sl[2] + sl[3]*sl[3])/(r*r);
}
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
double cell3d_surfaceExcessArea(){
double dA = 0.0;
for (int i = 0; i < nr; i++)
for (int k = 0; k < nth; k++){
const double q = surfaceSlopeSquared(i, k);
dA += rc[i]*drc[i]*dth*(q/(1 + __builtin_sqrt(1 + q)));
}
return dA;
}
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
static double gEff = CELL3D_GRAVITY_UNSET;
void cell3d_setGravity(double g){ gEff = g; }
double cell3d_gravity(){ return gEff; }
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
int cell3d_stepBegin(double dt){
if (!(dt > 0)){ lastError = 7; return lastError; }
cell3d_omegaFromW();
for (int i = 0; i < nr; i++)
for (int k = 0; k < nth; k++) Ht[ie(i, k)] = om[iw(i, k, nz)];
return cell3d_surfacePressure(psurf);
}
int cell3d_stepExplicit(int i0, int i1){
cell3d_viscousRows(1, i0, i1);
if (lastError) return lastError;
cell3d_advectRows(i0, i1);
return lastError;
}
int cell3d_stepFinish(double dt){
if (!(dt > 0)){ lastError = 7; return lastError; }
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
for (int k = 0; k < nth; k++)
for (int j = 0; j < nz; j++) u[iu(nr, k, j)] = 0.0;
for (int i = 0; i < nr; i++)
for (int k = 0; k < nth; k++) w[iw(i, k, 0)] = 0.0;
cell3d_axisU();
cell3d_omegaFromW();
cell3d_divergence(u, v, om, divw);
const double scale = rho/dt;
for (int c = 0; c < NP; c++) divw[c] *= scale;
for (int c = 0; c < NP; c++) p[c] = 0.0;
gradientWithSurface(p, gu, gv, gw, psurf);
cell3d_omegaOf(gu, gv, gw, gom);
cell3d_divergence(gu, gv, gom, cgq);
for (int c = 0; c < NP; c++) divw[c] -= cgq[c];
cell3d_solveP(divw, 1e-11, 400*(nr + nth + nz));
if (lastError) return lastError;
gradientWithSurface(p, gu, gv, gw, psurf);
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
for (int i = 0; i < nr; i++)
for (int k = 0; k < nth; k++){
const int e = ie(i, k);
eta[e] += dt*om[iw(i, k, nz)];
Ht[e] = om[iw(i, k, nz)];
}
cell3d_refreshMetric();
gEff = CELL3D_GRAVITY_UNSET;
return lastError;
}
int cell3d_step(double dt){
if (cell3d_stepBegin(dt)) return lastError;
if (cell3d_stepExplicit(0, nr - 1)) return lastError;
return cell3d_stepFinish(dt);
}
}
