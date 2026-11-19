extern "C" {
void* memset(void* dst, int c, unsigned long n){
unsigned char* q = (unsigned char*)dst;
for (unsigned long i = 0; i < n; i++) q[i] = (unsigned char)c;
return dst;
}
static const int ARENA = 3000000;
static double arena[ARENA];
static int arenaUsed = 0;
static int nr, nz, mMode, steps;
static double rho, nu, gamma_, gBase;
static double *rf, *zf, *rc, *zc, *drc, *dzc, *drf, *dzf;
static double *u, *v, *w, *p, *eta;
static double *us, *vs, *ws, *lu, *lv, *lw, *dv, *gu, *gv, *gw, *ps;
static double *cgr, *cgd, *cgq, *cgz, *pdiag, *dudz, *dvdz, *drive;
static double *vecIn, *vecOut;
static int cgIters; static double cgResidual;
static int lastError;
static double* take(int n){
if (arenaUsed + n > ARENA){ lastError = 1; return arena; }
double* q = arena + arenaUsed; arenaUsed += n;
for (int i = 0; i < n; i++) q[i] = 0.0;
return q;
}
static inline int iu(int i, int j){ return i*nz + j; }
static inline int iw(int i, int j){ return i*(nz + 1) + j; }
static inline int ip(int i, int j){ return i*nz + j; }
int setup(int NR, int NZ, int STEPS){
nr = NR; nz = NZ; steps = STEPS;
arenaUsed = 0; lastError = 0;
const int nu_ = (nr + 1)*nz, nw_ = nr*(nz + 1), np_ = nr*nz;
rf = take(nr + 1); zf = take(nz + 1); rc = take(nr); zc = take(nz);
drc = take(nr); dzc = take(nz); drf = take(nr + 1); dzf = take(nz + 1);
u = take(nu_); v = take(nu_); w = take(nw_); p = take(np_); eta = take(nr);
us = take(nu_); vs = take(nu_); ws = take(nw_);
lu = take(nu_); lv = take(nu_); lw = take(nw_);
dv = take(np_); gu = take(nu_); gv = take(nu_); gw = take(nw_);
ps = take(nr); cgr = take(np_); cgd = take(np_); cgq = take(np_);
cgz = take(np_); pdiag = take(np_);
dudz = take(nr + 1); dvdz = take(nr + 1);
drive = take(steps);
vecIn = take(2*(nr - 1)*nz + nw_ + nr + nz);
vecOut = take(2*(nr - 1)*nz + nw_ + nr + nz);
return lastError;
}
void setScalars(int m, double RHO, double NU, double GAMMA, double G){
mMode = m; rho = RHO; nu = NU; gamma_ = GAMMA; gBase = G;
}
double* ptrRf(){ return rf; } double* ptrZf(){ return zf; }
double* ptrRc(){ return rc; } double* ptrZc(){ return zc; }
double* ptrDrc(){ return drc; } double* ptrDzc(){ return dzc; }
double* ptrDrf(){ return drf; } double* ptrDzf(){ return dzf; }
double* ptrDrive(){ return drive; }
double* ptrVecIn(){ return vecIn; }
double* ptrVecOut(){ return vecOut; }
double* ptrEta(){ return eta; }
int getCgIters(){ return cgIters; }
double getCgResidual(){ return cgResidual; }
int getLastError(){ return lastError; }
static double wzSurface(int i){
const double w0 = w[iw(i, nz)], w1 = w[iw(i, nz-1)], w2 = w[iw(i, nz-2)];
const double a = zf[nz] - zf[nz-1], b = zf[nz] - zf[nz-2];
return (w0*(a + b)/(a*b)) - (w1*b/(a*(b - a))) + (w2*a/(b*(b - a)));
}
static double surfaceLaplacian(int i, int contactPinned){
const double r = rc[i];
double outer;
if (i == nr - 1)
outer = contactPinned ? rf[nr]*(0.0 - eta[nr-1])/drf[nr] : 0.0;
else
outer = rf[i+1]*(eta[i+1] - eta[i])/drf[i+1];
const double inner = (i == 0) ? 0.0 : rf[i]*(eta[i] - eta[i-1])/drf[i];
return (outer - inner)/(r*drc[i]) - (double)mMode*mMode*eta[i]/(r*r);
}
static void surfacePressure(double gNow, int contactPinned){
for (int i = 0; i < nr; i++)
ps[i] = rho*gNow*eta[i] - gamma_*surfaceLaplacian(i, contactPinned)
+ 2.0*rho*nu*wzSurface(i);
}
static void surfaceSlopes(){
for (int i = 0; i <= nr; i++){
if (i == 0 && mMode == 1){ dudz[0] = -w[iw(0, nz)]/drf[0]; dvdz[0] = -dudz[0]; continue; }
if (i == 0 || i == nr){ dudz[i] = 0.0; dvdz[i] = 0.0; continue; }
const double wl = w[iw(i-1, nz)], wrr = w[iw(i, nz)];
dudz[i] = -(wrr - wl)/drf[i];
const double wAtFace = (drc[i-1]*wrr + drc[i]*wl)/(drc[i-1] + drc[i]);
dvdz[i] = ((double)mMode/rf[i])*wAtFace;
}
}
static void lapUV(const double* f, const double* other, const double* dfdzTop,
double* out){
const int m = mMode;
for (int i = 1; i < nr; i++){
const double r = rf[i];
const double wIn = rc[i-1]/drc[i-1], wOut = rc[i]/drc[i];
const double vol = r*0.5*(drc[i-1] + drc[i]);
for (int j = 0; j < nz; j++){
const int c = iu(i, j);
const double fc = f[c];
const double fIn = f[iu(i-1, j)], fOut = f[iu(i+1, j)];
const double radial = (wOut*(fOut - fc) - wIn*(fc - fIn))/vol;
const double below = (j == 0) ? (fc - 0.0)/dzf[0]
: (fc - f[iu(i, j-1)])/dzf[j];
const double above = (j == nz - 1) ? dfdzTop[i]
: (f[iu(i, j+1)] - fc)/dzf[j+1];
const double vertical = (above - below)/dzc[j];
const double coupling = -(((double)m*m + 1.0)*fc + 2.0*m*other[c])/(r*r);
out[c] = radial + vertical + coupling;
}
}
}
static void lapAxis(const double* dfdzTop, double* outU, double* outV){
for (int j = 0; j < nz; j++){
const int c = iu(0, j), b = iu(1, j);
const double a = u[c];
const double radial = (u[b] - v[b] - 2.0*a)/(drc[0]*rc[0]);
const double below = (j == 0) ? (a - 0.0)/dzf[0] : (a - u[iu(0, j-1)])/dzf[j];
const double above = (j == nz - 1) ? dfdzTop[0] : (u[iu(0, j+1)] - a)/dzf[j+1];
outU[c] = radial + (above - below)/dzc[j];
outV[c] = -outU[c];
}
}
static void lapW(double* out){
const int m = mMode;
for (int i = 0; i < nr; i++){
const double r = rc[i];
const double inArea = rf[i], outArea = rf[i+1];
const double vol = r*drc[i];
for (int j = 1; j <= nz; j++){
const int c = iw(i, j);
const double wc = w[c];
const double fluxIn = (i == 0) ? 0.0
: inArea*(wc - w[iw(i-1, j)])/drf[i];
const double fluxOut = (i == nr - 1)
? outArea*(0.0 - wc)/drf[nr]
: outArea*(w[iw(i+1, j)] - wc)/drf[i+1];
const double radial = (fluxOut - fluxIn)/vol;
const double below = (j == 1) ? (wc - 0.0)/dzc[0]
: (wc - w[iw(i, j-1)])/dzc[j-1];
const double above = (j == nz) ? below + 2.0*(wzSurface(i) - below)
: (w[iw(i, j+1)] - wc)/dzc[j];
const double vertical = (above - below)/dzf[j];
out[c] = radial + vertical - (double)m*m*wc/(r*r);
}
}
}
static void divergence(const double* U, const double* V, const double* W,
double* out){
const int m = mMode;
for (int i = 0; i < nr; i++)
for (int j = 0; j < nz; j++){
const double vbar = 0.5*(V[iu(i, j)] + V[iu(i+1, j)]);
out[ip(i, j)] =
dzc[j]*(rf[i+1]*U[iu(i+1, j)] - rf[i]*U[iu(i, j)])
+ rc[i]*drc[i]*(W[iw(i, j+1)] - W[iw(i, j)])
+ (double)m*drc[i]*dzc[j]*vbar;
}
}
static void gradient(const double* q){
const int m = mMode;
const int nu_ = (nr + 1)*nz, nw_ = nr*(nz + 1);
for (int i = 0; i < nu_; i++){ gu[i] = 0.0; gv[i] = 0.0; }
for (int i = 0; i < nw_; i++) gw[i] = 0.0;
for (int i = 0; i < nr; i++)
for (int j = 0; j < nz; j++){
const double qc = q[ip(i, j)];
gu[iu(i+1, j)] += qc*dzc[j];
gu[iu(i, j)] -= qc*dzc[j];
gw[iw(i, j+1)] += qc*rc[i]*drc[i];
gw[iw(i, j )] -= qc*rc[i]*drc[i];
gv[iu(i, j)] += qc*(double)m*drc[i]*dzc[j]*0.5;
gv[iu(i+1, j)] += qc*(double)m*drc[i]*dzc[j]*0.5;
}
for (int i = 1; i < nr; i++){
const double span = drf[i];
const double halfWidth = 0.5*(drc[i-1] + drc[i]);
for (int j = 0; j < nz; j++){
gu[iu(i, j)] /= -(dzc[j]*span);
gv[iu(i, j)] /= -(rf[i]*dzc[j]*halfWidth);
}
}
for (int j = 0; j < nz; j++){
if (m == 1){
const double ga = gv[iu(0, j)]/(rc[0]*rc[0]*dzc[j]);
gu[iu(0, j)] = ga; gv[iu(0, j)] = -ga;
} else { gu[iu(0, j)] = 0.0; gv[iu(0, j)] = 0.0; }
gu[iu(nr, j)] = 0.0; gv[iu(nr, j)] = 0.0;
}
for (int i = 0; i < nr; i++){
gw[iw(i, 0)] = 0.0;
for (int j = 1; j <= nz; j++) gw[iw(i, j)] /= -(rc[i]*drc[i]*dzf[j]);
}
}
static void applyL(const double* q, double* out){
gradient(q);
divergence(gu, gv, gw, out);
}
static int pressureDiagonal(){
const int n = nr*nz;
for (int parity = 0; parity < 2; parity++){
for (int i = 0; i < n; i++) cgz[i] = 0.0;
for (int i = 0; i < nr; i++)
for (int j = 0; j < nz; j++)
if (((i + j) & 1) == parity) cgz[ip(i, j)] = 1.0;
applyL(cgz, cgq);
for (int i = 0; i < nr; i++)
for (int j = 0; j < nz; j++)
if (((i + j) & 1) == parity) pdiag[ip(i, j)] = cgq[ip(i, j)];
}
for (int i = 0; i < n; i++) if (!(pdiag[i] < 0.0)) return 3;
return 0;
}
static void solveP(const double* rhs, double tol, int maxIt){
const int n = nr*nz;
applyL(p, cgq);
double rr = 0.0;
for (int i = 0; i < n; i++){
cgr[i] = rhs[i] - cgq[i]; rr += cgr[i]*cgr[i];
}
const double rr0 = rr;
if (rr0 == 0.0){ cgIters = 0; cgResidual = 0.0; return; }
double rz = 0.0;
for (int i = 0; i < n; i++){
cgz[i] = cgr[i]/pdiag[i]; cgd[i] = cgz[i]; rz += cgr[i]*cgz[i];
}
int it = 0;
for (; it < maxIt; it++){
applyL(cgd, cgq);
double dq = 0.0;
for (int i = 0; i < n; i++) dq += cgd[i]*cgq[i];
if (dq == 0.0) break;
const double alpha = rz/dq;
double rr2 = 0.0;
for (int i = 0; i < n; i++){
p[i] += alpha*cgd[i]; cgr[i] -= alpha*cgq[i]; rr2 += cgr[i]*cgr[i];
}
if (__builtin_sqrt(rr2/rr0) < tol){ rr = rr2; it++; break; }
double rz2 = 0.0;
for (int i = 0; i < n; i++){ cgz[i] = cgr[i]/pdiag[i]; rz2 += cgr[i]*cgz[i]; }
const double beta = rz2/rz; rz = rz2; rr = rr2;
for (int i = 0; i < n; i++) cgd[i] = cgz[i] + beta*cgd[i];
}
cgIters = it; cgResidual = __builtin_sqrt(rr/rr0);
if (!(cgResidual < tol)) lastError = 2;
}
int prepare(){ return pressureDiagonal(); }
static void stepOnce(double dt, double gNow, int contactPinned){
const int nu_ = (nr + 1)*nz, nw_ = nr*(nz + 1);
surfaceSlopes();
lapUV(u, v, dudz, lu);
lapUV(v, u, dvdz, lv);
lapW(lw);
const int i0 = (mMode == 1) ? 0 : 1;
if (i0 == 0) lapAxis(dudz, lu, lv);
for (int i = 0; i < nu_; i++){ us[i] = u[i]; vs[i] = v[i]; }
for (int i = 0; i < nw_; i++) ws[i] = w[i];
for (int i = i0; i < nr; i++)
for (int j = 0; j < nz; j++){
const int c = iu(i, j);
us[c] = u[c] + dt*nu*lu[c];
vs[c] = v[c] + dt*nu*lv[c];
}
for (int i = 0; i < nr; i++)
for (int j = 1; j <= nz; j++){
const int c = iw(i, j);
ws[c] = w[c] + dt*nu*lw[c];
}
surfacePressure(gNow, contactPinned);
divergence(us, vs, ws, dv);
const int np_ = nr*nz;
for (int c = 0; c < np_; c++) dv[c] *= rho/dt;
for (int i = 0; i < nr; i++)
dv[ip(i, nz-1)] -= rc[i]*drc[i]*ps[i]/dzf[nz];
solveP(dv, 1e-11, 40*(nr + nz));
gradient(p);
for (int i = 0; i < nr; i++) gw[iw(i, nz)] += ps[i]/dzf[nz];
for (int i = i0; i < nr; i++)
for (int j = 0; j < nz; j++){
const int c = iu(i, j);
u[c] = us[c] - (dt/rho)*gu[c];
v[c] = vs[c] - (dt/rho)*gv[c];
}
for (int i = 0; i < nr; i++)
for (int j = 1; j <= nz; j++){
const int c = iw(i, j);
w[c] = ws[c] - (dt/rho)*gw[c];
}
for (int i = 0; i < nr; i++) w[iw(i, 0)] = 0.0;
for (int i = 0; i < nr; i++) eta[i] += dt*w[iw(i, nz)];
}
int applyPeriodMap(double dt, double href, double uref, int contactPinned){
lastError = 0;
const int nu_ = (nr + 1)*nz, nw_ = nr*(nz + 1), np_ = nr*nz;
for (int i = 0; i < nu_; i++){ u[i] = 0.0; v[i] = 0.0; }
for (int i = 0; i < nw_; i++) w[i] = 0.0;
for (int i = 0; i < np_; i++) p[i] = 0.0;
int q = 0;
for (int i = 1; i < nr; i++) for (int j = 0; j < nz; j++) u[iu(i, j)] = vecIn[q++]*uref;
for (int i = 1; i < nr; i++) for (int j = 0; j < nz; j++) v[iu(i, j)] = vecIn[q++]*uref;
for (int i = 0; i < nr; i++) for (int j = 1; j <= nz; j++) w[iw(i, j)] = vecIn[q++]*uref;
for (int i = 0; i < nr; i++) eta[i] = vecIn[q++]*href;
if (mMode == 1)
for (int j = 0; j < nz; j++){ u[iu(0, j)] = vecIn[q++]*uref; v[iu(0, j)] = -u[iu(0, j)]; }
for (int n = 0; n < steps; n++){
stepOnce(dt, drive[n], contactPinned);
if (lastError) return lastError;
}
q = 0;
for (int i = 1; i < nr; i++) for (int j = 0; j < nz; j++) vecOut[q++] = u[iu(i, j)]/uref;
for (int i = 1; i < nr; i++) for (int j = 0; j < nz; j++) vecOut[q++] = v[iu(i, j)]/uref;
for (int i = 0; i < nr; i++) for (int j = 1; j <= nz; j++) vecOut[q++] = w[iw(i, j)]/uref;
for (int i = 0; i < nr; i++) vecOut[q++] = eta[i]/href;
if (mMode == 1) for (int j = 0; j < nz; j++) vecOut[q++] = u[iu(0, j)]/uref;
return 0;
}
int stepFromVec(double dt, double gNow, double href, double uref, int contactPinned){
lastError = 0;
const int nu_ = (nr + 1)*nz, nw_ = nr*(nz + 1), np_ = nr*nz;
for (int i = 0; i < nu_; i++){ u[i] = 0.0; v[i] = 0.0; }
for (int i = 0; i < nw_; i++) w[i] = 0.0;
for (int i = 0; i < np_; i++) p[i] = 0.0;
int q = 0;
for (int i = 1; i < nr; i++) for (int j = 0; j < nz; j++) u[iu(i, j)] = vecIn[q++]*uref;
for (int i = 1; i < nr; i++) for (int j = 0; j < nz; j++) v[iu(i, j)] = vecIn[q++]*uref;
for (int i = 0; i < nr; i++) for (int j = 1; j <= nz; j++) w[iw(i, j)] = vecIn[q++]*uref;
for (int i = 0; i < nr; i++) eta[i] = vecIn[q++]*href;
if (mMode == 1)
for (int j = 0; j < nz; j++){ u[iu(0, j)] = vecIn[q++]*uref; v[iu(0, j)] = -u[iu(0, j)]; }
stepOnce(dt, gNow, contactPinned);
q = 0;
for (int i = 1; i < nr; i++) for (int j = 0; j < nz; j++) vecOut[q++] = u[iu(i, j)]/uref;
for (int i = 1; i < nr; i++) for (int j = 0; j < nz; j++) vecOut[q++] = v[iu(i, j)]/uref;
for (int i = 0; i < nr; i++) for (int j = 1; j <= nz; j++) vecOut[q++] = w[iw(i, j)]/uref;
for (int i = 0; i < nr; i++) vecOut[q++] = eta[i]/href;
if (mMode == 1) for (int j = 0; j < nz; j++) vecOut[q++] = u[iu(0, j)]/uref;
return lastError;
}
}
