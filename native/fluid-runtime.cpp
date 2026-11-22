#include "fluid-runtime.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

// Include the canonical implementation once. Native-only preprocessor hooks
// split its existing step without copying or changing the fluid equations.
#define CELL3D_NATIVE 1
#include "../dns/faraday_cell3d.cpp"

namespace faraday_native {
namespace {
NativeFluid* liveFluid=nullptr;
std::uint64_t nextGeneration=1;

void positiveFinite(double value,const char* name){
  if (!(std::isfinite(value) && value>0))
    throw std::invalid_argument(std::string(name)+" must be finite and positive");
}

void checkSpan(std::span<double> values,std::size_t count,const char* name){
  if (values.size()!=count)
    throw std::invalid_argument(std::string(name)+" has the wrong staggered field length");
}

void releaseStorage(){
  ::nativeDoubleBlocks.clear(); ::nativeIntBlocks.clear();
  ::arenaUsed=0; ::iarenaUsed=0; ::nativeBytes=0;
}

double vectorNorm(std::span<const double> a){
  double scale=0,sum=1;
  for(double x:a){
    if(!std::isfinite(x))throw std::runtime_error("implicit surface vector contains a non-finite value");
    if(x!=0){const double value=std::abs(x);
    if(scale<value){const double r=scale/value;sum=1+sum*r*r;scale=value;}
    else {const double r=value/scale;sum+=r*r;}}}
  const double result=scale*std::sqrt(sum);
  if(!std::isfinite(result))throw std::runtime_error("implicit surface vector norm overflowed");
  return result;
}
double vectorMaximum(std::span<const double> a){
  double result=0;for(double value:a){
    if(!std::isfinite(value))throw std::runtime_error("implicit surface vector contains a non-finite value");
    result=std::max(result,std::abs(value));}return result;
}
double vectorDot(std::span<const double> a,std::span<const double> b){
  double result=0;for(std::size_t i=0;i<a.size();i++)result+=a[i]*b[i];
  if(!std::isfinite(result))throw std::runtime_error("implicit surface dot product is non-finite");
  return result;
}

// Exact directional derivative of the canonical discrete-area curvature.
// Differentiating its existing scatter retains all nonlinear slope terms and
// the pinned-rim contribution; this is not a small-slope surface replacement.
void curvatureDirection(std::span<const double> at,std::span<const double> direction,
                        std::span<double> out){
  std::fill(out.begin(),out.end(),0);
  struct RestoreEta {double* saved;~RestoreEta(){::eta=saved;}} restore{::eta};
  for(int i=0;i<::nr;i++)for(int k=0;k<::nth;k++){
    double sl[6],ds[6];
    ::eta=const_cast<double*>(at.data());::etaSlopes(i,k,sl);
    ::eta=const_cast<double*>(direction.data());::etaSlopes(i,k,ds);
    const int e=::ie(i,k);const double rr=::rc[i]*::rc[i];
    const double X=1+sl[4]*sl[0]*sl[0]+sl[5]*sl[1]*sl[1]
      +.5*(sl[2]*sl[2]+sl[3]*sl[3])/rr;
    const double dX=2*sl[4]*sl[0]*ds[0]+2*sl[5]*sl[1]*ds[1]
      +(sl[2]*ds[2]+sl[3]*ds[3])/rr;
    const double wt=::rc[i]*::drc[i]*::dth/(2*std::sqrt(X));
    const double dwt=-.5*wt*dX/X;
    const double ci=2*sl[4]*(dwt*sl[0]+wt*ds[0]);
    const double co=2*sl[5]*(dwt*sl[1]+wt*ds[1]);
    if(i>0){out[e]+=ci/::drf[i];out[::ie(i-1,k)]-=ci/::drf[i];}
    if(i<::nr-1){out[e]-=co/::drf[i+1];out[::ie(i+1,k)]+=co/::drf[i+1];}
    else if(::pinned)out[e]-=co/::drf[::nr];
    const double cl=(dwt*sl[2]+wt*ds[2])/rr,ch=(dwt*sl[3]+wt*ds[3])/rr;
    out[e]+=(cl-ch)/::dth;out[::ie(i,k-1)]-=cl/::dth;out[::ie(i,k+1)]+=ch/::dth;
  }
  for(int i=0;i<::nr;i++)for(int k=0;k<::nth;k++)
    out[::ie(i,k)]/=-::rc[i]*::drc[i]*::dth;
}

// Restarted, reorthogonalized GMRES. The surface response need not be symmetric
// in an unweighted Euclidean norm. Every successful return checks A*x-b itself;
// a small recursively updated Arnoldi residual alone is insufficient.
std::vector<double> surfaceGmres(const std::function<std::vector<double>(std::span<const double>)>& apply,
    std::span<const double> rhs,double tolerance,std::size_t restart,std::size_t maximum,
    std::size_t& iterations){
  const std::size_t n=rhs.size();std::vector<double> x(n),r(rhs.begin(),rhs.end());
  iterations=0;
  while(vectorNorm(r)>tolerance){
    if(iterations>=maximum)throw std::runtime_error("Implicit surface linear solve did not converge");
    const std::size_t m=std::min(restart,maximum-iterations);
    std::vector<std::vector<double>> v(m+1,std::vector<double>(n));
    std::vector<double> h((m+1)*m),cs(m),sn(m),g(m+1),y(m);
    const double beta=vectorNorm(r);g[0]=beta;
    for(std::size_t j=0;j<n;j++)v[0][j]=r[j]/beta;
    std::size_t used=0;
    for(std::size_t j=0;j<m;j++){
      auto w=apply(v[j]);
      // A second MGS pass keeps the Krylov basis usable at the strict surface
      // residual, without adding a diagonal shift to the physical operator.
      for(int pass=0;pass<2;pass++)for(std::size_t i=0;i<=j;i++){
        const double q=vectorDot(w,v[i]);h[i*m+j]+=q;
        for(std::size_t k=0;k<n;k++)w[k]-=q*v[i][k];
      }
      const double next=vectorNorm(w);h[(j+1)*m+j]=next;
      if(next>0)for(std::size_t k=0;k<n;k++)v[j+1][k]=w[k]/next;
      for(std::size_t i=0;i<j;i++){
        const double a=h[i*m+j],b=h[(i+1)*m+j];
        h[i*m+j]=cs[i]*a+sn[i]*b;h[(i+1)*m+j]=-sn[i]*a+cs[i]*b;
      }
      const double a=h[j*m+j],b=h[(j+1)*m+j],length=std::hypot(a,b);
      if(!(length>0&&std::isfinite(length)))throw std::runtime_error("Implicit surface Krylov operator broke down");
      cs[j]=a/length;sn[j]=b/length;h[j*m+j]=length;h[(j+1)*m+j]=0;
      g[j+1]=-sn[j]*g[j];g[j]*=cs[j];iterations++;used=j+1;
      if(std::abs(g[j+1])<=tolerance||next==0)break;
    }
    for(std::size_t ii=used;ii-->0;){
      double q=g[ii];for(std::size_t j=ii+1;j<used;j++)q-=h[ii*m+j]*y[j];
      y[ii]=q/h[ii*m+ii];
    }
    for(std::size_t j=0;j<used;j++)for(std::size_t k=0;k<n;k++)x[k]+=y[j]*v[j][k];
    const auto ax=apply(x);for(std::size_t k=0;k<n;k++)r[k]=rhs[k]-ax[k];
  }
  return x;
}
}

NativeFluid::NativeFluid(const FluidConfig& requested):config_(requested){
  if (liveFluid) throw std::logic_error("the native fluid core already has a live instance");
  ::nativeCheckDimensions(config_.nr,config_.nth,config_.nz);
  positiveFinite(config_.R,"R"); positiveFinite(config_.h,"h");
  positiveFinite(config_.rho,"rho"); positiveFinite(config_.nu,"nu");
  positiveFinite(config_.gamma,"gamma"); positiveFinite(config_.dtheta,"dtheta");
  if (!(std::isfinite(config_.g) && config_.g>=0))
    throw std::invalid_argument("g must be finite and nonnegative");
  if (!config_.pinned)
    throw std::invalid_argument("the native physical cell requires a pinned contact line");
  const double turn=config_.dtheta*config_.nth;
  if (std::abs(turn-2*std::acos(-1.0))>16*std::numeric_limits<double>::epsilon())
    throw std::invalid_argument("dtheta must tile exactly one azimuthal turn");
  const std::size_t np=static_cast<std::size_t>(config_.nr)*config_.nth*config_.nz;
  const std::size_t nw=static_cast<std::size_t>(config_.nr)*config_.nth*(config_.nz+1);
  const std::size_t scratchBytes=(nw+2*np)*sizeof(double);
  if (scratchBytes>=config_.memoryBudgetBytes)
    throw std::length_error("native projection scratch exceeds the requested array-memory budget; grid unchanged");
  ::nativeBudget=config_.memoryBudgetBytes-scratchBytes;
  try {
    ::cell3d_init(config_.nr,config_.nth,config_.nz,1,config_.R,config_.h,
      config_.rho,config_.nu,config_.gamma,config_.g,config_.dtheta);
    checkError("native initialization");
    projectionOmega_.resize(nw);
    projectionRhs_.resize(np);
    projectionPressure_.resize(np);
    generation_=nextGeneration++;
    liveFluid=this;
  } catch (...) { releaseStorage(); throw; }
}

NativeFluid::~NativeFluid(){
  if (liveFluid==this){ releaseStorage(); liveFluid=nullptr; }
}

std::size_t NativeFluid::allocatedBytes() const noexcept {
  return ::nativeBytes+(projectionOmega_.size()+projectionRhs_.size()+projectionPressure_.size())*sizeof(double);
}

std::span<double> NativeFluid::array(int key){
  const int n=::cell3d_len(key); double* data=::cell3d_ptr(key);
  if (n<0 || !data) throw std::out_of_range("unknown native fluid array key "+std::to_string(key));
  return {data,static_cast<std::size_t>(n)};
}

std::span<const double> NativeFluid::array(int key) const {
  const int n=::cell3d_len(key); const double* data=::cell3d_ptr(key);
  if (n<0 || !data) throw std::out_of_range("unknown native fluid array key "+std::to_string(key));
  return {data,static_cast<std::size_t>(n)};
}

void NativeFluid::checkError(const char* operation) const {
  const int code=::cell3d_error();
  if (!code) return;
  std::ostringstream message;
  message << operation << " failed with canonical fluid error " << code;
  if (code==2) message << ": pressure residual " << ::cell3d_cgResidual()
    << " after " << ::cell3d_cgIters() << " iterations";
  if (code==3) message << ": nonpositive liquid depth at radial/azimuthal cell "
    << ::cell3d_errorI() << "/" << ::cell3d_errorK();
  throw std::runtime_error(message.str());
}

void NativeFluid::requireReady() const {
  if (!initialized_) throw std::logic_error("copy the canonical grid/preconditioner and call initialize() first");
  checkError("native fluid state");
}

void NativeFluid::initialize(){
  if (initialized_) throw std::logic_error("native fluid is already initialized");
  for (int key=0;key<=8;key++) for (double x:array(key))
    if (!std::isfinite(x)) throw std::invalid_argument("non-finite canonical grid value");
  if (::rf[0]!=0 || ::rf[::nr]!=config_.R || ::sf[0]!=0 || ::sf[::nz]!=1)
    throw std::invalid_argument("canonical radial/sigma faces do not match the specified vessel");
  for (int i=0;i<::nr;i++)
    if (!(::rf[i+1]>::rf[i] && ::rc[i]>0 && ::drc[i]>0))
      throw std::invalid_argument("canonical radial grid is not strictly increasing");
  for (int j=0;j<::nz;j++)
    if (!(::sf[j+1]>::sf[j] && ::sc[j]>0 && ::dsc[j]>0))
      throw std::invalid_argument("canonical sigma grid is not strictly increasing");
  for (int key:{6,7}) for (double x:array(key))
    if (!(x>0)) throw std::invalid_argument("canonical staggered spacings must be positive");
  for (int key:{9,10,11,12,13,14,20,62,63,64}) for (double x:array(key))
    if (!std::isfinite(x)) throw std::invalid_argument("non-finite initial state or pressure preconditioner");
  if (::twc[0]!=1 || ::tws[0]!=0)
    throw std::invalid_argument("canonical pressure-transform twiddles have not been copied");
  const int stride=::nz+1;
  for (std::size_t i=0;i<array(64).size();i+=static_cast<std::size_t>(stride))
    if (!(::pcband[i]>0)) throw std::invalid_argument("canonical flat-cell Cholesky factor has a nonpositive diagonal");
  ::cell3d_buildFamilies(); checkError("native family initialization");
  ::cell3d_refreshMetric(); checkError("native metric initialization");
  initialized_=true;
}

void NativeFluid::refreshMetric(){
  requireReady();
  ::cell3d_refreshMetric(); checkError("native metric refresh");
}

void NativeFluid::synchronizeVelocity(){
  requireReady();
  ::cell3d_axisU();::cell3d_omegaFromW();checkError("native velocity synchronization");
  for(int i=0;i<::nr;i++)for(int k=0;k<::nth;k++)
    ::Ht[::ie(i,k)]=::om[::iw(i,k,::nz)];
}

void NativeFluid::enableRigidProjection(){
  requireReady();
  if (pending_) throw std::logic_error("enable rigid projection before starting a coupled step");
  if (rigidProjection_) return;
  ::nativeRigidProjection=true; rigidProjection_=true;
  ::cell3d_axisU(); ::cell3d_omegaFromW();
  generation_=nextGeneration++;
}

FluidCheckpoint NativeFluid::checkpoint() const {
  requireReady();
  if (pending_) throw std::logic_error("checkpoint requires an accepted state, not a pending coupled predictor");
  FluidCheckpoint saved;
  saved.ownerGeneration=generation_; saved.time=time_;
  const int keys[7]={9,10,11,12,13,14,20};
  for (std::size_t j=0;j<7;j++){
    const auto values=array(keys[j]);
    saved.state[j].assign(values.begin(),values.end());
  }
  return saved;
}

void NativeFluid::restore(const FluidCheckpoint& saved){
  if (!initialized_) throw std::logic_error("cannot restore an uninitialized native fluid");
  if (saved.ownerGeneration!=generation_ || !(std::isfinite(saved.time) && saved.time>=0))
    throw std::invalid_argument("checkpoint belongs to a different native fluid or has invalid time");
  const int keys[7]={9,10,11,12,13,14,20};
  // Validate every value before mutating any state; a corrupted checkpoint must
  // not destroy the last recoverable fluid state.
  for (std::size_t j=0;j<7;j++){
    if (saved.state[j].size()!=array(keys[j]).size())
      throw std::invalid_argument("checkpoint has an invalid staggered state length");
    for (double value:saved.state[j])
      if (!std::isfinite(value)) throw std::invalid_argument("checkpoint contains a non-finite state");
  }
  for (double elevation:saved.state[5])
    if (!(config_.h+elevation>0)) throw std::invalid_argument("checkpoint surface reaches the floor");
  for (std::size_t j=0;j<7;j++)
    std::copy(saved.state[j].begin(),saved.state[j].end(),array(keys[j]).begin());
  ::cell3d_clearError();
  ::cell3d_refreshMetric(); checkError("native checkpoint restoration");
  ::gEff=CELL3D_GRAVITY_UNSET;
  time_=saved.time; pending_=false; pendingDt_=0;
}

void NativeFluid::step(double dt,double effectiveGravity){
  requireReady(); positiveFinite(dt,"dt");
  if(!(std::isfinite(time_+dt)&&time_+dt>time_))
    throw std::invalid_argument("dt must advance the representable physical clock");
  if (pending_) throw std::logic_error("finish the pending coupled step before starting another");
  if (!std::isfinite(effectiveGravity)) throw std::invalid_argument("effective gravity must be finite");
  ::cell3d_setGravity(effectiveGravity);
  ::cell3d_step(dt); checkError("native full fluid step");
  time_+=dt;
}

std::size_t NativeFluid::implicitSurfaceWorkspaceBytes(const ImplicitSurfaceOptions& options) const {
  requireReady();
  const long double count=8.L*(::NU+static_cast<long double>(::NV)+::NW+::NP)
    +(2.L*static_cast<long double>(options.krylovDimension)+40.L)*::NE
    +2.L*(static_cast<long double>(options.krylovDimension)+5.L)*options.krylovDimension;
  const long double bytes=count*sizeof(double);
  if(!(bytes<=std::numeric_limits<std::size_t>::max()))
    throw std::length_error("implicit surface workspace size overflows size_t");
  return static_cast<std::size_t>(bytes);
}

ImplicitSurfaceReport NativeFluid::stepImplicitSurface(double dt,double effectiveGravity,
    const ImplicitSurfaceOptions& options,const SurfaceResponseCallback& response){
  requireReady();positiveFinite(dt,"dt");
  if(!(std::isfinite(time_+dt)&&time_+dt>time_))
    throw std::invalid_argument("dt must advance the representable physical clock");
  if(pending_)throw std::logic_error("finish the pending coupled step before an implicit surface step");
  if(!std::isfinite(effectiveGravity))throw std::invalid_argument("effective gravity must be finite");
  positiveFinite(options.absoluteSurfaceTolerance,"absolute surface tolerance");
  positiveFinite(options.relativeSurfaceTolerance,"relative surface tolerance");
  positiveFinite(options.absolutePressureTolerance,"absolute surface pressure tolerance");
  positiveFinite(options.relativePressureTolerance,"relative surface pressure tolerance");
  positiveFinite(options.linearRelativeTolerance,"linear surface tolerance");
  if(options.relativeSurfaceTolerance>=1||options.relativePressureTolerance>=1||options.linearRelativeTolerance>=1
      ||!options.maximumNewtonIterations||!options.maximumLinearIterations||!options.krylovDimension)
    throw std::invalid_argument("implicit surface tolerances and iteration limits must be positive and meaningful");
  const auto limits=stepLimits(effectiveGravity-config_.g);
  if(dt>std::min({limits.viscous,limits.advective,limits.gravity}))
    throw std::invalid_argument("implicit capillarity does not remove viscous, advective or gravity step limits");
  // Budget temporary checkpoint, predictor, Krylov vectors and simultaneous
  // residual/Jacobian/line-search work before allocating any of them.
  if(implicitSurfaceWorkspaceBytes(options)>config_.memoryBudgetBytes-allocatedBytes())
    throw std::length_error("implicit surface workspace exceeds the requested memory budget; grid unchanged");
  const bool oldProjection=rigidProjection_;
  const auto beforeMode=checkpoint();
  ImplicitSurfaceReport report;report.dt=dt;
  try {
    enableRigidProjection();
    ::cell3d_setGravity(effectiveGravity);::cell3d_stepBegin(dt);checkError("implicit surface step begin");
    ::cell3d_stepExplicit(0,::nr-1);checkError("implicit surface explicit momentum terms");
    ::cell3d_nativePredictor(dt);checkError("implicit surface momentum predictor");
    std::array<std::vector<double>,3> predictor;
    for(int component=0;component<3;component++){
      const auto values=array(9+component);predictor[component].assign(values.begin(),values.end());
    }
    const auto& oldEta=beforeMode.state[5];std::vector<double> trial=oldEta,curvature(::NE);
    auto pressureAt=[&](std::span<const double> elevation){
      struct RestoreEta {double* saved;~RestoreEta(){::eta=saved;}} restore{::eta};
      ::eta=const_cast<double*>(elevation.data());::cell3d_curvature(curvature.data());
      for(int e=0;e<::NE;e++)::psurf[e]=config_.rho*effectiveGravity*elevation[e]
        -config_.gamma*curvature[e];
    };
    auto projectPressure=[&](bool momentum){
      for(int component=0;component<3;component++){
        auto values=array(9+component);
        if(momentum)std::copy(predictor[component].begin(),predictor[component].end(),values.begin());
        else std::fill(values.begin(),values.end(),0);
      }
      ::cell3d_axisU();::cell3d_omegaFromW();
      ::cell3d_nativeProject(dt);checkError("implicit surface pressure response");report.pressureSolves++;
      if(response)response(momentum?SurfaceResponseKind::physical:SurfaceResponseKind::tangent);
      for(int key:{9,10,11,13})for(double value:array(key))
        if(!std::isfinite(value))throw std::runtime_error("implicit joint response produced non-finite velocity or pressure");
      ::cell3d_axisU();::cell3d_omegaFromW();checkError("implicit joint surface response");
    };
    auto normalStress=[&](){
      std::vector<double> value(::NE);
      for(int i=0;i<::nr;i++)for(int k=0;k<::nth;k++)value[::ie(i,k)]=::surfaceNormalStress(i,k);
      vectorMaximum(value);return value;
    };
    auto pressureResponse=[&](bool momentum){
      // At one fixed trial geometry the normal stress N is linear in velocity,
      // and the pressure/rigid response is affine: u(q)=U0+Lq. Therefore the
      // complete endpoint condition q=q_cap+Nu(q) is solved by
      //     (I-NL) correction = N u(q_cap).
      // N is the full canonical n.E.n contraction. No flat-normal, lagged
      // velocity, diagonal shift or empirical stress coefficient is substituted.
      const std::vector<double> capillaryPressure(::psurf,::psurf+::NE);
      projectPressure(momentum);
      const auto rhs=normalStress();
      const double tolerance=options.absolutePressureTolerance+options.relativePressureTolerance
        *std::max(vectorMaximum(capillaryPressure),vectorMaximum(rhs));
      auto apply=[&](std::span<const double> value){
        std::copy(value.begin(),value.end(),::psurf);projectPressure(false);
        auto result=normalStress();
        for(std::size_t e=0;e<result.size();e++)result[e]=value[e]-result[e];
        return result;
      };
      std::size_t iterations=0;
      const auto correction=surfaceGmres(apply,rhs,.25*tolerance,options.krylovDimension,
        options.maximumLinearIterations,iterations);
      report.normalStressIterations+=iterations;
      for(int e=0;e<::NE;e++)::psurf[e]=capillaryPressure[e]+correction[e];
      projectPressure(momentum);
      const auto actual=normalStress();
      double defect=0;
      for(int e=0;e<::NE;e++)defect=std::max(defect,std::abs(::psurf[e]-capillaryPressure[e]-actual[e]));
      if(!(std::isfinite(defect)&&defect<=tolerance))
        throw std::runtime_error("implicit surface endpoint normal stress failed its true residual");
    };
    auto residual=[&](std::span<const double> elevation){
      for(double z:elevation)if(!std::isfinite(z)||!(config_.h+z>0))
        throw std::runtime_error("implicit surface trial leaves the single-valued wet domain");
      std::copy(elevation.begin(),elevation.end(),::eta);
      ::cell3d_refreshMetric();checkError("implicit trial endpoint metric");
      pressureAt(elevation);pressureResponse(true);
      std::vector<double> r(::NE);
      for(int i=0;i<::nr;i++)for(int k=0;k<::nth;k++){
        const int e=::ie(i,k);r[e]=elevation[e]-oldEta[e]-dt*::om[::iw(i,k,::nz)];
      }
      return r;
    };
    auto r=residual(trial);
    for(;;){
      const double tolerance=options.absoluteSurfaceTolerance+options.relativeSurfaceTolerance
        *std::max(vectorMaximum(oldEta),vectorMaximum(trial));
      report.surfaceResidual=vectorMaximum(r);
      if(report.surfaceResidual<=tolerance)break;
      if(report.newtonIterations>=options.maximumNewtonIterations)
        throw std::runtime_error("implicit nonlinear surface/pressure solve did not converge");
      auto jacobian=[&](std::span<const double> direction){
        curvatureDirection(trial,direction,curvature);
        for(int e=0;e<::NE;e++)::psurf[e]=config_.rho*effectiveGravity*direction[e]-config_.gamma*curvature[e];
        pressureResponse(false);std::vector<double> result(direction.begin(),direction.end());
        for(int i=0;i<::nr;i++)for(int k=0;k<::nth;k++)result[::ie(i,k)]-=dt*::om[::iw(i,k,::nz)];
        return result;
      };
      std::vector<double> rhs=r;for(double& value:rhs)value=-value;
      std::size_t iterations=0;
      const double linearTolerance=std::max(.05*tolerance,options.linearRelativeTolerance*vectorNorm(rhs));
      const auto correction=surfaceGmres(jacobian,rhs,linearTolerance,options.krylovDimension,
        options.maximumLinearIterations,iterations);
      report.linearIterations+=iterations;
      const double before=vectorNorm(r);double fraction=1;bool accepted=false;
      for(int backtrack=0;backtrack<32;backtrack++,fraction*=.5){
        std::vector<double> candidate=trial;
        bool wet=true;for(std::size_t e=0;e<candidate.size();e++){
          candidate[e]+=fraction*correction[e];wet=wet&&std::isfinite(candidate[e])&&config_.h+candidate[e]>0;
        }
        if(!wet)continue;
        auto next=residual(candidate);
        if(vectorNorm(next)<=(1-1e-4*fraction)*before||vectorMaximum(next)<=tolerance){
          trial=std::move(candidate);r=std::move(next);accepted=true;break;
        }
      }
      if(!accepted)throw std::runtime_error("implicit surface Newton line search did not decrease its true residual");
      report.newtonIterations++;
    }
    // Evaluate the converged state's velocity/pressure once more: a Jacobian
    // response must never be mistaken for the physical accepted velocity.
    r=residual(trial);report.surfaceResidual=vectorMaximum(r);
    if(report.surfaceResidual>options.absoluteSurfaceTolerance+options.relativeSurfaceTolerance
        *std::max(vectorMaximum(oldEta),vectorMaximum(trial)))
      throw std::runtime_error("implicit accepted surface failed its independently recomputed residual");
    std::vector<double> acceptedPressure(::psurf,::psurf+::NE);
    // The converged trial metric already carries this velocity's pressure and
    // continuity solve. Advancing eta again, or projecting after changing the
    // metric, would break the coupled equation just solved. Accept its bounded
    // kinematic residual and retain that exact endpoint geometry instead.
    synchronizeVelocity();
    pressureAt(array(FluidArray::eta));
    const auto finalNormal=normalStress();
    for(int e=0;e<::NE;e++)report.surfacePressureResidual=std::max(report.surfacePressureResidual,
      std::abs(acceptedPressure[e]-::psurf[e]-finalNormal[e]));
    const double pressureTolerance=options.absolutePressureTolerance+options.relativePressureTolerance
      *std::max(vectorMaximum(acceptedPressure),vectorMaximum(finalNormal));
    if(report.surfacePressureResidual>pressureTolerance)
      throw std::runtime_error("accepted implicit surface does not satisfy endpoint normal stress");
    std::copy(acceptedPressure.begin(),acceptedPressure.end(),::psurf);
    ::gEff=CELL3D_GRAVITY_UNSET;time_+=dt;
    return report;
  } catch (...) {
    ::nativeRigidProjection=oldProjection;rigidProjection_=oldProjection;generation_=beforeMode.ownerGeneration;
    restore(beforeMode);
    throw;
  }
}

void NativeFluid::predictor(double dt,double effectiveGravity){
  requireReady(); positiveFinite(dt,"dt");
  if(!(std::isfinite(time_+dt)&&time_+dt>time_))
    throw std::invalid_argument("dt must advance the representable physical clock");
  if (pending_) throw std::logic_error("a coupled native predictor is already pending");
  if (!std::isfinite(effectiveGravity)) throw std::invalid_argument("effective gravity must be finite");
  ::cell3d_setGravity(effectiveGravity);
  ::cell3d_stepBegin(dt); checkError("native step begin");
  ::cell3d_stepExplicit(0,::nr-1); checkError("native explicit right-hand side");
  ::cell3d_nativePredictor(dt); checkError("native explicit predictor");
  ::cell3d_nativeProject(dt); checkError("native free-surface pressure projection");
  pendingDt_=dt; pending_=true;
}

void NativeFluid::projectHomogeneous(std::span<double> du,std::span<double> dv,
    std::span<double> dw,double tolerance){
  requireReady(); positiveFinite(tolerance,"pressure tolerance");
  if (!(tolerance<1)) throw std::invalid_argument("pressure tolerance must be less than one");
  checkSpan(du,static_cast<std::size_t>(::NU),"du");
  checkSpan(dv,static_cast<std::size_t>(::NV),"dv");
  checkSpan(dw,static_cast<std::size_t>(::NW),"dw");
  auto overlaps=[](std::span<const double> a,std::span<const double> b){
    const auto x=reinterpret_cast<std::uintptr_t>(a.data());
    const auto y=reinterpret_cast<std::uintptr_t>(b.data());
    return x<=y ? y-x<a.size_bytes() : x-y<b.size_bytes();
  };
  if (overlaps(du,dv) || overlaps(du,dw) || overlaps(dv,dw))
    throw std::invalid_argument("homogeneous velocity components must use disjoint caller-owned storage");
  for (auto values:{du,dv,dw}){
    for (int key=0;key<=66;key++) if (overlaps(values,array(key)))
      throw std::invalid_argument("homogeneous velocity increments must not alias canonical fluid arrays");
    for (const auto* scratch:{&projectionOmega_,&projectionRhs_,&projectionPressure_})
      if (overlaps(values,*scratch))
        throw std::invalid_argument("homogeneous velocity increments must not alias native projection scratch");
  }
  // The caller supplies independent velocity rows; prescribed rows remain zero.
  // In rigid mode the axis is derived, never treated as an independent force row.
  for (int k=0;k<::nth;k++) for (int j=0;j<::nz;j++)
    if ((!rigidProjection_ && du[::iu(0,k,j)]!=0) || du[::iu(::nr,k,j)]!=0)
      throw std::invalid_argument("homogeneous increments must have zero auxiliary axis and prescribed rim rows");
  for (int i=0;i<::nr;i++) for (int k=0;k<::nth;k++)
    if (dw[::iw(i,k,0)]!=0)
      throw std::invalid_argument("homogeneous increments must have zero floor rows");
  for (auto values:{du,dv,dw}) for (double value:values)
    if (!std::isfinite(value)) throw std::invalid_argument("non-finite homogeneous velocity increment");
  if (rigidProjection_) ::nativeAxisOf(du.data());
  ::cell3d_omegaOf(du.data(),dv.data(),dw.data(),projectionOmega_.data());
  ::cell3d_divergence(du.data(),dv.data(),projectionOmega_.data(),projectionRhs_.data());
  std::fill(projectionPressure_.begin(),projectionPressure_.end(),0.0);
  struct RestorePressure { double* saved; ~RestorePressure(){ ::p=saved; } } restore{::p};
  ::p=projectionPressure_.data();
  ::cell3d_solveP(projectionRhs_.data(),tolerance,400*(::nr+::nth+::nz));
  checkError("native homogeneous pressure projection");
  ::cell3d_gradient(projectionPressure_.data(),::gu,::gv,::gw);
  for (std::size_t c=0;c<du.size();c++) du[c]-=::gu[c];
  for (std::size_t c=0;c<dv.size();c++) dv[c]-=::gv[c];
  for (std::size_t c=0;c<dw.size();c++) dw[c]-=::gw[c];
  if (rigidProjection_) ::nativeAxisOf(du.data());
}

void NativeFluid::finish(double dt){
  requireReady(); positiveFinite(dt,"dt");
  if (!pending_ || dt!=pendingDt_)
    throw std::logic_error("finish requires the exact time step of the pending native predictor");
  ::cell3d_axisU(); ::cell3d_omegaFromW();
  ::cell3d_nativeAdvanceSurface(dt); checkError("native coupled surface advance");
  time_+=dt; pending_=false; pendingDt_=0;
}

double NativeFluid::velocityMass(int component,std::size_t index) const {
  requireReady();
  const int key=component==0?9:component==1?10:component==2?11:-1;
  if (key<0 || index>=array(key).size()) throw std::out_of_range("invalid velocity mass degree of freedom");
  const std::size_t stride=component==2?static_cast<std::size_t>(::nz+1):static_cast<std::size_t>(::nz);
  const std::size_t j=index%stride, column=index/stride, k=column%static_cast<std::size_t>(::nth), i=column/static_cast<std::size_t>(::nth);
  if (component==0){
    if (i==0 || i==static_cast<std::size_t>(::nr)) return 0;
    return ::rho*(::radialDualColumnVolume(i,k)*::dsc[j]);
  }
  if (component==1) return ::rho*(::rc[i]*::drc[i]*::dth*::Hth[column]*::dsc[j]);
  if (j==0) return 0;
  (void)k;
  return ::rho*(::rc[i]*::drc[i]*::dth*::H[column]*::dsf[j]);
}

double NativeFluid::maxDivergence(){
  requireReady(); ::cell3d_omegaFromW();
  return ::cell3d_maxDivergence();
}

StepLimits NativeFluid::stepLimits(double peakAcceleration) const {
  requireReady();
  if (!std::isfinite(peakAcceleration)) throw std::invalid_argument("peak acceleration must be finite");
  double drMin=std::numeric_limits<double>::infinity();
  double dzMin=std::numeric_limits<double>::infinity(), Hmin=std::numeric_limits<double>::infinity();
  double massRateFactor=1;
  for (int i=0;i<::nr;i++) drMin=std::min(drMin,::drc[i]);
  for (int i=0;i<::nr;i++) for (int k=0;k<::nth;k++){
    const double depth=::H[::ie(i,k)]; Hmin=std::min(Hmin,depth);
    for (int j=0;j<::nz;j++) dzMin=std::min(dzMin,depth*::dsc[j]);
  }
  for(int i=1;i<::nr;i++)for(int k=0;k<::nth;k++)
    massRateFactor=std::max(massRateFactor,
      ::rf[i]*::drf[i]*::dth*::Hr[::ie(i,k)]/::radialDualColumnVolume(i,k));
  // Transcription of cell3dGridLimits, including the JS dual-mass correction.
  // Replacing a diagonal inertial mass by a smaller one increases a momentum
  // rate by at most max(M_old/M_new), and a wave frequency by its square root.
  // These retain the existing full-stencil stability estimates; this ratio
  // accounts for changed inertia and is not a new spectral proof for every ALE
  // geometry. It changes step admission only, never a physical coefficient.
  const double kr2=4/(drMin*drMin), kz2=4/(dzMin*dzMin);
  const double mMax=static_cast<double>(::nth>>1), ka2=(mMax*mMax)/(::rc[0]*::rc[0]);
  const double kSurf2=kr2+ka2, kSurf=std::sqrt(kSurf2);
  const double gravityBound=config_.g+std::abs(peakAcceleration);
  const double omegaSurf=std::sqrt((gravityBound*kSurf+::gamma_*kSurf2*kSurf/::rho)*std::tanh(kSurf*Hmin));
  double adv=0;
  for (int i=0;i<::nr;i++) for (int k=0;k<::nth;k++){
    const double radius=::rc[i], depth=::H[::ie(i,k)];
    for (int j=0;j<::nz;j++){
      const double components[6]={::u[::iu(i,k,j)],::u[::iu(i+1,k,j)],
        ::v[::iv(i,k,j)],::v[::iv(i,k+1,j)],::w[::iw(i,k,j)],::w[::iw(i,k,j+1)]};
      for (double value:components) if (!std::isfinite(value))
          throw std::runtime_error("non-finite physical velocity while computing native stability limits");
      const double uu=std::max(std::abs(components[0]),std::abs(components[1]));
      const double vv=std::max(std::abs(components[2]),std::abs(components[3]));
      const double ww=std::max(std::abs(components[4]),std::abs(components[5]));
      adv=std::max(adv,uu/::drc[i]+vv/(radius*::dth)+ww/(depth*::dsc[j]));
    }
  }
  const double frequencyFactor=std::sqrt(massRateFactor);
  return {(2/omegaSurf)/frequencyFactor,(2/(::nu*(kr2+kz2+ka2)))/massRateFactor,
    adv>0?(1/adv)/massRateFactor:std::numeric_limits<double>::infinity(),
    (std::min(drMin,::rc[0]*::dth)/std::sqrt(gravityBound*Hmin))/frequencyFactor,
    std::sqrt(kr2),std::sqrt(kz2),std::sqrt(ka2)};
}

double NativeFluid::stableStep(double peakAcceleration,double safety) const {
  positiveFinite(safety,"step safety factor");
  const auto limits=stepLimits(peakAcceleration);
  return safety*std::min({limits.capillary,limits.viscous,limits.advective,limits.gravity});
}

double NativeFluid::implicitSurfaceStepLimit(double peakAcceleration) const {
  const auto limits=stepLimits(peakAcceleration);
  return std::min({limits.viscous,limits.advective,limits.gravity});
}

int NativeFluid::pressureIterations() const noexcept { return ::cell3d_cgIters(); }
double NativeFluid::pressureResidual() const noexcept { return ::cell3d_cgResidual(); }

} // namespace faraday_native
