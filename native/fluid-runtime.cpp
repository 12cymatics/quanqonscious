#include "fluid-runtime.hpp"

#include <algorithm>
#include <cmath>
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
  if (pending_) throw std::logic_error("finish the pending coupled step before starting another");
  if (!std::isfinite(effectiveGravity)) throw std::invalid_argument("effective gravity must be finite");
  ::cell3d_setGravity(effectiveGravity);
  ::cell3d_step(dt); checkError("native full fluid step");
  time_+=dt;
}

void NativeFluid::predictor(double dt,double effectiveGravity){
  requireReady(); positiveFinite(dt,"dt");
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
    return ::rho*(::rf[i]*::drf[i]*::dth*::Hr[column]*::dsc[j]);
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
  for (int i=0;i<::nr;i++) drMin=std::min(drMin,::drc[i]);
  for (int i=0;i<::nr;i++) for (int k=0;k<::nth;k++){
    const double depth=::H[::ie(i,k)]; Hmin=std::min(Hmin,depth);
    for (int j=0;j<::nz;j++) dzMin=std::min(dzMin,depth*::dsc[j]);
  }
  // Transcription of cell3dGridLimits. These are operator stability bounds,
  // not a particle-speed cap or a modification of a physical coefficient.
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
  return {2/omegaSurf,2/(::nu*(kr2+kz2+ka2)),
    adv>0?1/adv:std::numeric_limits<double>::infinity(),
    std::min(drMin,::rc[0]*::dth)/std::sqrt(gravityBound*Hmin),
    std::sqrt(kr2),std::sqrt(kz2),std::sqrt(ka2)};
}

double NativeFluid::stableStep(double peakAcceleration,double safety) const {
  positiveFinite(safety,"step safety factor");
  const auto limits=stepLimits(peakAcceleration);
  return safety*std::min({limits.capillary,limits.viscous,limits.advective,limits.gravity});
}

int NativeFluid::pressureIterations() const noexcept { return ::cell3d_cgIters(); }
double NativeFluid::pressureResidual() const noexcept { return ::cell3d_cgResidual(); }

} // namespace faraday_native
