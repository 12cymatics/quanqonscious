// Include the production runtime in this verification translation unit so its
// canonical differential operators and exact curvature tangent can be tested
// directly, without introducing a public test-only API or a second fluid core.
#include "fluid-runtime.cpp"
#include "rigid-coupler.hpp"
#include <bit>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

using namespace faraday_native;
namespace {
void require(bool value,const std::string& why){if(!value)throw std::runtime_error(why);}
struct Fixture {
  FluidConfig config;std::vector<RigidGrain> grains;
  std::vector<std::pair<int,std::vector<double>>> fields;
  explicit Fixture(const char* path){
    std::ifstream in(path,std::ios::binary);require(bool(in),"Cannot open fixture");
    const auto bytes=[&](std::size_t n){std::string b(n,'\0');require(bool(in.read(b.data(),n)),"Truncated fixture");return b;};
    const auto integer=[&](std::size_t n){const auto b=bytes(n);std::uint64_t v=0;for(std::size_t i=0;i<n;i++)v|=std::uint64_t(static_cast<unsigned char>(b[i]))<<(8*i);return v;};
    const auto real=[&](){return std::bit_cast<double>(integer(8));};
    require(bytes(8)=="FNSCASE1"&&integer(4)==1,"Unsupported fixture");
    config.nr=integer(4);config.nth=integer(4);config.nz=integer(4);require(integer(4)==1,"Fixture must be pinned");config.pinned=true;
    config.memoryBudgetBytes=integer(8);config.R=real();config.h=real();config.rho=real();config.nu=real();config.gamma=real();config.g=real();config.dtheta=real();
    for(unsigned i=0;i<7;i++)(void)real();
    const auto ng=integer(4),na=integer(4),nj=integer(4);(void)bytes(nj);
    for(std::size_t i=0;i<ng;i++){RigidGrain g;g.id=integer(8);g.centre={real(),real(),real()};g.radius=real();g.density=real();g.velocity={real(),real(),real()};g.omega={real(),real(),real()};grains.push_back(g);}
    for(std::size_t i=0;i<na;i++){const int key=integer(4);const auto n=integer(8);std::vector<double> a(n);for(auto& x:a)x=real();fields.emplace_back(key,std::move(a));}
    require(in.peek()==std::char_traits<char>::eof()&&grains.size()==1,"Requires one separated sphere fixture");
  }
  std::unique_ptr<NativeFluid> fresh() const {
    auto f=std::make_unique<NativeFluid>(config);for(const auto& [key,a]:fields){auto out=f->array(key);require(out.size()==a.size(),"Fixture field size mismatch");std::copy(a.begin(),a.end(),out.begin());}f->initialize();
    // Finite initial deformation, not a microscopic field hiding geometry
    // errors: approximately 45 micrometres in this 500 micrometre layer.
    auto eta=f->array(FluidArray::eta);const auto rc=f->array(FluidArray::rc);
    for(int i=0;i<f->nr();i++)for(int k=0;k<f->nth();k++){
      const double r=rc[i]/config.R,theta=(k+.5)*config.dtheta;
      eta[i*f->nth()+k]=1.5e-4*(r*std::pow(1-r*r,2)*std::cos(theta)+.35*r*r*std::pow(1-r*r,2)*std::sin(2*theta));
    }
    f->refreshMetric();f->synchronizeVelocity();return f;
  }
};
double maximum(std::span<const double> a){double m=0;for(double x:a){require(std::isfinite(x),"Non-finite verification state");m=std::max(m,std::abs(x));}return m;}
long double volume(const NativeFluid& f){
  const auto rf=f.array(FluidArray::rf),h=f.array(FluidArray::H);long double sum=0;
  for(int i=0;i<f.nr();i++)for(int k=0;k<f.nth();k++)sum+=(static_cast<long double>(rf[i+1])*rf[i+1]-static_cast<long double>(rf[i])*rf[i])*.5L*f.config().dtheta*h[i*f.nth()+k];
  return sum;
}
std::vector<double> pack(const NativeFluid& f,const std::vector<FluidDof>& dofs){std::vector<double> out;for(const auto& d:dofs)out.push_back(f.array(9+d.family)[d.index]);return out;}
struct Predictor {
  std::array<std::vector<double>,3> velocity;std::vector<double> normal;
};
Predictor explicitPredictor(NativeFluid& f,double dt,double gravity){
  const auto saved=f.checkpoint();Predictor out;
  try{
    ::cell3d_setGravity(gravity);require(::cell3d_stepBegin(dt)==0,"Independent step begin failed");
    out.normal.resize(f.nr()*f.nth());for(int i=0;i<f.nr();i++)for(int k=0;k<f.nth();k++)out.normal[i*f.nth()+k]=::surfaceNormalStress(i,k);
    require(::cell3d_stepExplicit(0,f.nr()-1)==0&&::cell3d_nativePredictor(dt)==0,"Independent explicit predictor failed");
    for(int d=0;d<3;d++)out.velocity[d].assign(f.array(9+d).begin(),f.array(9+d).end());
    f.restore(saved);return out;
  }catch(...){f.restore(saved);throw;}
}
void curvatureDerivative(const Fixture& fixture){
  auto f=fixture.fresh();const std::vector<double> at(f->array(FluidArray::eta).begin(),f->array(FluidArray::eta).end());
  std::vector<double> direction(at.size()),analytic(at.size()),plus(at.size()),minus(at.size()),p(at.size()),m(at.size());
  for(std::size_t e=0;e<at.size();e++)direction[e]=std::sin(.37*e)+.2*std::cos(.13*e);
  faraday_native::curvatureDirection(at,direction,analytic);double previous=0;
  for(const double eps:{1e-7,1e-8,1e-9}){
    for(std::size_t e=0;e<at.size();e++){plus[e]=at[e]+eps*direction[e];minus[e]=at[e]-eps*direction[e];}
    auto saved=::eta;::eta=plus.data();::cell3d_curvature(p.data());::eta=minus.data();::cell3d_curvature(m.data());::eta=saved;
    double error=0;for(std::size_t e=0;e<at.size();e++)error=std::max(error,std::abs((p[e]-m[e])/(2*eps)-analytic[e]));error/=maximum(analytic);
    std::cout<<"{\"kind\":\"curvature-direction-finite-difference\",\"epsilonMetres\":"<<eps<<",\"relativeMaxError\":"<<error<<",\"surfaceAmplitudeMetres\":"<<maximum(at)<<"}\n";
    if(previous)require(error<previous/30,"Exact nonlinear curvature tangent lacks second-order central-difference agreement");
    previous=error;
  }
  require(previous<1e-6,"Nonlinear curvature tangent does not match independent finite differences");
}
struct RunResult {double dt=0,time=0,wall=0;std::vector<double> fluid,mass,eta,pressure;Six impulse{};RigidGrain grain;};
void physicalTime(const NativeFluid& fluid,double previous,double expected,
    const CoupledStepReport& report,double requested){
  const double tolerance=16*std::numeric_limits<double>::epsilon()*std::max(std::abs(expected),requested);
  require(std::isfinite(fluid.time())&&fluid.time()>previous,
    "Accepted coupled step did not advance finite physical time");
  require(report.dt==requested&&std::abs(fluid.time()-expected)<=tolerance
      &&std::abs(report.time-fluid.time())<=tolerance,
    "Accepted physical time or reported time differs from the requested step sequence");
}
void commonTime(const RunResult& a,const RunResult& b){
  const double scale=std::max({std::abs(a.time),std::abs(b.time),a.dt,b.dt});
  require(std::isfinite(a.time)&&std::isfinite(b.time)&&a.time>0&&b.time>0
      &&std::abs(a.time-b.time)<=32*std::numeric_limits<double>::epsilon()*scale,
    "Refinement or preconditioner results have different physical end times");
}
RunResult trajectory(const Fixture& fixture,double dt,std::size_t steps,bool cached=false){
  auto f=fixture.fresh();auto grains=fixture.grains;grains[0].centre.x=.4*grains[0].radius;
  CouplingOptions couplingOptions;if(cached)couplingOptions.preconditioner=ConstraintPreconditioner::projectedCached;
  RigidCoupler coupled(*f,grains,couplingOptions);require(coupled.stateDiagnostics().compatible,"Supplied resting finite-amplitude state is not compatible");
  const double gravity=fixture.config.g,initialAmplitude=maximum(f->array(FluidArray::eta));
  require(initialAmplitude>3e-5,"Finite-amplitude verification field was reduced");
  ImplicitSurfaceOptions options;options.absoluteSurfaceTolerance=1e-16;options.relativeSurfaceTolerance=1e-12;
  const auto initialLimits=f->stepLimits();
  double minimumRetainedLimit=std::min({initialLimits.viscous,initialLimits.advective,initialLimits.gravity});
  double maximumCapillaryRatio=0,maximumRetainedRatio=0;
  double maxKinematic=0,maxPressure=0,maxOldStressDefect=0,maxDiv=0,maxSlip=0,maxForceDefect=0,maxTorqueDefect=0,maxVolumeDefect=0,maxSurfaceUpdate=0;
  std::size_t totalNewton=0,totalLinear=0,totalPressure=0,totalConstraints=0,totalStress=0;
  Six accumulatedImpulse{};
  double expectedTime=f->time();
  const auto wall=std::chrono::steady_clock::now();
  for(std::size_t n=0;n<steps;n++){
    const auto limits=f->stepLimits();const double retainedLimit=std::min({limits.viscous,limits.advective,limits.gravity});
    require(dt<=.4*retainedLimit,"Coupled verification step exceeds retained viscous/advection/gravity safety bounds");
    minimumRetainedLimit=std::min(minimumRetainedLimit,retainedLimit);
    maximumCapillaryRatio=std::max(maximumCapillaryRatio,dt/limits.capillary);
    maximumRetainedRatio=std::max(maximumRetainedRatio,dt/retainedLimit);
    const double previousTime=f->time();expectedTime+=dt;
    const auto before=f->checkpoint();const auto old=coupled.grains()[0];const auto oldVolume=volume(*f);
    const auto predictor=explicitPredictor(*f,dt,gravity);
    const auto result=coupled.stepImplicitSurface(dt,gravity,options);const auto& now=coupled.grains()[0];
    require(result.coupling.dt==dt&&result.coupling.eventRefinements==0,"Implicit step changed requested dt");
    physicalTime(*f,previousTime,expectedTime,result.coupling,dt);
    require(result.surface.dt==dt,"Surface report changed the requested physical step");
    for(std::size_t d=0;d<3;d++)require(now.centre[d]==old.centre[d]+dt*old.velocity[d],"Declared old-velocity position drift was not retained");
    const auto accepted=f->checkpoint();f->synchronizeVelocity();require(accepted.state==f->checkpoint().state,"Accepted dependent velocity rows or surface rate are stale");
    const auto eta=f->array(FluidArray::eta),omega=f->array(FluidArray::om),ps=f->array(FluidArray::psurf);
    std::vector<double> curvature(eta.size());::cell3d_curvature(curvature.data());double kinematic=0,pressure=0,pressureScale=0;
    for(int i=0;i<f->nr();i++)for(int k=0;k<f->nth();k++){
      const int e=i*f->nth()+k;const double update=eta[e]-before.state[5][e];
      kinematic=std::max(kinematic,std::abs(update-dt*omega[(i*f->nth()+k)*(f->nz()+1)+f->nz()]));maxSurfaceUpdate=std::max(maxSurfaceUpdate,std::abs(update));
      const double capillary=fixture.config.rho*gravity*eta[e]-fixture.config.gamma*curvature[e];
      const double endpointStress=::surfaceNormalStress(i,k),target=capillary+endpointStress;
      pressure=std::max(pressure,std::abs(ps[e]-target));
      pressureScale=std::max({pressureScale,std::abs(capillary),std::abs(endpointStress),std::abs(target)});
      maxOldStressDefect=std::max(maxOldStressDefect,std::abs(ps[e]-capillary-predictor.normal[e]));
    }
    const double surfaceTolerance=options.absoluteSurfaceTolerance+options.relativeSurfaceTolerance*std::max(maximum(before.state[5]),maximum(eta));
    require(kinematic<=surfaceTolerance,"Actual endpoint velocity/geometry violates the nonlinear surface kinematic equation");
    require(pressure<=options.absolutePressureTolerance+options.relativePressureTolerance*pressureScale,
      "Endpoint pressure violates the actual final-velocity normal viscous traction condition");
    const auto dofs=independentFluidDofs(*f);RigidInterpolation interpolation(dofs,coupled.grains());
    const auto interpolated=interpolation.applyJ(pack(*f,dofs)),rigid=interpolation.applyK({rigidVelocity(now)});double slip=0,scale=0;
    for(std::size_t j=0;j<interpolated.size();j++){slip=std::max(slip,norm(interpolated[j]-rigid[j]));scale=std::max({scale,norm(interpolated[j]),norm(rigid[j])});}
    require(slip<=1e-13+1e-9*scale,"Actual trial-metric/endpoint-grain constraints are not satisfied");
    const double divergence=f->maxDivergence();require(divergence<=1e-8,"Actual endpoint metric is not incompressible");
    // Reconstruct the pressure term on the accepted geometry. Removing it and
    // the separately formed explicit predictor from M*(u_new-u_old) leaves only
    // the distributed grain impulse. Its total force/torque must be opposite
    // the actual excess-body momentum change, regardless of lambda bookkeeping.
    ::cell3d_gradientSurface(::p,::psurf,::gu,::gv,::gw);Vec3 fluidImpulse{},fluidTorque{};double reactionScale=0;
    for(const auto& d:dofs){
      const double impulse=d.mass*(f->array(9+d.family)[d.index]-predictor.velocity[d.family][d.index]+dt/fixture.config.rho*f->array(38+d.family)[d.index]);
      const Vec3 force=impulse*d.direction;fluidImpulse=fluidImpulse+force;fluidTorque=fluidTorque+cross(d.position,force);reactionScale+=std::abs(impulse);
    }
    const double me=(now.density-fixture.config.rho)*now.volume(),ie=.4*me*now.radius*now.radius;
    const Vec3 bodyImpulse=me*(now.velocity-old.velocity+Vec3{0,0,gravity*dt});
    const Vec3 bodyTorque=ie*(now.omega-old.omega)+cross(now.centre,bodyImpulse);
    const double forceDefect=norm(fluidImpulse+bodyImpulse),torqueDefect=norm(fluidTorque+bodyTorque);
    require(forceDefect<=1e-24+2e-8*reactionScale,"Actual liquid/body momentum exchange is unequal or pressure bookkeeping is wrong");
    require(torqueDefect<=1e-27+2e-8*fixture.config.R*reactionScale,"Actual liquid/body torque exchange is unequal");
    const double volumeDefect=static_cast<double>(std::abs(volume(*f)-oldVolume));
    const double volumeBound=std::acos(-1.0)*fixture.config.R*fixture.config.R*surfaceTolerance+1e-23;
    require(volumeDefect<=volumeBound,"Volume change exceeds the integrated nonlinear kinematic residual bound");
    maxKinematic=std::max(maxKinematic,kinematic);maxPressure=std::max(maxPressure,pressure);maxDiv=std::max(maxDiv,divergence);maxSlip=std::max(maxSlip,slip);maxForceDefect=std::max(maxForceDefect,forceDefect);maxTorqueDefect=std::max(maxTorqueDefect,torqueDefect);maxVolumeDefect=std::max(maxVolumeDefect,volumeDefect);
    totalNewton+=result.surface.newtonIterations;totalLinear+=result.surface.linearIterations;totalPressure+=result.surface.pressureSolves;totalConstraints+=result.coupling.constraintIterations;totalStress+=result.surface.normalStressIterations;
    for(std::size_t d=0;d<6;d++)accumulatedImpulse[d]+=result.coupling.rigidImpulse[0][d];
    std::cout<<"{\"kind\":\"accepted-joint-step\",\"step\":"<<n+1<<",\"dt\":"<<dt<<",\"time\":"<<f->time()<<",\"kinematicResidualMetres\":"<<kinematic<<",\"endpointNormalStressResidualPa\":"<<pressure<<",\"divergencePerSecond\":"<<divergence<<",\"slipMetresPerSecond\":"<<slip<<",\"pressureSolves\":"<<result.surface.pressureSolves<<",\"constraintIterations\":"<<result.coupling.constraintIterations<<"}\n";
  }
  require(maxSurfaceUpdate>1e-11,"Surface evolution is too small to test coupled endpoint geometry");
  RunResult out;out.dt=dt;out.time=f->time();out.wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall).count();out.impulse=accumulatedImpulse;out.grain=coupled.grains()[0];const auto dofs=independentFluidDofs(*f);out.fluid=pack(*f,dofs);for(const auto& d:dofs)out.mass.push_back(d.mass);out.eta.assign(f->array(FluidArray::eta).begin(),f->array(FluidArray::eta).end());out.pressure.assign(f->array(FluidArray::p).begin(),f->array(FluidArray::p).end());
  const auto preconditioner=coupled.preconditionerDiagnostics();
  if(cached)require(preconditioner.builds==1,"Projected preconditioner was not reused across the unchanged marker lattice");
  std::cout<<"{\"kind\":\"finite-amplitude-joint-implicit\",\"projectedCached\":"<<(cached?"true":"false")<<",\"preconditionerBuilds\":"<<preconditioner.builds<<",\"dt\":"<<dt<<",\"steps\":"<<steps<<",\"time\":"<<out.time<<",\"wallSeconds\":"<<out.wall<<",\"initialAmplitudeMetres\":"<<initialAmplitude<<",\"initialCapillaryLimitSeconds\":"<<initialLimits.capillary<<",\"minimumRetainedLimitSeconds\":"<<minimumRetainedLimit<<",\"maximumStepToCapillaryRatio\":"<<maximumCapillaryRatio<<",\"maximumStepToRetainedRatio\":"<<maximumRetainedRatio<<",\"maxSurfaceUpdateMetres\":"<<maxSurfaceUpdate<<",\"kinematicResidualMetres\":"<<maxKinematic<<",\"endpointNormalStressResidualPa\":"<<maxPressure<<",\"oldNormalStressDefectPa\":"<<maxOldStressDefect<<",\"divergencePerSecond\":"<<maxDiv<<",\"slipMetresPerSecond\":"<<maxSlip<<",\"forceImpulseDefectNs\":"<<maxForceDefect<<",\"torqueImpulseDefectNms\":"<<maxTorqueDefect<<",\"volumeDefectM3\":"<<maxVolumeDefect<<",\"newtonIterations\":"<<totalNewton<<",\"linearIterations\":"<<totalLinear<<",\"normalStressIterations\":"<<totalStress<<",\"pressureSolves\":"<<totalPressure<<",\"constraintIterations\":"<<totalConstraints<<"}\n";
  return out;
}
double fluidDifference(const RunResult& a,const RunResult& b){commonTime(a,b);require(a.fluid.size()==b.fluid.size(),"Refinement changed the spatial grid");long double sum=0,mass=0;for(std::size_t j=0;j<a.fluid.size();j++){const long double d=a.fluid[j]-b.fluid[j];sum+=a.mass[j]*d*d;mass+=a.mass[j];}return std::sqrt(static_cast<double>(sum/mass));}
RunResult explicitTrajectory(const Fixture& fixture,double dt,std::size_t steps){
  auto f=fixture.fresh();auto grains=fixture.grains;grains[0].centre.x=.4*grains[0].radius;
  const double initialAmplitude=maximum(f->array(FluidArray::eta));
  RigidCoupler coupled(*f,grains);require(coupled.stateDiagnostics().compatible,"Explicit comparison initial state is incompatible");
  require(dt<=f->stableStep(),"Explicit comparison exceeds its capillary stability limit");
  double maxD=0,maxSlip=0;std::size_t refinements=0;const auto wall=std::chrono::steady_clock::now();
  double expectedTime=f->time();
  for(std::size_t n=0;n<steps;n++){
    const double previousTime=f->time();expectedTime+=dt;
    const auto accepted=coupled.step(dt,fixture.config.g);refinements+=accepted.eventRefinements;
    require(accepted.dt==dt&&accepted.eventRefinements==0,"Explicit comparison reduced its requested step");
    physicalTime(*f,previousTime,expectedTime,accepted,dt);
    const auto saved=f->checkpoint();f->synchronizeVelocity();require(saved.state==f->checkpoint().state,"Explicit comparison endpoint arrays are stale");
    const auto dofs=independentFluidDofs(*f);RigidInterpolation interpolation(dofs,coupled.grains());
    const auto actual=interpolation.applyJ(pack(*f,dofs)),target=interpolation.applyK({rigidVelocity(coupled.grains()[0])});double scale=0,slip=0;
    for(std::size_t j=0;j<actual.size();j++){slip=std::max(slip,norm(actual[j]-target[j]));scale=std::max({scale,norm(actual[j]),norm(target[j])});}
    require(slip<=1e-13+1e-9*scale,"Explicit comparison endpoint no-slip failure");
    maxSlip=std::max(maxSlip,slip);maxD=std::max(maxD,f->maxDivergence());require(maxD<=1e-8,"Explicit comparison endpoint divergence failure");
  }
  RunResult out;out.dt=dt;out.time=f->time();out.wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall).count();out.grain=coupled.grains()[0];const auto dofs=independentFluidDofs(*f);out.fluid=pack(*f,dofs);for(const auto& d:dofs)out.mass.push_back(d.mass);out.eta.assign(f->array(FluidArray::eta).begin(),f->array(FluidArray::eta).end());
  std::cout<<"{\"kind\":\"finite-amplitude-explicit-comparison\",\"dt\":"<<dt<<",\"steps\":"<<steps<<",\"time\":"<<out.time<<",\"wallSeconds\":"<<out.wall<<",\"initialAmplitudeMetres\":"<<initialAmplitude<<",\"refinements\":"<<refinements<<",\"divergencePerSecond\":"<<maxD<<",\"slipMetresPerSecond\":"<<maxSlip<<"}\n";
  return out;
}
void refinement(const char* name,double a,double b,double floor=1e-12){const double order=std::log2(a/b);std::cout<<"{\"kind\":\"joint-implicit-temporal-refinement\",\"observable\":"<<std::quoted(name)<<",\"coarseDifference\":"<<a<<",\"fineDifference\":"<<b<<",\"order\":"<<order<<"}\n";require(std::isfinite(order)&&b>floor&&order>.65&&order<1.35,"Joint implicit temporal refinement does not show first-order convergence above the numerical floor");}
double scalarRmsDifference(const RunResult& a,const RunResult& b,bool surface){
  commonTime(a,b);const auto& left=surface?a.eta:a.pressure;const auto& right=surface?b.eta:b.pressure;
  require(left.size()==right.size()&&!left.empty(),"Scalar refinement fields have different spatial grids");
  long double sum=0;for(std::size_t j=0;j<left.size();j++){const long double d=left[j]-right[j];sum+=d*d;}
  return std::sqrt(static_cast<double>(sum/left.size()));
}
void comparePreconditioner(const Fixture& fixture){
  const auto diagonal=trajectory(fixture,1e-7,2),cached=trajectory(fixture,1e-7,2,true);
  commonTime(diagonal,cached);
  double velocity=0,eta=0,pressure=0,force=0,torque=0;
  for(std::size_t j=0;j<diagonal.fluid.size();j++)velocity=std::max(velocity,std::abs(diagonal.fluid[j]-cached.fluid[j]));
  for(std::size_t j=0;j<diagonal.eta.size();j++)eta=std::max(eta,std::abs(diagonal.eta[j]-cached.eta[j]));
  for(std::size_t j=0;j<diagonal.pressure.size();j++)pressure=std::max(pressure,std::abs(diagonal.pressure[j]-cached.pressure[j]));
  for(std::size_t d=0;d<6;d++)(d<3?force:torque)=std::max(d<3?force:torque,std::abs(diagonal.impulse[d]-cached.impulse[d]));
  const double body=norm(diagonal.grain.velocity-cached.grain.velocity),omega=norm(diagonal.grain.omega-cached.grain.omega),position=norm(diagonal.grain.centre-cached.grain.centre);
  const double velocityTolerance=20*(1e-13+1e-9*std::max(maximum(diagonal.fluid),maximum(cached.fluid)));
  const double me=(cached.grain.density-fixture.config.rho)*cached.grain.volume(),ie=.4*me*cached.grain.radius*cached.grain.radius;
  require(velocity<=velocityTolerance&&body<=velocityTolerance&&omega<=velocityTolerance/cached.grain.radius,"Cached preconditioning changed the converged fluid/body velocity beyond algebra tolerances");
  require(eta<=3e-16+2e-7*velocityTolerance&&position<=2e-7*velocityTolerance,"Cached preconditioning changed converged geometry beyond residual tolerances");
  require(pressure<=fixture.config.rho*fixture.config.R/1e-7*velocityTolerance,"Cached preconditioning changed accepted pressure beyond propagated velocity tolerance");
  require(force<=me*velocityTolerance&&torque<=ie*velocityTolerance/cached.grain.radius,"Cached preconditioning changed accepted impulse beyond propagated body tolerance");
  std::cout<<"{\"kind\":\"projected-preconditioner-state-comparison\",\"fluidMaxDifferenceMetresPerSecond\":"<<velocity<<",\"grainVelocityDifferenceMetresPerSecond\":"<<body<<",\"angularVelocityDifferencePerSecond\":"<<omega<<",\"positionDifferenceMetres\":"<<position<<",\"surfaceDifferenceMetres\":"<<eta<<",\"pressureDifferencePa\":"<<pressure<<",\"forceImpulseDifferenceNs\":"<<force<<",\"torqueImpulseDifferenceNms\":"<<torque<<",\"diagonalSeconds\":"<<diagonal.wall<<",\"cachedSeconds\":"<<cached.wall<<"}\n";
}
}
int main(int argc,char** argv){
  std::cout<<std::setprecision(17)<<std::unitbuf;
  try{
    require(argc==2||argc==3,"Usage: check-implicit-coupled prepared-case.fns [single|explicit|cached|preconditioner|above-capillary|above-capillary-single]");Fixture fixture(argv[1]);
    const std::string mode=argc==3?argv[2]:"full";
    require(mode=="full"||mode=="single"||mode=="explicit"||mode=="cached"||mode=="preconditioner"||mode=="above-capillary"||mode=="above-capillary-single","Unknown verification mode");
    if(argc==3&&std::string(argv[2])=="preconditioner"){
      comparePreconditioner(fixture);std::cout<<"{\"summary\":true,\"passed\":true,\"scope\":\"Matched-state projected preconditioner verification with actual fluid/body impulse checks\"}\n";return 0;
    }
    if(argc==3&&std::string(argv[2])=="explicit"){
      const auto coarse=explicitTrajectory(fixture,2e-8,10),middle=explicitTrajectory(fixture,1e-8,20),fine=explicitTrajectory(fixture,5e-9,40);
      refinement("explicit comparison fluid velocity RMS",fluidDifference(coarse,middle),fluidDifference(middle,fine));
      refinement("explicit comparison grain velocity",norm(coarse.grain.velocity-middle.grain.velocity),norm(middle.grain.velocity-fine.grain.velocity));
      std::cout<<"{\"summary\":true,\"passed\":true,\"scope\":\"Explicit comparison at the same finite amplitude and physical duration; error estimates require refinement, not a larger-step speedup claim\"}\n";return 0;
    }
    curvatureDerivative(fixture);
    const bool above=mode=="above-capillary"||mode=="above-capillary-single";
    const bool cached=mode=="cached"||above;
    double coarseDt=1e-7;
    if(above){
      auto f=fixture.fresh();const auto limits=f->stepLimits();
      coarseDt=1.5*limits.capillary;
      const double retained=std::min({limits.viscous,limits.advective,limits.gravity});
      require(std::isfinite(coarseDt)&&coarseDt>limits.capillary&&coarseDt<=.4*retained,
        "Above-capillary fixture violates retained viscous/advection/gravity safety bounds");
      std::cout<<"{\"kind\":\"coupled-step-bounds\",\"explicitCapillarySeconds\":"<<limits.capillary<<",\"retainedLimitSeconds\":"<<retained<<",\"coarseStepSeconds\":"<<coarseDt<<",\"ratioToCapillaryLimit\":"<<coarseDt/limits.capillary<<"}\n";
    }
    const auto coarse=trajectory(fixture,coarseDt,2,cached);
    if(mode=="full"||mode=="cached"||mode=="above-capillary"){
      const auto middle=trajectory(fixture,coarseDt/2,4,cached),fine=trajectory(fixture,coarseDt/4,8,cached);
      refinement("fluid velocity RMS",fluidDifference(coarse,middle),fluidDifference(middle,fine));
      refinement("grain velocity",norm(coarse.grain.velocity-middle.grain.velocity),norm(middle.grain.velocity-fine.grain.velocity));
      refinement("free surface elevation RMS",scalarRmsDifference(coarse,middle,true),scalarRmsDifference(middle,fine,true),1e-17);
      refinement("pressure RMS",scalarRmsDifference(coarse,middle,false),scalarRmsDifference(middle,fine,false),1e-8);
    }
    std::cout<<"{\"summary\":true,\"passed\":true,\"scope\":\"Joint finite-amplitude endpoint equations, reaction impulse balance and requested temporal checks; no contact or spatial-convergence claim\"}\n";return 0;
  }catch(const std::exception& e){std::cout<<"{\"summary\":true,\"passed\":false,\"error\":"<<std::quoted(e.what())<<"}\n";return 1;}
}
