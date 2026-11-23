#include "rigid-coupler.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// Independent fixed-physical-time verification of the explicit moving-geometry
// integrator. This is temporal verification on one spatial grid, not a claim of
// resolved lubrication, grid-converged hydrodynamics or measured sand behaviour.
using namespace faraday_native;
namespace {
void require(bool value,const std::string& why){if(!value)throw std::runtime_error(why);}
struct Fixture {
  FluidConfig config;
  std::vector<RigidGrain> grains;
  std::vector<std::pair<int,std::vector<double>>> fields;
  double acceleration=0;
  explicit Fixture(const char* path){
    std::ifstream in(path,std::ios::binary);require(bool(in),"Cannot open fixture");
    const auto bytes=[&](std::size_t n){std::string b(n,'\0');require(bool(in.read(b.data(),n)),"Truncated fixture");return b;};
    const auto integer=[&](std::size_t n){const auto b=bytes(n);std::uint64_t v=0;for(std::size_t i=0;i<n;i++)v|=std::uint64_t(static_cast<unsigned char>(b[i]))<<(8*i);return v;};
    const auto real=[&](){return std::bit_cast<double>(integer(8));};
    require(bytes(8)=="FNSCASE1"&&integer(4)==1,"Unsupported fixture");
    config.nr=integer(4);config.nth=integer(4);config.nz=integer(4);require(integer(4)==1,"Fixture must be pinned");config.pinned=true;
    config.memoryBudgetBytes=integer(8);config.R=real();config.h=real();config.rho=real();config.nu=real();config.gamma=real();config.g=real();config.dtheta=real();
    (void)real();acceleration=real();for(unsigned i=0;i<5;i++)(void)real();
    const auto ng=integer(4),na=integer(4),nj=integer(4);(void)bytes(nj);
    for(std::size_t i=0;i<ng;i++){RigidGrain g;g.id=integer(8);g.centre={real(),real(),real()};g.radius=real();g.density=real();g.velocity={real(),real(),real()};g.omega={real(),real(),real()};grains.push_back(g);}
    for(std::size_t i=0;i<na;i++){const int key=integer(4);const auto n=integer(8);std::vector<double> a(n);for(auto& x:a)x=real();fields.emplace_back(key,std::move(a));}
    require(in.peek()==std::char_traits<char>::eof(),"Trailing fixture data");require(grains.size()==1,"Requires one separated sphere");
  }
  std::unique_ptr<NativeFluid> fresh() const {
    auto f=std::make_unique<NativeFluid>(config);
    for(const auto& [key,a]:fields){auto out=f->array(key);require(out.size()==a.size(),"Fixture field size mismatch");std::copy(a.begin(),a.end(),out.begin());}
    f->initialize();return f;
  }
};
std::vector<double> packed(const NativeFluid& f,const std::vector<FluidDof>& dofs){std::vector<double> out;out.reserve(dofs.size());for(const auto& d:dofs)out.push_back(f.array(9+d.family)[d.index]);return out;}
long double volume(const NativeFluid& f){
  const auto r=f.array(FluidArray::rf),h=f.array(FluidArray::H);long double v=0;
  for(int i=0;i<f.nr();i++)for(int k=0;k<f.nth();k++)v+=(static_cast<long double>(r[i+1])*r[i+1]-static_cast<long double>(r[i])*r[i])*.5L*f.config().dtheta*h[i*f.nth()+k];
  return v;
}
struct Result {
  double dt=0,time=0,slip=0,divergence=0,bodyImpulseDefect=0,relativeVolumeDrift=0,wall=0;
  std::size_t steps=0,refinements=0;
  RigidGrain grain;
  std::vector<double> fluid,mass,eta,pressure;
};
void requireSynchronized(NativeFluid& f){
  const auto before=f.checkpoint();f.synchronizeVelocity();const auto after=f.checkpoint();
  require(before.time==after.time&&before.state==after.state,
    "Accepted state has stale dependent axis velocity, ALE flux or surface-rate arrays");
}
Result integrate(const Fixture& input,double dt,std::size_t count,bool offAxis=false){
  auto f=input.fresh();auto grains=input.grains;grains[0].velocity={0,0,-.001};
  if(offAxis){grains[0].centre.x=.8*grains[0].radius;grains[0].velocity={-.0005,.0003,-.001};grains[0].omega={.4,-.25,.2};}
  RigidCoupler coupled(*f,grains);CouplingOptions tolerance;
  // The same short, zero-gravity conditioning step makes the initial body and
  // fluid velocities constraint-compatible before EACH temporal refinement.
  // Its physical duration is included in the reported time, not hidden.
  const auto warm=coupled.step(1e-12,0);require(warm.dt==1e-12,"Conditioning step was refined");requireSynchronized(*f);
  require(dt<=f->stableStep(input.acceleration),"Requested verification step violates fluid stability");
  const auto initialVolume=volume(*f);const double startTime=f->time();Result r;r.dt=dt;
  const auto wallStart=std::chrono::steady_clock::now();
  for(std::size_t step=0;step<count;step++){
    const auto previous=coupled.grains()[0];
    const double gravity=input.config.g+input.acceleration;
    const auto accepted=coupled.step(dt,gravity);r.steps++;r.refinements+=accepted.eventRefinements;
    require(accepted.dt==dt,"A requested fixed-time refinement step was reduced");
    require(accepted.contacts==0,"Separated-sphere test unexpectedly reached contact");
    requireSynchronized(*f);
    const auto& current=coupled.grains()[0];
    // Newton--Euler impulse balance uses the independently computed excess
    // inertia and ALL reported impulses; it does not accept a residual report.
    const double me=(current.density-input.config.rho)*current.volume();
    const double ie=.4*me*current.radius*current.radius;
    const Six before=rigidVelocity(previous),after=rigidVelocity(current);
    require(accepted.rigidImpulse.size()==1,"Hydrodynamic impulse not reported");
    for(std::size_t d=0;d<6;d++){
      const double inertial=(d<3?me:ie)*(after[d]-before[d]+(d==2?gravity*dt:0));
      const double contact=accepted.contactImpulse.empty()?0:accepted.contactImpulse[0][d];
      const double defect=std::abs(inertial-accepted.rigidImpulse[0][d]-contact);
      const double scale=std::max(std::abs(inertial),std::abs(accepted.rigidImpulse[0][d])+std::abs(contact));
      r.bodyImpulseDefect=std::max(r.bodyImpulseDefect,defect);
      require(defect<1e-24+1e-8*scale,"Reported fluid/body impulses do not balance Newton--Euler");
    }
    // Build fresh interpolation using the actual endpoint fluid geometry and
    // grain centre. This can falsify stale old-geometry residual bookkeeping.
    const auto dofs=independentFluidDofs(*f);RigidInterpolation end(dofs,coupled.grains());
    const auto fluid=end.applyJ(packed(*f,dofs));const auto rigid=end.applyK({rigidVelocity(current)});
    double slip=0,scale=0;
    for(std::size_t i=0;i<fluid.size();i++){slip=std::max(slip,norm(fluid[i]-rigid[i]));scale=std::max({scale,norm(fluid[i]),norm(rigid[i])});}
    r.slip=std::max(r.slip,slip);r.divergence=std::max(r.divergence,f->maxDivergence());
    require(slip<=tolerance.absoluteVelocityTolerance+tolerance.relativeTolerance*scale,"Independent endpoint no-slip failure");
    require(r.divergence<=tolerance.maximumDivergence,"Independent endpoint incompressibility failure");
    const double volumeDrift=static_cast<double>(std::abs(volume(*f)-initialVolume)/initialVolume);
    r.relativeVolumeDrift=std::max(r.relativeVolumeDrift,volumeDrift);
  }
  r.time=f->time()-startTime;require(std::abs(r.time-dt*count)<=8e-15*dt*count,"Refinements compared at different physical times");
  r.grain=coupled.grains()[0];const auto dofs=independentFluidDofs(*f);r.fluid=packed(*f,dofs);for(const auto& d:dofs)r.mass.push_back(d.mass);
  r.eta.assign(f->array(FluidArray::eta).begin(),f->array(FluidArray::eta).end());r.pressure.assign(f->array(FluidArray::p).begin(),f->array(FluidArray::p).end());
  r.wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wallStart).count();
  std::cout<<"{\"kind\":\"trajectory\",\"offAxisLateralRotating\":"<<(offAxis?"true":"false")<<",\"dt\":"<<dt<<",\"steps\":"<<r.steps<<",\"physicalTime\":"<<r.time<<",\"conditioningTime\":"<<warm.dt<<",\"refinements\":"<<r.refinements<<",\"wallSeconds\":"<<r.wall<<",\"velocityZ\":"<<r.grain.velocity.z<<",\"centreZ\":"<<r.grain.centre.z<<",\"maxSlip\":"<<r.slip<<",\"maxDivergence\":"<<r.divergence<<",\"maxBodyImpulseDefect\":"<<r.bodyImpulseDefect<<",\"relativeVolumeDrift\":"<<r.relativeVolumeDrift<<"}\n";
  return r;
}
double fluidDifference(const Result& a,const Result& b){
  require(a.fluid.size()==b.fluid.size(),"Different spatial grids");long double sum=0,mass=0;
  for(std::size_t i=0;i<a.fluid.size();i++){const long double delta=a.fluid[i]-b.fluid[i];sum+=a.mass[i]*delta*delta;mass+=a.mass[i];}
  return std::sqrt(static_cast<double>(sum/mass));
}
double bodyDifference(const Result& a,const Result& b){return norm(a.grain.velocity-b.grain.velocity);}
double positionDifference(const Result& a,const Result& b){return norm(a.grain.centre-b.grain.centre);}
void refinement(const char* name,double coarse,double fine,double numericalFloor){
  const double ratio=coarse/fine,order=std::log2(ratio);
  std::cout<<"{\"kind\":\"temporal-refinement\",\"observable\":"<<std::quoted(name)<<",\"coarseDifference\":"<<coarse<<",\"fineDifference\":"<<fine<<",\"ratio\":"<<ratio<<",\"observedOrder\":"<<order<<"}\n";
  require(std::isfinite(order)&&fine>numericalFloor,"Temporal differences are below a meaningful numerical floor");
  require(order>.7&&order<1.3,"Fixed-time refinement does not show the claimed first-order integration");
}
}
int main(int argc,char** argv){
  std::cout<<std::setprecision(17)<<std::unitbuf;
  try{
    require(argc==2,"Usage: check-coupled-convergence prepared-case.fns");const Fixture input(argv[1]);
    const auto coarse=integrate(input,2e-8,8),middle=integrate(input,1e-8,16),fine=integrate(input,5e-9,32);
    // Require differences separated from algebraic/coordinate error floors so
    // a fortuitous ratio between nearly equal doubles cannot certify an order.
    refinement("fluid mass-weighted velocity RMS",fluidDifference(coarse,middle),fluidDifference(middle,fine),1e-12);
    refinement("grain translational velocity",bodyDifference(coarse,middle),bodyDifference(middle,fine),1e-12);
    refinement("grain position",positionDifference(coarse,middle),positionDifference(middle,fine),200*std::numeric_limits<double>::epsilon()*input.config.R);
    require(std::max({coarse.relativeVolumeDrift,middle.relativeVolumeDrift,fine.relativeVolumeDrift})<1e-12,"Closed-vessel volume drift exceeds precision-level bound");
    const auto lateral=integrate(input,2e-8,3,true);
    require(lateral.refinements==0&&lateral.relativeVolumeDrift<1e-12,"Off-axis finite-velocity state failed the unchanged acceptance bounds");
    std::cout<<"{\"summary\":true,\"passed\":true,\"scope\":\"First-order temporal refinement and independent endpoint/body/volume checks on one fixed grid; not spatial convergence or deposition validation\"}\n";
    return 0;
  }catch(const std::exception& e){std::cout<<"{\"summary\":true,\"passed\":false,\"error\":"<<std::quoted(e.what())<<"}\n";return 1;}
}
