#include "rigid-coupler.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace faraday_native;
namespace {
void require(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
struct Fixture {
  FluidConfig config;
  std::vector<RigidGrain> grains;
  std::vector<std::pair<int,std::vector<double>>> fields;
  double dt=0,acceleration=0,omega=0;
  explicit Fixture(const char* path){
    std::ifstream in(path,std::ios::binary);require(bool(in),"Cannot open prepared test fixture");
    const auto bytes=[&](std::size_t n){std::string b(n,'\0');require(bool(in.read(b.data(),n)),"Truncated test fixture");return b;};
    const auto integer=[&](std::size_t n){const auto b=bytes(n);std::uint64_t v=0;for(std::size_t i=0;i<n;i++)v|=std::uint64_t(static_cast<unsigned char>(b[i]))<<(8*i);return v;};
    const auto real=[&](){return std::bit_cast<double>(integer(8));};
    require(bytes(8)=="FNSCASE1"&&integer(4)==1,"Unsupported test fixture");
    config.nr=integer(4);config.nth=integer(4);config.nz=integer(4);require(integer(4)==1,"Fixture must be pinned");config.pinned=true;
    config.memoryBudgetBytes=integer(8);config.R=real();config.h=real();config.rho=real();config.nu=real();config.gamma=real();config.g=real();config.dtheta=real();
    omega=real();acceleration=real();(void)real();dt=real();(void)real();(void)real();(void)real();
    const auto ng=integer(4),na=integer(4),nj=integer(4);(void)bytes(nj);
    for(std::size_t i=0;i<ng;i++){RigidGrain g;g.id=integer(8);g.centre={real(),real(),real()};g.radius=real();g.density=real();g.velocity={real(),real(),real()};g.omega={real(),real(),real()};grains.push_back(g);}
    for(std::size_t i=0;i<na;i++){const int key=integer(4);const auto n=integer(8);std::vector<double> a(n);for(auto& x:a)x=real();fields.emplace_back(key,std::move(a));}
    require(in.peek()==std::char_traits<char>::eof(),"Trailing test fixture bytes");require(grains.size()==1,"The independent oracle requires one declared test sphere");
  }
  std::unique_ptr<NativeFluid> fresh() const {
    auto f=std::make_unique<NativeFluid>(config);for(const auto& [key,a]:fields){auto out=f->array(key);require(out.size()==a.size(),"Fixture field length mismatch");std::copy(a.begin(),a.end(),out.begin());}f->initialize();return f;
  }
};
bool same(const FluidCheckpoint& a,const FluidCheckpoint& b){return a.ownerGeneration==b.ownerGeneration&&a.time==b.time&&a.state==b.state;}
bool sameGrains(const std::vector<RigidGrain>& a,const std::vector<RigidGrain>& b){
  if(a.size()!=b.size())return false;
  for(std::size_t i=0;i<a.size();i++){if(a[i].id!=b[i].id||a[i].radius!=b[i].radius||a[i].density!=b[i].density)return false;for(std::size_t d=0;d<3;d++)if(a[i].centre[d]!=b[i].centre[d]||a[i].velocity[d]!=b[i].velocity[d]||a[i].omega[d]!=b[i].omega[d])return false;}return true;
}
double maxAbs(const std::vector<double>& a){double v=0;for(double x:a)v=std::max(v,std::abs(x));return v;}
double maxDiff(const std::vector<double>& a,const std::vector<double>& b){require(a.size()==b.size(),"Comparison lengths differ");double v=0;for(std::size_t i=0;i<a.size();i++)v=std::max(v,std::abs(a[i]-b[i]));return v;}
std::vector<double> pack(const NativeFluid& f,const std::vector<FluidDof>& dofs){std::vector<double> a;for(const auto& d:dofs)a.push_back(f.array(9+d.family)[d.index]);return a;}
struct ProjectedColumn {std::vector<double> fluid,pressure;Six body{};};
ProjectedColumn projectColumn(NativeFluid& f,const std::vector<FluidDof>& dofs,const RigidInterpolation& j,std::size_t column,double excessMass,double excessInertia){
  std::vector<Vec3> unit(j.markers().size());unit[column/3][column%3]=1;
  auto a=j.spreadTranspose(unit);std::vector<double> u(f.array(9).size()),v(f.array(10).size()),w(f.array(11).size());
  for(std::size_t i=0;i<a.size();i++){const auto& d=dofs[i];(d.family==0?u:d.family==1?v:w)[d.index]=a[i]/d.mass;}
  f.projectHomogeneous(u,v,w,1e-13);ProjectedColumn out;
  for(const auto& d:dofs)out.fluid.push_back((d.family==0?u:d.family==1?v:w)[d.index]);
  out.pressure.assign(f.lastProjectionPotential().begin(),f.lastProjectionPotential().end());
  out.body=j.spreadKTranspose(unit)[0];for(std::size_t d=0;d<6;d++)out.body[d]/=d<3?excessMass:excessInertia;
  return out;
}
// A direct dense Cholesky solve of the independently assembled kinetic-energy
// Gram matrix. No call into the production Schur/PCG implementation is made.
std::vector<double> denseSolve(std::vector<double> a,std::vector<double> b){
  const std::size_t n=b.size();require(a.size()==n*n,"Invalid dense matrix");
  for(std::size_t i=0;i<n;i++)for(std::size_t j=0;j<=i;j++){
    double sum=a[i*n+j];for(std::size_t k=0;k<j;k++)sum-=a[i*n+k]*a[j*n+k];
    if(i==j){require(sum>0&&std::isfinite(sum),"Independent constraint Gram matrix lost positive definiteness");a[i*n+i]=std::sqrt(sum);}else a[i*n+j]=sum/a[j*n+j];
  }
  for(std::size_t i=0;i<n;i++){for(std::size_t j=0;j<i;j++)b[i]-=a[i*n+j]*b[j];b[i]/=a[i*n+i];}
  for(std::size_t ii=n;ii-->0;){for(std::size_t j=ii+1;j<n;j++)b[ii]-=a[j*n+ii]*b[j];b[ii]/=a[ii*n+ii];}return b;
}
void denseOracle(const Fixture& input){
  auto f=input.fresh();CouplingOptions o;
  // Algebra-only oracle: seven volume markers make the dense direct solve
  // inexpensive. The trajectory test below separately uses default collocation.
  o.interpolation.markerSpacingFactor=4;
  RigidCoupler coupled(*f,input.grains,o);
  const auto dofs=independentFluidDofs(*f);RigidInterpolation j(dofs,input.grains,o.interpolation);
  require(j.markers().size()==7,"Algebra oracle fixture no longer has seven markers");
  const double me=(input.grains[0].density-input.config.rho)*input.grains[0].volume();const double ie=.4*me*input.grains[0].radius*input.grains[0].radius;
  const std::size_t n=3*j.markers().size();std::vector<ProjectedColumn> columns;for(std::size_t i=0;i<n;i++)columns.push_back(projectColumn(*f,dofs,j,i,me,ie));
  std::vector<double> matrix(n*n);
  for(std::size_t i=0;i<n;i++)for(std::size_t k=0;k<=i;k++){
    double sum=0;for(std::size_t d=0;d<dofs.size();d++)sum+=dofs[d].mass*columns[i].fluid[d]*columns[k].fluid[d];
    for(std::size_t d=0;d<6;d++)sum+=(d<3?me:ie)*columns[i].body[d]*columns[k].body[d];
    matrix[i*n+k]=matrix[k*n+i]=sum;
  }
  const double gravity=input.config.g+input.acceleration;const auto before=f->checkpoint();
  const auto result=coupled.step(input.dt,gravity);
  require(result.dt>0&&result.dt<=input.dt,"Oracle step did not advance");
  std::vector<double> rhs(n);for(std::size_t i=2;i<n;i+=3)rhs[i]=result.dt*gravity;
  const auto lambda=denseSolve(matrix,rhs);
  std::vector<double> expectedFluid(dofs.size()),expectedPressure(f->array(13).size());Six expectedBody{};expectedBody[2]=-result.dt*gravity;
  for(std::size_t c=0;c<n;c++){
    for(std::size_t d=0;d<dofs.size();d++)expectedFluid[d]-=lambda[c]*columns[c].fluid[d];
    for(std::size_t d=0;d<6;d++)expectedBody[d]+=lambda[c]*columns[c].body[d];
    for(std::size_t d=0;d<expectedPressure.size();d++)expectedPressure[d]-=input.config.rho*lambda[c]*columns[c].pressure[d]/result.dt;
  }
  const auto actualFluid=pack(*f,dofs);const Six actualBody=rigidVelocity(coupled.grains()[0]);
  const double fluidError=maxDiff(expectedFluid,actualFluid);double bodyError=0;for(std::size_t d=0;d<6;d++)bodyError=std::max(bodyError,std::abs(expectedBody[d]-actualBody[d]));
  const std::vector<double> actualPressure(f->array(13).begin(),f->array(13).end());const double pressureError=maxDiff(expectedPressure,actualPressure),pressureScale=maxAbs(expectedPressure);
  // Bounds reflect the production absolute velocity tolerance. They are tight
  // enough to reject missing pressure, reversed reaction, or full-mass counting.
  require(fluidError<2e-12,"Fluid correction differs from direct kinetic-energy oracle");
  require(bodyError<2e-9,"Rigid velocity differs from direct kinetic-energy oracle");
  require(pressureScale>1e-6&&pressureError<2e-5*pressureScale+1e-8,"Reduced pressure omits or reverses the grain reaction");
  require(!same(before,f->checkpoint()),"Nonzero constrained impulse did not change the fluid state");
  std::cout<<"{\"metric\":\"direct-energy-gram-oracle\",\"constraints\":"<<n<<",\"fluidError\":"<<fluidError<<",\"bodyError\":"<<bodyError<<",\"pressureError\":"<<pressureError<<",\"pressureScale\":"<<pressureScale<<"}\n";
}
}

int main(int argc,char** argv){
  std::cout<<std::setprecision(17)<<std::unitbuf;if(argc<2||argc>3){std::cerr<<"Usage: check-coupler prepared-verification-case.fns [test-name-substring]\n";return 2;}
  try{
    Fixture input(argv[1]);std::size_t passed=0,failed=0;
    const auto test=[&](const char* name,const std::function<void()>& run){if(argc==3&&std::string(name).find(argv[2])==std::string::npos)return;try{run();passed++;std::cout<<"{\"test\":"<<std::quoted(name)<<",\"passed\":true}\n";}catch(const std::exception& e){failed++;std::cout<<"{\"test\":"<<std::quoted(name)<<",\"passed\":false,\"error\":"<<std::quoted(e.what())<<"}\n";}};
    test("zero grains preserve the canonical full step exactly",[&]{
      auto f=input.fresh();const auto initial=f->checkpoint();f->step(input.dt,input.config.g+input.acceleration);const auto expected=f->checkpoint();f->restore(initial);RigidCoupler coupled(*f,{});const auto result=coupled.step(input.dt,input.config.g+input.acceleration);
      require(same(expected,f->checkpoint()),"Zero-grain branch differs from canonical state/time");require(result.markerCount==0&&!f->rigidProjectionEnabled(),"Zero-grain path enabled rigid physics");
    });
    test("unresolved construction refusal does not mutate fluid state or mode",[&]{
      auto f=input.fresh();const auto before=f->checkpoint();auto grains=input.grains;grains[0].radius/=100;bool refused=false;
      try{RigidCoupler c(*f,grains);}catch(const std::runtime_error&){refused=true;}
      require(refused,"Underresolved grain was accepted");require(same(before,f->checkpoint())&&!f->rigidProjectionEnabled(),"Failed construction mutated the fluid or projection mode");
    });
    test("invalid geometry construction refusal does not mutate fluid state or mode",[&]{
      auto f=input.fresh();const auto before=f->checkpoint();auto grains=input.grains;grains[0].centre.z=grains[0].radius*.5;bool refused=false;
      try{RigidCoupler c(*f,grains);}catch(const std::exception&){refused=true;}
      require(refused,"Overlapping grain was accepted");require(same(before,f->checkpoint())&&!f->rigidProjectionEnabled(),"Geometry refusal mutated the fluid or projection mode");
    });
    test("overbudget construction refusal does not mutate fluid state or mode",[&]{
      auto f=input.fresh();const auto before=f->checkpoint();CouplingOptions o;o.interpolation.workingMemoryBudgetBytes=1;bool refused=false;
      try{RigidCoupler c(*f,input.grains,o);}catch(const std::length_error&){refused=true;}
      require(refused,"One-byte interpolation budget was ignored");require(same(before,f->checkpoint())&&!f->rigidProjectionEnabled(),"Resource refusal mutated the fluid or projection mode");
    });
    test("joint constraint nonconvergence rolls back all accepted state",[&]{
      auto f=input.fresh();CouplingOptions o;o.maximumIterations=1;RigidCoupler coupled(*f,input.grains,o);const auto before=f->checkpoint();const auto grains=coupled.grains();bool refused=false;
      try{coupled.step(input.dt,input.config.g+input.acceleration);}catch(const std::runtime_error&){refused=true;}
      require(refused,"One-iteration nontrivial constraint unexpectedly accepted");require(same(before,f->checkpoint())&&sameGrains(grains,coupled.grains()),"Failed constraint changed accepted state");
    });
    test("post-advance rejection restores surface pressure time and grains",[&]{
      auto f=input.fresh();CouplingOptions o;o.maximumDivergence=1e-300;o.maximumEventRefinements=1;RigidCoupler coupled(*f,input.grains,o);const auto before=f->checkpoint();const auto grains=coupled.grains();bool refused=false;
      try{coupled.step(input.dt,input.config.g+input.acceleration);}catch(const std::runtime_error&){refused=true;}
      require(refused,"Deliberately unattainable final divergence bound was ignored");require(same(before,f->checkpoint())&&sameGrains(grains,coupled.grains()),"Rejected advanced candidate escaped rollback");
    });
    test("nonzero coupled momentum and reduced pressure match independent dense energy oracle",[&]{denseOracle(input);});
    test("default resolved sphere advances with two-way reactions and measured residuals",[&]{
      auto f=input.fresh();RigidCoupler coupled(*f,input.grains);const double initialZ=input.grains[0].centre.z;double minimumDt=std::numeric_limits<double>::infinity(),maximumSlip=0,maximumDivergence=0;std::size_t refinements=0;
      const auto start=std::chrono::steady_clock::now();
      for(unsigned step=0;step<20;step++){
        const auto old=coupled.grains()[0];const double gravity=input.config.g+input.acceleration*std::cos(input.omega*f->time());const auto result=coupled.step(input.dt,gravity);const auto& now=coupled.grains()[0];
        require(result.dt>0&&result.time==f->time(),"Physical clock did not advance coherently");require(result.markerCount>=27,"Default rigid-volume collocation was replaced");
        const double me=(now.density-input.config.rho)*now.volume();const double momentum=me*(now.velocity.z-old.velocity.z+result.dt*gravity);
        require(std::abs(momentum-result.rigidImpulse[0][2])<1e-25+1e-8*std::abs(momentum),"Excess-mass body equation does not balance the reported fluid reaction");
        require(result.slipResidual<=1e-13+1e-9*norm(now.velocity),"Accepted rigid slip exceeds default tolerance");require(result.divergence<=1e-8,"Accepted velocity is not divergence constrained");
        minimumDt=std::min(minimumDt,result.dt);maximumSlip=std::max(maximumSlip,result.slipResidual);maximumDivergence=std::max(maximumDivergence,result.divergence);refinements+=result.eventRefinements;
      }
      const auto& grain=coupled.grains()[0];require(grain.velocity.z<0&&grain.centre.z<initialZ,"Heavy submerged grain did not begin settling");
      const auto dofs=independentFluidDofs(*f);require(maxAbs(pack(*f,dofs))>0,"Grain moved without any fluid response");
      const std::vector<double> pressure(f->array(13).begin(),f->array(13).end());require(maxAbs(pressure)>0,"Grain moved without a pressure reaction");
      const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      std::cout<<"{\"metric\":\"default-initial-settling\",\"steps\":20,\"physicalTime\":"<<f->time()<<",\"wallSeconds\":"<<wall<<",\"minimumDt\":"<<minimumDt<<",\"refinements\":"<<refinements<<",\"displacementMetres\":"<<initialZ-grain.centre.z<<",\"velocityZ\":"<<grain.velocity.z<<",\"maxSlip\":"<<maximumSlip<<",\"maxDivergence\":"<<maximumDivergence<<"}\n";
    });
    test("finite settling speed records actual progress without projecting grain positions",[&]{
      auto f=input.fresh();auto grains=input.grains;grains[0].velocity.z=-.001;RigidCoupler coupled(*f,grains);const double initialZ=grains[0].centre.z;double minDt=input.dt;std::size_t refinements=0;const auto start=std::chrono::steady_clock::now();
      for(unsigned i=0;i<3;i++){const auto old=coupled.grains()[0];const auto r=coupled.step(input.dt,input.config.g);const auto& now=coupled.grains()[0];require(now.centre.z==old.centre.z+r.dt*now.velocity.z,"Accepted centre has been projected or clipped");require(r.dt>0&&r.time==f->time(),"Finite-speed case stopped advancing");minDt=std::min(minDt,r.dt);refinements+=r.eventRefinements;}
      std::cout<<"{\"metric\":\"finite-speed-settling\",\"steps\":3,\"physicalTime\":"<<f->time()<<",\"wallSeconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<",\"minimumDt\":"<<minDt<<",\"refinements\":"<<refinements<<",\"displacementMetres\":"<<initialZ-coupled.grains()[0].centre.z<<"}\n";
    });
    test("supported floor sphere balances weight contact and hydrodynamic impulses",[&]{
      auto f=input.fresh();auto grains=input.grains;grains[0].centre.z=grains[0].radius;CouplingOptions o;o.friction=.5;o.restitution=.3;RigidCoupler coupled(*f,grains,o);
      const double initialZ=grains[0].centre.z;const auto initial=f->checkpoint();const auto start=std::chrono::steady_clock::now();double maximumFluidSpeed=0,maximumBodySpeed=0;
      for(unsigned i=0;i<2;i++){
        const auto before=coupled.grains()[0];const auto r=coupled.step(input.dt,input.config.g);const auto& now=coupled.grains()[0];
        require(r.contacts==1&&r.contactImpulse.size()==1,"Supported grain did not receive a floor-contact response");
        require(r.contactImpulse[0][2]>0,"Floor impulse does not oppose the excess weight");
        const double me=(now.density-input.config.rho)*now.volume(),momentum=me*(now.velocity.z-before.velocity.z+r.dt*input.config.g),impulse=r.rigidImpulse[0][2]+r.contactImpulse[0][2];
        require(std::abs(momentum-impulse)<1e-25+1e-8*std::abs(momentum),"Supported body momentum omits contact-induced hydrodynamic impulse");
        require(now.centre.z==before.centre.z+r.dt*now.velocity.z,"Supported centre has been projected");
        require(now.centre.z>=initialZ,"Supported sphere penetrated floor");
        // Gravity and the normal support are collinear body impulses. Their
        // exact constrained responses cancel, leaving the fluid at rest.
        maximumFluidSpeed=std::max(maximumFluidSpeed,maxAbs(pack(*f,independentFluidDofs(*f))));
        maximumBodySpeed=std::max(maximumBodySpeed,norm(now.velocity));
      }
      require(f->time()>initial.time,"Supported contact did not advance time");
      require(maximumFluidSpeed<1e-12&&maximumBodySpeed<1e-12,"A supported sphere at rest generated unexplained flow or rigid motion");
      std::cout<<"{\"metric\":\"supported-floor-contact\",\"physicalTime\":"<<f->time()<<",\"wallSeconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<",\"displacementMetres\":"<<coupled.grains()[0].centre.z-initialZ<<",\"maximumFluidSpeed\":"<<maximumFluidSpeed<<",\"maximumBodySpeed\":"<<maximumBodySpeed<<"}\n";
    });
    std::cout<<"{\"summary\":true,\"passed\":"<<passed<<",\"failed\":"<<failed<<"}\n";return failed?1:0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
