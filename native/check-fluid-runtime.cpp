// Native half of check-fluid-runtime.mjs. Fixtures and binaries are temporary.
#include "fluid-runtime.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace faraday_native;

namespace {
static_assert(std::endian::native==std::endian::little,
  "This native/JS byte-parity fixture currently requires a little-endian host");
static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559,
  "Native/JS byte-parity requires IEEE 754 binary64 arithmetic");

void check(bool condition,const char* message){
  if (!condition) throw std::runtime_error(message);
}

template<class Error,class Call> void expectFailure(Call call,const char* message){
  bool refused=false;
  try { call(); } catch (const Error&) { refused=true; }
  check(refused,message);
}

template<class T> T read(std::istream& in){
  T value;
  in.read(reinterpret_cast<char*>(&value),sizeof(value));
  check(bool(in),"short native test fixture");
  return value;
}

template<class T> void write(std::ostream& out,T value){
  out.write(reinterpret_cast<const char*>(&value),sizeof(value));
}

void checkGuards(){
  FluidConfig config{64,128,40,true,.015,.004,1000,1e-6,.0728,
    9.80665,2*std::acos(-1.)/128,1024ULL*1024*1024};
  {
    NativeFluid fluid(config);
    check(fluid.allocatedBytes()>96000000,"allocation did not exceed the old WASM arena");
    expectFailure<std::logic_error>([&]{ NativeFluid other(config); },
      "a second live global fluid core was accepted");
    std::cout << "Dynamic allocation: " << fluid.allocatedBytes() << " bytes\n";
  }
  config.memoryBudgetBytes=1024;
  expectFailure<std::length_error>([&]{ NativeFluid fluid(config); },
    "explicit memory budget was ignored");
  config.memoryBudgetBytes=16ULL*1024*1024;
  expectFailure<std::length_error>([&]{ NativeFluid fluid(config); },
    "core allocation exceeding the remaining memory budget was accepted");
  config.memoryBudgetBytes=1024ULL*1024*1024;
  config.nr=1000000;
  expectFailure<std::length_error>([&]{ NativeFluid fluid(config); },
    "signed-index overflow request was accepted");
  // Constructor refusal must release the singleton and partially allocated core.
  config.nr=4; config.nth=8; config.nz=4; config.dtheta=2*std::acos(-1.)/8;
  NativeFluid recovered(config);
  check(recovered.allocatedBytes()>0,"core could not recover after constructor refusal");
}

// Independent reconstruction of the dependent axis used to expose the former
// regression: a pressure solve converged, then rebuilding the axis added div u.
void enforceAxis(NativeFluid& fluid){
  auto u=fluid.array(FluidArray::u),rf=fluid.array(FluidArray::rf);
  const int nth=fluid.nth(),nz=fluid.nz(),half=nth/2;
  std::vector<double> extrapolated(nth);
  for (int j=0;j<nz;j++){
    for (int k=0;k<nth;k++){
      const double a=u[(nth+k)*nz+j],b=u[(2*nth+k)*nz+j];
      extrapolated[k]=a-rf[1]*(b-a)/(rf[2]-rf[1]);
    }
    for (int k=0;k<nth;k++)
      u[k*nz+j]=.5*(extrapolated[k]-extrapolated[(k+half)%nth]);
  }
}

void checkProjection(NativeFluid& fluid,bool rigid,double dt){
  if (rigid){
    const auto oldMode=fluid.checkpoint();
    fluid.enableRigidProjection();
    expectFailure<std::invalid_argument>([&]{ fluid.restore(oldMode); },
      "checkpoint from a different projection operator was accepted");
  }
  const auto saved=fluid.checkpoint();
  const auto surface=fluid.array(FluidArray::psurf);
  const std::vector<double> savedSurface(surface.begin(),surface.end());

  // External zero u/w satisfy every prescribed boundary row. Canonical v has
  // no prescribed rows, so this must fail specifically because it aliases.
  std::vector<double> zeroU(fluid.array(9).size()),zeroW(fluid.array(11).size());
  expectFailure<std::invalid_argument>([&]{
    fluid.projectHomogeneous(zeroU,fluid.array(10),zeroW);
  },"canonical projection aliases were accepted");
  std::vector<double> overlap(std::max({fluid.array(9).size(),
    fluid.array(10).size(),fluid.array(11).size()}));
  expectFailure<std::invalid_argument>([&]{
    fluid.projectHomogeneous({overlap.data(),fluid.array(9).size()},
      {overlap.data(),fluid.array(10).size()},{overlap.data(),fluid.array(11).size()});
  },"overlapping projection components were accepted");

  std::array<std::vector<double>,3> a,b,pa,pb;
  for (int component=0;component<3;component++){
    a[component].resize(fluid.array(9+component).size());
    b[component].resize(a[component].size());
    for (std::size_t j=0;j<a[component].size();j++){
      if (fluid.velocityMass(component,j)>0){
        a[component][j]=std::sin(double(j)*.37+component);
        b[component][j]=std::cos(double(j)*.21-component);
      }
    }
    pa[component]=a[component]; pb[component]=b[component];
  }
  fluid.projectHomogeneous(pa[0],pa[1],pa[2]);
  fluid.projectHomogeneous(pb[0],pb[1],pb[2]);
  for (const auto* fields:{&pa,&pb}) for (const auto& component:*fields)
    for (double value:component) check(std::isfinite(value),"projection produced a non-finite velocity");
  double lhs=0,rhs=0,norm=0;
  for (int component=0;component<3;component++){
    for (std::size_t j=0;j<a[component].size();j++){
      const double mass=fluid.velocityMass(component,j);
      lhs+=mass*a[component][j]*pb[component][j];
      rhs+=mass*pa[component][j]*b[component][j];
      norm+=mass*(std::abs(a[component][j]*pb[component][j])+
        std::abs(pa[component][j]*b[component][j]));
    }
  }
  check(std::isfinite(lhs) && std::isfinite(rhs) && std::isfinite(norm)
      && norm>0 && std::abs(lhs-rhs)<1e-9*norm,
    "homogeneous pressure projector is not mass self-adjoint");
  const auto before=pa;
  fluid.projectHomogeneous(pa[0],pa[1],pa[2]);
  for (const auto& component:pa) for (double value:component)
    check(std::isfinite(value),"repeated projection produced a non-finite velocity");
  double maximumError=0;
  for (int component=0;component<3;component++)
    for (std::size_t j=0;j<pa[component].size();j++)
      maximumError=std::max(maximumError,std::abs(pa[component][j]-before[component][j]));
  check(maximumError<1e-8,"homogeneous pressure projector is not idempotent");
  const auto after=fluid.checkpoint();
  check(saved.state==after.state && saved.time==after.time,
    "homogeneous pressure projection mutated physical state");
  check(std::equal(savedSurface.begin(),savedSurface.end(),surface.begin()),
    "homogeneous pressure projection mutated prescribed surface pressure");

  if (rigid){
    for (int component=0;component<3;component++)
      std::copy(before[component].begin(),before[component].end(),fluid.array(9+component).begin());
    const double divBefore=fluid.maxDivergence();
    enforceAxis(fluid);
    const double divAfter=fluid.maxDivergence();
    check(divAfter<1e-5 && std::abs(divAfter-divBefore)<1e-10,
      "rigid projection diverges after dependent-axis reconstruction");
    fluid.restore(saved);
    fluid.predictor(dt,9.80665);
    const double predictorBefore=fluid.maxDivergence();
    enforceAxis(fluid);
    const double predictorAfter=fluid.maxDivergence();
    check(predictorAfter<1e-8 && std::abs(predictorAfter-predictorBefore)<1e-10,
      "rigid predictor diverges after dependent-axis reconstruction");
    fluid.restore(saved);
    std::cout << "Axis divergence before/after: " << divBefore << " / " << divAfter
      << "; predictor: " << predictorBefore << " / " << predictorAfter << '\n';
  }
  std::cout << "Projection relative symmetry error: " << std::abs(lhs-rhs)/norm
    << "; idempotence: " << maximumError << '\n';
}

void advance(NativeFluid& fluid,const std::string& mode,double dt,double gravity){
  if (mode=="full") fluid.step(dt,gravity);
  else if (mode=="split") { fluid.predictor(dt,gravity); fluid.finish(dt); }
  else if (mode=="checkpoint"){
    const auto saved=fluid.checkpoint();
    expectFailure<std::logic_error>([&]{ fluid.finish(dt); },
      "surface advance without a pending predictor was accepted");
    auto bad=saved; bad.state[0][0]=NAN;
    expectFailure<std::invalid_argument>([&]{ fluid.restore(bad); },
      "non-finite checkpoint was accepted");
    check(fluid.checkpoint().state==saved.state,"checkpoint validation was not transactional");
    fluid.array(FluidArray::eta)[0]=-fluid.config().h;
    expectFailure<std::runtime_error>([&]{ fluid.refreshMetric(); },
      "surface reaching the floor did not latch a canonical error");
    fluid.restore(saved);
    check(fluid.checkpoint().state==saved.state,"rollback did not recover a latched fluid error");
    fluid.predictor(dt,gravity);
    expectFailure<std::logic_error>([&]{ fluid.predictor(dt,gravity); },
      "a second pending predictor was accepted");
    expectFailure<std::logic_error>([&]{ fluid.step(dt,gravity); },
      "a full step during a pending predictor was accepted");
    expectFailure<std::logic_error>([&]{ fluid.finish(dt/2); },
      "surface advance accepted a different time step");
    fluid.array(FluidArray::u)[30]=123;
    fluid.restore(saved);
    check(fluid.checkpoint().state==saved.state,"fluid rollback did not restore accepted state");
    fluid.predictor(dt,gravity); fluid.finish(dt);
  } else throw std::invalid_argument("unknown native fluid test mode");
}
}

int main(int argc,char** argv){
  try {
    std::cout << std::setprecision(17);
    if (argc==2 && std::string(argv[1])=="guards") { checkGuards(); return 0; }
    check(argc==4,"usage: check-fluid-runtime fixture output mode");
    std::ifstream in(argv[1],std::ios::binary);
    std::ofstream out(argv[2],std::ios::binary);
    check(bool(in)&&bool(out),"cannot open native test fixture/output");
    FluidConfig config;
    config.nr=read<std::int32_t>(in); config.nth=read<std::int32_t>(in); config.nz=read<std::int32_t>(in);
    config.R=read<double>(in); config.h=read<double>(in); config.rho=read<double>(in);
    config.nu=read<double>(in); config.gamma=read<double>(in); config.g=read<double>(in);
    config.dtheta=read<double>(in);
    NativeFluid fluid(config);
    for (int key=0;key<67;key++){
      auto field=fluid.array(key);
      const auto count=read<std::uint32_t>(in);
      check(count==field.size(),"test fixture array length mismatch");
      in.read(reinterpret_cast<char*>(field.data()),field.size_bytes());
      check(bool(in),"short test fixture array");
    }
    fluid.initialize();
    const auto steps=read<std::uint32_t>(in);
    const double dt=read<double>(in);
    const std::string mode=argv[3];
    if (mode=="projection" || mode=="projection-rigid")
      checkProjection(fluid,mode=="projection-rigid",dt);
    else for (unsigned step=0;step<steps;step++) advance(fluid,mode,dt,read<double>(in));
    const auto limits=fluid.stepLimits(7);
    std::cout << "LIMITS " << limits.capillary << ' ' << limits.viscous << ' '
      << limits.advective << ' ' << limits.gravity << ' ' << limits.kRadial << ' '
      << limits.kVertical << ' ' << limits.kAzimuthal << ' ' << fluid.stableStep(7) << '\n';
    write(out,fluid.time());
    for (int key=0;key<67;key++){
      const auto field=fluid.array(key);
      write(out,std::uint32_t(field.size()));
      out.write(reinterpret_cast<const char*>(field.data()),field.size_bytes());
    }
    check(bool(out),"failed to write native test output");
    std::cout << mode << ": t=" << fluid.time() << ", array bytes=" << fluid.allocatedBytes() << '\n';
    return 0;
  } catch (const std::exception& error){
    std::cerr << "Native runtime check failed: " << error.what() << '\n';
    return 1;
  }
}
