#include "resolved-gap-flow.hpp"
#include <iomanip>
#include <iostream>
#include <string>

using namespace faraday_native;
using namespace faraday_native::resolved_gap_flow;
namespace {
unsigned assertions=0;
void require(bool value,const std::string& message){++assertions;if(!value)throw std::runtime_error(message);}
double vectorLength(Vec3 v){return std::hypot(v.x,v.y,v.z);}
template<class F>void refuses(F f,const std::string& message){bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught,message);}
Options options(unsigned n,double lengthScale=1,double velocityScale=1,double densityScale=1){
  Options o;o.nx=o.ny=o.nz=n;o.halfWidth=.4*lengthScale;
  o.density=densityScale;o.dynamicViscosity=.1*densityScale*velocityScale*lengthScale;
  o.referenceVelocity=velocityScale;o.nonlinearTolerance=1e-11;return o;
}
Problem spin(const Patch& patch,double angularVelocity){
  const auto frame=patch.chart().frame();const double gap=patch.chart().gap();Problem p;
  p.velocityFaceMask=16|32;p.force=[](Vec3){return Vec3{};};
  p.boundaryTraction=[](unsigned,Vec3,Vec3){return Vec3{};};
  p.boundaryVelocity=[frame,gap,angularVelocity](Vec3 point){
    const Vec3 r=point-frame.origin;
    return dot(r,frame.z)>gap/2?angularVelocity*cross(frame.z,r):Vec3{};
  };return p;
}
struct Balance {double capTorque=0,weakCapTorque=0,weakTorqueDefect=0,weakForceDefect=0,netFlux=0,advection=0;};
Balance balances(const Patch& patch,const SolveResult& solved,const Problem& problem){
  Balance b;const auto frame=patch.chart().frame();Vec3 advForce{},advTorque{},reaction{},reactionTorque{};
  for(const auto& s:patch.boundarySamples(solved.state)){
    const Vec3 r=s.point-frame.origin;
    advForce=advForce+s.weight*s.advectiveMomentumFlux;
    advTorque=advTorque+s.weight*cross(r,s.advectiveMomentumFlux);
    b.netFlux+=s.weight*s.normalVelocity;b.advection+=s.weight*dot(s.advectiveMomentumFlux,s.advectiveMomentumFlux);
    if(s.face==5)b.capTorque-=s.weight*dot(frame.z,cross(r,s.traction));
  }
  for(const auto& load:patch.boundaryLoads(solved.state,problem)){
    const Vec3 r=patch.velocityNodes()[load.node]-frame.origin;
    reaction=reaction+load.reaction;reactionTorque=reactionTorque+cross(r,load.reaction);
    if(dot(r,frame.z)>patch.chart().gap()/2)b.weakCapTorque-=dot(frame.z,cross(r,load.reaction));
  }
  b.weakForceDefect=vectorLength(reaction-advForce);
  b.weakTorqueDefect=std::abs(dot(frame.z,reactionTorque-advTorque));
  return b;
}
void hydrostatic(){
  Patch patch(resolved_gap::Chart::sphereWall(1,.05),options(2));Problem p;
  p.velocityFaceMask=16|32;p.boundaryVelocity=[](Vec3){return Vec3{};};
  p.force=[](Vec3){return Vec3{.13,-.09,0};};
  p.boundaryTraction=[](unsigned,Vec3 x,Vec3 n){return -(.23+.13*x.x-.09*x.y)*n;};
  p.pressureGauge=-99; // Ignored: physical traction fixes pressure instead.
  const auto result=patch.solve(p);double velocityError=0,pressureError=0;
  for(auto u:result.state.velocity)velocityError=std::max(velocityError,vectorLength(u));
  for(std::size_t i=0;i<result.state.pressure.size();++i){const auto x=patch.pressureNodes()[i];pressureError=std::max(pressureError,std::abs(result.state.pressure[i]-(.23+.13*x.x-.09*x.y)));}
  require(velocityError<2e-9,"traction hydrostatic equilibrium has no spurious velocity");
  require(pressureError<2e-8,"traction determines absolute pressure without gauge pinning");
  require(result.unknowns==3*5*5*3+27,"open faces retain velocity and all pressure unknowns");
  std::cout<<"{\"kind\":\"mixed-boundary-hydrostatic\",\"maximumVelocity\":"<<velocityError<<",\"maximumPressureError\":"<<pressureError<<"}\n";
}
void incompatibleBoundary(){
  Patch patch(resolved_gap::Chart::sphereWall(1,.05),options(2));Problem p;
  p.force=[](Vec3){return Vec3{};};p.boundaryVelocity=[](Vec3 x){return Vec3{x.x,0,0};};
  std::string fluxError;try{patch.solve(p);}catch(const std::exception& e){fluxError=e.what();}
  require(fluxError.find("violates an incompressibility equation")!=std::string::npos,"incompatible total flux is specifically rejected by the full continuity check");
  p.velocityFaceMask=48;refuses([&]{patch.solve(p);},"unspecified traction data must be refused");
}
void verticalGravity(){
  const Vec3 gradient{0,0,-.13};
  Patch patch(resolved_gap::Chart::sphereWall(1,.05),options(2));Problem p;
  p.velocityFaceMask=48;p.pressureGradientOffset=gradient;
  p.force=[gradient](Vec3){return gradient;};p.boundaryVelocity=[](Vec3){return Vec3{};};
  p.boundaryTraction=[gradient](unsigned,Vec3 x,Vec3 n){return -(.23+dot(gradient,x))*n;};
  const auto solved=patch.solve(p);double velocityError=0,pressureError=0;
  for(auto u:solved.state.velocity)velocityError=std::max(velocityError,vectorLength(u));
  for(double x:{-.3,-.1,.2,.35})for(double y:{-.32,-.08,.23})for(double s:{.1,.4,.8}){
    const auto point=patch.chart().metric(x,y,s).point;const auto field=patch.sample(solved.state,x,y,s);
    pressureError=std::max(pressureError,std::abs(field.pressure-(.23+dot(gradient,point))));
    require(vectorLength(field.pressureGradient-gradient)<2e-8,"physical vertical pressure gradient is reconstructed");
  }
  require(velocityError<2e-9&&pressureError<2e-8,"curved-gap vertical gravity balances with the explicit pressure decomposition");
  // Add the same conservative force and pressure to a nonzero full-NS flow.
  // Its velocity and reduced pressure must be unchanged, while reported
  // physical traction must acquire -phi*n. This catches simply deleting g.
  const auto baseProblem=spin(patch,.2);const auto base=patch.solve(baseProblem);auto gravity=baseProblem;
  gravity.pressureGradientOffset={.03,-.07,-.13};const auto g=gravity.pressureGradientOffset;
  gravity.force=[g](Vec3){return g;};gravity.boundaryTraction=[g](unsigned,Vec3 x,Vec3 n){return -dot(g,x)*n;};
  const auto shifted=patch.solve(gravity);double velocityDifference=0,coefficientDifference=0;
  for(std::size_t i=0;i<base.state.velocity.size();++i)velocityDifference=std::max(velocityDifference,vectorLength(base.state.velocity[i]-shifted.state.velocity[i]));
  for(std::size_t i=0;i<base.state.pressure.size();++i)coefficientDifference=std::max(coefficientDifference,std::abs(base.state.pressure[i]-shifted.state.pressure[i]));
  require(velocityDifference<2e-12&&coefficientDifference<2e-11,"conservative force and exact pressure shift preserve the full nonlinear flow");
  const auto before=patch.boundarySamples(base.state),after=patch.boundarySamples(shifted.state);
  for(std::size_t i=0;i<before.size();++i)require(vectorLength(after[i].traction-before[i].traction+dot(g,before[i].point)*before[i].normal)<2e-11,"physical boundary traction retains hydrostatic pressure");
  // On the cap, the integrated hydrostatic axial force must remain in the
  // physical weak reaction, despite using reduced pressure in the solve.
  Vec3 physicalTractionChange{},weakReactionChange{};
  for(std::size_t i=0;i<before.size();++i)if(before[i].face==5)physicalTractionChange=physicalTractionChange+before[i].weight*(after[i].traction-before[i].traction);
  const auto oldLoad=patch.boundaryLoads(base.state,baseProblem),newLoad=patch.boundaryLoads(shifted.state,gravity);
  for(std::size_t i=0;i<oldLoad.size();++i)if(patch.velocityNodes()[oldLoad[i].node].z>patch.chart().gap()/2)weakReactionChange=weakReactionChange+newLoad[i].reaction-oldLoad[i].reaction;
  require(vectorLength(weakReactionChange-physicalTractionChange)<2e-11,"hydrostatic load is retained in physical grain boundary reactions");
  std::cout<<"{\"kind\":\"vertical-gravity-pressure-decomposition\",\"maximumVelocity\":"<<velocityError<<",\"maximumPressureError\":"<<pressureError
    <<",\"nonlinearVelocityShiftError\":"<<velocityDifference<<",\"physicalReactionShiftError\":"<<vectorLength(weakReactionChange-physicalTractionChange)<<"}\n";
}
void physicalRotation(){
  std::vector<Balance> sequence;
  for(unsigned n:{2u,3u,4u}){
    Patch patch(resolved_gap::Chart::sphereWall(1,.05),options(n));const auto p=spin(patch,.2);
    const auto solved=patch.solve(p);const auto b=balances(patch,solved,p);sequence.push_back(b);
    require(solved.newtonIterations>1,"finite Reynolds number physical sphere rotation needs nonlinear iteration");
    require(b.capTorque<0&&b.weakCapTorque<0,"computed fluid torque opposes imposed sphere rotation");
    require(b.advection>1e-12,"physical rotating-sphere problem retains nonzero advective momentum transport");
    require(b.weakForceDefect<1e-10,"physical full-NS total weak force balance");
    require(b.weakTorqueDefect<1e-10,"physical full-NS axial weak torque balance");
    require(solved.maximumContinuityMoment/solved.continuityResidualScale<1e-10,"all mixed-boundary incompressibility equations converge");
    std::cout<<"{\"kind\":\"rotating-spherical-cap\",\"elementsPerAxis\":"<<n<<",\"unknowns\":"<<solved.unknowns<<",\"newtonIterations\":"<<solved.newtonIterations
      <<",\"capTorque\":"<<b.capTorque<<",\"weakCapTorque\":"<<b.weakCapTorque<<",\"weakForceDefect\":"<<b.weakForceDefect<<",\"weakTorqueDefect\":"<<b.weakTorqueDefect
      <<",\"boundaryFlux\":"<<b.netFlux<<",\"boundaryAdvectionSquareIntegral\":"<<b.advection<<"}\n";
  }
  require(std::abs(sequence[2].capTorque-sequence[2].weakCapTorque)<std::abs(sequence[0].capTorque-sequence[0].weakCapTorque),"stress torque approaches independently assembled weak reaction torque under refinement");
  require(std::abs(sequence[2].capTorque-sequence[1].capTorque)<std::abs(sequence[1].capTorque-sequence[0].capTorque),"physical cap torque differences decrease under refinement");
}
void unitSimilarity(){
  // Same dimensionless full-NS problem in reference units and SI units for
  // a 200 micrometre sphere in water (rho=1000 kg/m^3, mu=.001 Pa s).
  // A deliberately slow but nonzero rotation exposes false convergence from
  // a unit-sized force residual floor. No measured beach-sand input is claimed.
  constexpr double lengthScale=1e-4,velocityScale=.1,densityScale=1000,omega=.0002;
  Patch reference(resolved_gap::Chart::sphereWall(1,.05),options(2));const auto pr=spin(reference,omega);const auto base=reference.solve(pr);
  Patch physical(resolved_gap::Chart::sphereWall(lengthScale,.05*lengthScale),options(2,lengthScale,velocityScale,densityScale));
  const auto pp=spin(physical,omega*velocityScale/lengthScale);const auto si=physical.solve(pp);
  double velocityDifference=0,pressureDifference=0;
  for(std::size_t i=0;i<base.state.velocity.size();++i)velocityDifference=std::max(velocityDifference,vectorLength(si.state.velocity[i]/velocityScale-base.state.velocity[i]));
  const double pressureScale=densityScale*velocityScale*velocityScale;
  for(std::size_t i=0;i<base.state.pressure.size();++i)pressureDifference=std::max(pressureDifference,std::abs(si.state.pressure[i]/pressureScale-base.state.pressure[i]));
  require(si.newtonIterations>0,"micrometre SI problem must actually solve rather than accept its zero interior initial state");
  require(velocityDifference<2e-12&&pressureDifference<2e-11,"dimensionless and physical SI solutions agree after unit conversion");
  const auto b=balances(reference,base,pr),s=balances(physical,si,pp);
  const double torqueScale=pressureScale*lengthScale*lengthScale*lengthScale;
  require(std::abs(s.capTorque/torqueScale-b.capTorque)<2e-12,"computed hydrodynamic torque obeys exact unit similarity");
  std::cout<<"{\"kind\":\"micrometre-water-unit-similarity\",\"siNewtonIterations\":"<<si.newtonIterations<<",\"velocityDifference\":"<<velocityDifference
    <<",\"pressureDifference\":"<<pressureDifference<<",\"siCapTorqueNm\":"<<s.capTorque<<"}\n";
}
}
int main(){try{
  std::cout<<std::setprecision(12);hydrostatic();incompatibleBoundary();verticalGravity();physicalRotation();unitSimilarity();
  std::cout<<"{\"ok\":true,\"assertions\":"<<assertions<<",\"scope\":\"physical fixed-gap rotating cap with open traction sides; not whole-sphere drag or a joined moving vessel\"}\n";
  return 0;
}catch(const std::exception& e){std::cerr<<"resolved-gap-boundary check failed: "<<e.what()<<'\n';return 1;}}
