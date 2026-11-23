#include "resolved-gap-join.hpp"
#include <iomanip>
#include <iostream>
#include <string>

using namespace faraday_native;
using namespace faraday_native::resolved_gap_flow;
using resolved_gap_join::Pair;
namespace {
unsigned assertions=0;
void require(bool yes,const std::string& message){++assertions;if(!yes)throw std::runtime_error(message);}
template<class F>void refuses(F f,const std::string& message){bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught,message);}
double length(Vec3 x){return std::hypot(x.x,x.y,x.z);}
struct Setup {
  Options leftOptions,rightOptions,wholeOptions;
  resolved_gap::Chart chart=resolved_gap::Chart::sphereWall(1,.05);
  Setup(double dt=0){
    leftOptions.halfWidth=.2;leftOptions.halfWidthEta=.4;leftOptions.centerXi=-.2;
    leftOptions.density=1;leftOptions.dynamicViscosity=.1;leftOptions.timeStep=dt;
    rightOptions=leftOptions;rightOptions.centerXi=.2;
    wholeOptions=leftOptions;wholeOptions.nx=4;wholeOptions.halfWidth=.4;wholeOptions.centerXi=0;
  }
};
Problem arbitraryProblem(){
  Problem p;p.velocityFaceMask=48;p.pressureGradientOffset={.12,-.09,-.13};
  p.force=[](Vec3 x){return Vec3{.12+.1*x.y,-.09+.03*x.z,-.13+.07*x.x};};
  p.boundaryVelocity=[](Vec3 x){return Vec3{.1*x.y,-.1*x.x,0};};
  p.boundaryTraction=[](unsigned face,Vec3 x,Vec3 normal){return -(.23+.02*x.y)*normal+Vec3{.001*face,.03*x.x,0};};
  return p;
}
State arbitraryState(const Pair& pair,const Problem& p){
  State s=pair.initialState(p);
  for(std::size_t i=0;i<s.velocity.size();++i){const auto x=pair.velocityNodes()[i];
    s.velocity[i]={.2+.3*x.y+.1*x.z*x.z,-.1+.2*x.x*x.z,.07+.11*x.x*x.y};}
  for(std::size_t i=0;i<s.pressure.size();++i){const auto x=pair.pressureNodes()[i];s.pressure[i]=.19+.3*x.x-.17*x.y+.04*x.z;}
  return s;
}
struct Difference {double absolute=0,relative=0;};
Difference difference(const std::vector<double>& a,const std::vector<double>& b){
  require(a.size()==b.size(),"assembled dimensions agree");double error=0,scale=0,max=0;
  for(std::size_t i=0;i<a.size();++i){error=std::hypot(error,a[i]-b[i]);scale=std::hypot(scale,b[i]);max=std::max(max,std::abs(a[i]-b[i]));}
  return {max,error/std::max(scale,1e-30)};
}
void checkInterface(const Pair& pair,const State& s){
  double maximumVelocityJump=0,maximumAdvectiveFluxSum=0,maximumOffsetFluxSum=0,absoluteCrossingFlux=0;
  const auto samples=pair.interfaceSamples(s);require(!samples.empty(),"interface quadrature exists");
  for(const auto& sample:samples){const auto& a=sample.left;const auto& b=sample.right;
    require(length(a.point-b.point)<1e-14&&length(a.normal+b.normal)<1e-14&&std::abs(a.weight-b.weight)<1e-14,"interface points weights and opposite normals match");
    maximumVelocityJump=std::max(maximumVelocityJump,length(a.velocity-b.velocity));
    maximumAdvectiveFluxSum=std::max(maximumAdvectiveFluxSum,length(a.advectiveMomentumFlux+b.advectiveMomentumFlux));
    maximumOffsetFluxSum=std::max(maximumOffsetFluxSum,length(dot(s.pressureGradientOffset,a.point)*a.normal+dot(s.pressureGradientOffset,b.point)*b.normal));
    absoluteCrossingFlux+=a.weight*std::abs(a.normalVelocity);
  }
  require(maximumVelocityJump<1e-13,"shared interface velocity is continuous");
  require(maximumAdvectiveFluxSum<1e-13,"internal conservative advective momentum flux cancels");
  require(maximumOffsetFluxSum<1e-13,"internal affine-pressure traction cancels");
  require(absoluteCrossingFlux>1e-6,"interface is traversed by nonzero flow, not an artificial wall");
  std::cout<<"{\"kind\":\"shared-interface\",\"velocityJump\":"<<maximumVelocityJump<<",\"advectiveFluxSum\":"<<maximumAdvectiveFluxSum<<",\"affinePressureFluxSum\":"<<maximumOffsetFluxSum<<",\"absoluteCrossingFlux\":"<<absoluteCrossingFlux<<"}\n";
}
void assembledIdentity(){
  Setup setup(.3);Patch left(setup.chart,setup.leftOptions),right(setup.chart,setup.rightOptions),whole(setup.chart,setup.wholeOptions);Pair pair(left,right);
  const auto p=arbitraryProblem();const auto state=arbitraryState(pair,p);State previous=state;for(auto& u:previous.velocity)u=.83*u;
  require(pair.velocityNodes().size()==whole.velocityNodes().size()&&pair.pressureNodes().size()==whole.pressureNodes().size(),"one shared global lattice equals monolithic lattice");
  require(pair.velocityNodes().size()==225&&pair.pressureNodes().size()==45,"interface nodes counted once");
  const auto joined=pair.linearize(state,p,&previous),reference=whole.linearize(state,p,&previous);
  const auto r=difference(joined.residual,reference.residual),j=difference(joined.matrix,reference.matrix);
  require(r.relative<2e-12&&j.relative<2e-12,"complete nonlinear transient residual and Jacobian match monolithic assembly");
  const auto nv=3*joined.velocityNodes,n=joined.dimension();double adjoint=0,penalty=0;
  for(std::size_t i=0;i<nv;++i)for(std::size_t k=nv;k<n;++k)adjoint=std::max(adjoint,std::abs(joined.matrix[i*n+k]+joined.matrix[k*n+i]));
  for(std::size_t i=nv;i<n;++i)for(std::size_t k=nv;k<n;++k)penalty=std::max(penalty,std::abs(joined.matrix[i*n+k]));
  require(adjoint==0&&penalty==0,"joining preserves exact mixed adjoint and zero pressure penalty");
  std::cout<<"{\"kind\":\"monolithic-operator-identity\",\"unknowns\":"<<n<<",\"residualRelativeDifference\":"<<r.relative<<",\"matrixRelativeDifference\":"<<j.relative<<"}\n";
  checkInterface(pair,state);
}
// Independent Gauss-Legendre integration of the explicit Cartesian sphere
// height; no FEM quadrature, shapes, metrics, or interface maps enter it.
double exactVolume(){
  constexpr unsigned order=24;std::array<double,order> x{},w{};const double pi=std::acos(-1.);
  for(unsigned i=0;i<(order+1)/2;++i){double z=std::cos(pi*(i+.75)/(order+.5)),derivative=0;
    for(unsigned iteration=0;iteration<30;++iteration){double a=1,b=z;for(unsigned k=2;k<=order;++k){const double next=((2*k-1)*z*b-(k-1)*a)/k;a=b;b=next;}derivative=order*(z*b-a)/(z*z-1);const double next=z-b/derivative;if(std::abs(next-z)<1e-16){z=next;break;}z=next;}
    x[i]=-z;x[order-1-i]=z;w[i]=w[order-1-i]=2/((1-z*z)*derivative*derivative);
  }
  double volume=0;for(unsigned i=0;i<order;++i)for(unsigned j=0;j<order;++j){const double r2=.16*(x[i]*x[i]+x[j]*x[j]);volume+=.16*w[i]*w[j]*(.05+r2/(1+std::sqrt(1-r2)));}return volume;
}
void massCount(){
  Setup timed(.3),steady;Pair a(Patch(timed.chart,timed.leftOptions),Patch(timed.chart,timed.rightOptions));
  Pair b(Patch(steady.chart,steady.leftOptions),Patch(steady.chart,steady.rightOptions));
  Problem p;p.force=[](Vec3){return Vec3{};};p.boundaryVelocity=[](Vec3){return Vec3{};};const auto s=a.initialState(p);
  const auto m=a.linearize(s,p,&s),k=b.linearize(s,p);const auto n=m.dimension();double mass=0;
  for(std::size_t i=0;i<m.velocityNodes;++i)for(std::size_t j=0;j<m.velocityNodes;++j)mass+=.3*(m.matrix[(3*i)*n+3*j]-k.matrix[(3*i)*n+3*j]);
  const double volume=exactVolume();require(std::abs(mass-volume)/volume<2e-10,"assembled constant-field mass equals independently integrated physical liquid mass without overlap duplication");
  std::cout<<"{\"kind\":\"single-counted-fluid-mass\",\"assembledMass\":"<<mass<<",\"independentMass\":"<<volume<<",\"relativeDifference\":"<<std::abs(mass-volume)/volume<<"}\n";
}
Problem spin(){
  Problem p;p.velocityFaceMask=48;p.pressureGradientOffset={0,0,-.13};
  p.force=[](Vec3){return Vec3{0,0,-.13};};
  p.boundaryVelocity=[](Vec3 x){return x.z>.025?Vec3{-.2*x.y,.2*x.x,0}:Vec3{};};
  p.boundaryTraction=[](unsigned,Vec3 x,Vec3 n){return .13*x.z*n;};return p;
}
void physicalSolution(){
  Setup setup;Patch left(setup.chart,setup.leftOptions),right(setup.chart,setup.rightOptions),whole(setup.chart,setup.wholeOptions);Pair pair(left,right);const auto p=spin();
  const auto joined=pair.solve(p),reference=whole.solve(p);double uError=0,pError=0,reactionError=0;
  for(std::size_t i=0;i<joined.state.velocity.size();++i)uError=std::max(uError,length(joined.state.velocity[i]-reference.state.velocity[i]));
  for(std::size_t i=0;i<joined.state.pressure.size();++i)pError=std::max(pError,std::abs(joined.state.pressure[i]-reference.state.pressure[i]));
  const auto a=pair.boundaryLoads(joined.state,p),b=whole.boundaryLoads(reference.state,p);
  require(a.size()==b.size(),"exterior reaction node count matches monolithic");
  for(std::size_t i=0;i<a.size();++i){require(a[i].node==b[i].node,"exterior reaction topology matches");reactionError=std::max(reactionError,length(a[i].reaction-b[i].reaction));}
  require(joined.newtonIterations>1&&joined.unknowns==450,"joined solver iterates with every interface DOF free and one global physical pressure");
  require(uError<2e-11&&pError<2e-10&&reactionError<2e-11,"joined nonlinear state and physical reactions match monolithic solution");
  const auto assembled=pair.linearize(joined.state,p);double interfaceReaction=0,work=0;
  for(std::size_t k=1;k<4;++k)for(std::size_t j=1;j<4;++j){const auto node=(k*5+j)*9+4;
    const Vec3 force={assembled.residual[3*node],assembled.residual[3*node+1],assembled.residual[3*node+2]};
    interfaceReaction=std::max(interfaceReaction,length(force));work+=dot(force,joined.state.velocity[node]);}
  require(interfaceReaction<2e-11&&std::abs(work)<2e-11,"shared weak interface reactions and work balance without prescribing an interface velocity");
  // Derivative traces need not be continuous for a C0 finite element field.
  // Measure this discretization error rather than replace it with zero.
  const auto traction=[&](const Field& f,Vec3 normal){Vec3 t=-f.pressure*normal;
    for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)t[i]+=setup.wholeOptions.dynamicViscosity*(f.velocityGradient[i][j]+f.velocityGradient[j][i])*normal[j];
    return t;
  };
  double maximumStressJump=0,monolithicStressJumpDifference=0;
  for(const auto& sample:pair.interfaceSamples(joined.state)){
    const auto coordinate=setup.chart.coordinates(sample.left.point);const double h=2e-8;
    const auto trace=[&](double side,Vec3 normal){
      const auto one=whole.sample(reference.state,side*h,coordinate.y,coordinate.z);
      const auto two=whole.sample(reference.state,side*2*h,coordinate.y,coordinate.z);
      return 2*traction(one,normal)-traction(two,normal);
    };
    const Vec3 jump=sample.left.traction+sample.right.traction;
    // One-sided extrapolation avoids sample() choosing only the element on
    // the positive side of an exactly shared face. This is a diagnostic of
    // the monolithic C0 solution, not a boundary condition imposed on it.
    const Vec3 monolithicJump=trace(-1,sample.left.normal)+trace(1,sample.right.normal);
    maximumStressJump=std::max(maximumStressJump,length(jump));
    monolithicStressJumpDifference=std::max(monolithicStressJumpDifference,length(jump-monolithicJump));
  }
  require(monolithicStressJumpDifference<1e-10,"finite-resolution stress jump equals monolithic one-sided derivative jump");
  checkInterface(pair,joined.state);
  std::cout<<"{\"kind\":\"joined-physical-rotation\",\"newtonIterations\":"<<joined.newtonIterations<<",\"velocityDifference\":"<<uError<<",\"pressureDifference\":"<<pError<<",\"reactionDifference\":"<<reactionError<<",\"weakInterfaceReaction\":"<<interfaceReaction<<",\"interfaceWorkResidual\":"<<work<<",\"finiteElementStressJump\":"<<maximumStressJump<<",\"monolithicStressJumpDifference\":"<<monolithicStressJumpDifference<<"}\n";
}
void closedGaugeAndRefusals(){
  Setup setup;Patch left(setup.chart,setup.leftOptions),right(setup.chart,setup.rightOptions);Pair pair(left,right);
  Problem p;p.force=[](Vec3){return Vec3{0,0,-.13};};p.pressureGradientOffset={0,0,-.13};p.boundaryVelocity=[](Vec3){return Vec3{};};p.pressureGauge=dot(p.pressureGradientOffset,pair.pressureNodes().front());
  const auto rest=pair.solve(p);require(rest.unknowns==233,"closed joined problem fixes exactly one global pressure gauge");
  for(auto u:rest.state.velocity)require(length(u)<1e-13,"joined exact hydrostatic rest");
  p.boundaryVelocity=[](Vec3 x){return Vec3{x.x,0,0};};refuses([&]{pair.solve(p);},"closed incompatible boundary flux is refused");
  refuses([&]{Pair invalid(left,right,{100,128*1024*1024});},"global unknown budget is enforced before maps and dense matrices");
  refuses([&]{Pair invalid(left,right,{2500,1024});},"combined dense matrix budget is enforced");
  auto overlap=setup.rightOptions;overlap.centerXi=.19;refuses([&]{Pair invalid(left,Patch(setup.chart,overlap));},"overlapping patches are refused rather than double-counted");
  auto nonmatching=setup.rightOptions;nonmatching.ny=3;refuses([&]{Pair invalid(left,Patch(setup.chart,nonmatching));},"nonconforming face nodes are refused");
  auto otherMaterial=setup.rightOptions;otherMaterial.density=2;refuses([&]{Pair invalid(left,Patch(setup.chart,otherMaterial));},"different fluid material across same-fluid interface is refused");
  refuses([&]{Pair invalid(left,Patch(resolved_gap::Chart::sphereWall(1,.06),setup.rightOptions));},"different physical chart is refused");
  auto mismatch=rest.state;mismatch.pressureGradientOffset.z=0;refuses([&]{pair.linearize(mismatch,p);},"pressure decomposition mismatch is refused");
}
}
int main(){try{
  std::cout<<std::setprecision(12);assembledIdentity();massCount();physicalSolution();closedGaugeAndRefusals();
  std::cout<<"{\"ok\":true,\"assertions\":"<<assertions<<",\"scope\":\"conforming non-overlapping fixed-chart full NS pair; not joined to vessel mesh or moving grains\"}\n";return 0;
}catch(const std::exception& e){std::cerr<<"resolved-gap-join check failed: "<<e.what()<<'\n';return 1;}}
