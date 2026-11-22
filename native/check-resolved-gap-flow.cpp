#include "resolved-gap-flow.hpp"
#include <iomanip>
#include <iostream>
#include <string>

using namespace faraday_native;
using namespace faraday_native::resolved_gap_flow;
namespace {
unsigned assertions=0;
void require(bool value,const std::string& message){++assertions;if(!value)throw std::runtime_error(message);}
template<class F> void refuses(F operation,const std::string& name){bool caught=false;try{operation();}catch(const std::exception&){caught=true;}require(caught,name);}
double length(Vec3 v){return std::hypot(v.x,v.y,v.z);}
// Independent Cartesian polynomial derivatives via second-order forward
// differentiation. No mapped FEM shape function enters the exact solution.
struct Jet {double value=0;Vec3 gradient{};double laplacian=0;Jet()=default;Jet(double v):value(v){};};
Jet operator+(Jet a,Jet b){Jet c;c.value=a.value+b.value;c.gradient=a.gradient+b.gradient;c.laplacian=a.laplacian+b.laplacian;return c;}
Jet operator-(Jet a,Jet b){Jet c;c.value=a.value-b.value;c.gradient=a.gradient-b.gradient;c.laplacian=a.laplacian-b.laplacian;return c;}
Jet operator*(Jet a,Jet b){Jet c;c.value=a.value*b.value;c.gradient=a.value*b.gradient+b.value*a.gradient;c.laplacian=a.value*b.laplacian+b.value*a.laplacian+2*dot(a.gradient,b.gradient);return c;}
struct Exact {std::array<Jet,3> u;Jet p;};
Exact exact(Vec3 point){
  Jet x(point.x),y(point.y),z(point.z);x.gradient.x=1;y.gradient.y=1;z.gradient.z=1;
  // Curl of a polynomial vector potential plus constant velocity; all three
  // components vary in all three physical directions and div(u) is zero.
  return {{Jet(.24)+Jet(8)*(Jet(2)*x*x*x*y*z+Jet(.2)*x*z*z*z-Jet(.6)*x*x*y*y*y*z),
    Jet(.1)+Jet(8)*(Jet(-3)*x*x*y*y*z-Jet(.2)*y*z*z*z),
    Jet(.07)+Jet(4.8)*x*y*y*y*z*z},
    Jet(.2)*x*x+Jet(.3)*y*z+Jet(.1)*z*z*z+Jet(.17)*x-Jet(.08)*y+Jet(.04)*z};
}
Vec3 velocity(const Exact& e){return {e.u[0].value,e.u[1].value,e.u[2].value};}
Vec3 forcing(Vec3 point,const Options& o,double previousScale){
  const auto e=exact(point);const Vec3 u=velocity(e);Vec3 f=e.p.gradient;
  const double temporal=o.timeStep>0?(1-previousScale)/o.timeStep:0;
  for(unsigned a=0;a<3;++a)f[a]+=o.density*(temporal*u[a]+dot(u,e.u[a].gradient))-o.dynamicViscosity*e.u[a].laplacian;
  return f;
}
Problem problem(const Patch& patch,double previousScale=1){
  const auto o=patch.options();Problem p;
  p.boundaryVelocity=[](Vec3 x){return velocity(exact(x));};
  p.force=[o,previousScale](Vec3 x){return forcing(x,o,previousScale);};
  p.pressureGauge=exact(patch.pressureNodes().front()).p.value;return p;
}
const std::array<double,5> gx{.0469100770306680036,.230765344947158454,.5,.769234655052841546,.953089922969331996};
const std::array<double,5> gw{.118463442528094544,.239314335249683234,.284444444444444444,.239314335249683234,.118463442528094544};
struct Error {double velocity=0,pressure=0,gradient=0,divergence=0,traction=0,flux=0,nonlinear=0,weakBalance=0;};
Error errors(const Patch& patch,const State& state,const Problem& p,const State* old){
  Error e;double volume=0,area=0;Vec3 balance{};const auto o=patch.options();
  const double hx=2*o.halfWidth/o.nx,hy=2*o.halfWidth/o.ny,hz=1./o.nz;
  for(unsigned k=0;k<o.nz;++k)for(unsigned j=0;j<o.ny;++j)for(unsigned i=0;i<o.nx;++i)
    for(unsigned c=0;c<5;++c)for(unsigned b=0;b<5;++b)for(unsigned a=0;a<5;++a){
      const double x=-o.halfWidth+(i+gx[a])*hx,y=-o.halfWidth+(j+gx[b])*hy,s=(k+gx[c])*hz;
      const auto m=patch.chart().metric(x,y,s);const auto f=patch.sample(state,x,y,s);const auto ex=exact(m.point);
      const double weight=m.determinant*hx*hy*hz*gw[a]*gw[b]*gw[c];volume+=weight;
      const Vec3 du=f.velocity-velocity(ex);e.velocity+=weight*dot(du,du);e.pressure+=weight*std::pow(f.pressure-ex.p.value,2);
      double div=0,exactDiv=0;Vec3 conv{};
      for(unsigned d=0;d<3;++d){const Vec3 dg=f.velocityGradient[d]-ex.u[d].gradient;e.gradient+=weight*dot(dg,dg);div+=f.velocityGradient[d][d];exactDiv+=ex.u[d].gradient[d];conv[d]=dot(velocity(ex),ex.u[d].gradient);}
      require(std::abs(exactDiv)<1e-13,"manufactured Cartesian velocity is incompressible");
      e.divergence+=weight*div*div;e.nonlinear+=weight*dot(conv,conv);
    }
  // Independent re-evaluation at the assembly rule makes the weak momentum
  // balance check isolate conservation from a different quadrature rule.
  constexpr double x6[6]={.0337652428984239861,.169395306766867743,.380690406958401546,.619309593041598454,.830604693233132257,.966234757101576014};
  constexpr double w6[6]={.0856622461895851725,.180380786524069304,.233956967286345524,.233956967286345524,.180380786524069304,.0856622461895851725};
  for(unsigned k=0;k<o.nz;++k)for(unsigned j=0;j<o.ny;++j)for(unsigned i=0;i<o.nx;++i)
    for(unsigned c=0;c<6;++c)for(unsigned b=0;b<6;++b)for(unsigned a=0;a<6;++a){
      const double x=-o.halfWidth+(i+x6[a])*hx,y=-o.halfWidth+(j+x6[b])*hy,s=(k+x6[c])*hz;
      const auto m=patch.chart().metric(x,y,s);const auto f=patch.sample(state,x,y,s);const auto previous=old?patch.sample(*old,x,y,s).velocity:Vec3{};
      Vec3 total=-p.force(m.point);for(unsigned d=0;d<3;++d)total[d]+=(o.timeStep>0?o.density/o.timeStep*(f.velocity[d]-previous[d]):0);
      balance=balance+(m.determinant*hx*hy*hz*w6[a]*w6[b]*w6[c])*total;
    }
  for(const auto& b:patch.boundarySamples(state)){
    require(length(b.advectiveMomentumFlux-(o.density*b.normalVelocity)*b.velocity)<1e-15,"boundary advective momentum flux exposed");
    require(length(b.outwardMomentumFlux-b.advectiveMomentumFlux+b.traction)<1e-15,"boundary total outward momentum flux sign");
    balance=balance+(o.density*b.weight*b.normalVelocity)*b.velocity;
    const auto ex=exact(b.point);Vec3 traction=-ex.p.value*b.normal;
    for(unsigned a=0;a<3;++a)for(unsigned d=0;d<3;++d)traction[a]+=o.dynamicViscosity*(ex.u[a].gradient[d]+ex.u[d].gradient[a])*b.normal[d];
    const Vec3 difference=b.traction-traction;e.traction+=b.weight*dot(difference,difference);area+=b.weight;e.flux+=b.weight*b.normalVelocity;
  }
  Vec3 weak{};for(const auto& load:patch.boundaryLoads(state,p,old))weak=weak+load.reaction;
  e.weakBalance=length(weak-balance);
  e.velocity=std::sqrt(e.velocity/volume);e.pressure=std::sqrt(e.pressure/volume);e.gradient=std::sqrt(e.gradient/volume);
  e.divergence=std::sqrt(e.divergence/volume);e.nonlinear=std::sqrt(e.nonlinear/volume);e.traction=std::sqrt(e.traction/area);return e;
}
void matrixChecks(const Patch& patch,const SolveResult& solved,const Problem& p){
  const auto a=patch.linearize(solved.state,p);const auto n=a.dimension(),nv=3*a.velocityNodes;
  double adjoint=0,pressurePenalty=0;for(std::size_t i=0;i<nv;++i)for(std::size_t j=nv;j<n;++j)adjoint=std::max(adjoint,std::abs(a.matrix[i*n+j]+a.matrix[j*n+i]));
  for(std::size_t i=nv;i<n;++i)for(std::size_t j=nv;j<n;++j)pressurePenalty=std::max(pressurePenalty,std::abs(a.matrix[i*n+j]));
  require(adjoint==0,"mixed gradient is exactly negative transpose of divergence");require(pressurePenalty==0,"pressure block has no penalty or stabilization");
  std::vector<double> direction(n);for(std::size_t i=0;i<n;++i)direction[i]=.1*std::sin(static_cast<double>(i+1));
  State plus=solved.state,minus=solved.state;const double h=2e-6;
  for(std::size_t i=0;i<a.velocityNodes;++i)for(unsigned d=0;d<3;++d){plus.velocity[i][d]+=h*direction[3*i+d];minus.velocity[i][d]-=h*direction[3*i+d];}
  for(std::size_t i=0;i<a.pressureNodes;++i){plus.pressure[i]+=h*direction[nv+i];minus.pressure[i]-=h*direction[nv+i];}
  const auto ap=patch.linearize(plus,p),am=patch.linearize(minus,p);double difference=0,scale=0;
  for(std::size_t i=0;i<n;++i){double action=0;for(std::size_t j=0;j<n;++j)action+=a.matrix[i*n+j]*direction[j];const double fd=(ap.residual[i]-am.residual[i])/(2*h);difference+=(action-fd)*(action-fd);scale+=action*action;}
  require(std::sqrt(difference/scale)<2e-8,"full nonlinear Jacobian matches independent residual difference including boundary DOFs");
  // A constant pressure change must yield the corresponding physical traction
  // change while leaving velocity and normal flux unchanged.
  State shifted=solved.state;for(double& pressure:shifted.pressure)pressure+=.7;
  const auto before=patch.boundarySamples(solved.state),after=patch.boundarySamples(shifted);
  for(std::size_t i=0;i<before.size();++i){require(length(after[i].traction-before[i].traction+.7*before[i].normal)<1e-13,"pressure contribution to boundary traction");require(after[i].normalVelocity==before[i].normalVelocity,"pressure shift preserves boundary flux");}
}
Options options(unsigned n){Options o;o.nx=o.ny=o.nz=n;o.halfWidth=.35;o.density=2;o.dynamicViscosity=.13;return o;}
void printError(unsigned n,const SolveResult& r,const Error& e,bool transient=false){
  std::cout<<std::setprecision(10)<<"{\"elements\":"<<n<<",\"transient\":"<<(transient?"true":"false")<<",\"unknowns\":"<<r.unknowns<<",\"newtonIterations\":"<<r.newtonIterations
    <<",\"residual\":"<<r.finalResidual<<",\"continuityMoment\":"<<r.maximumContinuityMoment<<",\"velocityRms\":"<<e.velocity<<",\"pressureRms\":"<<e.pressure<<",\"gradientRms\":"<<e.gradient
    <<",\"divergenceRms\":"<<e.divergence<<",\"tractionRms\":"<<e.traction<<",\"netBoundaryFlux\":"<<e.flux<<",\"advectionRms\":"<<e.nonlinear<<",\"weakMomentumResidual\":"<<e.weakBalance<<"}\n";
}
}
int main(){
  try{
    const auto geometry=resolved_gap::Chart::spherePair(.9,1.2,.12);std::vector<Error> sequence;
    for(unsigned n:{2u,3u,4u}){
      Patch patch(geometry,options(n));const auto p=problem(patch);const auto start=patch.initialState(p);
      std::size_t interior=0;for(std::size_t i=0;i<start.velocity.size();++i){bool boundary=false;for(const auto& d:patch.boundaryDofs())if(d.node==i){boundary=true;break;}if(!boundary){++interior;require(length(start.velocity[i])==0,"interior starts at zero, not exact manufactured answer");}}
      require(interior>0,"unknown interior velocities exist");
      const auto solved=patch.solve(p);const auto e=errors(patch,solved.state,p,nullptr);printError(n,solved,e);sequence.push_back(e);
      require(solved.newtonIterations>1,"nonlinear manufactured solve iterates");require(solved.finalResidual<1e-10,"nonlinear residual converges");
      require(e.nonlinear>.01,"manufactured advection is materially nonzero");require(e.weakBalance<1e-9,"weak boundary reaction conserves full momentum");
      require(solved.maximumContinuityMoment<1e-9,"all continuity moments including omitted gauge row converge");
      if(n==2)matrixChecks(patch,solved,p);
    }
    for(std::size_t i=1;i<sequence.size();++i){
      require(sequence[i].velocity<sequence[i-1].velocity,"velocity error decreases with mesh refinement");
      require(sequence[i].pressure<sequence[i-1].pressure,"pressure error decreases with mesh refinement");
      require(sequence[i].gradient<sequence[i-1].gradient,"velocity-gradient error decreases with mesh refinement");
      require(sequence[i].traction<sequence[i-1].traction,"boundary-traction error decreases with mesh refinement");
    }
    require(sequence.back().velocity<sequence.front().velocity/4,"velocity error improves substantially over 2-to-4 refinement");
    // Physical stationary no-slip walls, pressure gradient balanced by a body
    // force. This uses the same nonlinear solver, not a Stokes-only path.
    Patch rest(geometry,options(2));Problem hydrostatic;
    hydrostatic.boundaryVelocity=[](Vec3){return Vec3{};};hydrostatic.force=[](Vec3){return Vec3{.13,-.09,0};};
    const auto anchor=rest.pressureNodes().front();hydrostatic.pressureGauge=.13*anchor.x-.09*anchor.y;
    const auto equilibrium=rest.solve(hydrostatic);
    double maximumRestVelocity=0,maximumRestPressureError=0;
    for(auto v:equilibrium.state.velocity)maximumRestVelocity=std::max(maximumRestVelocity,length(v));
    for(std::size_t i=0;i<rest.pressureNodes().size();++i){const auto point=rest.pressureNodes()[i];maximumRestPressureError=std::max(maximumRestPressureError,std::abs(equilibrium.state.pressure[i]-(.13*point.x-.09*point.y)));}
    std::cout<<"{\"hydrostaticVelocityMax\":"<<maximumRestVelocity<<",\"hydrostaticPressureErrorMax\":"<<maximumRestPressureError<<"}\n";
    require(maximumRestVelocity<2e-9,"physical no-slip equilibrium remains at rest");
    require(maximumRestPressureError<2e-8,"hydrostatic pressure and force balance");
    for(double x:{-rest.options().halfWidth,rest.options().halfWidth})for(double y:{-rest.options().halfWidth,rest.options().halfWidth})for(double s:{0.,1.})
      require(length(rest.sample(equilibrium.state,x,y,s).velocity)<1e-13,"physical boundary corner sample remains no slip");
    auto o=options(3);o.timeStep=.2;Patch transient(geometry,o);const auto p=problem(transient,.8);State previous=transient.initialState(p);
    for(std::size_t i=0;i<previous.velocity.size();++i)previous.velocity[i]=.8*velocity(exact(transient.velocityNodes()[i]));
    const auto step=transient.solve(p,nullptr,&previous);const auto error=errors(transient,step.state,p,&previous);printError(3,step,error,true);
    require(error.velocity<.003,"backward Euler nonzero acceleration manufactured solution");require(error.weakBalance<1e-9,"transient weak momentum conservation");
    refuses([&]{transient.solve(p);},"missing previous physical state refused");
    refuses([&]{auto excess=options(20);Patch large(geometry,excess);},"resource excess refused before allocation without coarsening");
    refuses([&]{auto invalid=options(2);invalid.density=0;Patch bad(geometry,invalid);},"zero-density Stokes substitution refused");
    refuses([&]{auto invalid=options(2);invalid.dynamicViscosity=0;Patch bad(geometry,invalid);},"missing physical viscosity refused");
    std::cout<<"{\"ok\":true,\"assertions\":"<<assertions<<",\"scope\":\"fixed-geometry 3D full Navier-Stokes reference patch; no joined vessel or depositing bed\"}\n";return 0;
  }catch(const std::exception& e){std::cerr<<"resolved-gap-flow check failed: "<<e.what()<<'\n';return 1;}
}
