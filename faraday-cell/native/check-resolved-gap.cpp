#include "resolved-gap.hpp"
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>

using namespace faraday_native;
using namespace faraday_native::resolved_gap;
namespace {
unsigned checks=0;
double maxGradientError=0,maxDivergenceError=0,maxGclError=0;
void require(bool condition,const std::string& name){
  ++checks;if(!condition)throw std::runtime_error(name);
}
void near(double actual,double expected,double tolerance,const std::string& name){
  require(std::isfinite(actual)&&std::abs(actual-expected)<=tolerance,name+
    ": actual="+std::to_string(actual)+" expected="+std::to_string(expected));
}
void near(Vec3 actual,Vec3 expected,double tolerance,const std::string& name){
  near(length(actual-expected),0,tolerance,name);
}
template<class F> void refuses(F operation,const std::string& name){
  bool refused=false;try{operation();}catch(const std::exception&){refused=true;}
  require(refused,name);
}
Vec3 basis(unsigned i){Vec3 out;out[i]=1;return out;}
Metric at(const Chart& c,Vec3 q){return c.metric(q.x,q.y,q.z);}
template<class F> Vec3 coordinateDerivative(F value,Vec3 q,double step){
  Vec3 out;for(unsigned i=0;i<3;++i)
    out[i]=(value(q+step*basis(i))-value(q-step*basis(i)))/(2*step);
  return out;
}
template<class F> Vec3 diagonalFluxDerivative(F value,Vec3 q,double step){
  Vec3 out;for(unsigned i=0;i<3;++i)
    out[i]=(value(q+step*basis(i))[i]-value(q-step*basis(i))[i])/(2*step);
  return out;
}
double polynomial(Vec3 p){return 2*p.x*p.x+p.x*p.y-3*p.y*p.z+4*p.z*p.z+5*p.x-2*p.y+p.z;}
Vec3 polynomialGradient(Vec3 p){return {4*p.x+p.y+5,p.x-3*p.z-2,-3*p.y+8*p.z+1};}
Vec3 quadraticVelocity(Vec3 p){return {p.x*p.x,p.y*p.y,p.z*p.z};}

void localChecks(const Chart& c){
  for(Vec3 q: {Vec3{0,0,.2},Vec3{.14,-.11,.37},Vec3{-.21,.09,.8}}){
    const auto m=at(c,q);const auto f=c.frame();
    const auto section=c.section(q.x,q.y);
    const auto sag=sphereSag(c.upperRadius(),q.x,q.y);
    const double ds=2e-6;
    const auto xp=sphereSag(c.upperRadius(),q.x+ds,q.y),xm=sphereSag(c.upperRadius(),q.x-ds,q.y);
    const auto yp=sphereSag(c.upperRadius(),q.x,q.y+ds),ym=sphereSag(c.upperRadius(),q.x,q.y-ds);
    near(sag.dxx,(xp.dx-xm.dx)/(2*ds),3e-11,"spherical Hessian xx finite difference");
    near(sag.dxy,(yp.dx-ym.dx)/(2*ds),3e-11,"spherical Hessian xy finite difference");
    near(sag.dyy,(yp.dy-ym.dy)/(2*ds),3e-11,"spherical Hessian yy finite difference");
    const Vec3 lower=c.metric(q.x,q.y,0).point,upper=c.metric(q.x,q.y,1).point;
    const Vec3 upperCentre=f.origin+(c.gap()+c.upperRadius())*f.z;
    near(length(upper-upperCentre),c.upperRadius(),2e-15,"upper surface implicit sphere");
    near(c.outwardUpperNormal(q.x,q.y),normalized(upperCentre-upper),2e-15,"upper normal implicit sphere");
    if(c.lowerRadius()>0){
      const Vec3 lowerCentre=f.origin-c.lowerRadius()*f.z;
      near(length(lower-lowerCentre),c.lowerRadius(),2e-15,"lower surface implicit sphere");
      near(c.outwardLowerNormal(q.x,q.y),normalized(lowerCentre-lower),2e-15,"lower normal implicit sphere");
    }else{
      near(dot(lower-f.origin,f.z),0,2e-15,"lower surface plane");
      near(c.outwardLowerNormal(q.x,q.y),-f.z,2e-15,"lower plane normal");
    }
    near(c.coordinates(m.point),q,2e-14,"inverse physical map");
    near(m.determinant,section.height,0,"Jacobian determinant positive gap");
    for(unsigned i=0;i<3;++i){
      const double h=2e-6;
      const Vec3 fd=(at(c,q+h*basis(i)).point-at(c,q-h*basis(i)).point)/(2*h);
      near(m.jacobian[i],fd,2e-10,"Jacobian independent coordinate difference");
      near(m.cofactor[i],cross(m.jacobian[(i+1)%3],m.jacobian[(i+2)%3]),3e-15,"cofactor independent cross product");
    }
    near(dot(m.jacobian[0],cross(m.jacobian[1],m.jacobian[2])),m.determinant,2e-15,"Jacobian triple product");
    const auto derivatives=coordinateDerivative([&](Vec3 v){return polynomial(at(c,v).point);},q,2e-6);
    const double gradientError=length(m.physicalGradient(derivatives)-polynomialGradient(m.point));
    maxGradientError=std::max(maxGradientError,gradientError);
    near(gradientError,0,2e-8,"physical polynomial gradient");
    const auto fluxDerivatives=diagonalFluxDerivative([&](Vec3 v){const auto p=at(c,v);return p.flux(quadraticVelocity(p.point));},q,2e-6);
    const double expectedDiv=2*(m.point.x+m.point.y+m.point.z);
    const double divergenceError=std::abs(m.divergence(fluxDerivatives)-expectedDiv);
    maxDivergenceError=std::max(maxDivergenceError,divergenceError);
    near(divergenceError,0,2e-9,"conservative mapped quadratic divergence");
    const Motion motion{{.17,-.3,.09},{.13,-.2,.4},-.037};
    const auto gridFluxDerivative=diagonalFluxDerivative([&](Vec3 v){const auto p=at(c,v);return p.flux(p.gridVelocity(motion));},q,2e-6);
    const double gclError=std::abs(m.divergence(gridFluxDerivative)*m.determinant-motion.gapRate);
    maxGclError=std::max(maxGclError,gclError);
    near(gclError,0,3e-11,"moving-grid geometric conservation law");
    near(m.relativeFlux(m.gridVelocity(motion),motion),{},0,"comoving fluid zero relative flux");
  }
}

template<class F> long double simpson(F value,long double lo,long double hi,unsigned intervals){
  require(intervals>0&&intervals%2==0,"Simpson even intervals");
  const long double h=(hi-lo)/intervals;long double sum=value(lo)+value(hi);
  for(unsigned i=1;i<intervals;++i)sum+=(i%2?4:2)*value(lo+i*h);
  return sum*h/3;
}
// Independent antiderivative, evaluated in long double at a moderate footprint.
long double independentVolume(const Chart& c,long double radius){
  const long double pi=std::acos(-1.L);
  auto cap=[&](long double a){return a==0?0:pi*a*radius*radius-
    2*pi/3*(a*a*a-std::pow(a*a-radius*radius,1.5L));};
  return pi*radius*radius*c.gap()+cap(c.lowerRadius())+cap(c.upperRadius());
}
double boundaryFlux(const Chart& c,double radius,unsigned intervals){
  const double pi=std::acos(-1.0);constexpr unsigned azimuths=32;
  double flux=0;
  for(unsigned j=0;j<azimuths;++j){
    const double phi=2*pi*j/azimuths,cs=std::cos(phi),sn=std::sin(phi);
    const auto caps=simpson([&](long double radial){
      const double r=static_cast<double>(radial);
      const auto bottom=c.metric(r*cs,r*sn,0),top=c.metric(r*cs,r*sn,1);
      return static_cast<long double>(r*(dot(top.cofactor[2],top.offset)-dot(bottom.cofactor[2],bottom.offset)));
    },0,radius,intervals);
    const auto side=simpson([&](long double sigma){
      const auto p=c.metric(radius*cs,radius*sn,static_cast<double>(sigma));
      const Vec3 area=radius*p.determinant*(cs*c.frame().x+sn*c.frame().y);
      return static_cast<long double>(dot(area,p.offset));
    },0,1,intervals);
    flux+=static_cast<double>(caps+side)*2*pi/azimuths;
  }
  return flux;
}
void integralChecks(const Chart& c){
  const double radius=.38;
  const double reference=static_cast<double>(independentVolume(c,radius));
  near(c.diskVolume(radius),reference,3e-16,"disk volume independent antiderivative");
  const double coarse=std::abs(boundaryFlux(c,radius,16)-3*reference);
  const double fine=std::abs(boundaryFlux(c,radius,64)-3*reference);
  require(fine<coarse/100,"surface flux quadrature converges");
  near(fine,0,2e-9,"divergence theorem over all gap boundaries");
}

struct BrennerResult{long double value=0,tailBound=0;unsigned terms=0;};
// TEST ORACLE ONLY: exact creeping-flow sphere/plane series; never a production
// force closure. The bound controls series truncation, not floating-point error.
BrennerResult brenner(long double q,long double tolerance){
  if(!(q>=.001L&&q<=10&&tolerance>0))throw std::invalid_argument("Outside verified Brenner oracle range");
  const long double sh=std::sqrt(q*(q+2)),a=std::log1p(q+sh),sh2=2*sh*(1+q);
  const long double r=std::exp(-2*a),oneMinusR=-std::expm1(-2*a),prefactor=4*sh/3;
  long double sum=0;
  for(unsigned n=1;n<=100000;++n){
    const long double t=2.L*n+1,e=std::exp(-t*a),oneMinusE=-std::expm1(-t*a);
    const long double den=oneMinusE*oneMinusE-t*t*sh*sh*e;
    if(!(den>0))throw std::runtime_error("Brenner denominator roundoff");
    const long double ratioMinusOne=(2*oneMinusE+t*sh2+t*t*sh*sh)*e/den;
    sum+=(static_cast<long double>(n)*(n+1)/((2.L*n-1)*(2.L*n+3)))*ratioMinusOne;
    if(t*a<2)continue;
    const long double t0=t+2,e0=std::exp(-t0*a),d=oneMinusR;
    const long double a0=1/d,a1=t0/d+2*r/(d*d);
    const long double a2=t0*t0/d+4*t0*r/(d*d)+4*r*(1+r)/(d*d*d);
    // Coefficient n(n+1)/((2n-1)(2n+3)) <= 2/5; denominator
    // increases after t*a>=2. Sum remaining constant/linear/quadratic
    // geometric tails after bounding 2*(1-e) by 2.
    const long double tail=prefactor*.4L/den*e0*(2*a0+sh2*a1+sh*sh*a2);
    if(tail<tolerance)return {prefactor*sum,tail,n};
  }
  throw std::runtime_error("Brenner oracle did not meet its truncation bound");
}
void referenceChecks(){
  // 75-decimal mpmath evaluation of the original hyperbolic series, independent
  // of this exponential rearrangement. Source equation and original rounded
  // values: https://basilisk.fr/sandbox/ghigo/src/test-stokes/sphere-towards-wall.c
  const std::array<std::pair<long double,long double>,6> cases{{
    {.4L,3.7356171627893659575800574123963567619L},
    {.1L,11.459157203409557229641019876155041348L},
    {.01L,101.89617232461348887084388795771547795L},
    {.001L,1002.3533268805824125008631898375225661L},
    {1.L,2.1255355667600402000907766576770883250L},
    {10.L,1.1135032339057078510323639380104748280L}}};
  for(auto [q,reference]:cases){
    const auto result=brenner(q,1e-13L);
    require(std::abs(result.value-reference)<2e-11L,"independent Brenner limiting-force reference");
    require(result.tailBound<1e-13L,"Brenner analytic truncation bound");
  }
}
void refusalChecks(){
  const double nan=std::numeric_limits<double>::quiet_NaN();
  refuses([]{Chart::sphereWall(1,0);},"zero gap refused");
  refuses([]{Chart::sphereWall(1,-.1);},"negative gap refused");
  refuses([]{Chart::sphereWall(0,.1);},"zero radius refused");
  refuses([]{Chart::spherePair(-1,1,.1);},"negative lower radius refused");
  refuses([&]{Chart::sphereWall(1,nan);},"nonfinite gap refused");
  refuses([]{Frame f;f.y=-f.y;Chart::sphereWall(1,.1,f);},"left handed frame refused");
  refuses([]{Frame f;f.x.x=2;Chart::sphereWall(1,.1,f);},"nonorthonormal frame refused");
  refuses([]{Frame::fromNormal({},{});},"zero normal refused");
  const auto c=Chart::sphereWall(1,.1);
  refuses([&]{c.metric(1,0,.5);},"equator refused");
  refuses([&]{c.metric(1.1,0,.5);},"exterior footprint refused");
  refuses([&]{c.metric(0,0,-.01);},"negative sigma refused");
  refuses([&]{c.metric(0,0,1.01);},"sigma above one refused");
  refuses([&]{c.metric(nan,0,.5);},"nonfinite coordinate refused");
  const auto outside=c.coordinates({0,0,.2});
  near(outside.z,2,0,"inverse map preserves exterior sigma instead of clamping");
}
}
int main(){
  try{
    const auto frame=Frame::fromNormal({.3,-.4,.2},{1,-2,3});
    for(const auto& chart:{Chart::sphereWall(1,.08),Chart::spherePair(.8,1.3,.06),
        Chart::sphereWall(1,.08,frame),Chart::spherePair(.8,1.3,.06,frame)}){
      localChecks(chart);integralChecks(chart);
    }
    const double theta=1e-8,expected=2*std::sin(theta/2)*std::sin(theta/2);
    near(sphereSag(1,std::sin(theta),0).value,expected,1e-31,"small sag retained without subtraction cancellation");
    near(Chart::sphereWall(1,1e-20).metric(0,0,.5).determinant,1e-20,0,"positive tiny centre gap retained without floor");
    near(Frame::fromNormal({},{1e-310,0,0}).z,{1,0,0},0,"subnormal direction normalized without reciprocal overflow");
    referenceChecks();refusalChecks();
    std::cout<<std::setprecision(9)<<"{\"ok\":true,\"checks\":"<<checks
      <<",\"maxGradientError\":"<<maxGradientError<<",\"maxDivergenceError\":"<<maxDivergenceError
      <<",\"maxGclError\":"<<maxGclError
      <<",\"scope\":\"exact positive-gap geometry and test-only Stokes reference; no production refinement or contact\"}\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<"resolved-gap check failed: "<<e.what()<<'\n';return 1;}
}
