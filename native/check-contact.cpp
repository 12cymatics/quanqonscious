#include "rigid-contact.hpp"
#include <algorithm>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace faraday_native;
namespace {
int passed=0;
void require(bool v,const std::string& m){if(!v)throw std::runtime_error(m);}
void near(double x,double y,double tolerance=1e-9){if(!(std::abs(x-y)<=tolerance))throw std::runtime_error("Mismatch "+std::to_string(x)+" versus "+std::to_string(y));}
void near(Vec3 a,Vec3 b,double tolerance=1e-9){near(a.x,b.x,tolerance);near(a.y,b.y,tolerance);near(a.z,b.z,tolerance);}
template<class F>void rejects(F fn){bool caught=false;try{fn();}catch(const std::exception&){caught=true;}require(caught,"Expected refusal");}
template<class F>void test(const char* label,F fn){fn();passed++;std::cout<<"PASS "<<label<<'\n';}
RigidGrain grain(Vec3 x,Vec3 v,double mass=1,double radius=1){RigidGrain g;g.centre=x;g.velocity=v;g.radius=radius;g.density=mass/((4.0/3)*std::acos(-1.0)*radius*radius*radius);return g;}

// Dry inverse masses appear ONLY in this independent analytic test fixture.
// Production solveContacts accepts a caller-supplied fluid/body mobility.
std::vector<double> analyticDryMobility(const std::vector<ContactGeometry>& contacts,const std::vector<RigidGrain>& grains){
  const std::size_t n=3*contacts.size();std::vector<double> W(n*n);
  for(std::size_t col=0;col<n;col++){
    std::vector<Six> impulse(grains.size(),Six{});Vec3 local;local[col%3]=1;
    accumulateContactImpulse(contacts[col/3],local,impulse);
    auto moved=grains;
    for(std::size_t i=0;i<moved.size();i++){
      moved[i].velocity={impulse[i][0]/grains[i].mass(),impulse[i][1]/grains[i].mass(),impulse[i][2]/grains[i].mass()};
      moved[i].omega={impulse[i][3]/grains[i].inertia(),impulse[i][4]/grains[i].inertia(),impulse[i][5]/grains[i].inertia()};
    }
    for(std::size_t row=0;row<contacts.size();row++){
      const auto v=relativeContactVelocity(contacts[row],moved);for(std::size_t d=0;d<3;d++)W[(3*row+d)*n+col]=v[d];
    }
  }return W;
}
ContactProblem dryProblem(const std::vector<ContactGeometry>& contacts,const std::vector<RigidGrain>& grains,double mu,double e){
  ContactProblem p;p.mobility=analyticDryMobility(contacts,grains);p.friction.assign(contacts.size(),mu);
  for(const auto& c:contacts){const auto v=relativeContactVelocity(c,grains);p.freeVelocity.insert(p.freeVelocity.end(),{v.x,v.y,v.z});}
  p.normalTarget=impactNormalTargets(p.freeVelocity,std::vector<double>(contacts.size(),e));return p;
}
std::vector<RigidGrain> applied(const std::vector<ContactGeometry>& contacts,const std::vector<RigidGrain>& grains,const ContactSolution& solution){
  std::vector<Six> impulse(grains.size(),Six{});for(std::size_t c=0;c<contacts.size();c++)accumulateContactImpulse(contacts[c],{solution.impulse[3*c],solution.impulse[3*c+1],solution.impulse[3*c+2]},impulse);
  auto out=grains;for(std::size_t i=0;i<out.size();i++){out[i].velocity=out[i].velocity+Vec3{impulse[i][0],impulse[i][1],impulse[i][2]}/grains[i].mass();out[i].omega=out[i].omega+Vec3{impulse[i][3],impulse[i][4],impulse[i][5]}/grains[i].inertia();}return out;
}
double energy(const std::vector<RigidGrain>& g){double e=0;for(const auto& x:g)e+=.5*x.mass()*dot(x.velocity,x.velocity)+.5*x.inertia()*dot(x.omega,x.omega);return e;}
Vec3 momentum(const std::vector<RigidGrain>& g){Vec3 p;for(const auto& x:g)p=p+x.mass()*x.velocity;return p;}
Vec3 angularMomentum(const std::vector<RigidGrain>& g){Vec3 p;for(const auto& x:g)p=p+cross(x.centre,x.mass()*x.velocity)+x.inertia()*x.omega;return p;}
ContactProblem permuteContacts(const ContactProblem& p,const std::vector<std::size_t>& permutation){
  const auto n=p.freeVelocity.size();ContactProblem q;q.mobility.resize(n*n);q.freeVelocity.resize(n);
  for(std::size_t a=0;a<permutation.size();a++){
    q.friction.push_back(p.friction[permutation[a]]);q.normalTarget.push_back(p.normalTarget[permutation[a]]);
    for(std::size_t d=0;d<3;d++){q.freeVelocity[3*a+d]=p.freeVelocity[3*permutation[a]+d];for(std::size_t b=0;b<permutation.size();b++)for(std::size_t k=0;k<3;k++)q.mobility[(3*a+d)*n+3*b+k]=p.mobility[(3*permutation[a]+d)*n+3*permutation[b]+k];}
  }return q;
}
}

int main(){try{
  test("pair geometry, right-handed basis and rotational contact velocity",[]{
    std::vector<RigidGrain> g{grain({0,0,1},{1,2,3}),grain({2,0,1},{-1,0,0})};g[0].omega={0,0,2};
    const auto c=spherePairContact(g[0],0,g[1],1);near(c.gap,0);near(c.normal,{-1,0,0});near(c.armA,{1,0,0});near(c.armB,{-1,0,0});near(cross(c.normal,c.tangent1),c.tangent2);
    near(contactWorldVector(c,relativeContactVelocity(c,g)),{2,4,3});
  });
  test("floor and cylinder geometry with explicit axis-normal ambiguity",[]{
    auto g=grain({2,0,1},{0,0,0});const auto floor=floorContact(g,0,0),wall=cylinderContact(g,0,3);near(floor.gap,0);near(wall.gap,0);near(wall.normal,{-1,0,0});near(wall.armA,{1,0,0});
    g.centre={0,0,1};const auto centre=cylinderContact(g,0,3);require(!centre.normalDefined,"Axis normal must be marked undefined");rejects([&]{contactCoordinates(centre,{1,0,0});});
  });
  test("pair event time is analytic and never repairs positions",[]{
    auto a=grain({0,0,0},{1,0,0}),b=grain({5,0,0},{-1,0,0});const auto beforeA=a.centre,beforeB=b.centre;near(*spherePairImpactTime(a,b,3),1.5);require(!spherePairImpactTime(a,b,1),"Event outside horizon");near(a.centre,beforeA);near(b.centre,beforeB);
    a.velocity={-1,0,0};b.velocity={1,0,0};require(!spherePairImpactTime(a,b,10),"Receding pair must not collide");
    b.centre={1,0,0};rejects([&]{spherePairImpactTime(a,b,10);});
  });
  test("floor, wall and grazing time-of-impact",[]{
    auto a=grain({0,0,3},{0,0,-2});near(*floorImpactTime(a,0,2),1);
    a.centre={0,0,1};a.velocity={2,0,0};near(*cylinderImpactTime(a,5,3),2);
    a.centre={4,0,1};a.velocity={0,1,0};near(*cylinderImpactTime(a,5,3),0);require(cylinderNeedsSustainedConstraint(a,5),"Curved sustained contact must be flagged");
    a.centre={0,0,0};a.velocity={1,0,0};auto b=grain({3,2,0},{0,0,0});near(*spherePairImpactTime(a,b,4),3);
  });
  test("unresolved positive lubrication gaps are reported without a closure",[]{
    auto a=grain({0,0,1.00001},{0,0,0});const auto c=floorContact(a,0,0);
    require(gapResolution(c,.0001)==GapResolution::positiveGapUnresolved,"Gap resolution flag");require(gapResolution(c,.000001)==GapResolution::separatedResolved,"Resolved gap flag");
    std::vector<Six> impulse(1,Six{});rejects([&]{accumulateContactImpulse(c,{1,0,0},impulse);});
    a.centre.z=1;require(gapResolution(floorContact(a,0,0),.01)==GapResolution::touching,"Touching flag");a.centre.z=.99;require(gapResolution(floorContact(a,0,0),.01)==GapResolution::overlapping,"Overlap flag");
  });
  test("unequal-mass elastic impact conserves momentum and kinetic energy",[]{
    const std::vector<RigidGrain> g{grain({0,0,0},{1,0,0},1),grain({2,0,0},{0,0,0},2)};const std::vector<ContactGeometry> c{spherePairContact(g[0],0,g[1],1)};
    const auto s=solveContacts(dryProblem(c,g,0,1));const auto out=applied(c,g,s);near(out[0].velocity.x,-1.0/3);near(out[1].velocity.x,2.0/3);near(momentum(out),momentum(g));near(energy(out),energy(g));near(s.energyChange,0);
  });
  test("perfectly inelastic impact reaches common normal velocity",[]{
    const std::vector<RigidGrain> g{grain({0,0,0},{1,0,0},1),grain({2,0,0},{0,0,0},2)};const std::vector<ContactGeometry> c{spherePairContact(g[0],0,g[1],1)};
    const auto s=solveContacts(dryProblem(c,g,0,0));const auto out=applied(c,g,s);near(out[0].velocity.x,1.0/3);near(out[1].velocity.x,1.0/3);near(momentum(out),momentum(g));require(energy(out)<energy(g),"Inelastic impact must dissipate energy");
  });
  test("sliding floor impact satisfies Coulomb direction and rotation",[]{
    const std::vector<RigidGrain> g{grain({0,0,1},{1,0,-2})};const std::vector<ContactGeometry> c{floorContact(g[0],0,0)};
    const auto p=dryProblem(c,g,.1,0);const auto s=solveContacts(p);const auto out=applied(c,g,s);near(out[0].velocity,{.8,0,0});near(out[0].omega,{0,.5,0});near(std::hypot(s.impulse[1],s.impulse[2]),.1*s.impulse[0]);require(s.energyChange<0,"Friction must dissipate");near(s.energyChange,energy(out)-energy(g));
  });
  test("sticking floor impact and rolling state",[]{
    std::vector<RigidGrain> g{grain({0,0,1},{1,0,-2})};const std::vector<ContactGeometry> c{floorContact(g[0],0,0)};auto s=solveContacts(dryProblem(c,g,1,0));near(s.velocity[1],0);near(s.velocity[2],0);require(std::hypot(s.impulse[1],s.impulse[2])<s.impulse[0],"Sticking impulse inside cone");
    g[0].omega={0,1,0};s=solveContacts(dryProblem(c,g,.5,0));near(s.impulse[1],0);near(s.impulse[2],0);
  });
  test("frictional pair conserves total angular and linear momentum",[]{
    const std::vector<RigidGrain> g{grain({0,0,0},{1,2,0}),grain({2,0,0},{-1,0,0},2)};const std::vector<ContactGeometry> c{spherePairContact(g[0],0,g[1],1)};const auto s=solveContacts(dryProblem(c,g,.4,.3));const auto out=applied(c,g,s);near(momentum(out),momentum(g));near(angularMomentum(out),angularMomentum(g));require(energy(out)<=energy(g),"Passive pair energy");
  });
  test("coupled off-diagonal mobility converges and contact order does not matter",[]{
    const std::size_t n=6;const double L[36]={1,0,0,0,0,0, .2,1.2,0,0,0,0, -.1,.3,.9,0,0,0, .15,-.1,.1,1.1,0,0, .1,.15,0,.2,1.3,0, -.05,.1,.2,-.15,.1,.8};
    ContactProblem p;p.mobility.assign(36,0);for(std::size_t i=0;i<n;i++)for(std::size_t j=0;j<n;j++)for(std::size_t k=0;k<n;k++)p.mobility[i*n+j]+=L[i*n+k]*L[j*n+k];
    p.freeVelocity={-1,2,-.5,-.7,-1,1.2};p.normalTarget={0,0};p.friction={.4,.3};const auto a=solveContacts(p),b=solveContacts(permuteContacts(p,{1,0}));
    for(std::size_t i=0;i<6;i++){near(a.impulse[i],b.impulse[(i+3)%6],2e-8);near(a.velocity[i],b.velocity[(i+3)%6],2e-8);}require(a.energyChange<0,"Coupled solution passivity");require(a.normalResidual<1e-8&&a.tangentResidual<1e-8,"Physical residuals");
  });
  test("tangent-basis rotation leaves physical impulse invariant",[]{
    std::vector<RigidGrain> g{grain({0,0,1},{1,2,-1})};auto c=floorContact(g[0],0,0);const auto p=dryProblem({c},g,.3,0);const auto s=solveContacts(p);const auto world=contactWorldVector(c,{s.impulse[0],s.impulse[1],s.impulse[2]});
    const double angle=.7;const Vec3 a=c.tangent1,b=c.tangent2;c.tangent1=std::cos(angle)*a+std::sin(angle)*b;c.tangent2=-std::sin(angle)*a+std::cos(angle)*b;const auto t=solveContacts(dryProblem({c},g,.3,0));near(world,contactWorldVector(c,{t.impulse[0],t.impulse[1],t.impulse[2]}));
  });
  test("redundant PSD contacts may have nonunique impulses but identical velocity",[]{
    ContactProblem p;p.mobility.assign(36,0);for(auto i:{0,3})for(auto j:{0,3})p.mobility[i*6+j]=1;p.freeVelocity={-1,0,0,-1,0,0};p.normalTarget={0,0};p.friction={0,0};const auto s=solveContacts(p);near(s.impulse[0]+s.impulse[3],1);near(s.velocity[0],0);near(s.velocity[3],0);
  });
  test("nonpassive simultaneous Newton restitution is refused",[]{
    ContactProblem p;p.mobility.assign(36,0);for(std::size_t i=0;i<6;i++)p.mobility[i*6+i]=1;p.mobility[3]=p.mobility[18]=-.9;p.freeVelocity={-1,0,0,.5,0,0};p.normalTarget=impactNormalTargets(p.freeVelocity,{1,1});p.friction={0,0};rejects([&]{solveContacts(p);});
  });
  test("small fine-grain energy scales are not hidden by a fixed energy allowance",[]{
    ContactProblem p;p.mobility.assign(36,0);for(std::size_t i=0;i<6;i++)p.mobility[i*6+i]=1e8;p.mobility[3]=p.mobility[18]=-.9e8;p.freeVelocity={-.01,0,0,.005,0,0};p.normalTarget=impactNormalTargets(p.freeVelocity,{1,1});p.friction={0,0};ContactSolveOptions o;o.velocityTolerance=1e-13;o.impulseTolerance=1e-20;rejects([&]{solveContacts(p,o);});
  });
  test("indefinite/asymmetric mobility, invalid coefficients and immovable closing contacts refuse",[]{
    ContactProblem p;p.mobility={1,0,0,0,1,0,0,0,1};p.freeVelocity={-1,0,0};p.normalTarget={0};p.friction={0};p.mobility[0]=-1;rejects([&]{solveContacts(p);});p.mobility[0]=1;p.mobility[1]=.1;rejects([&]{solveContacts(p);});p.mobility[1]=0;p.friction[0]=-.1;rejects([&]{solveContacts(p);});p.friction[0]=0;p.mobility.assign(9,0);rejects([&]{solveContacts(p);});rejects([&]{impactNormalTargets({-1,0,0},{1.01});});
  });
  test("iteration refusal returns no approximate impulse or mutated input",[]{
    ContactProblem p;p.mobility={1,.7,.2,.7,1,.3,.2,.3,1};p.freeVelocity={-1,2,1};p.normalTarget={0};p.friction={.5};const auto before=p.freeVelocity;ContactSolveOptions o;o.maximumIterations=1;rejects([&]{solveContacts(p,o);});require(p.freeVelocity==before,"Input mutation");
  });
  test("finite inputs that overflow derived contact arithmetic refuse",[]{
    ContactProblem p;p.mobility={1,0,0,0,1e300,0,0,0,1e300};p.freeVelocity={-1,2,1};p.normalTarget={0};p.friction={1};rejects([&]{solveContacts(p);});
  });
  test("actual coupled contact velocities must satisfy the law independently",[]{
    ContactProblem p;p.mobility={1,0,0,0,1,0,0,0,1};p.freeVelocity={-1,0,0};p.normalTarget={0};p.friction={.5};
    const auto s=solveContacts(p);verifyContactResponse(p,s.impulse,s.velocity);
    rejects([&]{verifyContactResponse(p,s.impulse,{-1e-4,0,0});});
    rejects([&]{verifyContactResponse(p,s.impulse,{0,1,0});});
    rejects([&]{verifyContactResponse(p,{1,2,0},s.velocity);});
    rejects([&]{verifyContactResponse(p,s.impulse,{0,std::numeric_limits<double>::quiet_NaN(),0});});
  });
  test("empty contact set is an identity solve",[]{const auto s=solveContacts(ContactProblem{});require(s.impulse.empty()&&s.velocity.empty(),"Empty result");near(s.energyChange,0);});
  std::cout<<passed<<" contact checks passed\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL after "<<passed<<" checks: "<<e.what()<<'\n';return 1;}}
