#include "rigid-interpolation.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>

using namespace faraday_native;
namespace {
int passed=0;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
void close(double a,double b,double tol,const char* message){
  if(!(std::abs(a-b)<=tol))throw std::runtime_error(std::string(message)+": "+std::to_string(a)+" != "+std::to_string(b));
}
void close(Vec3 a,Vec3 b,double tol,const char* message){for(int i=0;i<3;i++)close(a[i],b[i],tol,message);}
template<class F>void refusal(F f,const char* text){bool threw=false;try{f();}catch(const std::exception& e){threw=std::string(e.what()).find(text)!=std::string::npos;}require(threw,std::string("Expected refusal: ")+text);}
template<class F>void check(const char* name,F f){f();passed++;std::cout<<"PASS "<<name<<'\n';}
Vec3 field(Vec3 x){return {0.13+13*x.x-21*x.y+8*x.z,-0.29-19*x.x+3*x.y+11*x.z,0.17+7*x.x+17*x.y-6*x.z};}
std::vector<FluidDof> cylinder(){
  const int nr=16,nt=128,nz=20;const double dr=.001,dz=.001,dt=2*std::acos(-1.0)/nt,rho=1000;
  std::vector<FluidDof> out;
  for(int i=1;i<nr;i++)for(int k=0;k<nt;k++)for(int j=0;j<nz;j++){
    const double r=i*dr,t=(k+.5)*dt,z=(j+.5)*dz;
    out.push_back({0,static_cast<std::size_t>((i*nt+k)*nz+j),{r*std::cos(t),r*std::sin(t),z},{std::cos(t),std::sin(t),0},rho*r*dr*dt*dz,std::max({dr,r*dt,dz})});
  }
  for(int i=0;i<nr;i++)for(int k=0;k<nt;k++)for(int j=0;j<nz;j++){
    const double r=(i+.5)*dr,t=k*dt,z=(j+.5)*dz;
    out.push_back({1,static_cast<std::size_t>((i*nt+k)*nz+j),{r*std::cos(t),r*std::sin(t),z},{-std::sin(t),std::cos(t),0},rho*r*dr*dt*dz,std::max({dr,r*dt,dz})});
  }
  for(int i=0;i<nr;i++)for(int k=0;k<nt;k++)for(int j=1;j<=nz;j++){
    const double r=(i+.5)*dr,t=(k+.5)*dt,z=j*dz;
    out.push_back({2,static_cast<std::size_t>((i*nt+k)*(nz+1)+j),{r*std::cos(t),r*std::sin(t),z},{0,0,1},rho*r*dr*dt*(j==nz?.5*dz:dz),std::max({dr,r*dt,dz})});
  }
  return out;
}
}
int main(){try{
  const auto dofs=cylinder();
  const std::vector<RigidGrain> grains={{73,{.008,-.0002,.010},{.2,-.1,.05},{7,-11,13},.004,2650}};
  const RigidInterpolation J(dofs,grains);
  check("sphere markers are interior, symmetric and mesh-resolved",[&]{
    require(J.markers().size()>=27,"Insufficient volume marker coverage");Vec3 sum{};
    for(const auto& m:J.markers()){require(norm(m.offset)<grains[0].radius,"Marker outside sphere");sum=sum+m.offset;}
    close(sum,{},1e-16,"Symmetric volume marker centroid");
    for(const auto& s:J.stencils()){require(s.condition<1e8,"Condition bound");require(s.affineResidual<2e-11,"Moment residual");}
  });
  check("affine Cartesian fields reproduced across cylindrical periodic seam",[&]{
    std::vector<double> u;for(const auto& d:dofs)u.push_back(dot(d.direction,field(d.position)));
    const auto result=J.applyJ(u);
    for(std::size_t i=0;i<result.size();i++)close(result[i],field(J.markers()[i].position),2e-13,"Affine reconstruction");
  });
  check("constant translation and angular rigid motion reproduced",[&]{
    std::vector<double> u;for(const auto& d:dofs)u.push_back(dot(d.direction,grains[0].velocity+cross(grains[0].omega,d.position-grains[0].centre)));
    const auto sampled=J.applyJ(u),rigid=J.applyK({rigidVelocity(grains[0])});
    for(std::size_t i=0;i<sampled.size();i++)close(sampled[i],rigid[i],2e-13,"Rigid velocity reconstruction");
  });
  std::vector<Vec3> impulse(J.markers().size());
  for(std::size_t i=0;i<impulse.size();i++)impulse[i]={1e-6*std::sin(.37*i+.2),1e-6*std::cos(.29*i-.8),1e-6*std::sin(.19*i+1.1)};
  check("J transpose is the exact integrated-force power adjoint",[&]{
    std::vector<double> u(dofs.size());for(std::size_t i=0;i<u.size();i++)u[i]=.1*std::sin(.137*i);
    const auto ju=J.applyJ(u);const auto jt=J.spreadTranspose(impulse),adjoint=J.weightedAdjoint(impulse);
    double fluid=0,marker=0,weighted=0;
    for(std::size_t i=0;i<u.size();i++){fluid+=u[i]*jt[i];weighted+=dofs[i].mass*u[i]*adjoint[i];}
    for(std::size_t i=0;i<ju.size();i++)marker+=dot(ju[i],impulse[i]);
    close(fluid,marker,1e-19,"Integrated power identity");close(weighted,marker,1e-19,"Mass-weighted adjoint identity");
  });
  check("spread forces and torques equal marker forces and torques",[&]{
    const auto spread=J.spreadTranspose(impulse);Vec3 fluidForce{},fluidTorque{},markerForce{},markerTorque{};
    for(std::size_t i=0;i<spread.size();i++){const Vec3 f=spread[i]*dofs[i].direction;fluidForce=fluidForce+f;fluidTorque=fluidTorque+cross(dofs[i].position,f);}
    for(std::size_t i=0;i<impulse.size();i++){markerForce=markerForce+impulse[i];markerTorque=markerTorque+cross(J.markers()[i].position,impulse[i]);}
    close(fluidForce,markerForce,1e-17,"Total force");close(fluidTorque,markerTorque,1e-19,"Total torque about origin");
  });
  check("K transpose gives body reaction including angular impulse",[&]{
    const auto wrench=J.spreadKTranspose(impulse);const auto velocity=J.applyK({rigidVelocity(grains[0])});
    double body=0,marker=0;const Six q=rigidVelocity(grains[0]);
    for(int i=0;i<6;i++)body+=q[i]*wrench[0][i];
    for(std::size_t i=0;i<velocity.size();i++)marker+=dot(velocity[i],impulse[i]);
    close(body,marker,1e-19,"Rigid power identity");
    const auto spread=J.spreadTranspose(impulse);Vec3 fluidF{},fluidT{};
    for(std::size_t i=0;i<spread.size();i++){const Vec3 f=spread[i]*dofs[i].direction;fluidF=fluidF+f;fluidT=fluidT+cross(dofs[i].position,f);}
    const Vec3 bodyF={wrench[0][0],wrench[0][1],wrench[0][2]},bodyT={wrench[0][3],wrench[0][4],wrench[0][5]};
    close(fluidF-bodyF,{},1e-17,"Equal opposite fluid/body force");
    close(fluidT-bodyT-cross(grains[0].centre,bodyF),{},1e-19,"Equal opposite fluid/body torque");
  });
  check("interpolation invariant under complete DOF ordering reversal",[&]{
    std::vector<FluidDof> near;for(const auto& d:dofs)if(norm(d.position-grains[0].centre)<=.003)near.push_back(d);
    std::vector<std::size_t> ids(near.size());std::iota(ids.begin(),ids.end(),0);
    auto a=affineVelocityStencil(near,ids,grains[0].centre,.003);std::reverse(near.begin(),near.end());
    auto b=affineVelocityStencil(near,ids,grains[0].centre,.003);
    Vec3 ua{},ub{};for(const auto& e:a.entries){const auto& d=near[near.size()-1-e.dof];ua=ua+dot(d.direction,field(d.position))*e.coefficient;}
    for(const auto& e:b.entries){const auto& d=near[e.dof];ub=ub+dot(d.direction,field(d.position))*e.coefficient;}
    close(ua,ub,2e-13,"DOF-order invariant reconstructed velocity");
  });
  check("fine grains on the existing coarse grid are refused",[&]{
    auto fine=grains[0];fine.radius=62.5e-6;
    refusal([&]{sphereVolumeMarkers(fine,0,.001);},"Unresolved grain");
  });
  check("rank-deficient, insufficient and ill-conditioned support are refused",[&]{
    std::vector<FluidDof> planar;for(int i=-2;i<=2;i++)for(int j=-2;j<=2;j++)for(int c=0;c<3;c++){
      Vec3 e{};e[c]=1;planar.push_back({c,planar.size(),{.001*i,.001*j,0},e,1e-6,.001});}
    std::vector<std::size_t> ids(planar.size());std::iota(ids.begin(),ids.end(),0);
    refusal([&]{affineVelocityStencil(planar,ids,{},.01);},"Rank-deficient");
    refusal([&]{affineVelocityStencil(planar,{0,1,2},{},.01);},"Insufficient");
    std::vector<FluidDof> support;for(const auto& d:dofs)if(norm(d.position-grains[0].centre)<=.003)support.push_back(d);
    ids.resize(support.size());std::iota(ids.begin(),ids.end(),0);
    auto opts=RigidInterpolationOptions{};opts.maximumCondition=1;
    refusal([&]{affineVelocityStencil(support,ids,grains[0].centre,.003,opts);},"condition bound");
  });
  check("invalid geometry and inconsistent vector lengths are refused",[&]{
    refusal([&]{J.applyJ({1});},"length");refusal([&]{J.applyK({});},"count");
    refusal([&]{J.spreadTranspose({});},"length");
    auto bad=impulse;bad[0].x=std::numeric_limits<double>::quiet_NaN();
    refusal([&]{J.spreadTranspose(bad);},"non-finite");
    auto overlap=grains;overlap.push_back(grains[0]);
    refusal([&]{RigidInterpolation invalid(dofs,overlap);},"Overlapping");
    auto opts=RigidInterpolationOptions{};opts.markerSpacingFactor=1;
    refusal([&]{sphereVolumeMarkers(grains[0],0,.001,opts);},"at least two");
    opts={};opts.minimumCellsPerDiameter=4;
    refusal([&]{sphereVolumeMarkers(grains[0],0,.001,opts);},"eight cells");
  });
  check("mesh refinement increases volume collocation density",[&]{
    const auto coarse=sphereVolumeMarkers(grains[0],0,.001),fine=sphereVolumeMarkers(grains[0],0,.0005);
    require(fine.size()>5*coarse.size(),"Volume markers did not refine in all three dimensions");
  });
  check("smooth non-affine interpolation converges quadratically",[&]{
    const Vec3 point={.013,-.007,.009};
    const auto quadratic=[](Vec3 p){return Vec3{p.x*p.x+2*p.y*p.z,3*p.y*p.y-p.x*p.z,p.z*p.z+p.x*p.y};};
    std::vector<double> errors;
    for(double h:{.002,.001,.0005}){
      std::vector<FluidDof> local;
      for(int i=-2;i<=2;i++)for(int j=-2;j<=2;j++)for(int k=-2;k<=2;k++)for(int c=0;c<3;c++){
        Vec3 e{};e[c]=1;
        const Vec3 p=point+h*Vec3{i+.19,j-.13,k+.27};
        local.push_back({c,local.size(),p,e,1000*h*h*h,h});
      }
      std::vector<std::size_t> ids(local.size());std::iota(ids.begin(),ids.end(),0);
      const auto stencil=affineVelocityStencil(local,ids,point,5*h);
      Vec3 value{};for(const auto& e:stencil.entries)value=value+dot(local[e.dof].direction,quadratic(local[e.dof].position))*e.coefficient;
      errors.push_back(norm(value-quadratic(point)));
    }
    require(errors[0]>1e-8,"Manufactured quadratic field accidentally exact");
    require(errors[0]/errors[1]>3.9&&errors[1]/errors[2]>3.9,"Second-order convergence lost");
  });
  check("zero-grain interpolation is an empty operator",[&]{
    const std::vector<FluidDof> none;const RigidInterpolation empty(none,{});
    require(empty.applyJ({}).empty()&&empty.applyK({}).empty()&&empty.spreadTranspose({}).empty(),"Zero-grain operator nonempty");
  });
  std::cout<<passed<<" interpolation checks passed\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
