#include "resolved-gap-join.hpp"
#include <algorithm>
#include <limits>

namespace faraday_native::resolved_gap_join {
namespace {
std::size_t mul(std::size_t a,std::size_t b){
  if(b&&a>std::numeric_limits<std::size_t>::max()/b)throw std::length_error("Joined NS size overflow");
  return a*b;
}
std::size_t add(std::size_t a,std::size_t b){
  if(a>std::numeric_limits<std::size_t>::max()-b)throw std::length_error("Joined NS size overflow");
  return a+b;
}
std::size_t idx(std::size_t i,std::size_t j,std::size_t k,std::size_t nx,std::size_t ny){return (k*ny+j)*nx+i;}
bool same(Vec3 a,Vec3 b){return a.x==b.x&&a.y==b.y&&a.z==b.z;}
bool sameChart(const resolved_gap::Chart& a,const resolved_gap::Chart& b){
  const auto& x=a.frame();const auto& y=b.frame();
  return a.gap()==b.gap()&&a.lowerRadius()==b.lowerRadius()&&a.upperRadius()==b.upperRadius()
    &&same(x.origin,y.origin)&&same(x.x,y.x)&&same(x.y,y.y)&&same(x.z,y.z);
}
std::vector<double> denseSolve(std::vector<double> a,std::vector<double> b){
  const std::size_t n=b.size();double scale=0;
  for(double v:a){if(!std::isfinite(v))throw std::runtime_error("Non-finite joined NS matrix");scale=std::max(scale,std::abs(v));}
  if(!(scale>0))throw std::runtime_error("Empty or singular joined NS system");
  const double threshold=64*std::numeric_limits<double>::epsilon()*scale;
  for(std::size_t k=0;k<n;++k){
    std::size_t pivot=k;for(std::size_t i=k+1;i<n;++i)if(std::abs(a[i*n+k])>std::abs(a[pivot*n+k]))pivot=i;
    if(!(std::abs(a[pivot*n+k])>threshold))throw std::runtime_error("Joined NS system is singular or numerically unresolved");
    if(pivot!=k){for(std::size_t j=k;j<n;++j)std::swap(a[k*n+j],a[pivot*n+j]);std::swap(b[k],b[pivot]);}
    for(std::size_t i=k+1;i<n;++i){const double factor=a[i*n+k]/a[k*n+k];a[i*n+k]=0;
      for(std::size_t j=k+1;j<n;++j)a[i*n+j]-=factor*a[k*n+j];
      b[i]-=factor*b[k];}
  }
  for(std::size_t i=n;i-->0;){for(std::size_t j=i+1;j<n;++j)b[i]-=a[i*n+j]*b[j];b[i]/=a[i*n+i];
    if(!std::isfinite(b[i]))throw std::runtime_error("Non-finite joined NS increment");}
  return b;
}
}
struct Pair::Impl {
  std::array<Patch,2> patches;
  std::array<std::vector<std::size_t>,2> vm,pm,dm;
  std::vector<Vec3> vn,pn;std::vector<BoundaryDof> boundary;std::vector<unsigned> masks;
  double halfWidth=0;Limits limits;
  Impl(Patch left,Patch right,Limits requested):patches{std::move(left),std::move(right)},limits(requested){
    const auto& l=patches[0].options();const auto& r=patches[1].options();
    const auto& a=patches[0].bounds();const auto& b=patches[1].bounds();
    if(!sameChart(patches[0].chart(),patches[1].chart()))throw std::invalid_argument("Joined patches must use the same exact physical chart");
    if(a.xiUpper!=b.xiLower||a.etaLower!=b.etaLower||a.etaUpper!=b.etaUpper)
      throw std::invalid_argument("Joined patches must meet exactly without a gap, overlap, or footprint mismatch");
    if(l.ny!=r.ny||l.nz!=r.nz||2*l.halfWidth/l.nx!=2*r.halfWidth/r.nx)
      throw std::invalid_argument("Joined patches require conforming matching element sizes and face nodes");
    if(l.density!=r.density||l.dynamicViscosity!=r.dynamicViscosity||l.timeStep!=r.timeStep
      ||l.referenceVelocity!=r.referenceVelocity||l.nonlinearTolerance!=r.nonlinearTolerance
      ||l.maximumNewtonIterations!=r.maximumNewtonIterations)
      throw std::invalid_argument("Joined patch material, time integration, and numerical solve options must agree");
    const std::size_t nx=add(l.nx,r.nx),ny=l.ny,nz=l.nz;
    const auto vx=add(mul(2,nx),1),vy=add(mul(2,ny),1),vz=add(mul(2,nz),1);
    const auto nv=mul(mul(vx,vy),vz),np=mul(mul(add(nx,1),add(ny,1)),add(nz,1)),n=add(mul(3,nv),np);
    // At most the old full matrix, a trial full matrix, and one local matrix
    // coexist in Newton. A reduced dense solve is never larger than full.
    if(n>limits.maximumUnknowns||mul(mul(mul(3,n),n),sizeof(double))>limits.maximumMatrixBytes)
      throw std::length_error("Joined patch exceeds explicit global dense resource budget; no coarsening was applied");
    vn.resize(nv);pn.resize(np);masks.resize(nv);halfWidth=(b.xiUpper-a.xiLower)/2;
    for(unsigned side=0;side<2;++side){
      const auto& patch=patches[side];const auto& o=patch.options();const std::size_t offset=side?l.nx:0;
      vm[side].resize(patch.velocityNodes().size());pm[side].resize(patch.pressureNodes().size());
      for(std::size_t k=0;k<2*o.nz+1ull;++k)for(std::size_t j=0;j<2*o.ny+1ull;++j)for(std::size_t i=0;i<2*o.nx+1ull;++i){
        const auto local=idx(i,j,k,2*o.nx+1ull,2*o.ny+1ull),global=idx(i+2*offset,j,k,vx,vy);
        vm[side][local]=global;vn[global]=patch.velocityNodes()[local];
      }
      for(std::size_t k=0;k<o.nz+1ull;++k)for(std::size_t j=0;j<o.ny+1ull;++j)for(std::size_t i=0;i<o.nx+1ull;++i){
        const auto local=idx(i,j,k,o.nx+1ull,o.ny+1ull),global=idx(i+offset,j,k,nx+1,ny+1);
        pm[side][local]=global;pn[global]=patch.pressureNodes()[local];
      }
      dm[side].resize(3*vm[side].size()+pm[side].size());
      for(std::size_t i=0;i<vm[side].size();++i)for(unsigned d=0;d<3;++d)dm[side][3*i+d]=3*vm[side][i]+d;
      for(std::size_t i=0;i<pm[side].size();++i)dm[side][3*vm[side].size()+i]=3*nv+pm[side][i];
    }
    for(std::size_t k=0;k<vz;++k)for(std::size_t j=0;j<vy;++j)for(std::size_t i=0;i<vx;++i){
      const auto node=idx(i,j,k,vx,vy);masks[node]=(i==0?1u:0u)|(i+1==vx?2u:0u)|(j==0?4u:0u)|(j+1==vy?8u:0u)|(k==0?16u:0u)|(k+1==vz?32u:0u);
      if(masks[node])boundary.push_back({node,vn[node],masks[node]});
    }
  }
  void validate(const State& state) const {
    if(state.velocity.size()!=vn.size()||state.pressure.size()!=pn.size())throw std::invalid_argument("Joined state does not match global mesh");
    for(auto u:state.velocity)if(!finite(u))throw std::invalid_argument("Non-finite joined velocity");
    for(double p:state.pressure)if(!std::isfinite(p))throw std::invalid_argument("Non-finite joined pressure");
    if(!finite(state.pressureGradientOffset))throw std::invalid_argument("Non-finite joined pressure offset");
  }
  void validateProblem(const Problem& p,const State* previous) const {
    if(!p.force||!p.boundaryVelocity||!std::isfinite(p.pressureGauge)||!finite(p.pressureGradientOffset)||p.velocityFaceMask>63)
      throw std::invalid_argument("Invalid joined physical problem");
    if(p.velocityFaceMask!=63&&!p.boundaryTraction)throw std::invalid_argument("Joined traction boundaries require physical data");
    if(patches[0].options().timeStep>0&&!previous)throw std::invalid_argument("Joined transient solve needs previous global state");
    if(previous)validate(*previous);
  }
  Problem localProblem(const Problem& global,unsigned side) const {
    Problem p=global;
    // No external load is applied at the artificial interface. These flags
    // affect boundary loading only: linearize() retains EVERY local DOF.
    p.velocityFaceMask|=1u<<(side?0:1);return p;
  }
  State local(const State& global,unsigned side) const {
    if(side>1)throw std::out_of_range("Joined patch side must be zero or one");
    validate(global);
    State s;s.pressureGradientOffset=global.pressureGradientOffset;s.velocity.resize(vm[side].size());s.pressure.resize(pm[side].size());
    for(std::size_t i=0;i<s.velocity.size();++i)s.velocity[i]=global.velocity[vm[side][i]];
    for(std::size_t i=0;i<s.pressure.size();++i)s.pressure[i]=global.pressure[pm[side][i]];
    return s;
  }
};
Pair::Pair(Patch a,Patch b,Limits limits):impl_(std::make_shared<Impl>(std::move(a),std::move(b),limits)){}
const std::vector<Vec3>& Pair::velocityNodes() const{return impl_->vn;}
const std::vector<Vec3>& Pair::pressureNodes() const{return impl_->pn;}
const std::vector<BoundaryDof>& Pair::boundaryDofs() const{return impl_->boundary;}
const std::vector<std::size_t>& Pair::velocityMap(unsigned side) const {if(side>1)throw std::out_of_range("Invalid patch side");return impl_->vm[side];}
const std::vector<std::size_t>& Pair::pressureMap(unsigned side) const {if(side>1)throw std::out_of_range("Invalid patch side");return impl_->pm[side];}
State Pair::localState(const State& s,unsigned side) const{return impl_->local(s,side);}
State Pair::initialState(const Problem& p) const {
  // Previous velocity is needed only by time integration, not initialization.
  if(!p.boundaryVelocity||!std::isfinite(p.pressureGauge)||!finite(p.pressureGradientOffset)||p.velocityFaceMask>63)
    throw std::invalid_argument("Invalid joined initial boundary data");
  State s;s.pressureGradientOffset=p.pressureGradientOffset;s.velocity.resize(impl_->vn.size());
  s.pressure.assign(impl_->pn.size(),p.velocityFaceMask==63?p.pressureGauge-dot(p.pressureGradientOffset,impl_->pn.front()):0);
  for(const auto& b:impl_->boundary)if(b.faceMask&p.velocityFaceMask){s.velocity[b.node]=p.boundaryVelocity(b.point);if(!finite(s.velocity[b.node]))throw std::invalid_argument("Non-finite joined boundary velocity");}
  return s;
}
Linearization Pair::linearize(const State& state,const Problem& p,const State* previous) const {
  impl_->validate(state);impl_->validateProblem(p,previous);
  if(!same(state.pressureGradientOffset,p.pressureGradientOffset))throw std::invalid_argument("Joined state and problem pressure offsets differ");
  Linearization result;result.velocityNodes=impl_->vn.size();result.pressureNodes=impl_->pn.size();const auto n=result.dimension();
  result.matrix.assign(n*n,0);result.residual.assign(n,0);
  for(unsigned side=0;side<2;++side){
    const auto s=impl_->local(state,side);const auto old=previous?impl_->local(*previous,side):State{};
    const auto a=impl_->patches[side].linearize(s,impl_->localProblem(p,side),previous?&old:nullptr);const auto& map=impl_->dm[side];
    for(std::size_t i=0;i<map.size();++i){result.residual[map[i]]+=a.residual[i];
      for(std::size_t j=0;j<map.size();++j)result.matrix[map[i]*n+map[j]]+=a.matrix[i*map.size()+j];}
  }
  return result;
}
SolveResult Pair::solve(const Problem& p,const State* initial,const State* previous) const {
  impl_->validateProblem(p,previous);State s=initial?*initial:initialState(p);impl_->validate(s);
  for(const auto& b:impl_->boundary)if(b.faceMask&p.velocityFaceMask)s.velocity[b.node]=p.boundaryVelocity(b.point);
  if(p.velocityFaceMask==63)s.pressure[0]=p.pressureGauge-dot(p.pressureGradientOffset,impl_->pn.front());
  const auto& o=impl_->patches[0].options();const auto nv=3*impl_->vn.size();const double uScale=o.referenceVelocity;
  const double pScale=o.density*uScale*uScale+o.dynamicViscosity*uScale/impl_->halfWidth;
  SolveResult result;result.momentumResidualScale=pScale*impl_->halfWidth*impl_->halfWidth;
  result.continuityResidualScale=uScale*impl_->halfWidth*impl_->halfWidth;
  for(double scale:{pScale,result.momentumResidualScale,result.continuityResidualScale})if(!(scale>0&&std::isfinite(scale)))throw std::invalid_argument("Joined numerical normalization is not representable");
  std::vector<std::size_t> free;
  for(std::size_t i=0;i<impl_->vn.size();++i)if(!(impl_->masks[i]&p.velocityFaceMask))for(unsigned d=0;d<3;++d)free.push_back(3*i+d);
  for(std::size_t i=0;i<impl_->pn.size();++i)if(p.velocityFaceMask!=63||i!=0)free.push_back(nv+i);
  result.unknowns=free.size();
  const auto rowScale=[&](std::size_t i){return i<nv?result.momentumResidualScale:result.continuityResidualScale;};
  const auto columnScale=[&](std::size_t i){return i<nv?uScale:pScale;};
  const auto norm=[&](const Linearization& a){double r=0;for(auto i:free)r=std::hypot(r,a.residual[i]/rowScale(i));if(!std::isfinite(r))throw std::runtime_error("Non-finite joined residual");return r;};
  for(unsigned iteration=0;iteration<=o.maximumNewtonIterations;++iteration){
    auto a=linearize(s,p,previous);const double residual=norm(a);if(iteration==0)result.initialResidual=residual;
    const double target=o.nonlinearTolerance*std::max(1.,result.initialResidual);
    if(residual<=target){
      result.state=std::move(s);result.newtonIterations=iteration;result.finalResidual=residual;
      for(std::size_t i=nv;i<a.residual.size();++i)result.maximumContinuityMoment=std::max(result.maximumContinuityMoment,std::abs(a.residual[i]));
      if(result.maximumContinuityMoment/result.continuityResidualScale>target)throw std::runtime_error("Joined prescribed flux violates incompressibility; state refused");
      return result;
    }
    if(iteration==o.maximumNewtonIterations)throw std::runtime_error("Joined full NS Newton solve failed to converge");
    std::vector<double> matrix(free.size()*free.size()),rhs(free.size());
    for(std::size_t i=0;i<free.size();++i){rhs[i]=-a.residual[free[i]]/rowScale(free[i]);
      for(std::size_t j=0;j<free.size();++j)matrix[i*free.size()+j]=a.matrix[free[i]*a.dimension()+free[j]]*columnScale(free[j])/rowScale(free[i]);}
    const auto increment=denseSolve(std::move(matrix),std::move(rhs));bool accepted=false;double fraction=1;
    for(unsigned trial=0;trial<18;++trial){State candidate=s;
      for(std::size_t j=0;j<free.size();++j){const auto i=free[j];const double delta=fraction*columnScale(i)*increment[j];
        if(i<nv)candidate.velocity[i/3][i%3]+=delta;else candidate.pressure[i-nv]+=delta;}
      if(norm(linearize(candidate,p,previous))<residual*(1-1e-4*fraction)){s=std::move(candidate);accepted=true;break;}fraction*=.5;
    }
    if(!accepted)throw std::runtime_error("Joined NS Newton line search failed; no approximate state accepted");
  }
  throw std::logic_error("Unreachable joined NS state");
}
std::vector<BoundaryLoad> Pair::boundaryLoads(const State& s,const Problem& p,const State* previous) const {
  const auto a=linearize(s,p,previous);std::vector<BoundaryLoad> loads;
  for(const auto& b:impl_->boundary)if(b.faceMask&p.velocityFaceMask)loads.push_back({b.node,{a.residual[3*b.node],a.residual[3*b.node+1],a.residual[3*b.node+2]}});
  return loads;
}
std::vector<BoundarySample> Pair::boundarySamples(const State& s) const {
  std::vector<BoundarySample> samples;
  for(unsigned side=0;side<2;++side){
    for(const auto& b:impl_->patches[side].boundarySamples(impl_->local(s,side)))if(b.face!=(side?0u:1u))samples.push_back(b);
  }
  return samples;
}
std::vector<InterfaceSample> Pair::interfaceSamples(const State& s) const {
  std::array<std::vector<BoundarySample>,2> faces;
  for(unsigned side=0;side<2;++side)for(const auto& b:impl_->patches[side].boundarySamples(impl_->local(s,side)))if(b.face==(side?0u:1u))faces[side].push_back(b);
  if(faces[0].size()!=faces[1].size())throw std::logic_error("Conforming interface quadrature mismatch");
  std::vector<InterfaceSample> result;for(std::size_t i=0;i<faces[0].size();++i)result.push_back({faces[0][i],faces[1][i]});return result;
}
} // namespace faraday_native::resolved_gap_join
