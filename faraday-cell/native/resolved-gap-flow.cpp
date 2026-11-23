#include "resolved-gap-flow.hpp"
#include <algorithm>
#include <limits>
#include <numeric>

namespace faraday_native::resolved_gap_flow {
namespace {
// Curved spherical metrics are not polynomials. Six-point quadrature is
// verified against the stationary pressure/body-force balance as well as
// spatial refinement; low-order polynomial exactness alone is insufficient.
constexpr unsigned quadratureOrder=6;
constexpr double qx[quadratureOrder]={.0337652428984239861,.169395306766867743,.380690406958401546,.619309593041598454,.830604693233132257,.966234757101576014};
constexpr double qw[quadratureOrder]={.0856622461895851725,.180380786524069304,.233956967286345524,.233956967286345524,.180380786524069304,.0856622461895851725};
std::size_t product(std::size_t a,std::size_t b){
  if(b&&a>std::numeric_limits<std::size_t>::max()/b)throw std::length_error("Gap patch size overflow");
  return a*b;
}
std::size_t sumSize(std::size_t a,std::size_t b){
  if(a>std::numeric_limits<std::size_t>::max()-b)throw std::length_error("Gap patch size overflow");
  return a+b;
}
void budget(std::size_t n,const Options& o){
  if(n>o.maximumUnknowns||product(product(n,n),sizeof(double))>o.maximumMatrixBytes)
    throw std::length_error("Requested gap patch exceeds explicit dense-reference resource limit; no coarsening was applied");
}
std::array<double,3> q2(double s){return {2*(s-.5)*(s-1),4*s*(1-s),2*s*(s-.5)};}
std::array<double,3> dq2(double s){return {4*s-3,4-8*s,4*s-1};}
std::size_t index(unsigned i,unsigned j,unsigned k,unsigned nx,unsigned ny){return (static_cast<std::size_t>(k)*ny+j)*nx+i;}
double magnitude(const std::vector<double>& a){
  double result=0;for(double x:a)result=std::hypot(result,x);
  if(!std::isfinite(result))throw std::runtime_error("Mapped NS residual norm is not representable");
  return result;
}
std::vector<double> denseSolve(std::vector<double> a,std::vector<double> b){
  const std::size_t n=b.size();double largest=0;for(double x:a){
    if(!std::isfinite(x))throw std::runtime_error("Non-finite mapped NS matrix entry");
    largest=std::max(largest,std::abs(x));
  }
  if(!(largest>0&&std::isfinite(largest)))throw std::runtime_error("Invalid mapped NS matrix");
  const double threshold=64*std::numeric_limits<double>::epsilon()*largest;
  for(std::size_t k=0;k<n;++k){
    std::size_t pivot=k;for(std::size_t i=k+1;i<n;++i)if(std::abs(a[i*n+k])>std::abs(a[pivot*n+k]))pivot=i;
    if(!(std::abs(a[pivot*n+k])>threshold))throw std::runtime_error("Mapped mixed NS matrix is singular or numerically unresolved");
    if(pivot!=k){for(std::size_t j=k;j<n;++j)std::swap(a[k*n+j],a[pivot*n+j]);std::swap(b[k],b[pivot]);}
    const double diagonal=a[k*n+k];
    for(std::size_t i=k+1;i<n;++i){
      const double multiplier=a[i*n+k]/diagonal;a[i*n+k]=0;
      if(multiplier==0)continue;
      for(std::size_t j=k+1;j<n;++j)a[i*n+j]-=multiplier*a[k*n+j];
      b[i]-=multiplier*b[k];
    }
  }
  for(std::size_t i=n;i-->0;){
    for(std::size_t j=i+1;j<n;++j)b[i]-=a[i*n+j]*b[j];
    b[i]/=a[i*n+i];if(!std::isfinite(b[i]))throw std::runtime_error("Non-finite mapped NS increment");
  }
  return b;
}
struct Element {std::array<std::size_t,27> velocity;std::array<std::size_t,8> pressure;unsigned i,j,k;};
struct Point {
  std::array<double,27> n{};std::array<Vec3,27> gradient{};
  std::array<double,8> p{};std::array<Vec3,8> pGradient{};
  Vec3 position{};double weight=0;
};
struct Assembly {std::vector<double> matrix,residual;std::vector<std::ptrdiff_t> map;std::size_t n=0;};
}
struct Patch::Impl {
  resolved_gap::Chart geometry;Options opt;
  Bounds footprint;
  std::vector<Vec3> vn,pn;std::vector<BoundaryDof> boundary;
  std::vector<unsigned> masks;std::vector<Element> elements;
  std::vector<std::array<Point,quadratureOrder*quadratureOrder*quadratureOrder>> quadrature;
  double hx,hy,hz;std::size_t freeUnknowns=0;
  Impl(resolved_gap::Chart c,Options o):geometry(c),opt(o){
    if(o.nx<2||o.ny<2||o.nz<2)throw std::invalid_argument("Q2/Q1 patch requires at least two elements per axis");
    for(double v:{o.halfWidth,o.density,o.dynamicViscosity,o.nonlinearTolerance,o.referenceVelocity})
      if(!(v>0&&std::isfinite(v)))throw std::invalid_argument("Invalid positive mapped NS parameter");
    if(o.nonlinearTolerance>=1)throw std::invalid_argument("Mapped NS relative nonlinear tolerance must be smaller than one");
    if(!(o.timeStep>=0&&std::isfinite(o.timeStep))||o.maximumNewtonIterations==0)
      throw std::invalid_argument("Invalid mapped NS time step or Newton iteration limit");
    const double halfHeight=o.halfWidthEta.value_or(o.halfWidth);
    if(!(halfHeight>0&&std::isfinite(halfHeight)&&std::isfinite(o.centerXi)&&std::isfinite(o.centerEta)))
      throw std::invalid_argument("Invalid exact-chart footprint");
    footprint={o.centerXi-o.halfWidth,o.centerXi+o.halfWidth,o.centerEta-halfHeight,o.centerEta+halfHeight};
    for(double x:{footprint.xiLower,footprint.xiUpper})for(double y:{footprint.etaLower,footprint.etaUpper})
      if(!(std::hypot(x,y)<geometry.chartRadius()))throw std::invalid_argument("Patch footprint reaches spherical equator");
    // Check integer sizes and the declared resource budget BEFORE allocation.
    const std::size_t vx=2*static_cast<std::size_t>(o.nx)+1,vy=2*static_cast<std::size_t>(o.ny)+1,vz=2*static_cast<std::size_t>(o.nz)+1;
    if(vx>std::numeric_limits<unsigned>::max()||vy>std::numeric_limits<unsigned>::max()||vz>std::numeric_limits<unsigned>::max())
      throw std::length_error("Gap patch axis count overflow");
    const auto nVelocity=product(product(vx,vy),vz),nPressure=product(product(o.nx+1ull,o.ny+1ull),o.nz+1ull);
    freeUnknowns=sumSize(product(3,product(product(vx-2,vy-2),vz-2)),nPressure-1);
    budget(freeUnknowns,o);
    hx=2*o.halfWidth/o.nx;hy=2*halfHeight/o.ny;hz=1./o.nz;
    vn.resize(nVelocity);masks.resize(nVelocity);pn.resize(nPressure);
    for(unsigned k=0;k<vz;++k)for(unsigned j=0;j<vy;++j)for(unsigned i=0;i<vx;++i){
      const auto n=index(i,j,k,static_cast<unsigned>(vx),static_cast<unsigned>(vy));
      vn[n]=geometry.metric(footprint.xiLower+i*hx/2,footprint.etaLower+j*hy/2,k*hz/2).point;
      masks[n]=(i==0?1u:0u)|(i+1==vx?2u:0u)|(j==0?4u:0u)|(j+1==vy?8u:0u)|(k==0?16u:0u)|(k+1==vz?32u:0u);
      if(masks[n])boundary.push_back({n,vn[n],masks[n]});
    }
    for(unsigned k=0;k<=o.nz;++k)for(unsigned j=0;j<=o.ny;++j)for(unsigned i=0;i<=o.nx;++i)
      pn[index(i,j,k,o.nx+1,o.ny+1)]=geometry.metric(footprint.xiLower+i*hx,footprint.etaLower+j*hy,k*hz).point;
    for(unsigned k=0;k<o.nz;++k)for(unsigned j=0;j<o.ny;++j)for(unsigned i=0;i<o.nx;++i){
      Element e;e.i=i;e.j=j;e.k=k;unsigned n=0;
      for(unsigned c1=0;c1<3;++c1)for(unsigned b=0;b<3;++b)for(unsigned a=0;a<3;++a)
        e.velocity[n++]=index(2*i+a,2*j+b,2*k+c1,2*o.nx+1,2*o.ny+1);
      n=0;for(unsigned c1=0;c1<2;++c1)for(unsigned b=0;b<2;++b)for(unsigned a=0;a<2;++a)
        e.pressure[n++]=index(i+a,j+b,k+c1,o.nx+1,o.ny+1);
      elements.push_back(e);std::array<Point,quadratureOrder*quadratureOrder*quadratureOrder> points;n=0;
      for(unsigned c1=0;c1<quadratureOrder;++c1)for(unsigned b=0;b<quadratureOrder;++b)for(unsigned a=0;a<quadratureOrder;++a)
        points[n++]=point(e,qx[a],qx[b],qx[c1],qw[a]*qw[b]*qw[c1]);
      quadrature.push_back(std::move(points));
    }
  }
  Point point(const Element& e,double a,double b,double c,double weight=1) const {
    Point out;const auto metric=geometry.metric(footprint.xiLower+(e.i+a)*hx,footprint.etaLower+(e.j+b)*hy,(e.k+c)*hz);
    out.position=metric.point;out.weight=weight*hx*hy*hz*metric.determinant;
    const auto na=q2(a),nb=q2(b),nc=q2(c),da=dq2(a),db=dq2(b),dc=dq2(c);
    unsigned n=0;
    for(unsigned k=0;k<3;++k)for(unsigned j=0;j<3;++j)for(unsigned i=0;i<3;++i){
      out.n[n]=na[i]*nb[j]*nc[k];
      out.gradient[n++]=metric.physicalGradient({da[i]*nb[j]*nc[k]/hx,na[i]*db[j]*nc[k]/hy,na[i]*nb[j]*dc[k]/hz});
    }
    const std::array<double,2> pa{1-a,a},pb{1-b,b},pc{1-c,c},d{-1,1};n=0;
    for(unsigned k=0;k<2;++k)for(unsigned j=0;j<2;++j)for(unsigned i=0;i<2;++i){
      out.p[n]=pa[i]*pb[j]*pc[k];
      out.pGradient[n++]=metric.physicalGradient({d[i]*pb[j]*pc[k]/hx,pa[i]*d[j]*pc[k]/hy,pa[i]*pb[j]*d[k]/hz});
    }
    return out;
  }
  void validate(const State& state) const {
    if(state.velocity.size()!=vn.size()||state.pressure.size()!=pn.size())throw std::invalid_argument("Mapped NS state mesh mismatch");
    for(auto v:state.velocity)if(!finite(v))throw std::invalid_argument("Non-finite mapped NS velocity");
    for(auto p:state.pressure)if(!std::isfinite(p))throw std::invalid_argument("Non-finite mapped NS pressure");
    if(!finite(state.pressureGradientOffset))throw std::invalid_argument("Non-finite mapped NS pressure offset");
  }
  void validateProblem(const Problem& p,const State* previous) const {
    if(!p.force||!p.boundaryVelocity||!std::isfinite(p.pressureGauge))throw std::invalid_argument("Mapped NS forcing, boundary data and pressure gauge are required");
    if(p.velocityFaceMask>63|| (p.velocityFaceMask!=63&&!p.boundaryTraction))
      throw std::invalid_argument("Every non-velocity face requires physical traction data");
    if(!finite(p.pressureGradientOffset))throw std::invalid_argument("Non-finite prescribed pressure offset");
    if(opt.timeStep>0&&!previous)throw std::invalid_argument("Backward Euler requires an explicit previous state");
    if(previous)validate(*previous);
  }
  Field field(const State& s,const Element& e,const Point& p) const {
    Field f;for(unsigned n=0;n<27;++n){const auto v=s.velocity[e.velocity[n]];f.velocity=f.velocity+p.n[n]*v;
      for(unsigned a=0;a<3;++a)f.velocityGradient[a]=f.velocityGradient[a]+v[a]*p.gradient[n];}
    for(unsigned n=0;n<8;++n){f.pressure+=p.p[n]*s.pressure[e.pressure[n]];f.pressureGradient=f.pressureGradient+s.pressure[e.pressure[n]]*p.pGradient[n];}
    f.pressure+=dot(s.pressureGradientOffset,p.position);f.pressureGradient=f.pressureGradient+s.pressureGradientOffset;
    if(!finite(f.velocity)||!finite(f.pressureGradient)||!std::isfinite(f.pressure))throw std::overflow_error("Physical mapped NS field is not representable");
    for(auto gradient:f.velocityGradient)if(!finite(gradient))throw std::overflow_error("Physical mapped NS velocity gradient is not representable");
    return f;
  }
  template<class F> void forEachBoundary(F visit) const {
    for(const auto& e:elements)for(unsigned face=0;face<6;++face){
      if((face==0&&e.i!=0)||(face==1&&e.i+1!=opt.nx)||(face==2&&e.j!=0)||(face==3&&e.j+1!=opt.ny)||(face==4&&e.k!=0)||(face==5&&e.k+1!=opt.nz))continue;
      const unsigned axis=face/2;const double sign=face%2?1:-1;
      for(unsigned v=0;v<quadratureOrder;++v)for(unsigned u=0;u<quadratureOrder;++u){
        Vec3 local;local[axis]=face%2;local[(axis+1)%3]=qx[u];local[(axis+2)%3]=qx[v];
        const auto p=point(e,local.x,local.y,local.z);
        const auto m=geometry.metric(footprint.xiLower+(e.i+local.x)*hx,footprint.etaLower+(e.j+local.y)*hy,(e.k+local.z)*hz);
        const Vec3 areaVector=sign*m.cofactor[axis];const double area=resolved_gap::length(areaVector);
        const std::array<double,3> widths{hx,hy,hz};
        visit(e,p,face,areaVector/area,area*widths[(axis+1)%3]*widths[(axis+2)%3]*qw[u]*qw[v]);
      }
    }
  }
  Assembly assemble(const State& s,const Problem& problem,const State* previous,bool matrix,bool unrestricted) const {
    validate(s);validateProblem(problem,previous);Assembly out;
    if(s.pressureGradientOffset.x!=problem.pressureGradientOffset.x||s.pressureGradientOffset.y!=problem.pressureGradientOffset.y||s.pressureGradientOffset.z!=problem.pressureGradientOffset.z)
      throw std::invalid_argument("State and problem use different physical pressure decompositions");
    const auto nv=3*vn.size(),total=nv+pn.size();out.map.assign(total,-1);
    for(std::size_t n=0;n<vn.size();++n)if(unrestricted||!(masks[n]&problem.velocityFaceMask))for(unsigned a=0;a<3;++a)out.map[3*n+a]=out.n++;
    for(std::size_t n=0;n<pn.size();++n)if(unrestricted||problem.velocityFaceMask!=63||n!=0)out.map[nv+n]=out.n++;
    if(matrix){budget(out.n,opt);out.matrix.assign(product(out.n,out.n),0);}
    out.residual.assign(out.n,0);
    const auto addR=[&](std::size_t row,double value){if(out.map[row]>=0)out.residual[static_cast<std::size_t>(out.map[row])]+=value;};
    const auto addA=[&](std::size_t row,std::size_t col,double value){if(matrix&&out.map[row]>=0&&out.map[col]>=0)
      out.matrix[static_cast<std::size_t>(out.map[row])*out.n+static_cast<std::size_t>(out.map[col])]+=value;};
    const double mass=opt.timeStep>0?opt.density/opt.timeStep:0;
    for(std::size_t el=0;el<elements.size();++el){const auto& e=elements[el];
      for(const auto& p:quadrature[el]){
        const auto f=field(s,e,p);const Vec3 force=problem.force(p.position)-problem.pressureGradientOffset;
        double pressure=0;for(unsigned k=0;k<8;++k)pressure+=p.p[k]*s.pressure[e.pressure[k]];
        if(!finite(force))throw std::invalid_argument("Non-finite mapped NS body force");
        const Vec3 old=previous?field(*previous,e,p).velocity:Vec3{};
        const double div=f.velocityGradient[0].x+f.velocityGradient[1].y+f.velocityGradient[2].z;
        for(unsigned i=0;i<27;++i)for(unsigned a=0;a<3;++a){
          const auto row=3*e.velocity[i]+a;
          double viscous=0;for(unsigned b=0;b<3;++b)viscous+=opt.dynamicViscosity*p.gradient[i][b]*(f.velocityGradient[a][b]+f.velocityGradient[b][a]);
          // Conservative momentum convection. The matching exterior momentum
          // flux is assembled below, so weakly divergence-free Q2 velocities
          // still conserve integral momentum without assuming div(u)==0 pointwise.
          addR(row,p.weight*(p.n[i]*(mass*(f.velocity[a]-old[a])-force[a])
            -opt.density*f.velocity[a]*dot(f.velocity,p.gradient[i])+viscous-pressure*p.gradient[i][a]));
          if(matrix){
            for(unsigned j=0;j<27;++j)for(unsigned b=0;b<3;++b){
              double value=opt.dynamicViscosity*p.gradient[i][b]*p.gradient[j][a]-opt.density*p.n[j]*f.velocity[a]*p.gradient[i][b];
              if(a==b)value+=mass*p.n[i]*p.n[j]+opt.dynamicViscosity*dot(p.gradient[i],p.gradient[j])-opt.density*p.n[j]*dot(f.velocity,p.gradient[i]);
              addA(row,3*e.velocity[j]+b,p.weight*value);
            }
            for(unsigned k=0;k<8;++k){const double coupling=p.weight*p.p[k]*p.gradient[i][a];
              // The SAME quadrature contribution enters G=-B^T, with no
              // stabilization or independently approximated pressure gradient.
              addA(row,nv+e.pressure[k],-coupling);addA(nv+e.pressure[k],row,coupling);
            }
          }
        }
        for(unsigned k=0;k<8;++k)addR(nv+e.pressure[k],p.weight*p.p[k]*div);
      }
    }
    forEachBoundary([&](const Element& e,const Point& p,unsigned face,Vec3 normal,double weight){
      const auto f=field(s,e,p);const double un=dot(f.velocity,normal);
      const Vec3 traction=(problem.velocityFaceMask&(1u<<face))?Vec3{}:problem.boundaryTraction(face,p.position,normal);
      // On traction faces this is the transformed prescribed traction. On
      // velocity faces it reconstructs the physical nodal reaction; all
      // unconstrained test functions vanish there, so no equation is added.
      const Vec3 pressureOffsetTraction=dot(problem.pressureGradientOffset,p.position)*normal;
      if(!finite(traction))throw std::invalid_argument("Non-finite prescribed physical boundary traction");
      for(unsigned i=0;i<27;++i)if(p.n[i]!=0)for(unsigned a=0;a<3;++a){
        addR(3*e.velocity[i]+a,weight*p.n[i]*(opt.density*f.velocity[a]*un-traction[a]-pressureOffsetTraction[a]));
        if(matrix)for(unsigned j=0;j<27;++j)if(p.n[j]!=0)for(unsigned b=0;b<3;++b)
          addA(3*e.velocity[i]+a,3*e.velocity[j]+b,weight*opt.density*p.n[i]*p.n[j]*(f.velocity[a]*normal[b]+(a==b?un:0)));
      }
    });
    for(double value:out.residual)if(!std::isfinite(value))throw std::runtime_error("Non-finite mapped NS residual");
    return out;
  }
};
Patch::Patch(resolved_gap::Chart c,Options o):impl_(std::make_shared<Impl>(c,o)){}
const Options& Patch::options() const{return impl_->opt;}
const resolved_gap::Chart& Patch::chart() const{return impl_->geometry;}
const Bounds& Patch::bounds() const{return impl_->footprint;}
const std::vector<Vec3>& Patch::velocityNodes() const{return impl_->vn;}
const std::vector<Vec3>& Patch::pressureNodes() const{return impl_->pn;}
const std::vector<BoundaryDof>& Patch::boundaryDofs() const{return impl_->boundary;}
State Patch::initialState(const Problem& p) const {
  if(!p.boundaryVelocity||!std::isfinite(p.pressureGauge))throw std::invalid_argument("Missing mapped NS boundary data");
  if(p.velocityFaceMask>63)throw std::invalid_argument("Invalid mapped NS boundary face mask");
  if(!finite(p.pressureGradientOffset))throw std::invalid_argument("Non-finite prescribed pressure offset");
  State s;s.pressureGradientOffset=p.pressureGradientOffset;s.velocity.resize(impl_->vn.size());
  s.pressure.assign(impl_->pn.size(),p.velocityFaceMask==63?p.pressureGauge-dot(p.pressureGradientOffset,impl_->pn.front()):0);
  for(const auto& b:impl_->boundary)if(b.faceMask&p.velocityFaceMask){s.velocity[b.node]=p.boundaryVelocity(b.point);if(!finite(s.velocity[b.node]))throw std::invalid_argument("Non-finite prescribed patch boundary velocity");}
  return s;
}
SolveResult Patch::solve(const Problem& problem,const State* initial,const State* previous) const {
  impl_->validateProblem(problem,previous);State s=initial?*initial:initialState(problem);impl_->validate(s);
  // Boundary values are prescribed inputs, never inferred from the solution.
  for(const auto& b:impl_->boundary)if(b.faceMask&problem.velocityFaceMask)s.velocity[b.node]=problem.boundaryVelocity(b.point);
  if(problem.velocityFaceMask==63)s.pressure[0]=problem.pressureGauge-dot(problem.pressureGradientOffset,impl_->pn.front());
  SolveResult result;
  const auto& o=impl_->opt;const double velocityScale=o.referenceVelocity;
  const double pressureScale=o.density*velocityScale*velocityScale+o.dynamicViscosity*velocityScale/o.halfWidth;
  result.momentumResidualScale=pressureScale*o.halfWidth*o.halfWidth;
  result.continuityResidualScale=velocityScale*o.halfWidth*o.halfWidth;
  for(double scale:{pressureScale,result.momentumResidualScale,result.continuityResidualScale})
    if(!(scale>0&&std::isfinite(scale)))throw std::invalid_argument("Mapped NS normalization is not representable");
  const auto rowScales=[&](const Assembly& a){
    std::vector<double> scale(a.n);for(std::size_t i=0;i<a.map.size();++i)if(a.map[i]>=0)
      scale[static_cast<std::size_t>(a.map[i])]=i<3*impl_->vn.size()?result.momentumResidualScale:result.continuityResidualScale;
    return scale;
  };
  const auto scaledNorm=[&](const Assembly& a){
    const auto scales=rowScales(a);auto scaled=a.residual;for(std::size_t i=0;i<a.n;++i)scaled[i]/=scales[i];return magnitude(scaled);
  };
  for(unsigned iteration=0;iteration<=impl_->opt.maximumNewtonIterations;++iteration){
    auto assembled=impl_->assemble(s,problem,previous,true,false);result.unknowns=assembled.n;const double residual=scaledNorm(assembled);
    if(iteration==0)result.initialResidual=residual;
    const double target=impl_->opt.nonlinearTolerance*std::max(1.,result.initialResidual);
    if(residual<=target){
      result.state=std::move(s);result.newtonIterations=iteration;result.finalResidual=residual;
      const auto full=impl_->assemble(result.state,problem,previous,false,true);
      for(std::size_t n=3*impl_->vn.size();n<full.residual.size();++n)result.maximumContinuityMoment=std::max(result.maximumContinuityMoment,std::abs(full.residual[n]));
      if(result.maximumContinuityMoment/result.continuityResidualScale>target)
        throw std::runtime_error("Prescribed patch boundary flux violates an incompressibility equation; no incompatible state accepted");
      return result;
    }
    if(iteration==impl_->opt.maximumNewtonIterations)throw std::runtime_error("Mapped full NS Newton solve did not converge");
    // Solve the same Jacobian equations in dimensionless variables. This also
    // prevents a unit-sized absolute residual floor or a dimensionally mixed
    // pivot criterion from accepting zero flow at micron-scale SI inputs.
    const auto scales=rowScales(assembled);std::vector<double> columnScales(assembled.n);
    for(std::size_t i=0;i<assembled.map.size();++i)if(assembled.map[i]>=0)
      columnScales[static_cast<std::size_t>(assembled.map[i])]=i<3*impl_->vn.size()?velocityScale:pressureScale;
    for(std::size_t i=0;i<assembled.n;++i){assembled.residual[i]/=-scales[i];
      for(std::size_t j=0;j<assembled.n;++j)assembled.matrix[i*assembled.n+j]*=columnScales[j]/scales[i];}
    const auto increment=denseSolve(std::move(assembled.matrix),std::move(assembled.residual));
    bool accepted=false;double fraction=1;
    // Numerical Newton globalization, not a physical velocity/gap clamp.
    for(unsigned trial=0;trial<18;++trial){
      State candidate=s;
      for(std::size_t n=0;n<candidate.velocity.size();++n)for(unsigned a=0;a<3;++a){const auto m=assembled.map[3*n+a];if(m>=0)candidate.velocity[n][a]+=fraction*velocityScale*increment[static_cast<std::size_t>(m)];}
      for(std::size_t n=0;n<candidate.pressure.size();++n){const auto m=assembled.map[3*candidate.velocity.size()+n];if(m>=0)candidate.pressure[n]+=fraction*pressureScale*increment[static_cast<std::size_t>(m)];}
      const auto next=impl_->assemble(candidate,problem,previous,false,false);
      if(scaledNorm(next)<residual*(1-1e-4*fraction)){s=std::move(candidate);accepted=true;break;}
      fraction*=.5;
    }
    if(!accepted)throw std::runtime_error("Mapped full NS Newton line search failed; no approximate state accepted");
  }
  throw std::logic_error("Unreachable mapped NS state");
}
Field Patch::sample(const State& s,double xi,double eta,double sigma) const {
  impl_->validate(s);const auto& o=impl_->opt;
  const auto bounds=impl_->footprint;
  if(!(std::isfinite(xi)&&std::isfinite(eta)&&std::isfinite(sigma)&&xi>=bounds.xiLower&&xi<=bounds.xiUpper&&eta>=bounds.etaLower&&eta<=bounds.etaUpper&&sigma>=0&&sigma<=1))
    throw std::domain_error("Requested point lies outside mapped NS patch");
  const auto locate=[](double coordinate,double lower,double upper,unsigned n){
    if(coordinate==upper)return std::pair<unsigned,double>{n-1,1};
    const double t=n*((coordinate-lower)/(upper-lower));
    const unsigned i=t==n?n-1:static_cast<unsigned>(std::floor(t));
    return std::pair<unsigned,double>{i,t-i};
  };
  const auto [i,a]=locate(xi,bounds.xiLower,bounds.xiUpper,o.nx);const auto [j,b]=locate(eta,bounds.etaLower,bounds.etaUpper,o.ny);const auto [k,c]=locate(sigma,0,1,o.nz);
  const auto& e=impl_->elements[index(i,j,k,o.nx,o.ny)];return impl_->field(s,e,impl_->point(e,a,b,c));
}
std::vector<BoundarySample> Patch::boundarySamples(const State& s) const {
  impl_->validate(s);std::vector<BoundarySample> out;const auto& o=impl_->opt;
  impl_->forEachBoundary([&](const Element& e,const Point& p,unsigned face,Vec3 normal,double weight){
    const auto f=impl_->field(s,e,p);Vec3 traction=-f.pressure*normal;
    for(unsigned a=0;a<3;++a)for(unsigned b=0;b<3;++b)traction[a]+=o.dynamicViscosity*(f.velocityGradient[a][b]+f.velocityGradient[b][a])*normal[b];
    const double un=dot(f.velocity,normal);const Vec3 advective=(o.density*un)*f.velocity;
    out.push_back({face,p.position,normal,f.velocity,traction,advective,advective-traction,weight,f.pressure,un});
  });
  return out;
}
std::vector<BoundaryLoad> Patch::boundaryLoads(const State& s,const Problem& p,const State* previous) const {
  const auto a=impl_->assemble(s,p,previous,false,true);std::vector<BoundaryLoad> out;
  for(const auto& b:impl_->boundary)if(b.faceMask&p.velocityFaceMask)out.push_back({b.node,{a.residual[3*b.node],a.residual[3*b.node+1],a.residual[3*b.node+2]}});
  return out;
}
Linearization Patch::linearize(const State& s,const Problem& p,const State* previous) const {
  auto a=impl_->assemble(s,p,previous,true,true);
  return {impl_->vn.size(),impl_->pn.size(),std::move(a.matrix),std::move(a.residual)};
}
} // namespace faraday_native::resolved_gap_flow
