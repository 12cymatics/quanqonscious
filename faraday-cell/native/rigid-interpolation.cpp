#include "rigid-interpolation.hpp"
#include "memory-budget.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>

namespace faraday_native {
namespace {
constexpr std::size_t columns=12;
using Row=std::array<double,columns>;
using Square=std::array<Row,columns>;
void positive(double value,const char* name){
  if(!(value>0)||!std::isfinite(value))throw std::invalid_argument(std::string(name)+" must be finite and positive");
}
void validateOptions(const RigidInterpolationOptions& o){
  positive(o.minimumCellsPerDiameter,"minimumCellsPerDiameter");
  positive(o.markerSpacingFactor,"markerSpacingFactor");
  positive(o.supportRadiusFactor,"supportRadiusFactor");
  positive(o.maximumCondition,"maximumCondition");
  positive(o.affineTolerance,"affineTolerance");
  if(o.markerSpacingFactor<2)throw std::invalid_argument("Rigid collocation marker spacing must be at least two local fluid spacings");
  if(o.minimumCellsPerDiameter<8)throw std::invalid_argument("Resolved-grain interpolation requires at least eight cells across a diameter; refinement convergence is still required");
}
void validateDof(const FluidDof& d){
  if(d.family<0||d.family>2||!finite(d.position)||!finite(d.direction))
    throw std::invalid_argument("Invalid independent fluid DOF geometry/family");
  positive(d.mass,"fluid DOF mass");positive(d.spacing,"fluid DOF spacing");
  if(std::abs(dot(d.direction,d.direction)-1)>1e-11)
    throw std::invalid_argument("Fluid DOF direction must be a unit Cartesian vector");
}
Row affineRow(const FluidDof& d,Vec3 point,double scale){
  const Vec3 q=(d.position-point)/scale;
  const std::array<double,4> polynomial={1,q.x,q.y,q.z};
  Row r{};
  for(std::size_t c=0;c<3;c++)for(std::size_t j=0;j<4;j++)r[c*4+j]=d.direction[c]*polynomial[j];
  return r;
}
double vectorDot(const std::vector<double>& a,const std::vector<double>& b){
  double sum=0;for(std::size_t i=0;i<a.size();i++)sum+=a[i]*b[i];return sum;
}
// An implicit balanced kd-tree needs one index per velocity DOF rather than a
// global marker-by-grid scan or a Cartesian box spanning the entire cylinder.
class DofTree {
  const std::vector<FluidDof>& d_;
  std::vector<std::size_t> index_;
  void build(std::size_t begin,std::size_t end,unsigned depth){
    if(begin==end)return;
    const std::size_t mid=begin+(end-begin)/2,axis=depth%3;
    std::nth_element(index_.begin()+begin,index_.begin()+mid,index_.begin()+end,
      [&](std::size_t a,std::size_t b){const double x=d_[a].position[axis],y=d_[b].position[axis];return x==y?a<b:x<y;});
    build(begin,mid,depth+1);build(mid+1,end,depth+1);
  }
  void query(std::size_t begin,std::size_t end,unsigned depth,Vec3 p,double radius,
      std::vector<std::size_t>& out) const{
    if(begin==end)return;
    const std::size_t mid=begin+(end-begin)/2,which=index_[mid],axis=depth%3;
    const Vec3 delta=d_[which].position-p;
    if(dot(delta,delta)<=radius*radius)out.push_back(which);
    if(delta[axis]>=-radius)query(begin,mid,depth+1,p,radius,out);
    if(delta[axis]<= radius)query(mid+1,end,depth+1,p,radius,out);
  }
public:
  explicit DofTree(const std::vector<FluidDof>& d):d_(d),index_(d.size()){
    std::iota(index_.begin(),index_.end(),0);build(0,index_.size(),0);
  }
  std::vector<std::size_t> radius(Vec3 p,double r) const{
    std::vector<std::size_t> out;query(0,index_.size(),0,p,r,out);
    std::sort(out.begin(),out.end());return out;
  }
};
void requireMarkerVector(const std::vector<Vec3>& values,std::size_t n){
  if(values.size()!=n)throw std::invalid_argument("Marker vector length does not match interpolation geometry");
  for(Vec3 x:values)if(!finite(x))throw std::invalid_argument("Marker vector contains a non-finite value");
}
}

std::vector<RigidMarker> sphereVolumeMarkers(const RigidGrain& grain,
    std::size_t grainIndex,double fluidSpacing,const RigidInterpolationOptions& options){
  grain.validate();validateOptions(options);positive(fluidSpacing,"fluid spacing");
  const double resolved=2*grain.radius/fluidSpacing;
  if(resolved<options.minimumCellsPerDiameter){
    std::ostringstream s;s<<"Unresolved grain "<<grain.id<<": diameter spans "<<resolved
      <<" local fluid cells; at least "<<options.minimumCellsPerDiameter<<" required";
    throw std::runtime_error(s.str());
  }
  const double step=options.markerSpacingFactor*fluidSpacing;
  const double extent=std::floor(grain.radius/step);
  if(extent>static_cast<double>(std::numeric_limits<int>::max()/2))
    throw std::length_error("Grain marker index range cannot be represented");
  const int count=static_cast<int>(extent);
  std::vector<RigidMarker> out;
  const auto side=static_cast<std::size_t>(2*count+1);
  const auto cube=memoryProduct(memoryProduct(side,side),side);
  requireMemory(memoryProduct(cube,sizeof(RigidMarker)),options.workingMemoryBudgetBytes);
  out.reserve(cube);
  for(int i=-count;i<=count;i++)for(int j=-count;j<=count;j++)for(int k=-count;k<=count;k++){
    const Vec3 offset={step*i,step*j,step*k};
    if(dot(offset,offset)<grain.radius*grain.radius)out.push_back({grainIndex,grain.centre+offset,offset});
  }
  if(out.size()<4)throw std::runtime_error("Too few independent volume markers to resolve rigid grain motion");
  return out;
}

InterpolationStencil affineVelocityStencil(const std::vector<FluidDof>& dofs,
    const std::vector<std::size_t>& neighbours,Vec3 point,double supportRadius,
    const RigidInterpolationOptions& options){
  validateOptions(options);positive(supportRadius,"interpolation support radius");
  if(!finite(point))throw std::invalid_argument("Non-finite marker position");
  if(neighbours.size()<columns)throw std::runtime_error("Insufficient fluid support for all twelve affine Cartesian velocity coefficients");
  requireMemory(memoryProduct(neighbours.size(),512),options.workingMemoryBudgetBytes);
  std::vector<std::size_t> ids=neighbours;
  std::sort(ids.begin(),ids.end());
  if(std::adjacent_find(ids.begin(),ids.end())!=ids.end())throw std::invalid_argument("Duplicate fluid DOF in interpolation stencil");
  const std::size_t n=ids.size();
  // QR work, orthogonal basis, exact rows, temporary indices and final entries.
  // The bound includes all simultaneous arrays rather than only final stencils.
  requireMemory(memoryProduct(n,512),options.workingMemoryBudgetBytes);
  std::array<std::vector<double>,columns> work,Q;
  for(auto& x:work)x.resize(n);
  for(auto& x:Q)x.resize(n);
  std::vector<double> sqrtWeight(n);
  std::vector<Row> exact(n);
  for(std::size_t i=0;i<n;i++){
    if(ids[i]>=dofs.size())throw std::out_of_range("Interpolation DOF index outside geometry");
    const FluidDof& d=dofs[ids[i]];validateDof(d);
    const double d2=dot(d.position-point,d.position-point);
    if(d2>supportRadius*supportRadius*(1+16*std::numeric_limits<double>::epsilon()))
      throw std::invalid_argument("Interpolation neighbour is outside declared support");
    sqrtWeight[i]=std::exp(-2*d2/(supportRadius*supportRadius));
    exact[i]=affineRow(d,point,supportRadius);
    for(std::size_t j=0;j<columns;j++)work[j][i]=sqrtWeight[i]*exact[i][j];
  }
  Square R{};std::array<std::size_t,columns> permutation{};
  std::iota(permutation.begin(),permutation.end(),0);
  double largestNorm=0;
  for(const auto& c:work)largestNorm=std::max(largestNorm,std::sqrt(vectorDot(c,c)));
  const double rankTolerance=64*std::numeric_limits<double>::epsilon()*static_cast<double>(std::max(n,columns))*largestNorm;
  for(std::size_t k=0;k<columns;k++){
    std::size_t pivot=k;double pivotNorm=vectorDot(work[k],work[k]);
    for(std::size_t j=k+1;j<columns;j++){
      const double norm2=vectorDot(work[j],work[j]);if(norm2>pivotNorm){pivot=j;pivotNorm=norm2;}
    }
    if(pivot!=k){std::swap(work[pivot],work[k]);std::swap(permutation[pivot],permutation[k]);
      for(std::size_t i=0;i<k;i++)std::swap(R[i][pivot],R[i][k]);}
    // Twice-modified Gram-Schmidt prevents loss of orthogonality from making
    // an apparently well-conditioned interpolation with inaccurate moments.
    for(unsigned pass=0;pass<2;pass++)for(std::size_t i=0;i<k;i++){
      const double r=vectorDot(Q[i],work[k]);R[i][k]+=r;
      for(std::size_t a=0;a<n;a++)work[k][a]-=r*Q[i][a];
    }
    R[k][k]=std::sqrt(vectorDot(work[k],work[k]));
    if(!(R[k][k]>rankTolerance))throw std::runtime_error("Rank-deficient affine velocity support; no regularization or lower-order interpolation substituted");
    for(std::size_t a=0;a<n;a++)Q[k][a]=work[k][a]/R[k][k];
    for(std::size_t j=k+1;j<columns;j++){
      const double r=vectorDot(Q[k],work[j]);R[k][j]+=r;
      for(std::size_t a=0;a<n;a++)work[j][a]-=r*Q[k][a];
    }
  }
  Square inverse{};
  for(std::size_t column=0;column<columns;column++)
    for(int row=static_cast<int>(columns)-1;row>=0;row--){
      double x=static_cast<std::size_t>(row)==column?1:0;
      for(std::size_t j=static_cast<std::size_t>(row)+1;j<columns;j++)x-=R[row][j]*inverse[j][column];
      inverse[row][column]=x/R[row][row];
    }
  double nr=0,ni=0;
  for(std::size_t i=0;i<columns;i++){double a=0,b=0;for(std::size_t j=0;j<columns;j++){a+=std::abs(R[i][j]);b+=std::abs(inverse[i][j]);}nr=std::max(nr,a);ni=std::max(ni,b);}
  InterpolationStencil out;out.supportRadius=supportRadius;out.condition=nr*ni;
  if(!std::isfinite(out.condition)||out.condition>options.maximumCondition)
    throw std::runtime_error("Affine velocity support exceeds declared condition bound; no regularization substituted");
  std::array<std::size_t,columns> inversePermutation{};
  for(std::size_t i=0;i<columns;i++)inversePermutation[permutation[i]]=i;
  out.entries.resize(n);
  for(std::size_t i=0;i<n;i++){
    out.entries[i].dof=ids[i];
    for(std::size_t c=0;c<3;c++){
      double value=0;const std::size_t row=inversePermutation[c*4];
      for(std::size_t j=0;j<columns;j++)value+=inverse[row][j]*Q[j][i];
      out.entries[i].coefficient[c]=value*sqrtWeight[i];
    }
  }
  for(std::size_t c=0;c<3;c++)for(std::size_t j=0;j<columns;j++){
    double value=0;for(std::size_t i=0;i<n;i++)value+=out.entries[i].coefficient[c]*exact[i][j];
    out.affineResidual=std::max(out.affineResidual,std::abs(value-(j==c*4?1.0:0.0)));
  }
  if(!std::isfinite(out.affineResidual)||out.affineResidual>options.affineTolerance)
    throw std::runtime_error("Affine velocity reconstruction failed its constant/linear moment identities");
  return out;
}

RigidInterpolation::RigidInterpolation(const std::vector<FluidDof>& dofs,
    const std::vector<RigidGrain>& grains,const RigidInterpolationOptions& options)
    :dofs_(&dofs),grainCount_(grains.size()){
  validateOptions(options);
  for(const auto& d:dofs)validateDof(d);
  for(const auto& g:grains)g.validate();
  if(!grains.empty()&&dofs.empty())throw std::invalid_argument("No independent fluid velocity DOFs");
  for(std::size_t a=0;a<grains.size();a++)for(std::size_t b=0;b<a;b++)
    if(norm(grains[a].centre-grains[b].centre)<grains[a].radius+grains[b].radius)
      throw std::invalid_argument("Overlapping physical grains cannot be repaired by interpolation");
  const auto fixedScratch=memorySum({memoryProduct(dofs.size(),544),memoryProduct(grains.size(),sizeof(double)),sizeof(*this)});
  requireMemory(fixedScratch,options.workingMemoryBudgetBytes);
  DofTree tree(dofs);
  for(std::size_t a=0;a<grains.size();a++){
    const auto& g=grains[a];const auto inside=tree.radius(g.centre,g.radius);
    if(inside.empty())throw std::runtime_error("Unresolved grain contains no independent fluid velocity nodes");
    double spacing=0;
    for(std::size_t i:inside)spacing=std::max(spacing,dofs[i].spacing);
    auto markerOptions=options;
    const auto retained=allocatedBytes();
    requireMemory(memorySum({fixedScratch,retained}),options.workingMemoryBudgetBytes);
    markerOptions.workingMemoryBudgetBytes=options.workingMemoryBudgetBytes-fixedScratch-retained;
    auto markers=sphereVolumeMarkers(g,a,spacing,markerOptions);
    const auto localMarkers=memoryProduct(markers.capacity(),sizeof(RigidMarker));
    const auto totalMarkers=memorySum({markers_.size(),markers.size()});
    const auto growth=memoryProduct(totalMarkers,sizeof(RigidMarker)+sizeof(InterpolationStencil));
    requireMemory(memorySum({fixedScratch,retained,localMarkers,growth}),options.workingMemoryBudgetBytes);
    markers_.reserve(totalMarkers);stencils_.reserve(totalMarkers);
    const double support=options.supportRadiusFactor*spacing;
    for(const auto& marker:markers){
      auto neighbours=tree.radius(marker.position,support);
      for(std::size_t i:neighbours)
        if(2*g.radius/dofs[i].spacing<options.minimumCellsPerDiameter)
          throw std::runtime_error("Grain interpolation support crosses an underresolved fluid region");
      requireMemory(memorySum({fixedScratch,allocatedBytes(),localMarkers,memoryProduct(neighbours.size(),sizeof(InterpolationEntry))}),options.workingMemoryBudgetBytes);
      auto stencil=affineVelocityStencil(dofs,neighbours,marker.position,support,options);
      markers_.push_back(marker);stencils_.push_back(std::move(stencil));
    }
    grainSpacing_.push_back(spacing);
  }
}

std::size_t RigidInterpolation::allocatedBytes() const {
  auto bytes=memorySum({sizeof(*this),memoryProduct(markers_.capacity(),sizeof(RigidMarker)),memoryProduct(stencils_.capacity(),sizeof(InterpolationStencil)),memoryProduct(grainSpacing_.capacity(),sizeof(double))});
  for(const auto& stencil:stencils_)bytes=memorySum({bytes,memoryProduct(stencil.entries.capacity(),sizeof(InterpolationEntry))});
  return bytes;
}

std::vector<Vec3> RigidInterpolation::applyJ(const std::vector<double>& values) const{
  if(values.size()!=dofs_->size())throw std::invalid_argument("Fluid vector length does not match interpolation geometry");
  for(double x:values)if(!std::isfinite(x))throw std::invalid_argument("Non-finite fluid velocity");
  std::vector<Vec3> out(markers_.size());
  for(std::size_t a=0;a<stencils_.size();a++)for(const auto& e:stencils_[a].entries)out[a]=out[a]+values[e.dof]*e.coefficient;
  for(Vec3 x:out)if(!finite(x))throw std::runtime_error("Non-finite interpolated fluid velocity");
  return out;
}
std::vector<double> RigidInterpolation::spreadTranspose(const std::vector<Vec3>& values) const{
  requireMarkerVector(values,markers_.size());std::vector<double> out(dofs_->size());
  for(std::size_t a=0;a<stencils_.size();a++)for(const auto& e:stencils_[a].entries)out[e.dof]+=dot(e.coefficient,values[a]);
  for(double x:out)if(!std::isfinite(x))throw std::runtime_error("Non-finite spread fluid impulse");
  return out;
}
std::vector<double> RigidInterpolation::weightedAdjoint(const std::vector<Vec3>& values) const{
  auto out=spreadTranspose(values);
  for(std::size_t i=0;i<out.size();i++){
    out[i]/=(*dofs_)[i].mass;
    if(!std::isfinite(out[i]))throw std::runtime_error("Non-finite mass-weighted spread impulse");
  }
  return out;
}
std::vector<Vec3> RigidInterpolation::applyK(const std::vector<Six>& values) const{
  if(values.size()!=grainCount_)throw std::invalid_argument("Rigid velocity count does not match grains");
  for(const auto& q:values)for(double x:q)if(!std::isfinite(x))throw std::invalid_argument("Non-finite rigid velocity");
  std::vector<Vec3> out;out.reserve(markers_.size());
  for(const auto& m:markers_){const auto& q=values[m.grain];out.push_back(Vec3{q[0],q[1],q[2]}+cross({q[3],q[4],q[5]},m.offset));}
  for(Vec3 x:out)if(!finite(x))throw std::runtime_error("Non-finite rigid marker velocity");
  return out;
}
std::vector<Six> RigidInterpolation::spreadKTranspose(const std::vector<Vec3>& values) const{
  requireMarkerVector(values,markers_.size());std::vector<Six> out(grainCount_);
  for(std::size_t i=0;i<markers_.size();i++){
    auto& q=out[markers_[i].grain];const Vec3 f=values[i],t=cross(markers_[i].offset,f);
    q[0]+=f.x;q[1]+=f.y;q[2]+=f.z;q[3]+=t.x;q[4]+=t.y;q[5]+=t.z;
  }
  for(const auto& q:out)for(double x:q)if(!std::isfinite(x))throw std::runtime_error("Non-finite rigid grain impulse");
  return out;
}
} // namespace faraday_native
