#include "rigid-coupler.hpp"
#include "memory-budget.hpp"
#include <algorithm>
#include <limits>
#include <sstream>

namespace faraday_native {
namespace {
double markerDot(const std::vector<Vec3>& a,const std::vector<Vec3>& b){double s=0;for(std::size_t i=0;i<a.size();i++)s+=dot(a[i],b[i]);return s;}
double markerMax(const std::vector<Vec3>& a){double s=0;for(Vec3 v:a)s=std::max(s,norm(v));return s;}
std::vector<Six> velocities(const std::vector<RigidGrain>& g){std::vector<Six> out;for(const auto& x:g)out.push_back(rigidVelocity(x));return out;}
void assignVelocities(std::vector<RigidGrain>& g,const std::vector<Six>& q){for(std::size_t i=0;i<g.size();i++)setRigidVelocity(g[i],q[i]);}
}

std::vector<FluidDof> independentFluidDofs(const NativeFluid& f){
  const auto& c=f.config();const int nr=f.nr(),nt=f.nth(),nz=f.nz();
  const auto rf=f.array(0),sf=f.array(1),rc=f.array(2),sc=f.array(3),drc=f.array(4),dsc=f.array(5),drf=f.array(6),dsf=f.array(7);
  const auto H=f.array(15),Hr=f.array(16),Ht=f.array(17);
  std::vector<FluidDof> out;out.reserve(static_cast<std::size_t>(3*nr-1)*nt*nz);
  for(int family=0;family<3;family++)for(int i=family==0?1:0;i<nr;i++)for(int k=0;k<nt;k++){
    const double r=family==0?rf[i]:rc[i],theta=(k+(family==1?0:0.5))*c.dtheta;
    const double co=std::cos(theta),si=std::sin(theta);
    const double h=family==0?Hr[i*nt+k]:family==1?Ht[i*nt+k]:H[i*nt+k];
    for(int j=family==2?1:0;j<(family==2?nz+1:nz);j++){
      const std::size_t index=(static_cast<std::size_t>(i)*nt+k)*(family==2?nz+1:nz)+j;
      const double z=(family==2?sf[j]:sc[j])*h;
      const Vec3 direction=family==0?Vec3{co,si,0}:family==1?Vec3{-si,co,0}:Vec3{0,0,1};
      const double spacing=std::max({family==0?drf[i]:drc[i],r*c.dtheta,(family==2?dsf[j]:dsc[j])*h});
      out.push_back({family,index,{r*co,r*si,z},direction,f.velocityMass(family,index),spacing});
    }
  }
  return out;
}

RigidCoupler::RigidCoupler(NativeFluid& f,std::vector<RigidGrain> g,CouplingOptions o):fluid_(f),grains_(std::move(g)),options_(o){
  if(!(std::isfinite(o.relativeTolerance)&&o.relativeTolerance>0&&std::isfinite(o.absoluteVelocityTolerance)&&o.absoluteVelocityTolerance>0&&std::isfinite(o.maximumDivergence)&&o.maximumDivergence>0&&o.maximumIterations>0&&o.maximumEventRefinements>0))throw std::invalid_argument("Invalid rigid constraint tolerances");
  if(!(std::isfinite(o.friction)&&o.friction>=0&&std::isfinite(o.restitution)&&o.restitution>=0&&o.restitution<=1&&std::isfinite(o.resolvedGapCells)&&o.resolvedGapCells>0))throw std::invalid_argument("Contact coefficients and gap resolution must be explicitly physical");
  requireMemory(baseWorkingArrayBytes(),f.config().memoryBudgetBytes);
  if(grains_.empty())return;
  const double rho=f.config().rho;
  for(std::size_t i=0;i<grains_.size();i++){
    const auto& grain=grains_[i];grain.validate();
    if(!(grain.density>rho))throw std::invalid_argument("This excess-inertia formulation requires grains denser than the liquid");
    for(std::size_t j=0;j<i;j++)if(grains_[j].id==grain.id)throw std::invalid_argument("Duplicate grain id");
    const double m=(grain.density-rho)*grain.volume();inverseMass_.push_back(1/m);inverseInertia_.push_back(1/(0.4*m*grain.radius*grain.radius));
  }
  scratchU_.resize(f.array(9).size());scratchV_.resize(f.array(10).size());scratchW_.resize(f.array(11).size());
  // A rejected grain configuration must leave the caller's fluid operator and
  // checkpoints unchanged. Geometry/interpolation do not require the projector.
  if(!grains_.empty()){rebuild();validateGeometry();fluid_.enableRigidProjection();}
}

void RigidCoupler::rebuild(){
  const auto baseBytes=baseWorkingArrayBytes();
  requireMemory(baseBytes,fluid_.config().memoryBudgetBytes);
  interpolation_.reset();dofs_=independentFluidDofs(fluid_);
  auto interpolationOptions=options_.interpolation;
  interpolationOptions.workingMemoryBudgetBytes=std::min(interpolationOptions.workingMemoryBudgetBytes,fluid_.config().memoryBudgetBytes-baseBytes);
  interpolation_=std::make_unique<RigidInterpolation>(dofs_,grains_,interpolationOptions);
  requireMemory(workingArrayBytes(),fluid_.config().memoryBudgetBytes);
  diagonal_.assign(interpolation_->markers().size()*3,0);
  for(std::size_t n=0;n<interpolation_->markers().size();n++){
    const auto& m=interpolation_->markers()[n];
    for(std::size_t d=0;d<3;d++){
      double a=inverseMass_[m.grain]+inverseInertia_[m.grain]*(dot(m.offset,m.offset)-m.offset[d]*m.offset[d]);
      for(const auto& e:interpolation_->stencils()[n].entries)a+=e.coefficient[d]*e.coefficient[d]/dofs_[e.dof].mass;
      if(!(a>0&&std::isfinite(a)))throw std::runtime_error("Invalid positive constraint preconditioner");
      diagonal_[3*n+d]=a;
    }
  }
}
std::size_t RigidCoupler::baseWorkingArrayBytes() const {
  if(grains_.empty())return fluid_.allocatedBytes();
  const auto nr=static_cast<std::size_t>(fluid_.nr()),nt=static_cast<std::size_t>(fluid_.nth()),nz=static_cast<std::size_t>(fluid_.nz());
  const auto independent=memoryProduct(memoryProduct(nr*3-1,nt),nz);
  std::size_t checkpoint=0,scratch=0;
  for(const auto key:{9,10,11,12,13,14,20})checkpoint=memorySum({checkpoint,memoryProduct(fluid_.array(key).size(),sizeof(double))});
  for(const auto key:{9,10,11})scratch=memorySum({scratch,memoryProduct(fluid_.array(key).size(),sizeof(double))});
  // Includes an old/new DOF geometry during rebuild, the transactional state,
  // simultaneous field vectors in the nested Schur/contact solves, body copies
  // and container allowance. Stencils and marker vectors are added separately.
  return memorySum({fluid_.allocatedBytes(),checkpoint,scratch,
    memoryProduct(independent,2*sizeof(FluidDof)+16*sizeof(double)),
    memoryProduct(grains_.size(),16*sizeof(RigidGrain)+32*sizeof(Six)),4096});
}
std::size_t RigidCoupler::workingArrayBytes() const {
  return memorySum({baseWorkingArrayBytes(),interpolation_?interpolation_->allocatedBytes():0,
    interpolation_?memoryProduct(interpolation_->markers().size(),24*sizeof(Vec3)):0});
}
std::vector<double> RigidCoupler::packFluid() const {
  std::vector<double> a(dofs_.size());for(std::size_t i=0;i<a.size();i++)a[i]=fluid_.array(9+dofs_[i].family)[dofs_[i].index];return a;
}
void RigidCoupler::addFluid(const std::vector<double>& a,double scale){for(std::size_t i=0;i<a.size();i++)fluid_.array(9+dofs_[i].family)[dofs_[i].index]+=scale*a[i];}
std::vector<double> RigidCoupler::project(const std::vector<double>& a){
  std::fill(scratchU_.begin(),scratchU_.end(),0);std::fill(scratchV_.begin(),scratchV_.end(),0);std::fill(scratchW_.begin(),scratchW_.end(),0);
  for(std::size_t i=0;i<a.size();i++){const auto& d=dofs_[i];(d.family==0?scratchU_:d.family==1?scratchV_:scratchW_)[d.index]=a[i];}
  fluid_.projectHomogeneous(scratchU_,scratchV_,scratchW_);
  std::vector<double> out(a.size());for(std::size_t i=0;i<a.size();i++){const auto& d=dofs_[i];out[i]=(d.family==0?scratchU_:d.family==1?scratchV_:scratchW_)[d.index];}return out;
}
std::vector<Six> RigidCoupler::bodyResponse(const std::vector<Six>& impulse) const {
  auto a=impulse;for(std::size_t g=0;g<a.size();g++)for(std::size_t d=0;d<6;d++)a[g][d]*=d<3?inverseMass_[g]:inverseInertia_[g];return a;
}
std::vector<Vec3> RigidCoupler::applySchur(const std::vector<Vec3>& lambda){
  auto a=interpolation_->applyJ(project(interpolation_->weightedAdjoint(lambda)));
  const auto b=interpolation_->applyK(bodyResponse(interpolation_->spreadKTranspose(lambda)));
  for(std::size_t i=0;i<a.size();i++)a[i]=a[i]+b[i];
  return a;
}
std::vector<Vec3> RigidCoupler::solve(const std::vector<Vec3>& rhs,std::size_t& iterations){
  const double tolerance=options_.absoluteVelocityTolerance+options_.relativeTolerance*markerMax(rhs);
  std::vector<Vec3> x(rhs.size()),r=rhs,z(rhs.size()),d(rhs.size());iterations=0;
  if(markerMax(r)<=tolerance)return x;
  for(std::size_t i=0;i<r.size();i++)for(std::size_t j=0;j<3;j++)z[i][j]=r[i][j]/diagonal_[3*i+j];
  d=z;double rz=markerDot(r,z);
  for(;iterations<options_.maximumIterations;){
    const auto q=applySchur(d);const double denominator=markerDot(d,q);
    if(!(denominator>0&&std::isfinite(denominator)))throw std::runtime_error("Rigid constraint operator lost positive curvature; no diagonal regularization applied");
    const double alpha=rz/denominator;
    for(std::size_t i=0;i<x.size();i++){x[i]=x[i]+alpha*d[i];r[i]=r[i]-alpha*q[i];}
    iterations++;
    if(markerMax(r)<=tolerance || iterations%30==0){
      const auto actual=applySchur(x);for(std::size_t i=0;i<r.size();i++)r[i]=rhs[i]-actual[i];
      if(markerMax(r)<=tolerance)return x;
      // A fresh search after true-residual replacement avoids retaining a
      // direction conjugate to a recursively accumulated residual.
      for(std::size_t i=0;i<r.size();i++)for(std::size_t j=0;j<3;j++)z[i][j]=r[i][j]/diagonal_[3*i+j];
      d=z;rz=markerDot(r,z);continue;
    }
    for(std::size_t i=0;i<r.size();i++)for(std::size_t j=0;j<3;j++)z[i][j]=r[i][j]/diagonal_[3*i+j];
    const double next=markerDot(r,z),beta=next/rz;rz=next;
    for(std::size_t i=0;i<d.size();i++)d[i]=z[i]+beta*d[i];
  }
  std::ostringstream message;message<<"Joint fluid/grain constraints failed to converge: velocity residual "<<markerMax(r)<<" m/s after "<<iterations<<" iterations";throw std::runtime_error(message.str());
}
void RigidCoupler::constrain(std::vector<Six>& body,CoupledStepReport& report){
  report.rigidImpulse.assign(grains_.size(),{});
  std::size_t usedIterations=0;
  for(;;){
    auto residual=interpolation_->applyJ(packFluid());const auto rigid=interpolation_->applyK(body);
    const double scale=std::max(markerMax(residual),markerMax(rigid));
    for(std::size_t i=0;i<residual.size();i++)residual[i]=residual[i]-rigid[i];
    report.slipResidual=markerMax(residual);
    if(report.slipResidual<=options_.absoluteVelocityTolerance+options_.relativeTolerance*scale)return;
    if(usedIterations>=options_.maximumIterations)throw std::runtime_error("Joint fluid/grain defect correction exhausted the iteration budget");
    // The incoming mismatch can exceed the final rigid velocity. A solve
    // stopped relative to that mismatch alone may fail the final-velocity
    // criterion even as dt tends to zero. Re-form and correct its true defect,
    // preserving both reaction impulses and reduced-pressure bookkeeping.
    std::size_t iterations=0;const auto lambda=solve(residual,iterations);
    if(iterations==0)throw std::runtime_error("Joint fluid/grain defect correction made no progress");
    usedIterations+=iterations;report.constraintIterations+=iterations;
    if(usedIterations>options_.maximumIterations)throw std::runtime_error("Joint fluid/grain defect correction exhausted the iteration budget");
    addFluid(project(interpolation_->weightedAdjoint(lambda)),-1);
    const auto pressurePotential=fluid_.lastProjectionPotential();auto pressure=fluid_.array(FluidArray::p);
    for(std::size_t i=0;i<pressure.size();i++)pressure[i]-=fluid_.config().rho*pressurePotential[i]/pendingDt_;
    const auto impulse=interpolation_->spreadKTranspose(lambda);const auto response=bodyResponse(impulse);
    for(std::size_t g=0;g<body.size();g++)for(std::size_t d=0;d<6;d++){
      body[g][d]+=response[g][d];report.rigidImpulse[g][d]+=impulse[g][d];
    }
  }
}
RigidCoupler::Response RigidCoupler::constrainedBodyResponse(const std::vector<Six>& impulse){
  Response response;response.bodies=bodyResponse(impulse);response.fluid.assign(dofs_.size(),0);
  response.pressurePotential.assign(fluid_.array(FluidArray::p).size(),0);response.rigidImpulse.resize(grains_.size());
  for(;;){
    auto residual=interpolation_->applyJ(response.fluid);const auto rigid=interpolation_->applyK(response.bodies);
    const double scale=std::max(markerMax(residual),markerMax(rigid));
    for(std::size_t i=0;i<residual.size();i++)residual[i]=residual[i]-rigid[i];
    if(markerMax(residual)<=options_.absoluteVelocityTolerance+options_.relativeTolerance*scale)return response;
    if(response.iterations>=options_.maximumIterations)throw std::runtime_error("Wet contact response exhausted the rigid constraint iteration budget");
    std::size_t iterations=0;const auto lambda=solve(residual,iterations);
    if(iterations==0)throw std::runtime_error("Wet contact constraint correction made no progress");
    response.iterations+=iterations;
    if(response.iterations>options_.maximumIterations)throw std::runtime_error("Wet contact response exhausted the rigid constraint iteration budget");
    const auto fluidCorrection=project(interpolation_->weightedAdjoint(lambda));
    for(std::size_t i=0;i<response.fluid.size();i++)response.fluid[i]-=fluidCorrection[i];
    const auto potential=fluid_.lastProjectionPotential();for(std::size_t i=0;i<potential.size();i++)response.pressurePotential[i]-=potential[i];
    const auto rigidImpulse=interpolation_->spreadKTranspose(lambda),bodyCorrection=bodyResponse(rigidImpulse);
    for(std::size_t g=0;g<response.bodies.size();g++)for(std::size_t d=0;d<6;d++){
      response.bodies[g][d]+=bodyCorrection[g][d];response.rigidImpulse[g][d]+=rigidImpulse[g][d];
    }
  }
}
void RigidCoupler::validateGeometry() const {
  const auto H=fluid_.array(15);const double surface=*std::min_element(H.begin(),H.end());
  for(std::size_t a=0;a<grains_.size();a++){
    const auto& g=grains_[a];const double spacing=interpolation_->grainSpacing()[a];
    if(g.centre.z+g.radius+options_.resolvedGapCells*spacing>=surface)throw std::runtime_error("Grain/free-surface gap is unresolved or grain is not fully submerged; no wetting model substituted");
    const double allowed=options_.resolvedGapCells*spacing;
    for(const auto& contact:{floorContact(g,a,0),cylinderContact(g,a,fluid_.config().R)}){
      const auto state=gapResolution(contact,allowed);
      if(state==GapResolution::overlapping)throw std::runtime_error("Grain overlaps the vessel; positions were not projected");
      if(state==GapResolution::positiveGapUnresolved)throw std::runtime_error("Positive grain/vessel lubrication gap is unresolved; no subgrid lubrication law substituted");
    }
    for(std::size_t b=0;b<a;b++){
      const auto state=gapResolution(spherePairContact(g,a,grains_[b],b),std::max(allowed,options_.resolvedGapCells*interpolation_->grainSpacing()[b]));
      if(state==GapResolution::overlapping)throw std::runtime_error("Grains overlap; no packing repair applied");
      if(state==GapResolution::positiveGapUnresolved)throw std::runtime_error("Positive grain/grain lubrication gap is unresolved; refine the fluid mesh");
    }
  }
}
std::vector<ContactGeometry> RigidCoupler::touchingContacts() const {
  std::vector<ContactGeometry> out;
  for(std::size_t a=0;a<grains_.size();a++){
    for(auto c:{floorContact(grains_[a],a,0),cylinderContact(grains_[a],a,fluid_.config().R)})if(c.gap==0)out.push_back(c);
    for(std::size_t b=0;b<a;b++){auto c=spherePairContact(grains_[a],a,grains_[b],b);if(c.gap==0)out.push_back(c);}
  }return out;
}
double RigidCoupler::firstImpact(double horizon) const {
  double earliest=horizon;
  for(std::size_t a=0;a<grains_.size();a++){
    for(auto t:{floorImpactTime(grains_[a],0,horizon),cylinderImpactTime(grains_[a],fluid_.config().R,horizon)})if(t&&*t>0&&*t<earliest)earliest=*t;
    for(std::size_t b=0;b<a;b++){auto t=spherePairImpactTime(grains_[a],grains_[b],horizon);if(t&&*t>0&&*t<earliest)earliest=*t;}
  }return earliest;
}
void RigidCoupler::validateSweptGaps(const std::vector<RigidGrain>& before,double dt) const {
  for(std::size_t a=0;a<grains_.size();a++){
    const double required=options_.resolvedGapCells*interpolation_->grainSpacing()[a];
    const auto oldFloor=floorContact(before[a],a,0),newFloor=floorContact(grains_[a],a,0);
    const auto oldWall=cylinderContact(before[a],a,fluid_.config().R),newWall=cylinderContact(grains_[a],a,fluid_.config().R);
    if((oldFloor.gap>0&&std::min(oldFloor.gap,newFloor.gap)<required)||(oldWall.gap>0&&std::min(oldWall.gap,newWall.gap)<required))
      throw std::runtime_error("Step would cross an unresolved positive vessel lubrication gap; refine the mesh rather than jump to contact");
    for(std::size_t b=0;b<a;b++){
      const Vec3 separation=before[a].centre-before[b].centre,relative=grains_[a].velocity-grains_[b].velocity;
      const double speed2=dot(relative,relative),closest=speed2>0?std::min(dt,std::max(0.0,-dot(separation,relative)/speed2)):0;
      const double minimum=norm(separation+closest*relative)-before[a].radius-before[b].radius;
      const double old=norm(separation)-before[a].radius-before[b].radius;
      if(old>0&&minimum<std::max(required,options_.resolvedGapCells*interpolation_->grainSpacing()[b]))
        throw std::runtime_error("Step would cross an unresolved positive inter-grain lubrication gap; no collision shortcut applied");
    }
  }
}
void RigidCoupler::applyContacts(std::vector<Six>& body,CoupledStepReport& report){
  assignVelocities(grains_,body);const auto contacts=touchingContacts();report.contacts=contacts.size();if(contacts.empty())return;
  const std::size_t n=memoryProduct(contacts.size(),3);
  requireMemory(memorySum({workingArrayBytes(),memoryProduct(memoryProduct(n,n),3*sizeof(double)),memoryProduct(n,32*sizeof(double)+sizeof(ContactGeometry)),
    memoryProduct(n,memorySum({memoryProduct(dofs_.size(),sizeof(double)),memoryProduct(grains_.size(),sizeof(Six))}))}),fluid_.config().memoryBudgetBytes);
  ContactProblem problem;problem.mobility.resize(memoryProduct(n,n));problem.friction.assign(contacts.size(),options_.friction);
  for(const auto& c:contacts){const Vec3 v=relativeContactVelocity(c,grains_);problem.freeVelocity.insert(problem.freeVelocity.end(),{v.x,v.y,v.z});}
  problem.normalTarget.assign(contacts.size(),0);
  for(std::size_t i=0;i<contacts.size();i++){
    const auto& c=contacts[i];
    if(c.kind==ContactKind::cylinderWall&&cylinderNeedsSustainedConstraint(grains_[c.a],fluid_.config().R))throw std::runtime_error("Tangential motion at a touching curved wall requires position-consistent constrained integration");
    const bool sustained=std::any_of(sustainedContacts_.begin(),sustainedContacts_.end(),[&](const auto& old){return old.kind==c.kind&&old.a==c.a&&old.b==c.b;});
    if(!sustained)problem.normalTarget[i]=-options_.restitution*std::min(0.0,problem.freeVelocity[3*i]);
  }
  // For the mass-orthogonal constrained response R, the contact mobility is
  // W_ab = R_a^T M R_b. Assemble this kinetic-energy Gram matrix directly,
  // including the liquid and EXCESS body masses. This preserves reciprocity
  // without averaging an asymmetric matrix or altering its eigenvalues.
  // Retain only the velocity responses; their field memory is checked above.
  struct VelocityResponse {std::vector<double> fluid;std::vector<Six> bodies;};
  std::vector<VelocityResponse> responses;responses.reserve(n);
  for(std::size_t column=0;column<n;column++){
    std::vector<Six> impulses(grains_.size());Vec3 unitImpulse;unitImpulse[column%3]=1;accumulateContactImpulse(contacts[column/3],unitImpulse,impulses);
    auto response=constrainedBodyResponse(impulses);
    report.constraintIterations+=response.iterations;
    responses.push_back({std::move(response.fluid),std::move(response.bodies)});
  }
  for(std::size_t a=0;a<n;a++)for(std::size_t b=0;b<=a;b++){
    double energy=0;
    for(std::size_t i=0;i<dofs_.size();i++)energy+=responses[a].fluid[i]*dofs_[i].mass*responses[b].fluid[i];
    for(std::size_t g=0;g<grains_.size();g++)for(std::size_t d=0;d<6;d++)energy+=responses[a].bodies[g][d]*responses[b].bodies[g][d]/(d<3?inverseMass_[g]:inverseInertia_[g]);
    problem.mobility[a*n+b]=problem.mobility[b*n+a]=energy;
  }
  double mobilityScale=0;for(double x:problem.mobility)mobilityScale=std::max(mobilityScale,std::abs(x));
  for(std::size_t column=0;column<n;column++){
    auto testGrains=grains_;assignVelocities(testGrains,responses[column].bodies);
    for(std::size_t row=0;row<contacts.size();row++){
      const Vec3 actual=relativeContactVelocity(contacts[row],testGrains);
      for(std::size_t d=0;d<3;d++)if(std::abs(actual[d]-problem.mobility[(3*row+d)*n+column])>options_.absoluteVelocityTolerance+options_.relativeTolerance*mobilityScale)
        throw std::runtime_error("Wet contact energy mobility disagrees with the actual constrained response; refine the linear solve");
    }
  }
  const auto solved=solveContacts(problem);
  report.contactImpulse.assign(grains_.size(),{});
  for(std::size_t i=0;i<contacts.size();i++)accumulateContactImpulse(contacts[i],{solved.impulse[3*i],solved.impulse[3*i+1],solved.impulse[3*i+2]},report.contactImpulse);
  const auto response=constrainedBodyResponse(report.contactImpulse);
  report.constraintIterations+=response.iterations;
  addFluid(response.fluid,1);
  auto pressure=fluid_.array(FluidArray::p);for(std::size_t i=0;i<pressure.size();i++)pressure[i]+=fluid_.config().rho*response.pressurePotential[i]/pendingDt_;
  for(std::size_t g=0;g<body.size();g++)for(std::size_t d=0;d<6;d++){
    body[g][d]+=response.bodies[g][d];
    report.rigidImpulse[g][d]+=response.rigidImpulse[g][d];
  }
  auto actualGrains=grains_;assignVelocities(actualGrains,body);std::vector<double> actualContactVelocity;
  for(const auto& c:contacts){const Vec3 v=relativeContactVelocity(c,actualGrains);actualContactVelocity.insert(actualContactVelocity.end(),{v.x,v.y,v.z});}
  verifyContactResponse(problem,solved.impulse,actualContactVelocity);
}
CoupledStepReport RigidCoupler::step(double requestedDt,double effectiveGravity){
  if(!(std::isfinite(requestedDt)&&requestedDt>0&&std::isfinite(effectiveGravity)))throw std::invalid_argument("Invalid physical step or gravity");
  CoupledStepReport report;
  if(grains_.empty()){fluid_.step(requestedDt,effectiveGravity);report.dt=requestedDt;report.time=fluid_.time();report.divergence=fluid_.maxDivergence();return report;}
  const auto saved=fluid_.checkpoint();const auto oldGrains=grains_;const auto oldContacts=sustainedContacts_;double dt=requestedDt;
  // A grain initially touching and at rest is a support contact, not a new
  // impact caused by the next gravity predictor.
  if(sustainedContacts_.empty())for(const auto& c:touchingContacts())if(relativeContactVelocity(c,oldGrains).x>=0)sustainedContacts_.push_back(c);
  try{
    for(std::size_t attempt=0;attempt<=options_.maximumEventRefinements;attempt++){
      if(attempt){fluid_.restore(saved);grains_=oldGrains;}
      report={};report.eventRefinements=attempt;pendingDt_=dt;rebuild();validateGeometry();
      fluid_.predictor(dt,effectiveGravity);auto body=velocities(grains_);
      for(auto& q:body)q[2]-=dt*effectiveGravity;
      constrain(body,report);applyContacts(body,report);assignVelocities(grains_,body);
      const double event=firstImpact(dt);
      if(event<dt){dt=event;continue;}
      for(auto& g:grains_)g.centre=g.centre+dt*g.velocity;
      // No post-step position projection or grain redistribution is permitted.
      validateSweptGaps(oldGrains,dt);validateGeometry();fluid_.finish(dt);rebuild();validateGeometry();
      auto finalSlip=interpolation_->applyJ(packFluid());const auto finalRigid=interpolation_->applyK(body);
      const double scale=std::max(markerMax(finalSlip),markerMax(finalRigid));for(std::size_t i=0;i<finalSlip.size();i++)finalSlip[i]=finalSlip[i]-finalRigid[i];
      report.slipResidual=markerMax(finalSlip);report.divergence=fluid_.maxDivergence();
      if(report.slipResidual>options_.absoluteVelocityTolerance+options_.relativeTolerance*scale || report.divergence>options_.maximumDivergence){dt*=0.5;continue;}
      sustainedContacts_=touchingContacts();report.dt=dt;report.time=fluid_.time();report.markerCount=interpolation_->markers().size();
      return report;
    }
    throw std::runtime_error("Contact event refinement did not converge; no position repair applied");
  }catch(...){fluid_.restore(saved);grains_=oldGrains;sustainedContacts_=oldContacts;throw;}
}
} // namespace faraday_native
