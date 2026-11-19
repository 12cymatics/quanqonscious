#include "rigid-contact.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace faraday_native {
namespace {
void requireFinite(double x,const char* name){if(!std::isfinite(x))throw std::invalid_argument(std::string(name)+" must be finite");}
void requireArithmetic(double x,const char* name){if(!std::isfinite(x))throw std::overflow_error(std::string("Non-finite contact arithmetic: ")+name);}
void requireHorizon(double h){requireFinite(h,"Impact horizon");if(h<0)throw std::invalid_argument("Impact horizon must be nonnegative");}
void basis(ContactGeometry& c){
  const Vec3 ref=std::abs(c.normal.x)<0.8?Vec3{1,0,0}:Vec3{0,1,0};
  c.tangent1=unit(cross(c.normal,ref));c.tangent2=cross(c.normal,c.tangent1);
}
void requireBasis(const ContactGeometry& c){if(!c.normalDefined)throw std::invalid_argument("The wall normal is undefined on the cylinder axis; no wall contact exists there");}
void requireNoOverlap(double gap){if(gap<0)throw std::domain_error("Initial grain overlap: contact detection does not repair positions");}
std::optional<double> inHorizon(double t,double h){if(t>=0&&t<=h)return t;return std::nullopt;}
double vectorNorm(const std::vector<double>& x){double s=0;for(double v:x)s=std::hypot(s,v);return s;}
void verifyPSD(const std::vector<double>& input,std::size_t n){
  double scale=0;for(double x:input){requireFinite(x,"Contact mobility");scale=std::max(scale,std::abs(x));}
  if(scale==0)return;
  const double tolerance=128*std::numeric_limits<double>::epsilon()*scale*std::max<std::size_t>(1,n);
  std::vector<double> a=input;
  for(std::size_t i=0;i<n;i++)for(std::size_t j=0;j<i;j++)
    if(std::abs(a[i*n+j]-a[j*n+i])>tolerance)throw std::invalid_argument("Contact mobility is not symmetric");
  // Pivoted Cholesky used only as a check. The input matrix is not symmetrised,
  // regularised or modified by this check; null mobility is allowed.
  for(std::size_t k=0;k<n;k++){
    std::size_t p=k;for(std::size_t i=k+1;i<n;i++)if(a[i*n+i]>a[p*n+p])p=i;
    if(p!=k){for(std::size_t j=0;j<n;j++)std::swap(a[k*n+j],a[p*n+j]);for(std::size_t i=0;i<n;i++)std::swap(a[i*n+k],a[i*n+p]);}
    const double pivot=a[k*n+k];
    if(pivot < -tolerance)throw std::invalid_argument("Contact mobility is not positive semidefinite");
    if(pivot<=tolerance){
      for(std::size_t i=k;i<n;i++)for(std::size_t j=k;j<n;j++)
        if(std::abs(a[i*n+j])>tolerance)throw std::invalid_argument("Contact mobility has an indefinite null block");
      return;
    }
    for(std::size_t i=k+1;i<n;i++)for(std::size_t j=i;j<n;j++){
      const double v=a[i*n+j]-a[i*n+k]*a[j*n+k]/pivot;
      requireArithmetic(v,"mobility factorisation");
      a[i*n+j]=a[j*n+i]=v;
    }
  }
}
// Exact minimiser of 1/2 x^T A x+b^T x over ||x||<=radius, to scalar root
// tolerance. This yields x opposite the FINAL slip velocity on the boundary.
std::array<double,2> tangentBlock(double a,double b,double d,double f,double g,double radius){
  if(radius==0)return {0,0};
  const double high=0.5*(a+d+std::hypot(a-d,2*b));
  requireArithmetic(high,"tangent mobility eigenvalue");
  requireArithmetic(a*d-b*b,"tangent mobility determinant");
  if(a<0||d<0||high<0||a*d-b*b<0)throw std::domain_error("Tangential mobility is not positive semidefinite");
  if(high==0){const double n=std::hypot(f,g);return n==0?std::array<double,2>{0,0}:std::array<double,2>{-radius*f/n,-radius*g/n};}
  const double low=(a*d-b*b)/high;
  double ex,ey;
  if(b!=0){ex=b;ey=high-a;const double n=std::hypot(ex,ey);ex/=n;ey/=n;}
  else {ex=a>=d?1:0;ey=a>=d?0:1;}
  const double fh=f*ex+g*ey,fl=-f*ey+g*ex;
  const bool solvable=low>0||fl==0;
  if(solvable){
    const double xh=-fh/high,xl=low>0?-fl/low:0;
    if(std::hypot(xh,xl)<=radius)return {xh*ex-xl*ey,xh*ey+xl*ex};
  }
  double lo=0,hi=std::hypot(f,g)/radius;
  if(!(hi>0))return {0,0};
  for(unsigned i=0;i<120;i++){
    const double mid=0.5*(lo+hi);
    if(mid==lo||mid==hi)break;
    const double xh=-fh/(high+mid),xl=-fl/(low+mid);
    if(std::hypot(xh,xl)>radius)lo=mid;else hi=mid;
  }
  const double xh=-fh/(high+hi),xl=-fl/(low+hi);
  return {xh*ex-xl*ey,xh*ey+xl*ex};
}
ContactSolution residual(const ContactProblem& p,const std::vector<double>& lambda,const std::vector<double>& velocity,std::size_t iteration){
  const std::size_t n=lambda.size();ContactSolution r;r.impulse=lambda;r.velocity=velocity;r.iterations=iteration;
  for(std::size_t i=0;i<p.friction.size();i++){
    const std::size_t k=3*i;const double ln=lambda[k],wn=velocity[k]-p.normalTarget[i],wnn=p.mobility[k*n+k];
    const double normal=wnn>0?std::abs(ln-std::max(0.0,ln-wn/wnn))*wnn:std::max(0.0,-wn);
    r.normalResidual=std::max(r.normalResidual,normal);
    const double radius=p.friction[i]*ln,lt=std::hypot(lambda[k+1],lambda[k+2]);
    r.coneResidual=std::max(r.coneResidual,std::max(-ln,lt-radius));
    const double a=p.mobility[(k+1)*n+k+1],b=p.mobility[(k+1)*n+k+2],d=p.mobility[(k+2)*n+k+2];
    const double factor=0.5*(a+d+std::hypot(a-d,2*b));
    if(factor>0){
      double x=lambda[k+1]-velocity[k+1]/factor,y=lambda[k+2]-velocity[k+2]/factor;
      const double len=std::hypot(x,y);if(len>radius){x*=radius/len;y*=radius/len;}
      r.tangentResidual=std::max(r.tangentResidual,factor*std::hypot(lambda[k+1]-x,lambda[k+2]-y));
    }else{
      const double slip=std::hypot(velocity[k+1],velocity[k+2]);
      if(slip>0){const double err=std::hypot(lambda[k+1]+radius*velocity[k+1]/slip,lambda[k+2]+radius*velocity[k+2]/slip);r.coneResidual=std::max(r.coneResidual,err);}
    }
  }
  for(std::size_t i=0;i<n;i++)r.energyChange+=lambda[i]*0.5*(p.freeVelocity[i]+velocity[i]);
  requireArithmetic(r.energyChange,"contact energy");
  return r;
}
} // namespace

ContactGeometry spherePairContact(const RigidGrain& a,std::size_t ia,const RigidGrain& b,std::size_t ib){
  a.validate();b.validate();if(ia==ib)throw std::invalid_argument("A grain cannot contact itself");
  const Vec3 delta=a.centre-b.centre;const double distance=norm(delta);
  if(!(distance>0))throw std::domain_error("Coincident grain centres have no contact normal");
  requireFinite(distance,"Grain separation");
  ContactGeometry c;c.kind=ContactKind::spherePair;c.a=ia;c.b=ib;c.gap=distance-a.radius-b.radius;c.normal=delta/distance;
  // A shared contact point keeps pair impulses exactly torque-balanced. Away
  // from contact this is only a diagnostic point: impulse application refuses.
  c.armA=-(a.radius+0.5*c.gap)*c.normal;c.armB=(b.radius+0.5*c.gap)*c.normal;basis(c);return c;
}
ContactGeometry floorContact(const RigidGrain& a,std::size_t ia,double floorZ){
  a.validate();requireFinite(floorZ,"Floor height");ContactGeometry c;c.kind=ContactKind::floor;c.a=ia;
  c.gap=a.centre.z-a.radius-floorZ;c.normal={0,0,1};c.armA={0,0,floorZ-a.centre.z};basis(c);return c;
}
ContactGeometry cylinderContact(const RigidGrain& a,std::size_t ia,double radius){
  a.validate();requireFinite(radius,"Cylinder radius");if(!(radius>a.radius))throw std::invalid_argument("The cylinder must be wider than the grain");
  ContactGeometry c;c.kind=ContactKind::cylinderWall;c.a=ia;const double r=std::hypot(a.centre.x,a.centre.y);c.gap=radius-a.radius-r;
  c.normalDefined=r>0;c.normal=r>0?Vec3{-a.centre.x/r,-a.centre.y/r,0}:Vec3{-1,0,0};
  c.armA=-(a.radius+c.gap)*c.normal;basis(c);return c;
}
Vec3 contactCoordinates(const ContactGeometry& c,Vec3 v){requireBasis(c);return {dot(v,c.normal),dot(v,c.tangent1),dot(v,c.tangent2)};}
Vec3 contactWorldVector(const ContactGeometry& c,Vec3 v){requireBasis(c);return v.x*c.normal+v.y*c.tangent1+v.z*c.tangent2;}
Vec3 relativeContactVelocity(const ContactGeometry& c,const std::vector<RigidGrain>& grains){
  requireBasis(c);if(c.a>=grains.size()||(c.b!=fixedBody&&c.b>=grains.size()))throw std::out_of_range("Contact grain index");
  const auto& a=grains[c.a];Vec3 v=a.velocity+cross(a.omega,c.armA);
  if(c.b!=fixedBody){const auto& b=grains[c.b];v=v-b.velocity-cross(b.omega,c.armB);}
  return contactCoordinates(c,v);
}
void accumulateContactImpulse(const ContactGeometry& c,Vec3 local,std::vector<Six>& impulse,double gapTolerance){
  if(!finite(local))throw std::invalid_argument("Contact impulse must be finite");
  requireFinite(gapTolerance,"Event location tolerance");
  if(gapTolerance<0||!std::isfinite(c.gap)||std::abs(c.gap)>gapTolerance)
    throw std::domain_error("An impulse can only be applied at a located contact, never across a separated or overlapping gap");
  if(c.a>=impulse.size()||(c.b!=fixedBody&&c.b>=impulse.size()))throw std::out_of_range("Contact impulse grain index");
  const Vec3 f=contactWorldVector(c,local),ta=cross(c.armA,f);
  for(std::size_t j=0;j<3;j++){impulse[c.a][j]+=f[j];impulse[c.a][j+3]+=ta[j];}
  if(c.b!=fixedBody){const Vec3 tb=cross(c.armB,-f);for(std::size_t j=0;j<3;j++){impulse[c.b][j]-=f[j];impulse[c.b][j+3]+=tb[j];}}
}
std::optional<double> spherePairImpactTime(const RigidGrain& a,const RigidGrain& b,double horizon){
  requireHorizon(horizon);a.validate();b.validate();const Vec3 x=a.centre-b.centre,v=a.velocity-b.velocity;
  const double r=a.radius+b.radius,A=dot(v,v),B=dot(x,v),C=dot(x,x)-r*r;requireNoOverlap(C);
  if(C==0)return B<0?std::optional<double>(0):std::nullopt;
  requireFinite(A,"Relative speed squared");requireFinite(B,"Impact linear coefficient");requireFinite(C,"Impact separation squared");
  if(A==0||B>=0)return std::nullopt;
  const double disc=B*B-A*C;requireFinite(disc,"Impact discriminant");if(disc<0)return std::nullopt;
  return inHorizon(C/(-B+std::sqrt(disc)),horizon);
}
std::optional<double> floorImpactTime(const RigidGrain& a,double floorZ,double horizon){
  requireHorizon(horizon);const auto c=floorContact(a,0,floorZ);requireNoOverlap(c.gap);
  if(a.velocity.z>=0)return std::nullopt;
  return inHorizon(-c.gap/a.velocity.z,horizon);
}
std::optional<double> cylinderImpactTime(const RigidGrain& a,double radius,double horizon){
  requireHorizon(horizon);const auto c=cylinderContact(a,0,radius);requireNoOverlap(c.gap);
  const double limit=radius-a.radius,A=a.velocity.x*a.velocity.x+a.velocity.y*a.velocity.y;
  if(A==0)return std::nullopt;
  const double B=a.centre.x*a.velocity.x+a.centre.y*a.velocity.y;
  const double C=a.centre.x*a.centre.x+a.centre.y*a.centre.y-limit*limit;
  requireFinite(A,"Wall speed squared");requireFinite(B,"Wall impact linear coefficient");requireFinite(C,"Wall separation squared");
  // At an existing curved-wall contact, a nonzero purely tangential straight
  // trial exits immediately. Report t=0; the caller must integrate the sustained
  // geometric constraint, not pretend that this straight segment is admissible.
  if(C==0&&B>=0)return 0;
  const double discriminant=B*B-A*C;requireFinite(discriminant,"Wall impact discriminant");
  const double root=std::sqrt(discriminant);
  const double t=B>0?(-C)/(B+root):(-B+root)/A;
  return inHorizon(t,horizon);
}
bool cylinderNeedsSustainedConstraint(const RigidGrain& a,double radius){
  const auto c=cylinderContact(a,0,radius);
  return c.gap==0&&dot(a.velocity,c.normal)==0
    &&(a.velocity.x!=0||a.velocity.y!=0);
}
GapResolution gapResolution(const ContactGeometry& c,double minimum){
  requireFinite(c.gap,"Contact gap");requireFinite(minimum,"Minimum resolved gap");if(minimum<0)throw std::invalid_argument("Minimum resolved gap must be nonnegative");
  if(c.gap<0)return GapResolution::overlapping;
  if(c.gap==0)return GapResolution::touching;
  return c.gap<minimum?GapResolution::positiveGapUnresolved:GapResolution::separatedResolved;
}
std::vector<double> impactNormalTargets(const std::vector<double>& velocity,const std::vector<double>& restitution){
  if(velocity.size()!=3*restitution.size())throw std::invalid_argument("Restitution needs three velocities per contact");
  std::vector<double> out(restitution.size());for(std::size_t i=0;i<out.size();i++){
    const double e=restitution[i];requireFinite(e,"Restitution");if(e<0||e>1)throw std::invalid_argument("Restitution must be in [0,1]");
    for(std::size_t j=0;j<3;j++)requireFinite(velocity[3*i+j],"Pre-impact velocity");
    out[i]=-e*std::min(0.0,velocity[3*i]);
  }return out;
}
ContactSolution verifyContactResponse(const ContactProblem& p,const std::vector<double>& impulse,const std::vector<double>& velocity,const ContactSolveOptions& o){
  const std::size_t n=impulse.size();
  if(n!=3*p.friction.size()||velocity.size()!=n||p.freeVelocity.size()!=n||p.normalTarget.size()!=p.friction.size()||p.mobility.size()!=n*n)throw std::invalid_argument("Inconsistent actual contact response dimensions");
  if(!(std::isfinite(o.velocityTolerance)&&o.velocityTolerance>0&&std::isfinite(o.impulseTolerance)&&o.impulseTolerance>0&&std::isfinite(o.relativeTolerance)&&o.relativeTolerance>=0))throw std::invalid_argument("Invalid contact response tolerances");
  for(double x:impulse)requireFinite(x,"Actual contact impulse");
  for(double x:velocity)requireFinite(x,"Actual contact velocity");
  for(double x:p.freeVelocity)requireFinite(x,"Free contact velocity");
  for(double x:p.normalTarget){requireFinite(x,"Normal target");if(x<0)throw std::invalid_argument("A normal target cannot request penetration");}
  for(double x:p.friction){requireFinite(x,"Friction coefficient");if(x<0)throw std::invalid_argument("Friction must be nonnegative");}
  verifyPSD(p.mobility,n);
  const double vtol=o.velocityTolerance+o.relativeTolerance*std::max(vectorNorm(p.freeVelocity),vectorNorm(p.normalTarget));
  const double itol=o.impulseTolerance+o.relativeTolerance*vectorNorm(impulse);
  requireArithmetic(vtol,"actual contact velocity tolerance");requireArithmetic(itol,"actual contact impulse tolerance");
  auto r=residual(p,impulse,velocity,0);
  if(r.normalResidual>vtol||r.tangentResidual>vtol||r.coneResidual>itol)throw std::runtime_error("Actual coupled velocity violates contact complementarity/Coulomb residual tolerances");
  double workScale=0;for(std::size_t i=0;i<n;i++)workScale+=std::abs(impulse[i])*0.5*(std::abs(p.freeVelocity[i])+std::abs(velocity[i]));
  const double energyTolerance=itol*(vectorNorm(p.freeVelocity)+vectorNorm(velocity))+vtol*vectorNorm(impulse)+128*std::numeric_limits<double>::epsilon()*workScale;
  requireArithmetic(energyTolerance,"actual contact energy tolerance");
  if(r.energyChange>energyTolerance)throw std::domain_error("Actual coupled contact response adds energy");
  return r;
}
ContactSolution solveContacts(const ContactProblem& p,const ContactSolveOptions& o){
  const std::size_t contacts=p.friction.size(),n=3*contacts;
  if(n!=p.freeVelocity.size()||contacts!=p.normalTarget.size()||p.mobility.size()!=n*n)throw std::invalid_argument("Inconsistent contact problem dimensions");
  if(!(o.velocityTolerance>0&&o.impulseTolerance>0&&o.relativeTolerance>=0&&o.maximumIterations>0)
     ||!std::isfinite(o.velocityTolerance)||!std::isfinite(o.impulseTolerance)||!std::isfinite(o.relativeTolerance))throw std::invalid_argument("Invalid contact solve tolerances");
  for(double x:p.freeVelocity)requireFinite(x,"Free contact velocity");
  for(double x:p.normalTarget){requireFinite(x,"Normal target");if(x<0)throw std::invalid_argument("A normal target cannot request penetration");}
  for(double x:p.friction){requireFinite(x,"Friction coefficient");if(x<0)throw std::invalid_argument("Friction must be nonnegative");}
  verifyPSD(p.mobility,n);
  std::vector<double> impulse(n,0),velocity=p.freeVelocity;
  const double velocityScale=std::max(vectorNorm(p.freeVelocity),vectorNorm(p.normalTarget));
  const double vtol=o.velocityTolerance+o.relativeTolerance*velocityScale;
  requireArithmetic(vtol,"velocity residual tolerance");
  const auto add=[&](std::size_t j,double value){
    requireArithmetic(value,"contact impulse");
    const double delta=value-impulse[j];impulse[j]=value;
    for(std::size_t i=0;i<n;i++){velocity[i]+=p.mobility[i*n+j]*delta;requireArithmetic(velocity[i],"contact velocity update");}
  };
  for(std::size_t iteration=0;iteration<=o.maximumIterations;iteration++){
    // Re-form velocity, rather than trusting accumulated updates indefinitely.
    for(std::size_t i=0;i<n;i++){velocity[i]=p.freeVelocity[i];for(std::size_t j=0;j<n;j++)velocity[i]+=p.mobility[i*n+j]*impulse[j];requireArithmetic(velocity[i],"contact velocity");}
    ContactSolution r=residual(p,impulse,velocity,iteration);
    const double itol=o.impulseTolerance+o.relativeTolerance*vectorNorm(impulse);
    requireArithmetic(itol,"impulse residual tolerance");
    if(r.normalResidual<=vtol&&r.tangentResidual<=vtol&&r.coneResidual<=itol){
      double workScale=0;for(std::size_t i=0;i<n;i++)workScale+=std::abs(impulse[i])*0.5*(std::abs(p.freeVelocity[i])+std::abs(velocity[i]));
      const double energyTolerance=itol*(vectorNorm(p.freeVelocity)+vectorNorm(velocity))
        +vtol*vectorNorm(impulse)+128*std::numeric_limits<double>::epsilon()*workScale;
      requireArithmetic(energyTolerance,"energy tolerance");
      if(r.energyChange>energyTolerance)throw std::domain_error("The simultaneous contact/restitution solution adds energy; no corrective rescaling was applied");
      return r;
    }
    if(iteration==o.maximumIterations)break;
    for(std::size_t c=0;c<contacts;c++){
      const std::size_t k=3*c;const double W=p.mobility[k*n+k],gapVelocity=velocity[k]-p.normalTarget[c];
      if(W>0)add(k,std::max(0.0,impulse[k]-gapVelocity/W));
      else if(gapVelocity < -vtol)throw std::domain_error("Closing contact has zero normal mobility");
      const double a=p.mobility[(k+1)*n+k+1],b=p.mobility[(k+1)*n+k+2],d=p.mobility[(k+2)*n+k+2];
      const double f=velocity[k+1]-a*impulse[k+1]-b*impulse[k+2],g=velocity[k+2]-b*impulse[k+1]-d*impulse[k+2];
      const auto t=tangentBlock(a,b,d,f,g,p.friction[c]*impulse[k]);add(k+1,t[0]);add(k+2,t[1]);
    }
  }
  throw std::runtime_error("Contact complementarity/Coulomb solve did not converge; no approximate impulses were applied");
}
} // namespace faraday_native
