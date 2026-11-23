#pragma once
#include "rigid-types.hpp"
#include <algorithm>
#include <array>
#include <limits>

// Exact positive-gap geometry for a future fitted Navier--Stokes patch.
// This file does not supply a drag law, refine the production mesh, change any
// gap-refusal threshold, or continue a grain through unresolved fluid/contact.
namespace faraday_native::resolved_gap {
inline double length(Vec3 a){return std::hypot(a.x,a.y,a.z);}
inline void positive(double x,const char* label){
  if(!(x>0&&std::isfinite(x)))throw std::invalid_argument(label);
}
inline void finiteValue(double x,const char* label){
  if(!std::isfinite(x))throw std::overflow_error(label);
}
inline Vec3 normalized(Vec3 a){
  if(!finite(a))throw std::invalid_argument("Non-finite gap-frame direction");
  const double scale=std::max({std::abs(a.x),std::abs(a.y),std::abs(a.z)});
  positive(scale,"Gap-frame direction must be nonzero");
  a={a.x/scale,a.y/scale,a.z/scale};return a/length(a);
}

struct Frame {
  Vec3 origin{},x{1,0,0},y{0,1,0},z{0,0,1};
  void validate() const {
    if(!finite(origin)||!finite(x)||!finite(y)||!finite(z))
      throw std::invalid_argument("Non-finite gap frame");
    const double eps=128*std::numeric_limits<double>::epsilon();
    if(std::abs(dot(x,x)-1)>eps||std::abs(dot(y,y)-1)>eps||std::abs(dot(z,z)-1)>eps
       ||std::abs(dot(x,y))>eps||std::abs(dot(x,z))>eps||std::abs(dot(y,z))>eps
       ||length(cross(x,y)-z)>eps)
      throw std::invalid_argument("Gap frame must be right-handed and orthonormal");
  }
  static Frame fromNormal(Vec3 origin,Vec3 normal){
    Frame f;f.origin=origin;f.z=normalized(normal);
    const double ax=std::abs(f.z.x),ay=std::abs(f.z.y),az=std::abs(f.z.z);
    const Vec3 ref=ax<=ay&&ax<=az?Vec3{1,0,0}:ay<=az?Vec3{0,1,0}:Vec3{0,0,1};
    f.x=normalized(ref-dot(ref,f.z)*f.z);f.y=cross(f.z,f.x);f.validate();return f;
  }
  Vec3 vector(Vec3 q) const{return q.x*x+q.y*y+q.z*z;}
  Vec3 components(Vec3 v) const{return {dot(v,x),dot(v,y),dot(v,z)};}
};

struct SphereSag {
  double value=0,dx=0,dy=0,dxx=0,dxy=0,dyy=0;
};
inline SphereSag sphereSag(double radius,double x,double y){
  positive(radius,"Sphere radius must be finite and positive");
  if(!std::isfinite(x)||!std::isfinite(y))throw std::invalid_argument("Non-finite gap coordinate");
  const double q=std::hypot(x,y)/radius;
  if(!(q<1))throw std::domain_error("Gap chart reaches the sphere equator or lies outside it");
  const double s=std::sqrt((1-q)*(1+q)),root=radius*s;
  positive(root,"Gap-chart spherical derivative is unrepresentable");
  SphereSag out;out.value=radius*(q*q/(1+s));
  out.dx=x/root;out.dy=y/root;
  out.dxx=(1+out.dx*out.dx)/root;out.dxy=out.dx*out.dy/root;
  out.dyy=(1+out.dy*out.dy)/root;
  for(double v:{out.value,out.dx,out.dy,out.dxx,out.dxy,out.dyy})
    finiteValue(v,"Non-finite spherical gap metric");
  return out;
}

struct Section {
  double lower=0,upper=0,height=0,lowerX=0,lowerY=0,upperX=0,upperY=0;
  double heightX=0,heightY=0;
};
struct Motion {
  Vec3 originVelocity{},angularVelocity{};
  double gapRate=0;
};
struct Metric {
  Vec3 point,offset;
  std::array<Vec3,3> jacobian,cofactor;
  double determinant=0,zXi=0,zEta=0,sigma=0;
  Frame frame;
  Vec3 physicalGradient(Vec3 coordinateDerivative) const {
    if(!finite(coordinateDerivative))throw std::invalid_argument("Non-finite coordinate gradient");
    const double vertical=coordinateDerivative.z/determinant;
    const Vec3 out=frame.vector({coordinateDerivative.x-zXi*vertical,
      coordinateDerivative.y-zEta*vertical,vertical});
    if(!finite(out))throw std::overflow_error("Non-finite physical gap gradient");
    return out;
  }
  Vec3 flux(Vec3 physicalVelocity) const {
    if(!finite(physicalVelocity))throw std::invalid_argument("Non-finite gap velocity");
    const Vec3 out={dot(cofactor[0],physicalVelocity),dot(cofactor[1],physicalVelocity),dot(cofactor[2],physicalVelocity)};
    if(!finite(out))throw std::overflow_error("Non-finite gap face flux");
    return out;
  }
  double divergence(Vec3 diagonalFluxDerivatives) const {
    if(!finite(diagonalFluxDerivatives))throw std::invalid_argument("Non-finite flux derivative");
    const double out=(diagonalFluxDerivatives.x+diagonalFluxDerivatives.y+diagonalFluxDerivatives.z)/determinant;
    finiteValue(out,"Non-finite gap divergence");return out;
  }
  Vec3 gridVelocity(const Motion& m) const {
    if(!finite(m.originVelocity)||!finite(m.angularVelocity)||!std::isfinite(m.gapRate))
      throw std::invalid_argument("Non-finite prescribed gap-chart motion");
    const Vec3 out=m.originVelocity+cross(m.angularVelocity,offset)+(sigma*m.gapRate)*frame.z;
    if(!finite(out))throw std::overflow_error("Non-finite gap grid velocity");
    return out;
  }
  Vec3 relativeFlux(Vec3 velocity,const Motion& motion) const{return flux(velocity-gridVelocity(motion));}
};

class Chart {
  Frame frame_;
  double gap_,lowerRadius_,upperRadius_;
  Chart(Frame frame,double gap,double lowerRadius,double upperRadius)
      :frame_(frame),gap_(gap),lowerRadius_(lowerRadius),upperRadius_(upperRadius){
    frame_.validate();positive(gap_,"A fitted liquid gap requires strictly positive separation");
    positive(upperRadius_,"Upper sphere radius must be positive");
    if(lowerRadius_!=0)positive(lowerRadius_,"Lower sphere radius must be positive");
  }
public:
  static Chart sphereWall(double radius,double gap,Frame frame={}){return Chart(frame,gap,0,radius);}
  static Chart spherePair(double lowerRadius,double upperRadius,double gap,Frame frame={}){
    positive(lowerRadius,"Lower sphere radius must be positive");return Chart(frame,gap,lowerRadius,upperRadius);
  }
  const Frame& frame() const{return frame_;}
  double gap() const{return gap_;}
  double lowerRadius() const{return lowerRadius_;}
  double upperRadius() const{return upperRadius_;}
  double chartRadius() const{return lowerRadius_==0?upperRadius_:std::min(lowerRadius_,upperRadius_);}
  Section section(double x,double y) const {
    const auto hi=sphereSag(upperRadius_,x,y);
    const auto lo=lowerRadius_==0?SphereSag{}:sphereSag(lowerRadius_,x,y);
    Section out;out.lower=-lo.value;out.upper=gap_+hi.value;out.height=gap_+lo.value+hi.value;
    out.lowerX=-lo.dx;out.lowerY=-lo.dy;out.upperX=hi.dx;out.upperY=hi.dy;
    out.heightX=lo.dx+hi.dx;out.heightY=lo.dy+hi.dy;
    positive(out.height,"Non-finite or collapsed liquid gap");
    finiteValue(out.upper,"Non-finite upper surface");finiteValue(out.heightX,"Non-finite gap slope");
    finiteValue(out.heightY,"Non-finite gap slope");return out;
  }
  Metric metric(double x,double y,double sigma) const {
    if(!(std::isfinite(sigma)&&sigma>=0&&sigma<=1))throw std::domain_error("Gap sigma must be in [0,1]");
    const auto s=section(x,y);
    const double z=s.lower+sigma*s.height,zx=s.lowerX+sigma*s.heightX,zy=s.lowerY+sigma*s.heightY;
    Metric out;out.frame=frame_;out.sigma=sigma;out.determinant=s.height;out.zXi=zx;out.zEta=zy;
    out.offset=frame_.vector({x,y,z});out.point=frame_.origin+out.offset;
    out.jacobian={frame_.x+zx*frame_.z,frame_.y+zy*frame_.z,s.height*frame_.z};
    out.cofactor={s.height*frame_.x,s.height*frame_.y,frame_.z-zx*frame_.x-zy*frame_.y};
    if(!finite(out.point)||!finite(out.offset))throw std::overflow_error("Gap position is not representable");
    return out;
  }
  Vec3 coordinates(Vec3 point) const {
    if(!finite(point))throw std::invalid_argument("Non-finite physical position");
    const Vec3 p=frame_.components(point-frame_.origin);const auto s=section(p.x,p.y);
    // This is an inverse coordinate map, not an inside/outside test; do not
    // clip sigma to make an exterior point appear inside the liquid.
    const Vec3 out={p.x,p.y,(p.z-s.lower)/s.height};
    if(!finite(out))throw std::overflow_error("Inverse gap coordinate is not representable");
    return out;
  }
  Vec3 outwardLowerNormal(double x,double y) const {
    const auto s=section(x,y);return normalized(frame_.vector({s.lowerX,s.lowerY,-1}));
  }
  Vec3 outwardUpperNormal(double x,double y) const {
    const auto s=section(x,y);return normalized(frame_.vector({-s.upperX,-s.upperY,1}));
  }
  double diskVolume(double radius) const {
    positive(radius,"Gap footprint radius must be positive");
    if(!(radius<chartRadius()))throw std::domain_error("Gap footprint reaches sphere equator");
    const double pi=std::acos(-1.0);
    const auto contribution=[&](double a){
      if(a==0)return 0.0;
      const double sag=sphereSag(a,radius,0).value;
      return pi*sag*sag*(a-(2.0/3.0)*sag);
    };
    const double out=pi*radius*radius*gap_+contribution(lowerRadius_)+contribution(upperRadius_);
    positive(out,"Gap disk volume is not representable");return out;
  }
};
} // namespace faraday_native::resolved_gap
