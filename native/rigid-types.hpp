#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace faraday_native {
struct Vec3 {
  double x=0, y=0, z=0;
  double& operator[](std::size_t i) { return i==0?x:i==1?y:z; }
  double operator[](std::size_t i) const { return i==0?x:i==1?y:z; }
};
inline Vec3 operator+(Vec3 a,Vec3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
inline Vec3 operator-(Vec3 a,Vec3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
inline Vec3 operator-(Vec3 a){return {-a.x,-a.y,-a.z};}
inline Vec3 operator*(double s,Vec3 a){return {s*a.x,s*a.y,s*a.z};}
inline Vec3 operator*(Vec3 a,double s){return s*a;}
inline Vec3 operator/(Vec3 a,double s){return (1/s)*a;}
inline double dot(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline Vec3 cross(Vec3 a,Vec3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline double norm(Vec3 a){return std::sqrt(dot(a,a));}
inline Vec3 unit(Vec3 a){const double n=norm(a);if(!(n>0))throw std::invalid_argument("Zero direction");return a/n;}
inline bool finite(Vec3 a){return std::isfinite(a.x)&&std::isfinite(a.y)&&std::isfinite(a.z);}

// Sphere geometry is a declared case input, not a claim about an unmeasured
// beach sample. Every length is in metres and every density in kg/m^3.
struct RigidGrain {
  std::size_t id=0;
  Vec3 centre, velocity, omega;
  double radius=0, density=0;
  double volume() const {return (4.0/3.0)*std::acos(-1.0)*radius*radius*radius;}
  double mass() const {return density*volume();}
  double inertia() const {return 0.4*mass()*radius*radius;}
  void validate() const {
    if(!(radius>0&&density>0)||!std::isfinite(radius)||!std::isfinite(density)
       ||!finite(centre)||!finite(velocity)||!finite(omega))
      throw std::invalid_argument("Grain geometry, density and state must be finite, with positive radius and density");
  }
};
using Six = std::array<double,6>;
inline Six rigidVelocity(const RigidGrain& g){return {g.velocity.x,g.velocity.y,g.velocity.z,g.omega.x,g.omega.y,g.omega.z};}
inline void setRigidVelocity(RigidGrain& g,const Six& q){g.velocity={q[0],q[1],q[2]};g.omega={q[3],q[4],q[5]};}
} // namespace faraday_native
