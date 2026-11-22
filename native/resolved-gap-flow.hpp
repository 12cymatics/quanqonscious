#pragma once
#include "resolved-gap.hpp"
#include <functional>
#include <memory>
#include <optional>

// Reference-size, fixed-geometry, mapped 3D Navier--Stokes patch. Q2 velocity /
// Q1 pressure; no pressure penalty, empirical gap force or geometry clipping.
// The vessel interface and moving-mesh/grain evolution are NOT joined here.
namespace faraday_native::resolved_gap_flow {
struct Options {
  unsigned nx=2,ny=2,nz=2;
  double halfWidth=.3,density=1,dynamicViscosity=.1,timeStep=0;
  // Numerical normalization only; this does not prescribe fluid velocity.
  // With L=halfWidth and U=referenceVelocity, equations are scaled by
  // P L^2 and U L^2, where P=rho U^2+mu U/L. State values remain in SI units.
  double referenceVelocity=1;
  // Exact chart-coordinate footprint; changing the center does not translate
  // the sphere geometry. Omitted eta width uses halfWidth.
  double centerXi=0,centerEta=0;
  std::optional<double> halfWidthEta;
  double nonlinearTolerance=1e-11;
  unsigned maximumNewtonIterations=16;
  std::size_t maximumUnknowns=2500,maximumMatrixBytes=128*1024*1024;
};
struct Problem {
  // Force per unit physical volume, including any specified body force.
  std::function<Vec3(Vec3)> force,boundaryVelocity;
  // Bits 0..5 select velocity faces (xi-/+, eta-/+, sigma-/+). Every other
  // face supplies physical traction sigma*n, including pressure. At a shared
  // edge a velocity condition takes precedence for that velocity node.
  unsigned velocityFaceMask=63;
  std::function<Vec3(unsigned,Vec3,Vec3)> boundaryTraction;
  // Exact affine pressure decomposition: p_total=p_Q1+dot(G,x). The physical
  // force remains supplied in force(); assembly uses force-G and transforms
  // physical boundary traction consistently. Useful for uniform gravity:
  // physical z is not in the mapped Q1 space on an exact curved chart.
  Vec3 pressureGradientOffset{};
  // Used only when all faces prescribe velocity. Any traction face fixes the
  // absolute pressure, so that case retains every pressure/continuity DOF.
  double pressureGauge=0;
};
struct State {
  std::vector<Vec3> velocity;
  // Coefficients of p_Q1. sample() and boundarySamples() reconstruct p_total.
  std::vector<double> pressure;
  Vec3 pressureGradientOffset{};
};
struct Field {
  Vec3 velocity{},pressureGradient{};
  std::array<Vec3,3> velocityGradient{};
  double pressure=0;
};
struct BoundaryDof {std::size_t node=0;Vec3 point{};unsigned faceMask=0;};
struct Bounds {double xiLower=0,xiUpper=0,etaLower=0,etaUpper=0;};
struct BoundarySample {
  unsigned face=0;Vec3 point{},normal{},velocity{},traction{};
  // Per unit physical area, outward from the patch. Geometry is fixed here:
  // advective = rho*u*(u.n), total = advective - sigma*n. A moving patch
  // instead needs rho*u*((u-gridVelocity).n), with its ALE/GCL equations.
  Vec3 advectiveMomentumFlux{},outwardMomentumFlux{};
  double weight=0,pressure=0,normalVelocity=0;
};
struct BoundaryLoad {std::size_t node=0;Vec3 reaction{};};
struct SolveResult {
  State state;
  unsigned newtonIterations=0;
  std::size_t unknowns=0;
  double initialResidual=0,finalResidual=0,maximumContinuityMoment=0;
  // Residual norms above are dimensionless after equation-wise scaling;
  // maximumContinuityMoment retains physical volume/time units.
  double momentumResidualScale=0,continuityResidualScale=0;
};
struct Linearization {
  // Uneliminated element assembly: 3*velocityNodes, then ALL pressureNodes.
  // Boundary velocity/normal flux DOFs and the pressure constant are retained
  // for a future shared interface/global gauge solve. Physical boundary or
  // interface equations/loads must be supplied before treating this as a
  // complete coupled boundary-value problem.
  std::size_t velocityNodes=0,pressureNodes=0;
  std::vector<double> matrix,residual;
  std::size_t dimension() const{return 3*velocityNodes+pressureNodes;}
};
class Patch {
  struct Impl;std::shared_ptr<Impl> impl_;
public:
  Patch(resolved_gap::Chart chart,Options options={});
  const Options& options() const;
  const resolved_gap::Chart& chart() const;
  const Bounds& bounds() const;
  const std::vector<Vec3>& velocityNodes() const;
  const std::vector<Vec3>& pressureNodes() const;
  const std::vector<BoundaryDof>& boundaryDofs() const;
  State initialState(const Problem& problem) const;
  SolveResult solve(const Problem& problem,const State* initial=nullptr,
                    const State* previous=nullptr) const;
  Field sample(const State& state,double xi,double eta,double sigma) const;
  std::vector<BoundarySample> boundarySamples(const State& state) const;
  std::vector<BoundaryLoad> boundaryLoads(const State& state,const Problem& problem,
                                         const State* previous=nullptr) const;
  Linearization linearize(const State& state,const Problem& problem,
                           const State* previous=nullptr) const;
};
} // namespace faraday_native::resolved_gap_flow
