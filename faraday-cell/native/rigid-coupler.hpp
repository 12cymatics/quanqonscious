#pragma once
#include "fluid-runtime.hpp"
#include "rigid-contact.hpp"
#include "rigid-interpolation.hpp"
#include <memory>

namespace faraday_native {
struct CouplingOptions {
  RigidInterpolationOptions interpolation;
  double relativeTolerance=1e-9;
  double absoluteVelocityTolerance=1e-13;
  double maximumDivergence=1e-8;
  std::size_t maximumIterations=3000;
  double friction=0, restitution=0;
  double resolvedGapCells=2;
  std::size_t maximumEventRefinements=30;
};
struct CoupledStepReport {
  double dt=0, time=0, slipResidual=0, divergence=0;
  std::size_t constraintIterations=0, markerCount=0, eventRefinements=0, contacts=0;
  // Integrated hydrodynamic constraint impulse, including contact-induced
  // fluid reactions. Contact impulse is reported separately (force, torque).
  std::vector<Six> rigidImpulse, contactImpulse;
};
std::vector<FluidDof> independentFluidDofs(const NativeFluid& fluid);

// Fictitious-domain rigid volume constraints. Fluid fills the vessel, so body
// inertia here is ONLY the excess over the fluid occupying each sphere.
// Hydrodynamic drag/added mass/history arise from resolved NS, not force laws.
class RigidCoupler {
public:
  RigidCoupler(NativeFluid&,std::vector<RigidGrain>,CouplingOptions={});
  const std::vector<RigidGrain>& grains() const {return grains_;}
  // Conservative peak working-array estimate; excludes allocator/OS overhead.
  std::size_t workingArrayBytes() const;
  CoupledStepReport step(double requestedDt,double effectiveGravity);
private:
  NativeFluid& fluid_;
  std::vector<RigidGrain> grains_;
  CouplingOptions options_;
  std::vector<FluidDof> dofs_;
  std::unique_ptr<RigidInterpolation> interpolation_;
  std::vector<double> inverseMass_,inverseInertia_;
  std::vector<double> scratchU_,scratchV_,scratchW_;
  std::vector<double> diagonal_;
  double pendingDt_=0;
  std::vector<ContactGeometry> sustainedContacts_;
  std::size_t baseWorkingArrayBytes() const;
  void rebuild();
  std::vector<double> packFluid() const;
  void addFluid(const std::vector<double>&,double scale);
  std::vector<double> project(const std::vector<double>&);
  std::vector<Six> bodyResponse(const std::vector<Six>&) const;
  std::vector<Vec3> applySchur(const std::vector<Vec3>&);
  std::vector<Vec3> solve(const std::vector<Vec3>&,std::size_t& iterations);
  struct Response {std::vector<double> fluid;std::vector<Six> bodies;std::vector<double> pressurePotential;std::vector<Six> rigidImpulse;std::size_t iterations=0;};
  Response constrainedBodyResponse(const std::vector<Six>&);
  void constrain(std::vector<Six>&,CoupledStepReport&);
  void validateGeometry() const;
  std::vector<ContactGeometry> touchingContacts() const;
  double firstImpact(double horizon) const;
  void validateSweptGaps(const std::vector<RigidGrain>& before,double dt) const;
  void applyContacts(std::vector<Six>&,CoupledStepReport&);
};
} // namespace faraday_native
