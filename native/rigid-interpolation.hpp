#pragma once
#include "rigid-types.hpp"
#include <cstddef>
#include <limits>
#include <vector>

namespace faraday_native {

// One independent staggered velocity component. index is its index in the
// corresponding native fluid family; a packed vector uses this list's order.
// Prescribed axis/wall/floor values are not independent DOFs and must be omitted.
struct FluidDof {
  int family=0;
  std::size_t index=0;
  Vec3 position, direction;
  double mass=0;
  double spacing=0; // largest physical cell spacing local to this node, metres
};
struct RigidMarker {
  std::size_t grain=0;
  Vec3 position, offset;
};
struct InterpolationEntry {
  std::size_t dof=0;
  Vec3 coefficient;
};
struct InterpolationStencil {
  std::vector<InterpolationEntry> entries;
  double supportRadius=0, condition=0, affineResidual=0;
};
struct RigidInterpolationOptions {
  double minimumCellsPerDiameter=8;
  double markerSpacingFactor=2;
  double supportRadiusFactor=3;
  double maximumCondition=1e8;
  double affineTolerance=2e-11;
  // Dedicated working-array allowance supplied by the coupled runtime, after
  // accounting for fluid fields and the simultaneous constraint workspace.
  std::size_t workingMemoryBudgetBytes=std::numeric_limits<std::size_t>::max();
};

// These are volumetric collocation points, not an exact sphere-volume
// quadrature. Markers become denser with mesh refinement. The requested minimum
// cells/diameter is a refusal gate, not a substitute for a convergence study.
std::vector<RigidMarker> sphereVolumeMarkers(const RigidGrain& grain,
    std::size_t grainIndex,double fluidSpacing,
    const RigidInterpolationOptions& options={});

// Affine Cartesian reconstruction of directional staggered velocities. A
// column-pivoted, reorthogonalized QR factorization is used directly; no normal
// equations, diagonal regularization, coefficient clipping or force cap.
InterpolationStencil affineVelocityStencil(const std::vector<FluidDof>& dofs,
    const std::vector<std::size_t>& neighbours,Vec3 point,double supportRadius,
    const RigidInterpolationOptions& options={});

class RigidInterpolation {
public:
  // The caller owns dofs, whose geometry/masses must remain unchanged for this
  // object's lifetime. Rebuild after moving the metric or marker positions.
  RigidInterpolation(const std::vector<FluidDof>& dofs,
      const std::vector<RigidGrain>& grains,
      const RigidInterpolationOptions& options={});
  const std::vector<RigidMarker>& markers() const{return markers_;}
  const std::vector<InterpolationStencil>& stencils() const{return stencils_;}
  const std::vector<double>& grainSpacing() const{return grainSpacing_;}
  std::size_t fluidDofCount() const{return dofs_->size();}
  std::size_t grainCount() const{return grainCount_;}
  std::size_t allocatedBytes() const;

  std::vector<Vec3> applyJ(const std::vector<double>& fluidVelocity) const;
  // Multipliers here are integrated Cartesian forces/impulses at markers.
  // Thus J^T returns integrated directional DOF forces/impulses. Dividing by
  // each physical DOF mass gives the weighted adjoint M_f^-1 J^T.
  std::vector<double> spreadTranspose(const std::vector<Vec3>& markerImpulse) const;
  std::vector<double> weightedAdjoint(const std::vector<Vec3>& markerImpulse) const;
  std::vector<Vec3> applyK(const std::vector<Six>& grainVelocity) const;
  std::vector<Six> spreadKTranspose(const std::vector<Vec3>& markerImpulse) const;
private:
  const std::vector<FluidDof>* dofs_;
  std::size_t grainCount_;
  std::vector<RigidMarker> markers_;
  std::vector<InterpolationStencil> stencils_;
  std::vector<double> grainSpacing_;
};
} // namespace faraday_native
