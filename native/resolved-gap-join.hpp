#pragma once
#include "resolved-gap-flow.hpp"

// A conforming, non-overlapping pair of fixed exact-chart NS patches.
// This reference joins two patches of the SAME discretization. It does not
// connect this FEM mesh to the vessel's staggered grid or move/remesh grains.
namespace faraday_native::resolved_gap_join {
using resolved_gap_flow::BoundaryDof;
using resolved_gap_flow::BoundaryLoad;
using resolved_gap_flow::BoundarySample;
using resolved_gap_flow::Linearization;
using resolved_gap_flow::Patch;
using resolved_gap_flow::Problem;
using resolved_gap_flow::SolveResult;
using resolved_gap_flow::State;
struct Limits {
  std::size_t maximumUnknowns=2500;
  // Bounds the concurrently live dense matrices, not merely one matrix.
  std::size_t maximumMatrixBytes=128*1024*1024;
};
struct InterfaceSample {BoundarySample left,right;};
class Pair {
  struct Impl;std::shared_ptr<Impl> impl_;
public:
  Pair(Patch left,Patch right,Limits limits={});
  const std::vector<Vec3>& velocityNodes() const;
  const std::vector<Vec3>& pressureNodes() const;
  const std::vector<BoundaryDof>& boundaryDofs() const;
  // Topological maps preserve one shared velocity and pressure unknown at
  // every interface node. No averaging of independent solutions is involved.
  const std::vector<std::size_t>& velocityMap(unsigned side) const;
  const std::vector<std::size_t>& pressureMap(unsigned side) const;
  State localState(const State& global,unsigned side) const;
  State initialState(const Problem& problem) const;
  Linearization linearize(const State&,const Problem&,const State* previous=nullptr) const;
  SolveResult solve(const Problem&,const State* initial=nullptr,const State* previous=nullptr) const;
  std::vector<BoundaryLoad> boundaryLoads(const State&,const Problem&,const State* previous=nullptr) const;
  std::vector<BoundarySample> boundarySamples(const State&) const;
  std::vector<InterfaceSample> interfaceSamples(const State&) const;
};
} // namespace faraday_native::resolved_gap_join
