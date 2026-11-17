#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <span>
#include <vector>

namespace faraday_native {

// The indices are the canonical cell3d_ptr/cell3d_len table used by the existing
// JavaScript/WebAssembly bridge. The caller copies its canonical grid and flat
// pressure preconditioner into these arrays before initialize().
enum class FluidArray : int {
  rf=0, sf=1, rc=2, sc=3, drc=4, dsc=5, drf=6, dsf=7, rx=8,
  u=9, v=10, w=11, om=12, p=13, eta=14,
  H=15, Hr=16, Hth=17, Hdr=18, Hdth=19, Ht=20,
  Hx=21, Hxr=22, Hxt=23, Ex=24, Er=25, Eth=26, Edr=27, Edth=28,
  Exr=29, Ext=30, Tx=31, Txr=32, Txt=33, Tr=34, Tth=35, Tdr=36, Tdth=37,
  gu=38, gv=39, gw=40, gom=41, div=42, cgr=43, cgd=44, cgq=45, cgz=46,
  lapU=47, lapV=48, lapW=49, advU=50, advV=51, advW=52,
  fsr=53, fst=54, fsz=55, kap=56, psurf=57,
  hcolP=58, hcolU=59, hcolV=60, hcolW=61,
  twc=62, tws=63, pcband=64, pchatc=65, pchats=66
};

struct FluidConfig {
  int nr=0, nth=0, nz=0;
  bool pinned=true;
  double R=0, h=0, rho=0, nu=0, gamma=0, g=0, dtheta=0;
  // Resource limit, not a numerical limiter: an over-budget request fails.
  std::size_t memoryBudgetBytes=1024ULL*1024ULL*1024ULL;
};

struct FluidCheckpoint {
  std::uint64_t ownerGeneration=0;
  double time=0;
  // u, v, w, om, p, eta, Ht, in that order.
  std::array<std::vector<double>,7> state;
};

struct StepLimits {
  double capillary=0, viscous=0, advective=0, gravity=0;
  double kRadial=0, kVertical=0, kAzimuthal=0;
};

// The current canonical solver has one global field arena. This wrapper permits
// exactly one live instance and releases every native allocation on destruction.
// It is not thread-safe; parallel explicit rows must be coordinated externally.
class NativeFluid {
public:
  explicit NativeFluid(const FluidConfig&);
  ~NativeFluid();
  NativeFluid(const NativeFluid&)=delete;
  NativeFluid& operator=(const NativeFluid&)=delete;
  NativeFluid(NativeFluid&&)=delete;
  NativeFluid& operator=(NativeFluid&&)=delete;

  const FluidConfig& config() const noexcept { return config_; }
  int nr() const noexcept { return config_.nr; }
  int nth() const noexcept { return config_.nth; }
  int nz() const noexcept { return config_.nz; }
  double time() const noexcept { return time_; }
  std::size_t allocatedBytes() const noexcept;
  std::span<double> array(int key);
  std::span<const double> array(int key) const;
  std::span<double> array(FluidArray key) { return array(static_cast<int>(key)); }
  std::span<const double> array(FluidArray key) const { return array(static_cast<int>(key)); }

  // Required inputs: grid keys 0..8 and 62..64. State keys 9..14 and 20
  // are initially zero, or may be copied from a canonical initial condition.
  void initialize();
  void refreshMetric();
  // Opt in when a rigid-grain constraint solve is present. The pressure
  // operator then eliminates the dependent axis velocity and includes that
  // elimination's transpose. Ordinary zero-grain/WASM arithmetic is unchanged.
  // Enabling changes the operator and invalidates earlier checkpoints.
  void enableRigidProjection();
  bool rigidProjectionEnabled() const noexcept { return rigidProjection_; }
  FluidCheckpoint checkpoint() const;
  void restore(const FluidCheckpoint&);

  // Identical zero-grain step; effective gravity is supplied by the caller so
  // native and JS/WASM comparisons use identical transcendental values.
  void step(double dt, double effectiveGravity);

  // Canonical predictor AND inhomogeneous free-surface pressure projection.
  // Leaves a divergence-free predictor and stores surface pressure at old eta.
  // Does not advance eta or time. Coupled constraints are solved next.
  void predictor(double dt, double effectiveGravity);

  // Apply P = I-G(DG)^-1 D to supplied physical velocity increments, using
  // homogeneous free-surface pressure. Does not alter canonical u/v/w/p/eta.
  // Auxiliary axis/rim/floor rows are not independent force degrees of freedom.
  // Components require disjoint caller-owned storage, with no canonical aliases.
  void projectHomogeneous(std::span<double> du, std::span<double> dv,
                          std::span<double> dw, double tolerance=1e-11);
  // q from the most recent homogeneous projection (D G q = D delta-u).
  // When its corrected increment is actually added to the physical state,
  // add rho*q/dt to canonical reduced pressure p. Iterative trial projections
  // must not change physical p. The next projection overwrites this span.
  std::span<const double> lastProjectionPotential() const noexcept {
    return projectionPressure_;
  }

  // Consume the final jointly corrected physical u/v/w, then advance eta/time.
  // Must follow predictor with the same dt; no extra fluid predictor is applied.
  void finish(double dt);

  // rho times the control volume used by the canonical gradient. Returns zero
  // for prescribed/auxiliary u axis+rim and w floor rows. component=0/1/2.
  double velocityMass(int component, std::size_t index) const;
  double maxDivergence();
  // Same discrete capillary/viscous/advective/gravity bounds as the JS solver.
  // Supply the drive's peak acceleration, rather than instantaneous gravity.
  StepLimits stepLimits(double peakAcceleration=0) const;
  double stableStep(double peakAcceleration=0,double safety=0.4) const;
  int pressureIterations() const noexcept;
  double pressureResidual() const noexcept;

private:
  FluidConfig config_;
  std::uint64_t generation_=0;
  double time_=0, pendingDt_=0;
  bool initialized_=false, pending_=false;
  bool rigidProjection_=false;
  std::vector<double> projectionOmega_, projectionRhs_, projectionPressure_;
  void requireReady() const;
  void checkError(const char* operation) const;
};

} // namespace faraday_native
