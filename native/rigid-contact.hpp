#pragma once
#include "rigid-types.hpp"
#include <limits>
#include <optional>
#include <string>

namespace faraday_native {

constexpr std::size_t fixedBody = std::numeric_limits<std::size_t>::max();
enum class ContactKind { spherePair, floor, cylinderWall };

// The normal points in the direction of a compressive impulse on body a.
// Contact velocities are velocity(a, armA) - velocity(b, armB), expressed
// in the right-handed (normal, tangent1, tangent2) basis. A fixed wall is b.
struct ContactGeometry {
  ContactKind kind=ContactKind::spherePair;
  std::size_t a=0, b=fixedBody;
  double gap=0;
  Vec3 normal{}, tangent1{}, tangent2{}, armA{}, armB{};
  bool normalDefined=true;
};
ContactGeometry spherePairContact(const RigidGrain& a,std::size_t ia,
                                  const RigidGrain& b,std::size_t ib);
ContactGeometry floorContact(const RigidGrain& a,std::size_t ia,double floorZ);
ContactGeometry cylinderContact(const RigidGrain& a,std::size_t ia,double cylinderRadius);
Vec3 contactCoordinates(const ContactGeometry& c,Vec3 worldVector);
Vec3 contactWorldVector(const ContactGeometry& c,Vec3 localVector);
Vec3 relativeContactVelocity(const ContactGeometry& c,const std::vector<RigidGrain>& grains);
// Add equal-and-opposite forces/torques to the supplied body impulse array.
// The caller applies these through its COUPLED fluid/body mobility, not through
// the inverse dry grain mass. No body state is changed here.
// gapTolerance is an explicitly supplied event-location error tolerance, in
// metres. It must never stand in for a lubrication/contact-distance model.
void accumulateContactImpulse(const ContactGeometry& c,Vec3 localImpulse,std::vector<Six>& bodyImpulses,double gapTolerance=0);

// Exact contact times for a CONSTANT centre velocity during the trial segment.
// Accelerated trajectories require event refinement by the caller. Starting
// overlaps are errors; no contact query changes or projects grain positions.
std::optional<double> spherePairImpactTime(const RigidGrain& a,const RigidGrain& b,double horizon);
std::optional<double> floorImpactTime(const RigidGrain& a,double floorZ,double horizon);
std::optional<double> cylinderImpactTime(const RigidGrain& a,double cylinderRadius,double horizon);
bool cylinderNeedsSustainedConstraint(const RigidGrain& a,double cylinderRadius);

enum class GapResolution { separatedResolved, positiveGapUnresolved, touching, overlapping };
GapResolution gapResolution(const ContactGeometry& c,double minimumResolvedGap);

struct ContactProblem {
  // Row-major, symmetric positive-semidefinite Delassus matrix. There are
  // three coordinates per contact, ordered normal, tangent1, tangent2.
  // It must include the liquid response to body impulses for a wet simulation.
  std::vector<double> mobility;
  std::vector<double> freeVelocity;
  std::vector<double> normalTarget;
  std::vector<double> friction;
};
struct ContactSolveOptions {
  double velocityTolerance=1e-10;       // m/s
  double impulseTolerance=1e-14;        // N s
  double relativeTolerance=1e-10;
  std::size_t maximumIterations=50000;
};
struct ContactSolution {
  std::vector<double> impulse, velocity;
  std::size_t iterations=0;
  double normalResidual=0;             // m/s
  double tangentResidual=0;            // m/s
  double coneResidual=0;               // N s
  double energyChange=0;               // J, relative to fixed contact frames
};

// Newton normal restitution for newly impacting contacts only. Sustained
// contacts use target zero. Coefficients must be supplied, not inferred.
std::vector<double> impactNormalTargets(const std::vector<double>& beforeVelocity,
                                        const std::vector<double>& restitution);

// Solves unilateral normal contact and the isotropic Coulomb law against
// FINAL tangential velocity. The disk/nonnegative projections are the stated
// physical admissible contact sets, not numerical caps on arbitrary forces.
// The tangential block is solved with its full 2x2 mobility, not a diagonal
// approximation. Iteration or passive-energy failure throws; no state or
// impulse is repaired. Newton restitution at simultaneous impacts is a stated
// constitutive choice and is refused if its solution would add contact energy.
ContactSolution solveContacts(const ContactProblem& problem,
                              const ContactSolveOptions& options={});
// Independently test the actual coupled velocity after applying solved
// impulses. This does not substitute the mobility matrix's predicted velocity.
ContactSolution verifyContactResponse(const ContactProblem& problem,
    const std::vector<double>& impulse,const std::vector<double>& actualVelocity,
    const ContactSolveOptions& options={});
} // namespace faraday_native
