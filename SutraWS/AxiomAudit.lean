import SutraWS.Sutra
import SutraWS.State
import SutraWS.Interval
import SutraWS.SutraSemantics
import SutraWS.Proofs
import SutraWS.Exhaustive
import SutraWS.Vertex
import SutraWS.VertexProofs
import SutraWS.Contracts
import SutraWS.WheelerGeometry
import SutraWS.RenderCertificates
import SutraWS.DECComplex
import Lean

/-!
# Axiom audit

Every theorem in the `SutraWS` namespace must rest on the Lean kernel plus
mathlib's three standard axioms (`propext`, `Classical.choice`, `Quot.sound`)
and nothing else.

Two axioms are specifically excluded:

* `sorryAx` — an admitted goal. A file full of `sorry` still compiles, and a
  `lake build` that only checks for errors reports it as proved.
* `Lean.ofReduceBool` / `Lean.ofReduceNat` — introduced by `native_decide`,
  which discharges a goal by *running the compiled program* instead of by
  kernel reduction, putting the Lean compiler and runtime into the trusted
  base. mathlib forbids it for that reason. This library was originally proved
  that way at 178 sites across three files, putting `ofReduceBool` into the
  trust base of 212 of its 2052 theorems;
  `norm_num [Sutra.all, sumDelta, delta]` closes all three obligation shapes
  without it. `decide` cannot: `Rat`'s `DecidableEq` reduces through
  `Nat.gcd`'s well-founded recursion and the kernel gives up.

This is a build-time gate, not a report: an offending theorem fails
`lake build`. It enumerates the environment rather than a written-down list,
so a theorem added later is covered without editing this file.
-/

open Lean Elab Command

namespace SutraWS

/-- The permitted trust base, as an allowlist rather than a blocklist of the
axioms we happen to know about. A blocklist passes a theorem that rests on some
newly declared project `axiom`, because that name is not on it; naming what is
allowed puts everything else outside by construction. -/
private def permittedAxioms : Array Name :=
  #[`propext, `Classical.choice, `Quot.sound]

private def axiomsOf (env : Environment) (c : Name) : Array Name :=
  (((CollectAxioms.collect c).run env).run {}).2.axioms

run_cmd do
  let env ← getEnv
  let mut offenders : Array (Name × Name) := #[]
  let mut audited := 0
  for (name, info) in env.constants.toList do
    unless (`SutraWS).isPrefixOf name do continue
    unless info.isThm do continue
    unless !name.isInternalDetail do continue
    audited := audited + 1
    for ax in axiomsOf env name do
      unless permittedAxioms.contains ax do
        offenders := offenders.push (name, ax)
  if !offenders.isEmpty then
    let lines := (offenders.map fun (n, ax) => s!"  {n} uses {ax}").toList
    throwError "SutraWS axiom audit FAILED ({offenders.size} theorem(s) of \
      {audited} resting on an axiom outside the declared trust base \
      {permittedAxioms.toList}):\n{"\n".intercalate lines}"
  else
    logInfo s!"SutraWS axiom audit: {audited} theorems, all on {permittedAxioms.toList}"

end SutraWS
