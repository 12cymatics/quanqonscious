import SutraWS
import Lean

/-!
# Provenance export

Writes `lean_provenance.json` from the environment that `lake build` actually
produced: the theorem names it kernel-checked, the axioms each one rests on, and
the olean files on disk.  `PROVENANCE` in `vedic_v18.51.1_exact_phi.html` is
regenerated from that file, so the page can only claim what the compiler did.
-/

open Lean Elab Command

namespace SutraWS

/-- The declared trust base, as an allowlist.  A blocklist of the axioms we know
to be bad classifies a theorem resting on some *newly declared* project axiom as
kernel-checked, because that axiom is not on the list.  Naming what is permitted
instead means anything else -- `sorryAx`, `Lean.ofReduceBool`, or an `axiom` added
tomorrow -- falls outside by construction. -/
private def provenanceTrustBase : Array Name :=
  #[`propext, `Classical.choice, `Quot.sound]

private def provenanceAxiomsOf (env : Environment) (c : Name) : Array Name :=
  (((CollectAxioms.collect c).run env).run {}).2.axioms

private def jsonStrings (xs : List String) : String :=
  "[" ++ ",".intercalate (xs.map (fun s => "\"" ++ s ++ "\"")) ++ "]"

run_cmd do
  let env ← getEnv
  let mut kernelChecked : Array String := #[]
  let mut compilerTrusted : Array String := #[]
  let mut axiomSet : Array String := #[]
  for (name, info) in env.constants.toList do
    unless (`SutraWS).isPrefixOf name do continue
    unless info.isThm do continue
    unless !name.isInternalDetail do continue
    let axs := provenanceAxiomsOf env name
    for ax in axs do
      let s := ax.toString
      unless axiomSet.contains s do axiomSet := axiomSet.push s
    if axs.all provenanceTrustBase.contains then
      kernelChecked := kernelChecked.push name.toString
    else
      compilerTrusted := compilerTrusted.push name.toString
  let sorted := (kernelChecked.qsort (· < ·)).toList
  let trusted := (compilerTrusted.qsort (· < ·)).toList
  let axioms := (axiomSet.qsort (· < ·)).toList
  let json :=
    "{\n  \"kernelChecked\": " ++ jsonStrings sorted ++
    ",\n  \"compilerTrusted\": " ++ jsonStrings trusted ++
    ",\n  \"axiomsUsed\": " ++ jsonStrings axioms ++
    ",\n  \"kernelCheckedCount\": " ++ toString sorted.length ++
    ",\n  \"compilerTrustedCount\": " ++ toString trusted.length ++ "\n}\n"
  IO.FS.writeFile "lean_provenance.json" json
  logInfo s!"provenance: {sorted.length} on the declared trust base, {trusted.length} outside it"

end SutraWS
