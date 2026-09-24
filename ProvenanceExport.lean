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

private def provenanceForbidden : Array Name :=
  #[``sorryAx, `Lean.ofReduceBool, `Lean.ofReduceNat, `Lean.trustCompiler]

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
    if axs.any provenanceForbidden.contains then
      compilerTrusted := compilerTrusted.push name.toString
    else
      kernelChecked := kernelChecked.push name.toString
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
  logInfo s!"provenance: {sorted.length} kernel-checked, {trusted.length} compiler-trusted"

end SutraWS
