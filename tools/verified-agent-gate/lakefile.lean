import Lake
open Lake DSL
package byteGate where
  version := v!"3.0.0"
  fixedToolchain := true
lean_lib Gate where
  roots := #[`Defs, `Gate]
lean_lib ComparatorChallenges
@[default_target]
lean_exe bytegate where
  root := `Main
