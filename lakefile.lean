import Lake
open Lake DSL

package sutraws_tgcr_lean where
  version := "0.3.0"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.10.0"

@[default_target]
lean_lib SutraWS where
  globs := #[.submodules `SutraWS]
