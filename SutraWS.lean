import SutraWS.Sutra
import SutraWS.State
import SutraWS.Interval
import SutraWS.SutraSemantics
import SutraWS.Proofs
import SutraWS.Exhaustive
import SutraWS.Vertex
import SutraWS.VertexProofs
import SutraWS.Contracts

/-!
# SutraWS — root module

Two layers:

* **Vertex substrate** (`Vertex`, `VertexProofs`, `Contracts`) — the Z₂⁴ tesseract field of the
  v18 kernel, its Hadamard duality, and the α → 0 identity guarantee for all 29 operators.
* **`LeanState` bookkeeping** (`Sutra`, `State`, `SutraSemantics`, `Proofs`, `Exhaustive`) — the
  7-rational counter transcribed from `simulation v18:693-704`, whose content is the triangular
  identity Σδ(1..29) = 435.
-/
