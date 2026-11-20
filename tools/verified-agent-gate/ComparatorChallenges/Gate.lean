import Defs
namespace ByteGate
theorem check_sound (c : Claims) (h : check c = true) : Accepted c := by sorry
theorem check_complete (c : Claims) (h : Accepted c) : check c = true := by sorry
theorem byte_preservation (c : Claims) (h : check c = true) : c.source = c.expected := by sorry
theorem length_preservation (c : Claims) (h : check c = true) : c.claimedLength = c.expected.length := by sorry
theorem changed_bytes_rejected (c : Claims) (h : c.source ≠ c.expected) : check c = false := by sorry
theorem replay_rejected (c : Claims) (h : c.id ∈ c.used) : check c = false := by sorry
theorem bad_signature_rejected (c : Claims) (h : c.signatureOK = false) : check c = false := by sorry
theorem bad_digest_rejected (c : Claims) (h : c.digestOK = false) : check c = false := by sorry
theorem noncanonical_rejected (c : Claims) (h : c.canonicalOK = false) : check c = false := by sorry
theorem changed_code_rejected (c : Claims) (h : c.filesOK = false) : check c = false := by sorry
theorem wrong_parent_rejected (c : Claims) (h : c.parent ≠ c.expectedParent) : check c = false := by sorry
theorem unknown_parent_rejected (c : Claims) (h : ¬ ParentKnown c) : check c = false := by sorry
theorem accepted_once (c : Claims) : check (consume c) = false := by sorry
theorem chain_preserves_bytes (source : List UInt8) (history : List Claims)
    (h : Chain source history) : ∀ c ∈ history, c.source = source := by sorry
end ByteGate
