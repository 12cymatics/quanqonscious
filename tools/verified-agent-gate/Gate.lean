import Defs
namespace ByteGate
theorem check_sound (c : Claims) (h : check c = true) : Accepted c := of_decide_eq_true h
theorem check_complete (c : Claims) (h : Accepted c) : check c = true := decide_eq_true h
theorem byte_preservation (c : Claims) (h : check c = true) : c.source = c.expected :=
  (check_sound c h).1
theorem length_preservation (c : Claims) (h : check c = true) : c.claimedLength = c.expected.length := by
  have a := check_sound c h
  rw [← a.1]
  exact a.2.1
theorem changed_bytes_rejected (c : Claims) (h : c.source ≠ c.expected) : check c = false := by
  apply Bool.eq_false_iff.mpr
  intro hc
  exact h (byte_preservation c hc)
theorem replay_rejected (c : Claims) (h : c.id ∈ c.used) : check c = false := by
  apply Bool.eq_false_iff.mpr
  intro hc
  exact (check_sound c hc).2.2.2.2.2.2.1 h
theorem bad_signature_rejected (c : Claims) (h : c.signatureOK = false) : check c = false := by
  apply Bool.eq_false_iff.mpr
  intro hc
  have hs := (check_sound c hc).2.2.1
  simp [h] at hs
theorem bad_digest_rejected (c : Claims) (h : c.digestOK = false) : check c = false := by
  apply Bool.eq_false_iff.mpr
  intro hc
  have hs := (check_sound c hc).2.2.2.1
  simp [h] at hs
theorem noncanonical_rejected (c : Claims) (h : c.canonicalOK = false) : check c = false := by
  apply Bool.eq_false_iff.mpr
  intro hc
  have hs := (check_sound c hc).2.2.2.2.1
  simp [h] at hs
theorem changed_code_rejected (c : Claims) (h : c.filesOK = false) : check c = false := by
  apply Bool.eq_false_iff.mpr
  intro hc
  have hs := (check_sound c hc).2.2.2.2.2.1
  simp [h] at hs
theorem wrong_parent_rejected (c : Claims) (h : c.parent ≠ c.expectedParent) : check c = false := by
  apply Bool.eq_false_iff.mpr
  intro hc
  exact h (check_sound c hc).2.2.2.2.2.2.2.1
theorem unknown_parent_rejected (c : Claims) (h : ¬ ParentKnown c) : check c = false := by
  apply Bool.eq_false_iff.mpr
  intro hc
  exact h (check_sound c hc).2.2.2.2.2.2.2.2
theorem accepted_once (c : Claims) : check (consume c) = false :=
  replay_rejected (consume c) (by simp [consume])
theorem chain_preserves_bytes (source : List UInt8) (history : List Claims)
    (h : Chain source history) : ∀ c ∈ history, c.source = source := by
  induction h with
  | empty => simp
  | @append history c _ hs hc ih =>
      intro e he
      rcases List.mem_cons.mp he with rfl | hm
      · exact (byte_preservation e hc).trans hs
      · exact ih e hm
end ByteGate
