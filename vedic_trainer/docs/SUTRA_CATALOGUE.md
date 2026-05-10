# Sutra catalogue

The 29 sutras over **Z₂⁴** (16 vertices, 4-bit indices). Each entry
states the formula, the corresponding Python function, and any
interpretation choices made where the briefing spec was truncated. The
authoritative implementation is `vedic/kernel/sutras_exact.py` — when
this catalogue and the spec disagree, the simulator is the authority.

The complement of `v` is `v̄ = v ⊕ 0b1111`.

## Main sutras (S1 .. S16)

| ID  | Name                          | Formula                                                   | Function                            |
| --- | ----------------------------- | --------------------------------------------------------- | ----------------------------------- |
| 1   | EkAdhikena                    | `Ψ_{v ⊕ 0001}`                                            | `s1_eka_adhikena`                   |
| 2   | NikhilamComplement            | `Ψ_{v̄}`                                                   | `s2_nikhilam`                       |
| 3   | UrdhvaTiryak                  | `Σ_{a⊕b=v} Ψ_a · Φ_b` (XOR convolution)                   | `s3_urdhva_tiryak`                  |
| 4   | ParavartyaYojayet             | `Ψ_v − Ψ_{v ⊕ 0001}`                                      | `s4_paravartya`                     |
| 5   | ShunyamSamya                  | `Ψ_v − (1/16) Σ_u Ψ_u`                                    | `s5_shunyam_samya`                  |
| 6   | AnurupyaShunyam               | `Ψ_v − Ψ_0 · δ_{v,0}`                                     | `s6_anurupya_shunyam`               |
| 7   | SankalanaVyavakalana          | `(S(Ψ), A(Ψ))` symmetric / antisymmetric pair             | `s7_sankalana_vyavakalana`          |
| 8   | PuranapuranabhyamFill         | `Ψ_v + Ψ_{v̄}` if `v < v̄` else 0                          | `s8_puranapuranabhyam_fill`         |
| 9   | ChalanaKalanabhyam            | `Σ_{k=0..3} (Ψ_{v ⊕ (1<<k)} − Ψ_v)` (4-cube Laplacian)    | `s9_chalana_kalanabhyam`            |
| 10  | YavadunamTavadunikrtya        | `(Ψ_v − 1)²`                                              | `s10_yavadunam_tavadunikrtya`       |
| 11  | VyastiSamasti                 | `Ψ_v − (1/4)·shell_mean(v)`                               | `s11_vyasti_samasti`                |
| 12  | ShesanyankenaCharamena        | `Ψ_v if (v & 0b1000) else 0`                              | `s12_shesanyankena_charamena`       |
| 13  | SopantyadvayamantyamLast2     | `Ψ_v if (v & 0b1100)==0b1100 else 0`                      | `s13_sopantyadvayamantyam_last2`    |
| 14  | EkanyunenaPurvena             | `Ψ_{(v − 1) mod 16}`                                      | `s14_ekanyunena_purvena`            |
| 15  | GunitasamucchayaProduct       | `2^popcount(v) · Ψ_v`                                     | `s15_gunitasamucchaya_product`      |
| 16  | GunakaSamucchaya              | `Ψ_v / 2^popcount(v)`                                     | `s16_gunaka_samucchaya`             |

## Sub-sutras (S17 .. S29)

| ID  | Name                          | Formula                                                   | Function                            |
| --- | ----------------------------- | --------------------------------------------------------- | ----------------------------------- |
| 17  | AnurupyenaProportion          | `Ψ · Φ_0 / Ψ_0`  (precondition: `Ψ_0 ≠ 0`)                | `s17_anurupyena_proportion`         |
| 18  | AdyamadyenaAntyamantyena      | `Ψ_0 · Ψ_15` (scalar)                                     | `s18_adyamadyena_antyamantyena`     |
| 19  | LopanaSthapanabhyam           | `Ψ_v − Ψ_{v & 0b1110} + Ψ_{v | 0b0001}`                   | `s19_lopana_sthapanabhyam`          |
| 20  | VilokanamSpect                | `⟨Ψ, h₁⟩ · h₁ / 16`  with `h₁[v] = (−1)^{v & 1}`          | `s20_vilokanam_spect`               |
| 21  | DhvajankaFlag                 | `|Ψ_v|`                                                   | `s21_dhvajanka_flag`                |
| 22  | ParityComplement              | `Ψ_{v_i} − Ψ_{v̄_i}` for the 8 pairs (v, v̄) with v < v̄    | `s22_parity_complement`             |
| 23  | DwandwaYoga                   | `Ψ_v · Φ_{v̄} + Ψ_{v̄} · Φ_v`                              | `s23_dwandwa_yoga`                  |
| 24  | KevalaihSaptakam              | `Ψ_v if (v % 7 != 0) else 0`                              | `s24_kevalaih_saptakam`             |
| 25  | VestanaCircular               | `Ψ_{σ(v)}`  (σ = bit-rotate-left-1 on 4 bits)             | `s25_vestana_circular`              |
| 26  | YavadunamSquare               | `Ψ_v²`                                                    | `s26_yavadunam_square`              |
| 27  | SamuccayaGunitah              | `Π_{popcount(v) even} Ψ_v − Π_{popcount(v) odd} Ψ_v`      | `s27_samuccaya_gunitah`             |
| 28  | LopanaRestore                 | inverse of S19 on im(S19), canonical right-inverse         | `s28_lopana_restore`                |
| 29  | MeanDrive                     | `(Ψ_v + mean(Ψ)) / 2`                                     | `s29_mean_drive`                    |

## Interpretation notes (where the briefing spec was truncated)

- **S6 — `AnurupyaShunyam`**: the spec writes
  `Ψ_v − (⟨Ψ, e₀⟩ / ⟨e₀, e₀⟩) e_{0,v}`. With `e₀` being the standard
  basis vector at vertex 0, ⟨Ψ, e₀⟩ = Ψ_0 and ⟨e₀, e₀⟩ = 1, giving the
  Kronecker subtraction above. The result has `(S6 Ψ)_0 = 0` and
  preserves all other coordinates.
- **S19 — `LopanaSthapanabhyam`**: the spec line was cut at
  "`Ψ_v − Ψ_{v & 0b1110} + Ψ_{v`". The natural completion that yields
  an explicit linear-algebra inverse on its image (required by S28) is
  `Ψ_v − Ψ_{v & 0b1110} + Ψ_{v | 0b0001}`. The (v & 0b1110, v | 0b0001)
  partner pair is the bit-0-clear / bit-0-set sibling on the 4-cube.
  The interaction-matrix test `S28 ∘ S19 = id on im(S19)` verifies the
  choice.
- **S21 — `DhvajankaFlag`**: the spec line ends "`(i.e. ` "; we resolve
  to `|Ψ_v|`. Idempotency `S21 ∘ S21 = S21` and non-negativity are
  verified by the interaction-matrix tests.

## 30 algebraic identities

`vedic/kernel/interaction_matrix.py` exposes 30 closed-form identities
that the operators must satisfy (S1∘S1 = id, S2∘S2 = id, S15∘S16 = id,
S28∘S19 = id on the canonical pre-image, S2 acts as ±1 on S/A
sub-spaces, etc.). The interaction-matrix test runs them on 50
randomized (Ψ, Φ) pairs.

## Conservation residuals

| Residual | Meaning                                                        |
| -------- | -------------------------------------------------------------- |
| R1       | trace_sum mod 435  (T(29) closure)                             |
| R2       | Σ_{v<v̄}(Ψ_v + Ψ_{v̄}) − Σ_v Ψ_v                                |
| R3       | mean(S29 Ψ) − mean(Ψ)                                          |
| R4       | ⟨S(Ψ), A(Ψ)⟩                                                   |

R2/R3/R4 are algebraic identities and evaluate to exact zero in ℚ. R1
closes whenever the trace counter is a positive integer multiple of
T(29) = 435.
