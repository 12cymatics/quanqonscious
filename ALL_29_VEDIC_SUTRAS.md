# All 29 Vedic Sutras

The 16 sutras and 13 sub-sutras of Bharati Krishna Tirthaji's *Vedic
Mathematics*, in the book's order. Each row names the method the book applies
the sutra to, and where that method is implemented.

The full statement of every method, with its book references and worked
examples, is in `docs/TRUE_SUTRA_REGISTER.md`.

## Where they are implemented

- `vedic_book.py`: the registry. It records each sutra's id, name, primary
  application and printed pages (1992 revised edition), and
  `vedic_book.evaluate(sid, ...)` runs the primary application in exact
  `sympy` rationals.
- `true_sutras.py`: every book application of every sutra (digit-level
  multiplication and division procedures, recurring decimals, the six meanings
  of Śūnyam Sāmyasamuccaye, osculation, and so on), in exact integers and
  `Fraction`s.
- `core/operators/sutra_ops.py`: the 29 field operators. Each applies its
  sutra's book output as exact rational edge rotations on the lattice (see
  `vedic_field.py`).
- `vedic_sutras_complete.hpp`: the C++ implementation, in the same order.

## The 29

| Id | Sutra | Meaning | Primary application (`vedic_book.py`) | Further applications (`true_sutras.py`) |
|----|-------|---------|----------------------------------------|------------------------------------------|
| 1 | Ekādhikena Pūrveṇa | By one more than the previous one | `ekadhikena`: repetend of 1/D, D ending in 9 | `s1_recurring_decimal_division`, `s1_recurring_decimal_multiplication`, `s1_square_ending_in_5` |
| 2 | Nikhilam Navataścaramaṃ Daśataḥ | All from 9 and the last from 10 | `nikhilam`: product about a base | `s2_complement`, `s2_multiply`, `s2_divide` |
| 3 | Ūrdhva-Tiryagbhyām | Vertically and crosswise | `urdhva`: crosswise coefficients | `s3_multiply`, `s3_polynomial_multiply`, `s3_square_by_duplex` |
| 4 | Parāvartya Yojayet | Transpose and apply | `paravartya`: polynomial division | `s4_divide`, `s4_polynomial_divide`, `s4_solve_linear`, `s4_solve_simultaneous`, `s4_cyclic_fraction_root` |
| 5 | Śūnyaṃ Sāmyasamuccaye | When the Samuccaya is the same, it is zero | `shunyam`: equal products of independent terms | `s5_common_factor`, `s5_independent_terms`, `s5_equal_numerators`, `s5_fraction_equation_roots`, `s5_reciprocal_sums`, `s5_cubic_binomials` |
| 6 | (Ānurūpye) Śūnyamanyat | If one is in ratio, the other is zero | `anurupye`: proportional right-hand side | `s6_solve`, `s6_solve_system` |
| 7 | Saṅkalana-Vyavakalanābhyām | By addition and by subtraction | `sankalana`: interchanged coefficients | `s7_solve` |
| 8 | Pūraṇāpūraṇābhyām | By completion or non-completion | `purana`: completing the square | `s8_complete_square`, `s8_complete_cube`, `s8_solve_cubic` |
| 9 | Calana-Kalanābhyām | By sequential motion (differential) | `calana`: polynomial derivative | `s9_differential` |
| 10 | Yāvadūnam | Whatever the extent of its deficiency | `yavadunam`: square near a base | `s10_square`, `s10_cube` |
| 11 | Vyaṣṭisamaṣṭiḥ | Part and whole | `vyashti`: biquadratic by the average binomial | `s11_sum_of_fourth_powers` |
| 12 | Śeṣāṇyaṅkena Caramena | The remainders by the last digit | `sesanyankena`: digits of 1/D | `s12_recurring_decimal` |
| 13 | Sopāntyadvayamantyam | The ultimate and twice the penultimate | `sopantya`: 2P + L = 0 | `s13_solve`, `s13_solve_binomials` |
| 14 | Ekanyūnena Pūrveṇa | By one less than the previous one | `ekanyunena`: multiplication by nines | `s14_multiply_by_nines` |
| 15 | Guṇitasamuccayaḥ | The product of the sum is the sum of the product | `gunitasamuccaya`: coefficient sums | `s15_product_of_sums`, `s15_check` |
| 16 | Guṇakasamuccayaḥ | The factors of the sum is the sum of the factors | `gunakasamuccaya`: derivative of a product of factors | `s16_differentials` |
| 17 | Ānurūpyeṇa | Proportionately | `anurupyena`: working base | `us1_multiply`, `us1_split_middle` |
| 18 | Śiṣyate Śeṣasaṃjñaḥ | The remainder remains constant | `shishyate`: successive division, constant remainder | `us2_constant_remainder` |
| 19 | Ādyamādyenāntyamantyena | The first by the first and the last by the last | `adyam`: second factor of a quadratic | `us3_factor_quadratic` |
| 20 | Kevalaiḥ Saptakaṃ Guṇyāt | For seven the multiplicand is 143 | `kevalaih`: repetend of n/7 | `us4_sevenths` |
| 21 | Veṣṭanam | By osculation | `vestanam`: divisibility by osculation | `us5_positive_osculator`, `us5_negative_osculator`, `us5_divisible` |
| 22 | Yāvadūnaṃ Tāvadūnam | Lessen by the deficiency | `tavadunam_cube`: cube near a base | `us6_cube` |
| 23 | Yāvadūnaṃ Tāvadūnīkṛtya Vargañca Yojayet | Lessen by the deficiency and set up its square | `extended_square`: square near a base | `us7_square` |
| 24 | Antyayordaśake'pi | Also where the last digits total ten | `antyayordasakepi`: last parts totalling a power of ten | `us8_multiply` |
| 25 | Antyayoreva | Only the last terms | `antyayoreva`: ratio equations | `us9_last_terms`, `us9_series_sum` |
| 26 | Samuccayaguṇitaḥ | The sum of the products | `samuccayagunitah`: coefficient sum of a product | `us10_sum_of_product` |
| 27 | Lopanasthāpanābhyām | By alternate elimination and retention | `lopana`: homogeneous quadratic factors | `us11_factor_homogeneous` |
| 28 | Vilokanam | By mere observation | `vilokanam`: reciprocal pairs | `us12_reciprocal` |
| 29 | Guṇitasamuccayaḥ Samuccayaguṇitaḥ | The product of the sum is the sum of the products | `gunitasamuccaya_samuccayagunitah`: both sides of the identity | `us13_verify` |

## Earlier versions of this document

This file used to describe 29 lattice smoothing and blending updates, each
filed under a sutra's name. Those formulas were not the sutras' methods: none
could produce its own worked example, and the list departed from the book's
order and contents. It placed Ānurūpyeṇa at 6, left out Śiṣyate Śeṣasaṃjñaḥ and
Yāvadūnaṃ Tāvadūnam, and added two entries the book does not have. The code it
described has been replaced; the git history keeps the old text.
