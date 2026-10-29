# True Sutra Register — the 16 Sutras and 13 Sub-Sutras

This register states, for each of the 29 Vedic sutras, the method that the
source text actually applies it to. It is the reference for restoring sutra
formulas wherever a function carries a sutra's name but computes something
else.

## Source

- **Primary text:** Jagadguru Swami Sri Bharati Krsna Tirthaji Maharaja,
  *Vedic Mathematics, or Sixteen Simple Mathematical Formulae from the Vedas*,
  ed. V. S. Agrawala, Motilal Banarsidass, 1965.
- **Copies used:**
  - archive.org item `VedicMathematicsJagatguru`, OCR text file
    "vedic mathematics jagatguru_djvu.txt" (503,411 bytes, SHA-256
    `9692415c2c7811055157fcd9f725929eb2cba62e1918e0c63e7ed3b453619ca7`),
    downloaded 2026-10-28. "OCR line N" below means line N of that file.
  - archive.org item `vedic-mathematics-bharati-krishna-tirth-ji-maharaj` (1992
    revised edition), OCR text file "Vedic Mathematics - Bharati Krishna Tirth
    Ji Maharaj_djvu.txt" (577,246 bytes, SHA-256
    `cab2faf7361a1527e590b02ae8e218a150dc3434698cb8a8d6644dff8ed2bd1d`),
    downloaded 2026-10-29. "p. N" below means printed page N of this edition.
    These are the page numbers `vedic_book.py` records.
- Both OCR texts are noisy. Every formula below was checked against the book's
  own worked example, and each worked example was recomputed.
- **List of 29:** the book's table "Sixteen Sutras and their Corollaries"
  (OCR lines 403-530). The editor notes that the list "has been compiled from
  stray references in the text".

## Numbering

`S1`–`S16` are the sutras and `US1`–`US13` the sub-sutras, in the book's
order. The C++ header `vedic_sutras_complete.hpp` uses this same order in
its namespaces. `ALL_29_VEDIC_SUTRAS.md` and `core/operators/sutra_ops.py`
do not; see [Departures from the book's list](#departures-from-the-books-list).

## Evidence label

Every entry is labelled **book**: the book states the method and works an
example. An earlier version of this register marked US2 and US6 as named in the
list without a worked method. The 1992 edition works both, as recorded below.

## Implementations

- `vedic_book.py` holds each sutra's primary application.
- `true_sutras.py` holds every application listed here.
- `tests/test_vedic_book_cross.py` checks that the two agree wherever they
  implement the same application.

---

## Sutras

### S1 Ekādhikena Pūrveṇa — "By one more than the previous one" (book)

1. **Recurring decimal of n/D, D = 10a + 9** (Ch. 1, Ch. 26; OCR 17230-17240).
   The ekādhika is E = a + 1.
   - *Division method, left to right:* start with dividend n. At each step
     q = dividend div E and r = dividend mod E. Emit q. The next dividend is
     10·r + q. Stop when the dividend returns to n.
   - *Multiplication method, right to left:* the last digit of the repetend of
     n/D is n (for 0 < n < 10). Repeatedly multiply by E, adding the carry.
   - 1/19 = 0.(052631578947368421).
2. **Square of a number ending in 5** (second corollary of Nikhilam, Ch. 3;
   OCR 4694-4745): (10a + 5)² = a(a + 1) | 25. Examples: 15² = 2|25 and
   195² = 380|25.
3. **Osculator.** The ekādhika of a divisor ending in 9 is its positive
   osculator (Ch. 29). This is used by US5.

### S2 Nikhilam Navataścaramaṃ Daśataḥ — "All from 9 and the last from 10" (book)

1. **Complement:** subtract every digit from 9 and the last non-zero digit
   from 10; trailing zeros stay zero. This gives 10ᵏ − N.
2. **Multiplication near a base B = 10ᵏ** (Ch. 3). Let d_a = a − B and
   d_b = b − B (signed). Then

       a × b = (a + d_b) | d_a·d_b

   where the right portion has k digits and carries or borrows into the left.
   98 × 97 = 95 | 06 = 9506.
3. **Division by a divisor just below 10ᵏ** (Ch. 4). Use the complement
   digits of the divisor. Bring each quotient-column digit down, multiply it by
   the complement digits and add the products to the following columns. Then
   correct the remainder until 0 ≤ r < divisor.

### S3 Ūrdhva-Tiryagbhyām — "Vertically and crosswise" (book)

1. **Multiplication** (Ch. 3). For digit vectors a and b, column
   k = Σ_{i+j=k} a_i·b_j, with carries propagated from the right.
   123 × 456: columns [4, 13, 28, 27, 18] → 56088.
2. **Polynomial multiplication:** the same convolution on coefficients, with
   no carries.
3. **Duplex squaring** (Ch. 33): D(a) = a², D(ab) = 2ab, D(abc) = 2ac + b²,
   D(abcd) = 2ad + 2bc, and so on. This is the vertical-and-crosswise product
   of a number with itself.

### S4 Parāvartya Yojayet — "Transpose and apply" (book)

1. **Division by a divisor just above 10ᵏ** (Ch. 5). Transpose the
   divisor's excess digits over 10ᵏ with their signs changed, then apply them
   column by column to the dividend. Negative digits (vinculum) are normalised
   at the end, and the remainder is corrected so that 0 ≤ r < divisor.
2. **Polynomial division:** the same transposed-coefficient procedure.
3. **Simple equations** (Ch. 11): ax + b = cx + d gives x = (d − b)/(a − c).
4. **Simultaneous equations, cyclic rule** (Ch. 15; OCR 12914-13000). For
   ax + by = p and cx + dy = q:

       x = (b·q − d·p)/(b·c − a·d),  y = (p·c − q·a)/(b·c − a·d)

   For 2x + 3y = 8 and 4x + 5y = 14: x = 1, y = 2.
5. **Fraction sums** (Ch. 16, first type; OCR 13447-13470).
   N₁/((x+a)(x+b)) + N₂/((x+b)(x+c)) + N₃/((x+c)(x+a)) = 0 gives

       x = −(N₁·c + N₂·a + N₃·b)/(N₁ + N₂ + N₃)

   That is, each numerator is multiplied by the factor absent from its
   denominator.

### S5 Śūnyaṃ Sāmyasamuccaye — "When the Samuccaya is the same, that Samuccaya is zero" (book)

Ch. 12, OCR 10624-11356. There are six meanings:

1. A factor common to all terms: 9(x+1) = 7(x+1) gives x + 1 = 0.
2. Product of the independent terms: (x+a)(x+b) = (x+c)(x+d) with ab = cd
   gives x = 0. Example: (x+7)(x+9) = (x+3)(x+21).
3. Sum of the denominators of two fractions with the same numerical numerator:
   m/(ax+b) + m/(cx+d) = 0 gives (a+c)x + (b+d) = 0.
4. N₁/D₁ = N₂/D₂ with N₁ + N₂ = D₁ + D₂, up to a numerical factor: equate
   that sum to 0. Example: (2x+9)/(2x+7) = (2x+7)/(2x+9) gives x = −4.
5. The quadratic case of 4: when also N₁ − D₁ = N₂ − D₂, the difference set to
   0 gives the second root.
6. 1/D₁ + 1/D₂ = 1/D₃ + 1/D₄ with D₁ + D₂ = D₃ + D₄: equate that sum to 0.
   Example: 1/(x−7) + 1/(x−9) = 1/(x−6) + 1/(x−10) gives x = 8. Disguised
   forms are first brought to this shape by transposing, by dividing out
   numerators, or by scaling the x-coefficients.

A further cubic form is (x − 2a)³ + (x − 2b)³ = 2(x − a − b)³, which gives
x = a + b (OCR 11300-11340).

### S6 (Ānurūpye) Śūnyamanyat — "If one is in ratio, the other one is zero" (book)

Ch. 15, OCR 13059-13110. For ax + by = p and cx + dy = q: if b : d = p : q,
then x = 0 and y = p/b. Symmetrically, if a : c = p : q, then y = 0 and
x = p/a. The rule extends to n unknowns: an unknown whose coefficient column is
proportional to the right-hand side takes the value p/coefficient, and every
other unknown is 0.

### S7 Saṅkalana-Vyavakalanābhyām — "By addition and by subtraction" (book)

Ch. 15, OCR 13110-13130. The x- and y-coefficients of the two equations are
interchanged, possibly with a common change of sign: a₁x + b₁y = p₁ and
a₂x + b₂y = p₂ with (a₂, b₂) = ±(b₁, a₁). Adding and subtracting the
equations gives two equations, each a multiple of x + y or of x − y.

For ax + by = p and bx + ay = q this is

    (a + b)(x + y) = p + q,   (a − b)(x − y) = p − q

Example: 45x − 23y = 113 and 23x − 45y = 91 add to 68(x − y) = 204 and
subtract to 22(x + y) = 22. So x − y = 3, x + y = 1, x = 2 and y = −1.

### S8 Pūraṇāpūraṇābhyām — "By the completion or non-completion" (book)

Ch. 18, OCR 14732-14800. Complete the square, cube or fourth power. For
x³ + px² + qx + r = 0, set y = x + p/3; this gives y³ + Py + Q = 0. For
example, x³ + 6x² + 11x + 6 = 0 becomes (x+2)³ − (x+2) = 0, so x + 2 ∈
{0, 1, −1}. For the quadratic case, ax² + bx + c = a(x + b/2a)² − (b² −
4ac)/4a.

### S9 Calana-Kalanābhyām — "By sequential motion" (differential) (book)

Ch. 17, OCR 14120-14135. For ax² + bx + c = 0, the first differential equals
plus or minus the square root of the discriminant:

    D₁ = 2ax + b = ±√(b² − 4ac)

### S10 Yāvadūnam — "Whatever the extent of its deficiency" (book)

1. **Squaring near a base** (Ch. 32 revision of the first corollary): with
   d = n − B, n² = (n + d) | d², where the right portion has k digits.
2. **Cubing** (Ch. 32; OCR 23934-23960): with d = n − B,

       n³ = (n + 2d) | 3d² | d³

   where each right portion has k digits and carries pass leftward.
   104³ = 112 | 48 | 64.

### S11 Vyaṣṭisamaṣṭiḥ — "Part and whole" (book)

Ch. 19, OCR 15138-15190. For (x + m + n)⁴ + (x + m − n)⁴ = p, substitute the
average binomial a = x + m. The odd powers cancel, leaving

    2a⁴ + 12a²n² + 2n⁴ = p

which is a quadratic in a². Example: (x+7)⁴ + (x+5)⁴ = 706 gives a = x + 6,
then a⁴ + 6a² − 352 = 0, so a² = 16 and x = −2 or −10.

### S12 Śeṣāṇyaṅkena Caramena — "The remainders by the last digit" (book)

Ch. 26, OCR 17040-17200. For n/D with gcd(D, 10) = 1, take the remainders
r_k = n·10ᵏ mod D for k = 1, 2, …. Then each quotient digit is

    q_k = (r_k · ℓ) mod 10

where ℓ is the last digit of the answer, fixed by ℓ·D ≡ −1 (mod 10). Examples:
1/7 has remainders 3, 2, 6, 4, 5, 1, and ×7 gives 142857. 1/13 uses ×3 to give
076923.

### S13 Sopāntyadvayamantyam — "The ultimate and twice the penultimate" (book)

Ch. 16, second type, OCR 13525-13560. For

    1/(AB) + 1/(AC) = 1/(AD) + 1/(BC)

with the linear factors A, B, C, D in arithmetic progression, the solution is
2C + D = 0, i.e. 2P + L = 0. Examples: A = x+3 gives 2(x+5) + (x+6) = 0, so
x = −16/3. A = 2x+1 with step x+1 gives 13x + 10 = 0.

### S14 Ekanyūnena Pūrveṇa — "By one less than the previous one" (book)

Ch. 3, third corollary, OCR 4814-5000. Multiplication by 10ᵏ − 1 (all nines):

1. N has k digits: N × (10ᵏ − 1) = (N − 1) | (10ᵏ − N). Example:
   777 × 999 = 776|223.
2. N has fewer digits: pad N with zeros to k digits and apply case 1.
3. N has more digits: write N = E·10ᵏ + R. Then the left portion is
   N − (E + 1) and the right portion is 10ᵏ − R (Nikhilam on R), with carry.
   Example: 37 × 9 = 33|3.

### S15 Guṇitasamuccayaḥ — "The product of the sum is equal to the sum of the product" (book)

Ch. 7 end, OCR 9614-9640: Sc(product) = Π Sc(factors), where Sc is the sum
of the coefficients. Example: (x+1)(x+2)(x+3) = x³ + 6x² + 11x + 6, and
2·3·4 = 1 + 6 + 11 + 6 = 24. The book uses it to verify multiplications,
divisions and factorisations, and to fill in an unknown coefficient.

### S16 Guṇakasamuccayaḥ — "The factors of the sum is equal to the sum of the factors" (book)

Ch. 22, OCR 15798-15870. If E = Π(x + a_i), its first differential is the
sum of the products of the factors taken n − 1 at a time. In general, the j-th
differential is j! × Σ over the products of the factors taken n − j at a time.
Example: x² + 3x + 2 = (x+1)(x+2), and D₁ = 2x + 3 = (x+1) + (x+2).

---

## Sub-sutras

### US1 Ānurūpyeṇa — "Proportionately" (book)

1. **Working base as a multiple or sub-multiple of 10ᵏ** (Ch. 3). Multiply by
   Nikhilam about W = f·10ᵏ or 10ᵏ/f, then multiply or divide the left portion
   by f. Examples: 49² with W = 100/2 gives 48/2 | 01 = 2401; 29² with
   W = 10×3 gives 28×3 | 1 = 841.
2. **Factorising ax² + bx + c** (Ch. 7, with US3; OCR 9500-9560). Split
   b = b₁ + b₂ so that a : b₁ = b₂ : c. That ratio, in lowest terms, is the
   first factor.

### US2 Śiṣyate Śeṣasaṃjñaḥ — "The remainder remains constant" (book)

p. 19, Ch. 2, in the explanation of proportionate working bases. The book
divides 65 successively by 2, 4, 8, 16, 32 and 64. The quotients 32, 16, 8, 4,
2 and 1 decrease in proportion as the divisor increases, but the remainder 1
"remains constant. And this is why it is rightly called the remainder". The
same principle is why, under US1, only the left-hand portion is multiplied or
divided by the working-base ratio. The right-hand vertical product is left
unaffected.

The application: dividing N successively by divisors in proportion gives a
constant remainder. It is refused when the remainders differ.

### US3 Ādyamādyenāntyamantyena — "The first by the first and the last by the last" (book)

Ch. 7, OCR 9500-9580. With the first factor (p x + q) found by US1, the second
factor is (a/p) x + (c/q). Example: 2x² + 5x + 2 = (x + 2)(2x + 1).

### US4 Kevalaiḥ Saptakaṃ Guṇyāt — "For seven the multiplicand is 143" (book)

Ch. 26, OCR 17620-17640. Since 7 × 143 = 1001, n/7 = 143n/1001, and Ekanyūnena
(S14) gives 143n × 999 as the six-digit repetend. 1/7 = 0.(142857).

### US5 Veṣṭanam — "By osculation" (book)

Ch. 29-30. For a divisor D ending in 9, the positive osculator is the ekādhika
of D (S1). For other endings, first multiply D up to one ending in 9.
Repeatedly replace N by (N without its last digit) + osculator × (last
digit). D divides N if and only if it divides the result. The negative
osculator of D = 10a + 1 is a, with N → (N without its last digit) − a × (last
digit).

### US6 Yāvadūnaṃ Tāvadūnam — "Lessen by the deficiency" (book)

Cubing near a base. p. xliii (illustrative samples) names the "Yavadunam-
Tavadunam Sutra" for the cube of 9989; pp. 288-289 (Ch. 32) give the
procedure. With deficiency d = B − n:

    n³ = (n − 2d) | 3d² | −d³

where each right portion has k digits and carries or borrows into the next.
9989³ = 9967 | 0363 | −1331 = 9967 | 0362 | 8669 = 996703628669.

### US7 Yāvadūnaṃ Tāvadūnīkṛtya Vargañca Yojayet — "Lessen by the deficiency and set up the square of the deficiency" (book)

Ch. 3, first corollary, OCR 4360-4690: n² = (n − d) | d², with working bases
as in US1. Examples: 97² = 94|09 and 108² = 116|64.

### US8 Antyayordaśake'pi — "Also where the last digits total 10" (book)

Ch. 3, OCR 4756-4790. For numbers with the same previous part a and last parts
b, c with b + c = 10ᵏ:

    product = a(a + 1) | b·c

where the right portion has 2k digits. Examples: 27 × 23 = 6|21 and
884 × 816 = 72|1344 = 721344.

### US9 Antyayoreva — "Only the last terms" (book)

Ch. 16, third to fifth types, OCR 13562-13700.

1. (AB + C)/(AD + E) = B/D gives B/D = C/E. Example: (x²+x+1)/(x²+3x+3) =
   (x+1)/(x+3) gives (x+1)/(x+3) = 1/3.
2. Σ 1/(D_i·D_{i+1}) with the D_i in arithmetic progression equals
   n/(D₁·D_{n+1}): the numerator is the sum of the numerators, and the
   denominator is the product of the first and last binomials.

### US10 Samuccayaguṇitaḥ — "The sum of the products" (book)

Ch. 7-9: Sc of a product, read off as the sum of the coefficients of the
expanded product. It is the side of S15's identity that is computed from the
product.

### US11 Lopanasthāpanābhyām — "By alternate elimination and retention" (book)

Ch. 8, OCR 9648-9760. For a homogeneous quadratic in x, y, z, …:

1. Set z = 0 and factorise the quadratic in x and y with US3.
2. Set y = 0 and factorise the quadratic in x and z.
3. Fill in the gaps: E = (x + 2y + 3z)(2x + 3y + z) for
   2x² + 6y² + 3z² + 7xy + 11yz + 7zx.

### US12 Vilokanam — "By mere observation" (book)

Ch. 17, first special type (reciprocals), OCR 14140-14175. If
x + 1/x = a + 1/a, then x = a or 1/a. Example: x + 1/x = 17/4 = 4 + 1/4 gives
x = 4 or 1/4. The book also uses it for observing roots in simultaneous
quadratics (Ch. 21).

### US13 Guṇitasamuccayaḥ Samuccayaguṇitaḥ — "The product of the sum is the sum of the products" (book)

Ch. 7, OCR 9614-9640: Sc(product) = Π Sc(factors), asserted as an identity
between both sides (S15 × US10) and used to check every multiplication,
division and factorisation.

---

## Departures from the book's list

- `ALL_29_VEDIC_SUTRAS.md` and `core/operators/sutra_ops.py` used to number the
  sutras differently from the book. Both now follow the book's order.
  - They placed Ānurūpyeṇa (book US1) at 6 and Ānurūpye Śūnyamanyat (book S6)
    at 17.
  - They omitted Śiṣyate Śeṣasaṃjñaḥ (US2) and Yāvadūnaṃ Tāvadūnam (US6).
  - They added "Yavadunam Tavadum Vilokanam" and "Dwandwa Yoga", which are not
    in the book's list. Dvandva, the duplex, is the book's squaring method
    under S3.
  - Every one of those 29 formulas was a lattice smoothing or blending update.
    None reproduced its own worked example.
- `vedic_sutras_complete.hpp` follows the book's list. Several of its
  namespaces implement standard algorithms instead of the sutra's method:
  - Floyd cycle detection under US2;
  - linear interpolation under US1;
  - Gaussian elimination under S7;
  - a dot product under US10;
  - last-digit-of-power under US9.
- The supplied corpus file `vedic2.txt` defines its own 16, with glosses that
  differ from the book (for example, "Chalana-Kalanābhyām: By shifting").
