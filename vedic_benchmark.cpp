/**
 * VEDIC SUTRAS BENCHMARK & DEMONSTRATION
 * Shows performance and special mathematical properties
 */

#include "vedic_sutras_complete.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

using namespace vedic;
using namespace std::chrono;

// Timing helper
template<typename Func>
double time_execution(Func f, int iterations = 1000) {
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        f();
    }
    auto end = high_resolution_clock::now();
    return duration_cast<nanoseconds>(end - start).count() / static_cast<double>(iterations);
}

int main() {
    std::cout << R"(
╔══════════════════════════════════════════════════════════════════════════════╗
║                    VEDIC SUTRAS - PERFORMANCE & SPECIAL PROPERTIES           ║
╚══════════════════════════════════════════════════════════════════════════════╝
)" << std::endl;

    // =========================================================================
    // BENCHMARK 1: Nikhilam vs Standard Multiplication
    // =========================================================================
    std::cout << "═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "BENCHMARK 1: S2 Nikhilam (Near-Base Multiplication)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════\n\n";

    std::cout << "WHAT'S SPECIAL: Numbers near powers of 10 multiply faster!\n";
    std::cout << "Instead of digit-by-digit, use deficiencies from base.\n\n";

    // Test cases near 100
    std::vector<std::pair<int, int>> test_cases = {
        {98, 97}, {99, 98}, {97, 96}, {103, 107}, {998, 997}
    };

    for (auto& [a, b] : test_cases) {
        BigInt ba(a), bb(b);
        size_t digits = std::max(util::digit_count(ba), util::digit_count(bb));
        BigInt base = util::pow10(digits);

        auto result = S2_Nikhilam::multiply(ba, bb, base);

        std::cout << a << " × " << b << " (base " << base << "):\n";
        std::cout << "   Deficiencies: " << result.deficiency_a << ", " << result.deficiency_b << "\n";
        std::cout << "   Formula: (" << a << " - " << result.deficiency_b << ") × " << base
                  << " + " << result.deficiency_a << " × " << result.deficiency_b << "\n";
        std::cout << "   = " << result.cross_term << " + " << result.product_term
                  << " = " << result.product << "\n\n";
    }

    // Timing comparison
    BigInt n1(9998), n2(9997);
    // Both arms consume their result identically. This used to guard only the
    // standard arm with `volatile`, leaving the Vedic arm's product unused and
    // free for the optimiser to delete -- an unequal comparison that biased in
    // the Vedic arm's favour. See vedic_benchmark_fair.cpp, which measures all
    // five of the sutras VEDIC_SUTRAS_AUTHENTIC_COMPLETE.md claimed speedups
    // for and reports the ratio these two lines never formed.
    static BigInt sink = 0;
    double nikhilam_time = time_execution([&]() {
        sink += S2_Nikhilam::multiply(n1, n2, BigInt(10000)).product;
    }, 10000);

    double standard_time = time_execution([&]() {
        sink += n1 * n2;
    }, 10000);

    std::cout << "TIMING (9998 × 9997, 10000 iterations):\n";
    std::cout << "   Nikhilam:  " << std::fixed << std::setprecision(1) << nikhilam_time << " ns/op\n";
    std::cout << "   Standard:  " << standard_time << " ns/op\n";
    std::cout << "   Ratio:     " << std::setprecision(2)
              << (standard_time / nikhilam_time) << "x  (>1 means Nikhilam wins)\n\n";

    // =========================================================================
    // BENCHMARK 2: Ūrdhva-Tiryagbhyām (Crosswise)
    // =========================================================================
    std::cout << "═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "BENCHMARK 2: S3 Ūrdhva-Tiryagbhyām (Vertical-Crosswise)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════\n\n";

    std::cout << "WHAT'S SPECIAL: Parallel digit computation - all positions at once!\n";
    std::cout << "Each digit position computed independently, then combined.\n\n";

    auto urdhva = S3_Urdhva::multiply(BigInt(123), BigInt(456));
    std::cout << "123 × 456 = " << urdhva.product << "\n\n";
    std::cout << "Cross products by position:\n";
    std::cout << "   Position 0 (1×4):                    " << urdhva.cross_products[0] << "\n";
    std::cout << "   Position 1 (1×5 + 2×4):              " << urdhva.cross_products[1] << "\n";
    std::cout << "   Position 2 (1×6 + 2×5 + 3×4):        " << urdhva.cross_products[2] << "\n";
    std::cout << "   Position 3 (2×6 + 3×5):              " << urdhva.cross_products[3] << "\n";
    std::cout << "   Position 4 (3×6):                    " << urdhva.cross_products[4] << "\n\n";

    std::cout << "Visual pattern:\n";
    std::cout << "       1  2  3\n";
    std::cout << "     × 4  5  6\n";
    std::cout << "    ─────────────\n";
    std::cout << "       ↓  ╲╱  ↓    (crosswise at each step)\n\n";

    // =========================================================================
    // BENCHMARK 3: Yāvadūnam (Squaring by Deficiency)
    // =========================================================================
    std::cout << "═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "BENCHMARK 3: S10 Yāvadūnam (Squaring Near Base)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════\n\n";

    std::cout << "WHAT'S SPECIAL: Square any number near a base with simple arithmetic!\n";
    std::cout << "Formula: n² = (n ± d) × base + d² where d = |n - base|\n\n";

    std::vector<int> squares = {97, 98, 99, 102, 103, 997};
    for (int n : squares) {
        auto sq = S10_Yavadunam::square(BigInt(n));
        std::cout << n << "² = " << sq.square << "\n";
        std::cout << "   Base: " << sq.base << ", Deficiency: " << sq.deficiency << "\n";
        std::cout << "   = " << sq.left_part << " × " << sq.base << " + " << sq.right_part << "\n\n";
    }

    // Speed of light demonstration
    std::cout << "PHYSICS APPLICATION - Speed of Light Squared:\n";
    auto c_sq = S10_Yavadunam::square(BigInt(299792458), BigInt(300000000));
    std::cout << "c = 299,792,458 m/s\n";
    std::cout << "Base = 300,000,000 (3×10⁸)\n";
    std::cout << "Deficiency = " << c_sq.deficiency << "\n";
    std::cout << "c² = " << c_sq.left_part << " × 300000000 + " << c_sq.right_part << "\n";
    std::cout << "   = " << c_sq.square << " m²/s² (EXACT - zero floating point!)\n\n";

    // =========================================================================
    // BENCHMARK 4: Continued Fractions (Golden Ratio)
    // =========================================================================
    std::cout << "═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "BENCHMARK 4: S13 Sopāntyadvayamantyam (Continued Fractions)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════\n\n";

    std::cout << "WHAT'S SPECIAL: Recursive computation using 'ultimate + twice penultimate'\n";
    std::cout << "p_n = a_n × p_{n-1} + p_{n-2}  (Fibonacci-like recurrence)\n\n";

    std::cout << "Golden Ratio φ = [1; 1, 1, 1, ...] converges via Fibonacci:\n\n";
    std::cout << "n    Convergent      Decimal          Fibonacci\n";
    std::cout << "─────────────────────────────────────────────────\n";

    for (int n = 1; n <= 12; ++n) {
        auto conv = S13_Sopantya::golden_ratio_convergent(n);
        double approx = static_cast<double>(conv.numerator) / static_cast<double>(conv.denominator);
        std::cout << std::setw(2) << n << "   "
                  << std::setw(5) << conv.numerator << "/" << std::setw(4) << std::left << conv.denominator
                  << std::right << "   " << std::fixed << std::setprecision(10) << approx
                  << "   F(" << n+1 << ")/F(" << n << ")\n";
    }
    std::cout << "\nTrue φ = 1.6180339887...\n\n";

    // =========================================================================
    // BENCHMARK 5: Divisibility by Osculation
    // =========================================================================
    std::cout << "═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "BENCHMARK 5: US5 Veṣṭanam (Divisibility by Osculation)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════\n\n";

    std::cout << "WHAT'S SPECIAL: Test divisibility without division!\n";
    std::cout << "Use 'osculators' - magic multipliers that reduce the number.\n\n";

    auto osc7 = US5_Vestanam::find_negative_osculator(BigInt(7));
    auto osc13 = US5_Vestanam::find_negative_osculator(BigInt(13));
    auto osc17 = US5_Vestanam::find_positive_osculator(BigInt(17));

    std::cout << "Osculators found:\n";
    std::cout << "   Divisor 7:  osculator = " << (osc7 ? osc7->str() : "none") << " (negative)\n";
    std::cout << "   Divisor 13: osculator = " << (osc13 ? osc13->str() : "none") << " (negative)\n";
    std::cout << "   Divisor 17: osculator = " << (osc17 ? osc17->str() : "none") << " (positive)\n\n";

    std::cout << "Example: Is 1234567 divisible by 7?\n";
    std::cout << "   Using osculator 2: repeatedly compute (rest - 2×last_digit)\n";
    std::cout << "   1234567 → 123456 - 2×7 = 123442\n";
    std::cout << "   123442  → 12344 - 2×2 = 12340\n";
    std::cout << "   ... continues until small number\n";
    bool div7 = US5_Vestanam::divisibility_by_osculation(BigInt(1234567), BigInt(7), BigInt(2), false);
    std::cout << "   Result: " << (div7 ? "YES" : "NO") << " (1234567 = 7 × 176366 + 5)\n\n";

    // =========================================================================
    // BENCHMARK 6: Special Multiplication Patterns
    // =========================================================================
    std::cout << "═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "BENCHMARK 6: US8 Antyayordaśake'pi (Last Digits Sum to 10)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════\n\n";

    std::cout << "WHAT'S SPECIAL: When last digits sum to 10 and tens are same,\n";
    std::cout << "multiply instantly: n(n+1) for left, last digits for right!\n\n";

    std::vector<std::pair<int, int>> special_pairs = {
        {23, 27}, {34, 36}, {43, 47}, {58, 52}, {61, 69}, {75, 75}
    };

    for (auto& [a, b] : special_pairs) {
        auto result = US8_Antyayor::multiply_sum_to_ten(BigInt(a), BigInt(b));
        std::cout << a << " × " << b << " = " << result.product;
        if (result.applicable) {
            std::cout << "  ← INSTANT! (" << result.common_part << "×" << (result.common_part + 1)
                      << "=" << result.common_part * (result.common_part + 1)
                      << ", " << result.digit_a << "×" << result.digit_b << "="
                      << result.digit_a * result.digit_b << ")";
        }
        std::cout << "\n";
    }

    // =========================================================================
    // BENCHMARK 7: Exact Quadratic Solutions
    // =========================================================================
    std::cout << "\n═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "BENCHMARK 7: S8 Pūraṇāpūraṇābhyām (Complete the Square)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════\n\n";

    std::cout << "WHAT'S SPECIAL: Exact rational roots - no floating point errors!\n\n";

    std::vector<std::tuple<int, int, int>> quadratics = {
        {1, -5, 6},    // x² - 5x + 6 = 0
        {1, -7, 12},   // x² - 7x + 12 = 0
        {2, -7, 3},    // 2x² - 7x + 3 = 0
        {6, -7, 2},    // 6x² - 7x + 2 = 0
    };

    for (auto& [a, b, c] : quadratics) {
        auto roots = S8_Purana::solve_quadratic(Rational(a), Rational(b), Rational(c));
        std::cout << a << "x² + (" << b << ")x + " << c << " = 0\n";
        if (roots.exact && roots.root1 && roots.root2) {
            std::cout << "   Roots: x = " << *roots.root1 << ", x = " << *roots.root2 << " (EXACT)\n";
        }
        std::cout << "   Discriminant: " << roots.discriminant << "\n\n";
    }

    // =========================================================================
    // BENCHMARK 8: Multiply by 9s
    // =========================================================================
    std::cout << "═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "BENCHMARK 8: S14 Ekanyūnena (Multiply by 9, 99, 999...)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════\n\n";

    std::cout << "WHAT'S SPECIAL: n × 999...9 = n × 10^k - n (just shift and subtract!)\n\n";

    std::vector<std::pair<int, int>> nines_tests = {
        {12, 2}, {123, 3}, {1234, 4}, {7, 3}
    };

    for (auto& [n, nines] : nines_tests) {
        auto result = S14_Ekanyunena::multiply_by_nines(BigInt(n), nines);
        std::string nines_str(nines, '9');
        std::cout << n << " × " << nines_str << " = " << result.product << "\n";
        std::cout << "   = " << n << " × 10^" << nines << " - " << n << "\n";
        std::cout << "   = " << result.left_part << " - " << result.right_part << "\n\n";
    }

    // =========================================================================
    // SUMMARY
    // =========================================================================
    std::cout << R"(
═══════════════════════════════════════════════════════════════════════════════
                              SUMMARY: WHY VEDIC MATH?
═══════════════════════════════════════════════════════════════════════════════

✓ EXACT ARITHMETIC: Arbitrary-precision rationals, zero floating-point error
✓ PATTERN RECOGNITION: Special cases computed instantly
✓ PARALLEL COMPUTATION: Cross-products computed independently
✓ MENTAL MATH: Designed for human calculation, not just computers
✓ MATHEMATICAL ELEGANCE: Deep connections (Fibonacci, continued fractions)
✓ VERIFICATION BUILT-IN: Multiple methods cross-check results

The 29 Vedic Sutras provide a complete mathematical toolkit that combines
computational efficiency with mathematical beauty.
═══════════════════════════════════════════════════════════════════════════════
)";

    return 0;
}
