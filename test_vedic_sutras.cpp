/**
 * VEDIC SUTRAS TEST RUNNER
 * Verifies all 29 sutras (16 main + 13 sub-sutras)
 *
 * Compile: g++ -std=c++17 -O3 -I/path/to/boost test_vedic_sutras.cpp -o test_vedic
 * Run: ./test_vedic
 */

#include "vedic_sutras_complete.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           VEDIC SUTRAS COMPLETE TEST SUITE                        ║\n";
    std::cout << "║           All 29 Sutras (16 Main + 13 Sub-Sutras)                 ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n\n";

    auto results = vedic::tests::run_all_tests();

    int passed = 0, failed = 0;

    std::cout << "MAIN SUTRAS (S1-S16):\n";
    std::cout << "─────────────────────────────────────────────────────────────────────\n";

    for (size_t i = 0; i < 16 && i < results.size(); ++i) {
        std::cout << std::setw(35) << std::left << results[i].name << " ";
        if (results[i].passed) {
            std::cout << "✓ PASS\n";
            passed++;
        } else {
            std::cout << "✗ FAIL\n";
            failed++;
        }
    }

    std::cout << "\nSUB-SUTRAS (US1-US13):\n";
    std::cout << "─────────────────────────────────────────────────────────────────────\n";

    for (size_t i = 16; i < results.size(); ++i) {
        std::cout << std::setw(35) << std::left << results[i].name << " ";
        if (results[i].passed) {
            std::cout << "✓ PASS\n";
            passed++;
        } else {
            std::cout << "✗ FAIL\n";
            failed++;
        }
    }

    std::cout << "\n═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "SUMMARY: " << passed << "/" << (passed + failed) << " tests passed\n";

    if (failed == 0) {
        std::cout << "STATUS: ALL TESTS PASSED ✓\n";
    } else {
        std::cout << "STATUS: " << failed << " TESTS FAILED ✗\n";
    }
    std::cout << "═══════════════════════════════════════════════════════════════════════\n";

    // Additional verification examples
    std::cout << "\n╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    VERIFICATION EXAMPLES                          ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n\n";

    // S2: Nikhilam
    std::cout << "S2 Nikhilam: 98 × 97 = ";
    auto nikhilam_result = vedic::S2_Nikhilam::multiply(
        vedic::BigInt(98), vedic::BigInt(97), vedic::BigInt(100)
    );
    std::cout << nikhilam_result.product << "\n";
    std::cout << "   Deficiencies: " << nikhilam_result.deficiency_a
              << ", " << nikhilam_result.deficiency_b << "\n";
    std::cout << "   Cross term: " << nikhilam_result.cross_term << "\n";
    std::cout << "   Product term: " << nikhilam_result.product_term << "\n\n";

    // S10: Yavadunam (c²)
    std::cout << "S10 Yāvadūnam: c² = 299792458² = ";
    auto c_squared = vedic::S10_Yavadunam::c_squared();
    std::cout << c_squared << "\n";
    std::cout << "   Expected: 89875517873681764\n";
    std::cout << "   Match: " << (c_squared == vedic::BigInt("89875517873681764") ? "YES ✓" : "NO ✗") << "\n\n";

    // S3: Urdhva-Tiryagbhyam
    std::cout << "S3 Ūrdhva-Tiryagbhyām: 123 × 456 = ";
    auto urdhva_result = vedic::S3_Urdhva::multiply(vedic::BigInt(123), vedic::BigInt(456));
    std::cout << urdhva_result.product << "\n";
    std::cout << "   Cross products: ";
    for (const auto& cp : urdhva_result.cross_products) {
        std::cout << cp << " ";
    }
    std::cout << "\n\n";

    // S8: Purana (Completing the square)
    std::cout << "S8 Pūraṇāpūraṇābhyām: x² - 5x + 6 = 0\n";
    auto quad_roots = vedic::S8_Purana::solve_quadratic(
        vedic::Rational(1), vedic::Rational(-5), vedic::Rational(6)
    );
    if (quad_roots.exact && quad_roots.root1 && quad_roots.root2) {
        std::cout << "   Roots: x = " << *quad_roots.root1
                  << ", x = " << *quad_roots.root2 << "\n";
    }
    std::cout << "   Discriminant: " << quad_roots.discriminant << "\n\n";

    // S13: Continued fractions (Golden ratio)
    std::cout << "S13 Sopāntyadvayamantyam: Golden ratio convergent (n=10)\n";
    auto golden = vedic::S13_Sopantya::golden_ratio_convergent(10);
    std::cout << "   F(12)/F(11) = " << golden.numerator << "/" << golden.denominator << "\n";
    std::cout << "   ≈ " << static_cast<double>(golden.numerator) / static_cast<double>(golden.denominator) << "\n\n";

    // US8: Antyayordashake'pi
    std::cout << "US8 Antyayordaśake'pi: 43 × 47 = ";
    auto antyayor_result = vedic::US8_Antyayor::multiply_sum_to_ten(
        vedic::BigInt(43), vedic::BigInt(47)
    );
    std::cout << antyayor_result.product << "\n";
    std::cout << "   Method applicable: " << (antyayor_result.applicable ? "YES" : "NO") << "\n";
    std::cout << "   Common part: " << antyayor_result.common_part << "\n";
    std::cout << "   Last digits: " << antyayor_result.digit_a << " + "
              << antyayor_result.digit_b << " = 10\n\n";

    // S7: Gaussian elimination
    std::cout << "S7 Saṅkalana-Vyavakalanābhyām: Solve 2x + 3y = 8, x - y = 1\n";
    std::vector<std::vector<vedic::Rational>> A = {
        {vedic::Rational(2), vedic::Rational(3)},
        {vedic::Rational(1), vedic::Rational(-1)}
    };
    std::vector<vedic::Rational> b = {vedic::Rational(8), vedic::Rational(1)};
    auto gauss_result = vedic::S7_Sankalana::gaussian_eliminate(A, b);
    if (gauss_result) {
        std::cout << "   x = " << (*gauss_result)[0] << "\n";
        std::cout << "   y = " << (*gauss_result)[1] << "\n";
    }

    std::cout << "\n═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "                    VERIFICATION COMPLETE                              \n";
    std::cout << "═══════════════════════════════════════════════════════════════════════\n";

    return failed > 0 ? 1 : 0;
}
