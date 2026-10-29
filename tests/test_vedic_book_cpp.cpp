#include "../vedic_sutras_complete.hpp"
#include <cassert>

int main() {
    using namespace vedic;
    auto digits = S1_Ekadhikena::divide_by_nine_ender(BigInt(1019)).recurring;
    assert(digits.size() == 1018);
    BigInt numerator = 0;
    for (int digit : digits) numerator = 10 * numerator + digit;
    assert(numerator * 1019 == util::pow10(digits.size()) - 1);
    auto general = S1_Ekadhikena::divide_recurring(BigInt(1), BigInt(1019));
    assert(general.recurring == digits);
    for (bool transpose : {false, true}) {
        bool rejected = false;
        try {
            if (transpose) S1_Ekadhikena::divide_by_nine_ender(BigInt(19), 17);
            else S1_Ekadhikena::divide_recurring(BigInt(1), BigInt(19), 17);
        } catch (const std::length_error&) {
            rejected = true;
        }
        assert(rejected);
    }
    assert(S1_Ekadhikena::divide_by_nine_ender(BigInt(19), 18).recurring.size() == 18);
    assert(S1_Ekadhikena::divide_recurring(BigInt(1), BigInt(19), 18).recurring.size() == 18);
    assert(S13_Sopantya::test());
    assert(S12_Sesanyankena::test());
    for (int d : {3, 7, 11, 13, 17, 19, 27, 101, 1009}) {
        auto computed = S12_Sesanyankena::remainder_digits(BigInt(d));
        auto divided = S1_Ekadhikena::divide_recurring(BigInt(1), BigInt(d));
        assert(computed == divided.recurring);
    }
    assert(S16_Gunakasamuccaya::test());
    assert((S16_Gunakasamuccaya::differentiate_product({{Rational(2)}, {Rational(3)}})
            == std::vector<Rational>{Rational(0)}));
    for (int n = 1; n < 7; ++n) {
        auto computed = US4_Kevalaih::sevenths_repetend(n);
        assert(Rational(util::from_digits(computed), BigInt(999999)) == Rational(n, 7));
    }
    const auto both = S13_Sopantya::solve_arithmetic_progression(
        {Rational(1), Rational(2)}, {Rational(1), Rational(1)});
    assert((both.roots == std::vector<Rational>{Rational(-1), Rational(-10, 13)}));
    for (const auto& root : both.roots) {
        Rational a = 1 + 2 * root;
        Rational d = 1 + root;
        Rational A = a, B = a + d, C = a + 2 * d, D = a + 3 * d;
        assert(Rational(1) / (A * B) + Rational(1) / (A * C)
               == Rational(1) / (A * D) + Rational(1) / (B * C));
    }
    auto identity = S13_Sopantya::solve_arithmetic_progression(
        {Rational(1), Rational(1)}, {Rational(0), Rational(0)});
    assert(identity.identity);
    assert(identity.excluded == std::vector<Rational>{Rational(-1)});
    auto pole = S13_Sopantya::solve_arithmetic_progression(
        {Rational(0), Rational(1)}, {Rational(0), Rational(1)});
    assert(!pole.identity && pole.roots.empty());
    auto cf = ContinuedFractions::nth_convergent({BigInt(3), BigInt(7), BigInt(15), BigInt(1)}, 3);
    assert(cf.numerator == 355 && cf.denominator == 113);
    bool rejected = false;
    try {
        ContinuedFractions::nth_convergent({}, 0);
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    assert(rejected);
}
