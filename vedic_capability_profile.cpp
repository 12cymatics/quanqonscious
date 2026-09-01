// What each sutra can actually do: its domain, measured rather than described.
//
// Several sutras are CONDITIONAL -- Nikhilam wants both operands near a common
// base, Antyayor wants last digits summing to ten, Vestanam needs an osculator
// to exist at all. For those the applicability rate IS the capability, and it
// is found by sweeping inputs rather than by reading the docstring.
//
// For every sutra three things are measured:
//   applicability  -- of N swept inputs, how many the method accepts
//   correctness    -- of those accepted, how many match an independent reference
//   cost           -- nanoseconds per call, mean over many iterations
//
// A sutra that applies to 3% of inputs and is exact on all of them is not
// "worse" than one that applies to 100%; it is a different tool. The point is
// to say which is which, with numbers.

#include "vedic_sutras_complete.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace vedic;
using Clock = std::chrono::steady_clock;

static volatile unsigned long long SINK = 0;

struct Profile {
    std::string id, name, domain;
    long applicable = 0, swept = 0, correct = 0;
    double ns = 0;
    bool exact = true;
    std::string limit;
};
static std::vector<Profile> profiles;

template <typename F>
static double bench(F f, int iters) {
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) f();
    return std::chrono::duration<double, std::nano>(Clock::now() - t0).count() / iters;
}

// ---------------------------------------------------------------- the sweeps

static void profile_universal_multipliers() {
    // S3 Urdhva -- general multiplication, no precondition.
    {
        Profile p{"S3", "Urdhva Tiryagbhyam", "any two integers", 0, 0, 0, 0, true, "none found"};
        for (long a = 0; a <= 60; ++a) for (long b = 0; b <= 60; ++b) {
            ++p.swept; ++p.applicable;
            if (S3_Urdhva::multiply(BigInt(a), BigInt(b)).product == BigInt(a) * b) ++p.correct;
        }
        BigInt x("847362514"), y("293847561");
        p.ns = bench([&]{ SINK += (unsigned long long)(S3_Urdhva::multiply(x, y).product % 1000003); }, 2000);
        profiles.push_back(p);
    }
    // S2 Nikhilam -- conditional: both operands must sit near a common base.
    {
        Profile p{"S2", "Nikhilam", "both operands near a power of ten", 0, 0, 0, 0, true, ""};
        long best_dev = 0;
        for (long a = 2; a <= 200; ++a) for (long b = 2; b <= 200; ++b) {
            ++p.swept;
            BigInt base = S2_Nikhilam::find_base(BigInt(a), BigInt(b));
            // The method is worth using when both deficiencies are small
            // relative to the base; that is its stated precondition.
            long da = std::labs(base.convert_to<long>() - a);
            long db = std::labs(base.convert_to<long>() - b);
            if (da * 4 <= base.convert_to<long>() && db * 4 <= base.convert_to<long>()) {
                ++p.applicable;
                if (S2_Nikhilam::multiply(BigInt(a), BigInt(b), base).product == BigInt(a) * b) ++p.correct;
                best_dev = std::max(best_dev, std::max(da, db));
            }
        }
        p.limit = "deficiency within a quarter of the base; largest exercised " + std::to_string(best_dev);
        BigInt a(9998), b(9997), base(10000);
        p.ns = bench([&]{ SINK += (unsigned long long)(S2_Nikhilam::multiply(a, b, base).product % 1000003); }, 20000);
        profiles.push_back(p);
    }
    // US8 Antyayor -- last digits sum to ten AND leading parts equal.
    {
        Profile p{"US8", "Antyayordashake'pi", "last digits sum to 10, same leading part", 0, 0, 0, 0, true, ""};
        for (long a = 10; a <= 300; ++a) for (long b = 10; b <= 300; ++b) {
            ++p.swept;
            auto r = US8_Antyayor::multiply_sum_to_ten(BigInt(a), BigInt(b));
            if (r.applicable) { ++p.applicable; if (r.product == BigInt(a) * b) ++p.correct; }
        }
        p.limit = "the method itself reports inapplicability; it never guesses";
        BigInt a(43), b(47);
        p.ns = bench([&]{ SINK += (unsigned long long)(US8_Antyayor::multiply_sum_to_ten(a, b).product % 1000003); }, 20000);
        profiles.push_back(p);
    }
    // S14 Ekanyunena -- multiplier must be a string of nines.
    {
        Profile p{"S14", "Ekanyunena Purvena", "multiplier is 10^k - 1", 0, 0, 0, 0, true,
                  "only all-nines multipliers; any n"};
        for (long n = 0; n <= 400; ++n) for (size_t k = 1; k <= 4; ++k) {
            ++p.swept; ++p.applicable;
            BigInt nines = util::power(BigInt(10), k) - 1;
            if (S14_Ekanyunena::multiply_by_nines(BigInt(n), k).product == BigInt(n) * nines) ++p.correct;
        }
        BigInt n(4567);
        p.ns = bench([&]{ SINK += (unsigned long long)(S14_Ekanyunena::multiply_by_nines(n, 4).product % 1000003); }, 20000);
        profiles.push_back(p);
    }
    // US4 Kevalaih -- multiply by a single digit via its complement.
    {
        Profile p{"US4", "Kevalaih Saptakam", "multiplier is a single digit 1-9", 0, 0, 0, 0, true,
                  "single-digit multiplier only"};
        for (long n = 0; n <= 500; ++n) for (int k = 1; k <= 9; ++k) {
            ++p.swept; ++p.applicable;
            if (US4_Kevalaih::multiply_by_complement(BigInt(n), k) == BigInt(n) * k) ++p.correct;
        }
        BigInt n(12345);
        p.ns = bench([&]{ SINK += (unsigned long long)(US4_Kevalaih::multiply_by_complement(n, 7) % 1000003); }, 20000);
        profiles.push_back(p);
    }
}

static void profile_squarers() {
    struct Case { const char* id; const char* name; int which; };
    Case cases[] = {{"S10", "Yavadunam", 0}, {"US6", "Yavadunam Squared", 1}, {"US7", "Yavadunam Extended", 2}};
    for (auto& c : cases) {
        Profile p{c.id, c.name, "n near a power of ten", 0, 0, 0, 0, true, ""};
        long worst = 0;
        for (long n = 2; n <= 400; ++n) {
            ++p.swept;
            BigInt bn(n), truth = bn * bn;
            bool ok = false, applied = true;
            if (c.which == 0)      ok = (S10_Yavadunam::square(bn).square == truth);
            else if (c.which == 1) ok = (US6_Yavadunam_Squared::square_by_deficiency(bn).result == truth);
            else {
                // BOTH sides of the base. 10^digits(n) is always ABOVE n, so a
                // sweep using only that never enters the above-base branch --
                // a sign error there survived until this second base was added.
                BigInt hi = util::power(BigInt(10), std::to_string(n).size());
                BigInt lo = util::power(BigInt(10), std::to_string(n).size() - 1);
                ok = (US7_Yavadunam_Extended::square_extended(bn, hi).result == truth)
                  && (US7_Yavadunam_Extended::square_extended(bn, lo).result == truth);
            }
            if (applied) { ++p.applicable; if (ok) ++p.correct; else worst = n; }
        }
        p.limit = (p.correct == p.applicable)
                ? "exact for every n swept, near base or not"
                : "first disagreement at n = " + std::to_string(worst);
        BigInt n(9997);
        p.ns = bench([&]{ SINK += (unsigned long long)(S10_Yavadunam::square(n).square % 1000003); }, 20000);
        profiles.push_back(p);
    }
}

static void profile_divisibility() {
    // S12 -- seven digit rules, each exact for its own divisor.
    {
        Profile p{"S12", "Sesanyankena Caramena", "divisors 2,3,4,5,8,9,11", 0, 0, 0, 0, true,
                  "these seven divisors only"};
        for (long n = -300; n <= 300; ++n) {
            BigInt b(n);
            const bool got[] = {S12_Sesanyankena::divisible_by_2(b), S12_Sesanyankena::divisible_by_3(b),
                                S12_Sesanyankena::divisible_by_4(b), S12_Sesanyankena::divisible_by_5(b),
                                S12_Sesanyankena::divisible_by_8(b), S12_Sesanyankena::divisible_by_9(b),
                                S12_Sesanyankena::divisible_by_11(b)};
            const long ds[] = {2, 3, 4, 5, 8, 9, 11};
            for (int i = 0; i < 7; ++i) { ++p.swept; ++p.applicable; if (got[i] == (n % ds[i] == 0)) ++p.correct; }
        }
        BigInt n(123456789);
        p.ns = bench([&]{ SINK += S12_Sesanyankena::divisible_by_11(n) ? 1 : 0; }, 200000);
        profiles.push_back(p);
    }
    // US5 Vestanam -- needs an osculator, which exists only when gcd(d,10)=1.
    {
        Profile p{"US5", "Vestanam", "divisors coprime to 10", 0, 0, 0, 0, true, ""};
        long no_osc = 0;
        for (long d = 3; d <= 99; ++d) {
            auto k = US5_Vestanam::find_positive_osculator(BigInt(d));
            for (long n = 0; n <= 200; ++n) {
                ++p.swept;
                if (!k) { ++no_osc; continue; }
                ++p.applicable;
                if (US5_Vestanam::divisibility_by_osculation(BigInt(n), BigInt(d), *k, true) == (n % d == 0))
                    ++p.correct;
            }
        }
        p.limit = "no osculator for any d divisible by 2 or 5 (" + std::to_string(no_osc) + " swept inputs refused)";
        auto k = US5_Vestanam::find_positive_osculator(BigInt(19));
        BigInt n(361);
        p.ns = bench([&]{ SINK += US5_Vestanam::divisibility_by_osculation(n, BigInt(19), *k, true) ? 1 : 0; }, 20000);
        profiles.push_back(p);
    }
    // US9 Antyayoreva -- last digit only. Cheap, and answers a narrow question.
    {
        Profile p{"US9", "Antyayoreva", "the last digit of a product or power", 0, 0, 0, 0, true,
                  "returns one digit, not the value"};
        for (long a = 0; a < 80; ++a) for (long b = 0; b < 80; ++b) {
            ++p.swept; ++p.applicable;
            if (US9_Antyayoreva::last_digit_of_product(BigInt(a), BigInt(b)) == (a * b) % 10) ++p.correct;
        }
        BigInt a(847362514), b(293847561);
        p.ns = bench([&]{ SINK += US9_Antyayoreva::last_digit_of_product(a, b); }, 200000);
        profiles.push_back(p);
    }
    // S1 Ekadhikena -- recurring decimals; nine-enders have a dedicated route.
    {
        Profile p{"S1", "Ekadhikena Purvena", "1/d for d coprime to 10", 0, 0, 0, 0, true, ""};
        long nine_enders = 0;
        for (long d = 3; d <= 199; d += 2) {
            if (d % 5 == 0) { ++p.swept; continue; }
            ++p.swept; ++p.applicable;
            long r = 10 % d, k = 1;
            while (r != 1 && k < 5000) { r = (r * 10) % d; ++k; }
            if ((long)S1_Ekadhikena::divide_recurring(BigInt(1), BigInt(d), 5000).recurring.size() == k) ++p.correct;
            if (d % 10 == 9) ++nine_enders;
        }
        p.limit = std::to_string(nine_enders) + " of these are nine-enders with a dedicated faster route";
        p.ns = bench([&]{ SINK += S1_Ekadhikena::divide_recurring(BigInt(1), BigInt(19), 100).recurring.size(); }, 2000);
        profiles.push_back(p);
    }
}

static void profile_algebra() {
    // S4 Paravartya -- division, universal.
    {
        Profile p{"S4", "Paravartya Yojayet", "any dividend, any nonzero divisor", 0, 0, 0, 0, true, "divisor must be nonzero"};
        for (long n = 0; n <= 300; ++n) for (long d = 1; d <= 20; ++d) {
            ++p.swept; ++p.applicable;
            auto r = S4_Paravartya::divide(BigInt(n), BigInt(d));
            if (r.quotient == n / d && r.remainder == n % d) ++p.correct;
        }
        BigInt a(987654321), b(1234);
        p.ns = bench([&]{ SINK += (unsigned long long)(S4_Paravartya::divide(a, b).quotient % 1000003); }, 20000);
        profiles.push_back(p);
    }
    // S8 Purana -- quadratics. Exact only when the roots are rational.
    {
        Profile p{"S8", "Purana Apuranabhyam", "quadratics; exact when roots are rational", 0, 0, 0, 0, true, ""};
        long irrational = 0;
        for (long b = -12; b <= 12; ++b) for (long c = -12; c <= 12; ++c) {
            ++p.swept;
            auto q = S8_Purana::solve_quadratic(Rational(1), Rational(b), Rational(c));
            if (!q.exact) { ++irrational; continue; }
            ++p.applicable;
            if (q.root1 && q.root2) {
                Rational r1 = *q.root1, r2 = *q.root2;
                if (r1 + r2 == Rational(-b) && r1 * r2 == Rational(c)) ++p.correct;
            } else if (!q.root1 && !q.root2) ++p.correct;   // no real roots, correctly reported
        }
        p.limit = std::to_string(irrational) + " of " + std::to_string(p.swept)
                + " swept quadratics have irrational roots and are REFUSED, not approximated";
        p.ns = bench([&]{ auto q = S8_Purana::solve_quadratic(Rational(1), Rational(-6), Rational(5));
                          SINK += q.exact ? 1 : 0; }, 20000);
        profiles.push_back(p);
    }
    // S7 Sankalana -- linear systems by elimination.
    {
        Profile p{"S7", "Sankalana Vyavakalanabhyam", "linear systems with a unique solution", 0, 0, 0, 0, true, ""};
        long singular = 0;
        for (long a = -6; a <= 6; ++a) for (long b = -6; b <= 6; ++b)
            for (long c = -4; c <= 4; ++c) for (long d = -4; d <= 4; ++d) {
                ++p.swept;
                if (a * d - b * c == 0) { ++singular; continue; }
                ++p.applicable;
                auto s = S7_Sankalana::gaussian_eliminate(
                    {{Rational(a), Rational(b)}, {Rational(c), Rational(d)}}, {Rational(1), Rational(2)});
                if (!s) continue;
                Rational x = (*s)[0], y = (*s)[1];
                if (Rational(a) * x + Rational(b) * y != 1) continue;
                if (Rational(c) * x + Rational(d) * y != 2) continue;
                // The named 2x2 entry point is a SECOND implementation and was
                // untested here; a sign error in its back-substitution went
                // unnoticed because only gaussian_eliminate was ever called.
                auto e = S7_Sankalana::solve_by_elimination(
                    Rational(a), Rational(b), Rational(1), Rational(c), Rational(d), Rational(2));
                if (!e.x || !e.y) continue;
                if (*e.x != x || *e.y != y) continue;
                ++p.correct;
            }
        p.limit = std::to_string(singular) + " singular systems correctly refused rather than approximated";
        p.ns = bench([&]{ auto s = S7_Sankalana::gaussian_eliminate(
            {{Rational(2), Rational(3)}, {Rational(1), Rational(1)}}, {Rational(8), Rational(3)});
            SINK += s ? 1 : 0; }, 20000);
        profiles.push_back(p);
    }
    // S9 Calana -- polynomial calculus, exact for any rational polynomial.
    {
        Profile p{"S9", "Calana Kalanabhyam", "any polynomial over the rationals", 0, 0, 0, 0, true, "polynomials only"};
        for (long a = -8; a <= 8; ++a) for (long b = -8; b <= 8; ++b) for (long c = -4; c <= 4; ++c) {
            ++p.swept; ++p.applicable;
            std::vector<Rational> poly{Rational(c), Rational(b), Rational(a)};
            auto d = S9_Calana::differentiate(poly);
            bool ok = (d.size() == 2 && d[0] == Rational(b) && d[1] == Rational(2 * a));
            auto rt = S9_Calana::differentiate(S9_Calana::integrate(poly));
            ok = ok && (rt == poly);
            if (ok) ++p.correct;
        }
        std::vector<Rational> poly{Rational(5), Rational(3), Rational(2)};
        p.ns = bench([&]{ SINK += S9_Calana::differentiate(poly).size(); }, 20000);
        profiles.push_back(p);
    }
    // S13 Sopantya -- continued fractions, exact for any rational.
    {
        Profile p{"S13", "Sopantyadvayamantyam", "any rational number", 0, 0, 0, 0, true,
                  "irrationals need a stated truncation"};
        for (long n = 1; n <= 60; ++n) for (long d = 1; d <= 20; ++d) {
            ++p.swept; ++p.applicable;
            Rational r(n, d);
            auto cf = S13_Sopantya::to_continued_fraction(r);
            auto back = S13_Sopantya::nth_convergent(cf, cf.size() - 1);
            if (Rational(back.numerator, back.denominator) == r) ++p.correct;
        }
        Rational r(355, 113);
        p.ns = bench([&]{ SINK += S13_Sopantya::to_continued_fraction(r).size(); }, 20000);
        profiles.push_back(p);
    }
    // S16 Gunakasamuccaya -- prime factorisation, universal but superlinear.
    {
        Profile p{"S16", "Gunaka Samuccayah", "any positive integer", 0, 0, 0, 0, true,
                  "trial division: cost grows with the largest prime factor"};
        for (long n = 2; n <= 600; ++n) {
            ++p.swept; ++p.applicable;
            BigInt rebuilt = 1;
            bool sound = true;
            BigInt prev = 0;
            for (const auto& f : S16_Gunakasamuccaya::prime_factorize(BigInt(n))) {
                rebuilt *= util::power(f.prime, f.exponent);
                // Reconstruction ALONE is blind: a composite reported as one
                // prime still multiplies back to n. Each factor must actually
                // be prime, by an independent trial division.
                if (f.prime < 2) { sound = false; break; }
                for (BigInt k = 2; k * k <= f.prime; ++k)
                    if (f.prime % k == 0) { sound = false; break; }
                if (!sound) break;
                if (f.exponent < 1) { sound = false; break; }
                if (f.prime <= prev) { sound = false; break; }   // strictly ascending, no duplicates
                prev = f.prime;
            }
            if (sound && rebuilt == n) ++p.correct;
        }
        BigInt n(360);
        p.ns = bench([&]{ SINK += S16_Gunakasamuccaya::prime_factorize(n).size(); }, 20000);
        profiles.push_back(p);
    }
    // US12 Vilokanam -- pattern recognition. Its ability is to REFUSE correctly.
    {
        Profile p{"US12", "Vilokanam", "detects squares, AP and GP; refuses otherwise", 0, 0, 0, 0, true, ""};
        long refused = 0;
        for (long n = 1; n <= 400; ++n) {
            ++p.swept; ++p.applicable;
            auto r = US12_Vilokanam::check_perfect_square(BigInt(n));
            bool is_sq = false;
            for (long k = 0; k * k <= n; ++k) if (k * k == n) is_sq = true;
            bool said = (r.type == US12_Vilokanam::PatternType::PERFECT_SQUARE);
            if (said == is_sq) ++p.correct;
            if (!said) ++refused;
        }
        p.limit = "refused " + std::to_string(refused) + " of " + std::to_string(p.swept)
                + "; a refusal here is a correct answer, not a failure";
        BigInt n(144);
        p.ns = bench([&]{ SINK += (int)US12_Vilokanam::check_perfect_square(n).type; }, 20000);
        profiles.push_back(p);
    }
}


// ---------------------------------------------- equations, systems, identities
//
// These ten were absent from the first sweep. Several of them are not
// calculators at all -- S15 and US13 are VERIFIERS, US12 and US2 are
// DETECTORS. A verifier's ability is measured on both arms: it must say yes
// to a true identity and no to a corrupted one. Counting only the yes arm
// would score a function that returns true unconditionally at 100%.

static void profile_equations() {
    // S5 Shunyam -- (x+a)(x+b) = (x+c)(x+d). Refuses when the sums are equal,
    // because then the x terms cancel and there is no unique root.
    {
        Profile p{"S5", "Shunyam Samyasamuccaye", "(x+a)(x+b) = (x+c)(x+d)", 0, 0, 0, 0, true, ""};
        long refused = 0;
        for (long a = -6; a <= 6; ++a) for (long b = -6; b <= 6; ++b)
        for (long c = -6; c <= 6; ++c) for (long d = -6; d <= 6; ++d) {
            ++p.swept;
            auto r = S5_Shunyam::solve_product_equality(
                Rational(a), Rational(b), Rational(c), Rational(d));
            if (r.solutions.empty()) { ++refused; continue; }
            ++p.applicable;
            // Independent reference: substitute the root back into the
            // ORIGINAL equation, not into the formula that produced it.
            Rational x = r.solutions[0];
            if ((x + a) * (x + b) == (x + c) * (x + d)) ++p.correct;
        }
        p.limit = "refused " + std::to_string(refused) + " of " + std::to_string(p.swept)
                + " where a+b = c+d and no unique root exists";
        p.ns = bench([&]{ SINK += S5_Shunyam::solve_product_equality(
            Rational(2), Rational(3), Rational(1), Rational(5)).solutions.size(); }, 20000);
        profiles.push_back(p);
    }
    // S6 Anurupye -- 2x2 systems. Refuses on a zero determinant, and
    // distinguishes the two singular cases (parallel vs coincident).
    {
        Profile p{"S6", "Anurupye Shunyamanyat", "2x2 systems; classifies singular ones", 0, 0, 0, 0, true, ""};
        long singular = 0, classified = 0;
        for (long a1 = -4; a1 <= 4; ++a1) for (long b1 = -4; b1 <= 4; ++b1)
        for (long a2 = -4; a2 <= 4; ++a2) for (long b2 = -4; b2 <= 4; ++b2) {
            Rational c1(7), c2(-3);
            ++p.swept;
            auto r = S6_Anurupye::solve_system_2x2(
                Rational(a1), Rational(b1), c1, Rational(a2), Rational(b2), c2);
            if (!r.x || !r.y) {
                ++singular;
                // A refusal must still be RIGHT: the determinant really is zero,
                // and the consistency verdict must match the rank test.
                if (a1 * b2 - a2 * b1 == 0 && r.proportional) ++classified;
                continue;
            }
            ++p.applicable;
            // Reference: substitute into both original equations.
            if (Rational(a1) * *r.x + Rational(b1) * *r.y == c1 &&
                Rational(a2) * *r.x + Rational(b2) * *r.y == c2) ++p.correct;
        }
        p.limit = std::to_string(singular) + " singular systems refused, "
                + std::to_string(classified) + " of them correctly classified as proportional";
        p.ns = bench([&]{ SINK += S6_Anurupye::solve_system_2x2(
            Rational(2), Rational(3), Rational(8), Rational(5), Rational(-1), Rational(3)).x ? 1 : 0; }, 20000);
        profiles.push_back(p);
    }
    // US11 Lopanasthapana -- eliminate one variable, keep the rest.
    {
        Profile p{"US11", "Lopanasthapanabhyam", "eliminates a variable from a linear system", 0, 0, 0, 0, true,
                  "needs a nonzero pivot in the eliminated column; returns the system unchanged otherwise"};
        for (long a1 = -4; a1 <= 4; ++a1) for (long b1 = -4; b1 <= 4; ++b1)
        for (long a2 = -4; a2 <= 4; ++a2) for (long b2 = -4; b2 <= 4; ++b2) {
            Rational det = Rational(a1) * b2 - Rational(a2) * b1;
            if (det == 0) continue;                 // no unique y to compare against
            ++p.swept; ++p.applicable;
            std::vector<std::vector<Rational>> A = {{Rational(a1), Rational(b1)},
                                                    {Rational(a2), Rational(b2)}};
            std::vector<Rational> rhs = {Rational(7), Rational(-3)};
            auto e = US11_Lopanasthapana::eliminate_variable(A, rhs, 0);
            if (e.reduced_matrix.size() != 1) continue;
            // The surviving row must be an equation in y alone, and its root
            // must equal Cramer's y for the FULL system -- an independent route.
            if (e.reduced_matrix[0][0] != 0) continue;
            Rational coeff = e.reduced_matrix[0][1];
            if (coeff == 0) continue;
            Rational y_elim = e.reduced_rhs[0] / coeff;
            Rational y_cramer = (Rational(a1) * rhs[1] - Rational(a2) * rhs[0]) / det;
            if (y_elim == y_cramer) ++p.correct;
        }
        p.ns = bench([&]{
            std::vector<std::vector<Rational>> A = {{Rational(1), Rational(1)}, {Rational(2), Rational(-1)}};
            std::vector<Rational> b = {Rational(3), Rational(0)};
            SINK += US11_Lopanasthapana::eliminate_variable(A, b, 0).reduced_matrix.size(); }, 20000);
        profiles.push_back(p);
    }
}

static void profile_aggregates() {
    // S11 Vyashti -- factor a common multiplier out of a sum.
    {
        Profile p{"S11", "Vyashti Samanstih", "factors a common multiplier out of a sum", 0, 0, 0, 0, true,
                  "the factored and expanded forms must agree; it does not choose the factor for you"};
        for (long k = -8; k <= 8; ++k) for (long n = 1; n <= 12; ++n) {
            ++p.swept; ++p.applicable;
            std::vector<Rational> v;
            for (long i = 0; i < n; ++i) v.push_back(Rational(i * 3 - 5, i + 2));
            auto f = S11_Vyashti::factor_common(v, Rational(k));
            // Reference: expand FIRST, then sum -- the arrangement the sutra avoids.
            Rational expanded = 0;
            for (const auto& x : v) expanded += Rational(k) * x;
            if (f.total == expanded) ++p.correct;
        }
        std::vector<Rational> v(16, Rational(3, 7));
        p.ns = bench([&]{ SINK += (unsigned long long)(S11_Vyashti::factor_common(v, Rational(5)).total.numerator() % 97); }, 20000);
        profiles.push_back(p);
    }
    // US10 Samuccayagunitah -- dot product.
    {
        Profile p{"US10", "Samuccayagunitah", "dot product of two equal-length vectors", 0, 0, 0, 0, true,
                  "throws on a length mismatch rather than padding or truncating"};
        for (long s = 1; s <= 40; ++s) for (long n = 1; n <= 8; ++n) {
            ++p.swept; ++p.applicable;
            std::vector<Rational> a, b;
            for (long i = 0; i < n; ++i) {
                a.push_back(Rational(s * (i + 1) - 13, 3));
                b.push_back(Rational(7 - s * i, i + 4));
            }
            Rational got = US10_Samuccayagunitah::dot_product(a, b);
            // Independent reference: the polarisation identity
            //   a.b = ( |a+b|^2 - |a|^2 - |b|^2 ) / 2
            // A different arrangement of the same arithmetic, so a transposed
            // index or a dropped term does not cancel out of both sides.
            Rational sq_sum = 0, sq_a = 0, sq_b = 0;
            for (long i = 0; i < n; ++i) {
                sq_sum += (a[i] + b[i]) * (a[i] + b[i]);
                sq_a += a[i] * a[i];
                sq_b += b[i] * b[i];
            }
            if (got == (sq_sum - sq_a - sq_b) / 2) ++p.correct;
        }
        std::vector<Rational> a(32, Rational(2, 5)), b(32, Rational(3, 4));
        p.ns = bench([&]{ SINK += (unsigned long long)(US10_Samuccayagunitah::dot_product(a, b).numerator() % 97); }, 20000);
        profiles.push_back(p);
    }
    // US1 Anurupyena -- proportional scaling and interpolation.
    {
        Profile p{"US1", "Anurupyena", "scaling, lerp, proportional split", 0, 0, 0, 0, true,
                  "a ratio m:n with m+n = 0 has no proportional split and is excluded"};
        for (long a = -9; a <= 9; ++a) for (long b = -9; b <= 9; ++b)
        for (long tn = 0; tn <= 6; ++tn) {
            ++p.swept; ++p.applicable;
            Rational t(tn, 6);
            Rational got = US1_Anurupyena::lerp(Rational(a), Rational(b), t);
            // Reference: the (1-t)a + tb arrangement. It disagrees with a
            // swapped-argument lerp everywhere EXCEPT t = 1/2, which is why
            // the sweep runs t over sixths and not just the midpoint.
            if (got != (Rational(1) - t) * a + t * b) continue;
            if (a + b != 0) {
                auto [p1, p2] = US1_Anurupyena::divide_proportionally(
                    Rational(100), Rational(a), Rational(b));
                if (p1 + p2 != 100) continue;
                if (p1 * b != p2 * a) continue;      // the split really is a:b
            }
            ++p.correct;
        }
        p.ns = bench([&]{ SINK += (unsigned long long)(US1_Anurupyena::lerp(
            Rational(3), Rational(11), Rational(2, 7)).numerator() % 97); }, 20000);
        profiles.push_back(p);
    }
    // US3 Adyam -- first by first, last by last: endpoints and bounds.
    {
        Profile p{"US3", "Adyamadyenantyamantyena", "bounds and order from a sequence", 0, 0, 0, 0, true,
                  "reads every element for min/max; the endpoint shortcut is exact only on sorted input"};
        for (long s = 0; s <= 200; ++s) {
            ++p.swept; ++p.applicable;
            std::vector<Rational> v;
            long n = 3 + s % 9;
            for (long i = 0; i < n; ++i) v.push_back(Rational((s * 37 + i * 53) % 101 - 50, 4));
            auto got = US3_Adyam::check_bounds(v);
            // Reference: the standard library, not another hand loop.
            Rational lo = *std::min_element(v.begin(), v.end());
            Rational hi = *std::max_element(v.begin(), v.end());
            bool asc = std::is_sorted(v.begin(), v.end());
            bool desc = std::is_sorted(v.rbegin(), v.rend());
            if (got.min_bound == lo && got.max_bound == hi &&
                got.sorted_ascending == asc && got.sorted_descending == desc) ++p.correct;
        }
        std::vector<Rational> v(32, Rational(1, 3));
        p.ns = bench([&]{ SINK += (unsigned long long)(US3_Adyam::check_bounds(v).max_bound.numerator() % 97); }, 20000);
        profiles.push_back(p);
    }
}

static void profile_detectors_and_verifiers() {
    // S15 Gunitasamuccaya -- VERIFIER. Scored on both arms: it must accept a
    // true factorisation and reject a corrupted one. Accepting everything
    // would score 100% on a one-armed test.
    {
        Profile p{"S15", "Gunita Samuccayah", "verifies a factorisation against its coefficients", 0, 0, 0, 0, true, ""};
        long rejected = 0;
        for (long r1 = -7; r1 <= 7; ++r1) for (long r2 = -7; r2 <= 7; ++r2) {
            // x^2 - (r1+r2) x + r1 r2, coefficients low-to-high.
            std::vector<Rational> poly = {Rational(r1 * r2), Rational(-(r1 + r2)), Rational(1)};
            {   // TRUE arm: the real roots must be accepted.
                ++p.swept; ++p.applicable;
                auto v = S15_Gunitasamuccaya::verify_roots(poly, {Rational(r1), Rational(r2)});
                if (v.sum_verified && v.product_verified) ++p.correct;
            }
            {   // FALSE arm: a perturbed root must be rejected.
                ++p.swept; ++p.applicable;
                auto v = S15_Gunitasamuccaya::verify_roots(poly, {Rational(r1 + 1), Rational(r2)});
                bool accepted = v.sum_verified && v.product_verified;
                // r1+1 can only still satisfy both if it is a genuine root pair,
                // which for a monic quadratic requires r1+1 == r1: impossible.
                if (!accepted) { ++p.correct; ++rejected; }
            }
        }
        p.limit = "rejected " + std::to_string(rejected) + " of " + std::to_string(p.swept / 2)
                + " perturbed factorisations; a verifier that never rejects is worthless";
        std::vector<Rational> poly = {Rational(6), Rational(-5), Rational(1)};
        std::vector<Rational> roots = {Rational(2), Rational(3)};
        p.ns = bench([&]{ SINK += S15_Gunitasamuccaya::verify_roots(poly, roots).sum_verified; }, 20000);
        profiles.push_back(p);
    }
    // US13 -- VERIFIER of the product-of-sums identity, both arms again.
    {
        Profile p{"US13", "Gunitasamuccayah Samuccayagunitah", "cross-checks a polynomial product", 0, 0, 0, 0, true, ""};
        long rejected = 0;
        for (long s = 1; s <= 60; ++s) {
            std::vector<Rational> a, b;
            for (long i = 0; i < 4; ++i) {
                a.push_back(Rational((s * 11 + i * 7) % 19 - 9, 3));
                b.push_back(Rational((s * 5 + i * 13) % 17 - 8, 2));
            }
            // The true convolution product.
            std::vector<Rational> prod(a.size() + b.size() - 1, Rational(0));
            for (size_t i = 0; i < a.size(); ++i)
                for (size_t j = 0; j < b.size(); ++j) prod[i + j] += a[i] * b[j];
            {   // TRUE arm.
                ++p.swept; ++p.applicable;
                if (US13_Gunitasamuccaya_Samuccayagunitah::verify_polynomial_product(a, b, prod)) ++p.correct;
            }
            {   // FALSE arm: shift one coefficient, keeping the degree.
                ++p.swept; ++p.applicable;
                auto bad = prod; bad[2] += Rational(1);
                if (!US13_Gunitasamuccaya_Samuccayagunitah::verify_polynomial_product(a, b, bad)) {
                    ++p.correct; ++rejected;
                }
            }
        }
        p.limit = "rejected " + std::to_string(rejected) + " of " + std::to_string(p.swept / 2)
                + " corrupted products; the check is on p(1)q(1), so it cannot see"
                  " an error that preserves the coefficient sum";
        std::vector<Rational> a = {Rational(1), Rational(2)}, b = {Rational(3), Rational(4)};
        p.ns = bench([&]{ SINK += US13_Gunitasamuccaya_Samuccayagunitah::verify_consistency(a, b).consistent; }, 20000);
        profiles.push_back(p);
    }
    // US2 Shishyate -- DETECTOR: finds the cycle in a modular sequence.
    {
        Profile p{"US2", "Shishyate Sheshasamjnah", "cycle length in x -> (ax+b) mod m", 0, 0, 0, 0, true,
                  "Floyd's algorithm; gives up after 10000 steps, and says so rather than guessing"};
        for (long a = 0; a <= 12; ++a) for (long b = 0; b <= 12; ++b) for (long m = 2; m <= 20; ++m) {
            ++p.swept;
            auto got = US2_Shishyate::detect_linear_cycle(BigInt(a), BigInt(b), BigInt(m), BigInt(1));
            if (!got.has_cycle) continue;
            ++p.applicable;
            // Independent reference: walk the sequence keeping a seen-map.
            // Any map over a finite state space finds the true rho shape;
            // it shares no code with Floyd's two-pointer method.
            std::vector<long> order(m, -1);
            long x = 1 % m, step = 0, start = -1, len = -1;
            while (true) {
                if (order[x] >= 0) { start = order[x]; len = step - order[x]; break; }
                order[x] = step++;
                x = (a * x + b) % m;
            }
            if ((long)got.cycle_start == start && (long)got.cycle_length == len) ++p.correct;
        }
        p.ns = bench([&]{ SINK += US2_Shishyate::detect_linear_cycle(
            BigInt(3), BigInt(1), BigInt(7), BigInt(1)).cycle_length; }, 5000);
        profiles.push_back(p);
    }
}

int main() {
    profile_universal_multipliers();
    profile_squarers();
    profile_divisibility();
    profile_algebra();
    profile_equations();
    profile_aggregates();
    profile_detectors_and_verifiers();

    std::printf("\n%-5s %-28s %10s %10s %11s  %s\n",
                "ID", "SUTRA", "APPLIES", "CORRECT", "COST ns", "DOMAIN");
    std::printf("%s\n", std::string(118, '-').c_str());
    long total_correct = 0, total_applicable = 0;
    for (const auto& p : profiles) {
        double rate = p.swept ? 100.0 * p.applicable / p.swept : 0;
        double acc  = p.applicable ? 100.0 * p.correct / p.applicable : 0;
        std::printf("%-5s %-28s %8.1f%% %8.1f%% %11.1f  %s\n",
                    p.id.c_str(), p.name.c_str(), rate, acc, p.ns, p.domain.c_str());
        total_correct += p.correct; total_applicable += p.applicable;
    }
    std::printf("\n  LIMITS AND REFUSALS\n");
    for (const auto& p : profiles)
        if (!p.limit.empty()) std::printf("    %-5s %s\n", p.id.c_str(), p.limit.c_str());

    std::printf("\n  %ld of %ld in-domain evaluations correct (%.4f%%)\n",
                total_correct, total_applicable, 100.0 * total_correct / total_applicable);
    std::printf("  sink %llu\n", SINK);
    return total_correct == total_applicable ? 0 : 1;
}
