// Independent verification of all 29 sutras, and of what their combinations compute.
//
// The header ships its own `tests::run_all_tests()`, which reports 29/29. That
// is 29 single-case assertions written beside the code they check, and several
// assert a property rather than a value -- S1_Ekadhikena::test() checks only
// that the period of 1/19 has LENGTH 18, never that the digits are right. So it
// is evidence the code runs, not evidence it is correct, and nothing here
// relies on it.
//
// Every expectation below is computed independently: by direct arithmetic on
// the same types (the non-sutra route), or from a mathematical fact stated in
// the case itself. Part two is the interesting half -- pairs of sutras that
// reach the same quantity by different routes and must agree.

#include "vedic_sutras_complete.hpp"
#include <cstdio>
#include <string>
#include <vector>

using namespace vedic;

static int passed = 0, failed = 0;
static std::vector<std::string> failures;

static void check(bool ok, const std::string& what) {
    if (ok) ++passed;
    else { ++failed; failures.push_back(what); }
}

// Reference polynomial routines, written directly so nothing under test is
// used to judge anything else under test.
static std::vector<Rational> ref_poly_mul(const std::vector<Rational>& p,
                                          const std::vector<Rational>& q) {
    std::vector<Rational> r(p.size() + q.size() - 1, Rational(0));
    for (size_t i = 0; i < p.size(); ++i)
        for (size_t j = 0; j < q.size(); ++j) r[i + j] += p[i] * q[j];
    return r;
}
static Rational ref_poly_eval(const std::vector<Rational>& p, const Rational& x) {
    Rational acc(0), pw(1);
    for (const auto& c : p) { acc += c * pw; pw *= x; }
    return acc;
}
static BigInt ref_gcd(BigInt a, BigInt b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b != 0) { BigInt t = a % b; a = b; b = t; }
    return a;
}

// ------------------------------------------------- part one: the 29, one by one

static void part_one() {
    std::printf("\n=== PART ONE: all 29 sutras against independent references ===\n");

    // S1 Ekadhikena Purvena -- recurring decimals. The header's own test checks
    // only the PERIOD LENGTH of 1/19; these check the digits.
    {
        auto r = S1_Ekadhikena::divide_recurring(BigInt(1), BigInt(19), 100);
        // 1/19 = 0.(052631578947368421)
        const int expect[] = {0,5,2,6,3,1,5,7,8,9,4,7,3,6,8,4,2,1};
        bool ok = r.recurring.size() == 18;
        for (size_t i = 0; ok && i < 18; ++i) ok = r.recurring[i] == expect[i];
        check(ok, "S1 1/19 recurring digits");
        auto s = S1_Ekadhikena::divide_recurring(BigInt(1), BigInt(7), 100);
        const int e7[] = {1,4,2,8,5,7};
        bool ok7 = s.recurring.size() == 6;
        for (size_t i = 0; ok7 && i < 6; ++i) ok7 = s.recurring[i] == e7[i];
        check(ok7, "S1 1/7 recurring digits");
        // The nine-ender path must agree with the general one, digit for digit.
        auto t = S1_Ekadhikena::divide_by_nine_ender(BigInt(49), 200);
        auto g = S1_Ekadhikena::divide_recurring(BigInt(1), BigInt(49), 200);
        check(t.recurring.size() == 42 && t.recurring == g.recurring,
              "S1 1/49 nine-ender == general path, period 42");
    }

    // S2 Nikhilam -- multiplication by base complement.
    {
        struct { const char* a; const char* b; } cs[] = {
            {"98","97"}, {"9998","9997"}, {"103","104"}, {"88","93"}, {"9","8"}};
        bool ok = true;
        for (auto& c : cs) {
            BigInt a(c.a), b(c.b);
            ok = ok && S2_Nikhilam::multiply(a, b).product == a * b;
            ok = ok && S2_Nikhilam::multiply_extended(a, b) == a * b;
        }
        check(ok, "S2 Nikhilam multiply == a*b on 5 pairs");
        check(S2_Nikhilam::find_base(BigInt(98), BigInt(97)) == 100, "S2 find_base(98,97)==100");
    }

    // S3 Urdhva Tiryagbhyam -- general multiplication, polynomials, matrices.
    {
        bool ok = true;
        const char* xs[] = {"0","7","847362514","999999","123456789012345"};
        for (auto x : xs) for (auto y : xs) {
            BigInt a(x), b(y);
            ok = ok && S3_Urdhva::multiply(a, b).product == a * b;
        }
        check(ok, "S3 Urdhva multiply == a*b on 25 pairs");

        std::vector<Rational> p{Rational(1), Rational(2), Rational(3)};
        std::vector<Rational> q{Rational(4), Rational(5)};
        check(S3_Urdhva::polynomial_multiply(p, q) == ref_poly_mul(p, q),
              "S3 polynomial_multiply matches direct convolution");

        // (2x2)(2x2): [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        std::vector<Rational> A{Rational(1),Rational(2),Rational(3),Rational(4)};
        std::vector<Rational> B{Rational(5),Rational(6),Rational(7),Rational(8)};
        auto M = S3_Urdhva::matrix_multiply(A, 2, 2, B, 2, 2);
        check(M.size() == 4 && M[0] == 19 && M[1] == 22 && M[2] == 43 && M[3] == 50,
              "S3 matrix_multiply 2x2");
    }

    // S4 Paravartya Yojayet -- division.
    {
        bool ok = true;
        struct { const char* n; const char* d; } cs[] = {
            {"987654321","1234"}, {"100","7"}, {"0","5"}, {"144","12"}, {"999999","1001"}};
        for (auto& c : cs) {
            BigInt n(c.n), d(c.d);
            auto r = S4_Paravartya::divide(n, d);
            ok = ok && r.quotient == n / d && r.remainder == n % d;
        }
        check(ok, "S4 divide quotient+remainder on 5 pairs");

        // (x^2+3x+2) / (x+1) = (x+2) exactly.
        std::vector<Rational> num{Rational(2), Rational(3), Rational(1)};
        std::vector<Rational> den{Rational(1), Rational(1)};
        auto pd = S4_Paravartya::polynomial_divide(num, den);
        bool rem0 = true;
        for (const auto& c : pd.remainder) rem0 = rem0 && c == 0;
        check(ref_poly_mul(pd.quotient, den) == num && rem0,
              "S4 polynomial_divide: q*d == n with zero remainder");
    }

    // S5 Shunyam Samyasamuccaye -- when the sum is the same.
    {
        auto r = S5_Shunyam::solve_product_equality(Rational(2), Rational(3), Rational(1), Rational(4));
        check(r.sum_equality_applies && r.solutions.empty(), "S5 2+3==1+4 but 6!=4, no solution");
        // ax + b = cx + d  ->  x = (d-b)/(a-c). 3x+1 = x+7 -> x = 3.
        auto s = S5_Shunyam::solve_linear(Rational(3), Rational(1), Rational(1), Rational(7));
        check(s.has_value() && *s == 3, "S5 solve_linear 3x+1=x+7 -> 3");
        auto t = S5_Shunyam::solve_linear(Rational(2), Rational(5), Rational(2), Rational(9));
        check(!t.has_value(), "S5 solve_linear parallel -> no solution");
    }

    // S6 Anurupye -- proportion.
    {
        check(S6_Anurupye::ratios_equal(Rational(1), Rational(2), Rational(3), Rational(6)),
              "S6 1:2 == 3:6");
        check(!S6_Anurupye::ratios_equal(Rational(1), Rational(2), Rational(3), Rational(7)),
              "S6 1:2 != 3:7");
        // 2x + 3y = 8 ; x + y = 3  ->  x = 1, y = 2
        auto r = S6_Anurupye::solve_system_2x2(Rational(2), Rational(3), Rational(8),
                                               Rational(1), Rational(1), Rational(3));
        check(r.x.has_value() && *r.x == 1 && r.y.has_value() && *r.y == 2,
              "S6 solve_system_2x2 -> (1,2)");
    }

    // S7 Sankalana Vyavakalanabhyam -- elimination.
    {
        auto r = S7_Sankalana::solve_by_elimination(Rational(2), Rational(3), Rational(8),
                                                    Rational(1), Rational(1), Rational(3));
        check(r.x.has_value() && *r.x == 1 && *r.y == 2, "S7 elimination -> (1,2)");
        // 3x+2y+z=10 ; 2x+3y+z=11 ; x+y+z=6  ->  (1,2,3)
        std::vector<std::vector<Rational>> A{
            {Rational(3),Rational(2),Rational(1)},
            {Rational(2),Rational(3),Rational(1)},
            {Rational(1),Rational(1),Rational(1)}};
        std::vector<Rational> b{Rational(10), Rational(11), Rational(6)};
        auto g = S7_Sankalana::gaussian_eliminate(A, b);
        check(g.has_value() && (*g)[0] == 1 && (*g)[1] == 2 && (*g)[2] == 3,
              "S7 gaussian_eliminate 3x3 -> (1,2,3)");
    }

    // S8 Purana Apuranabhyam -- completing the square.
    {
        // x^2 - 6x + 5 : vertex (3, -4), roots 1 and 5.
        auto cs = S8_Purana::complete_the_square(Rational(1), Rational(-6), Rational(5));
        check(cs.a == 1 && cs.h == 3 && cs.k == -4, "S8 complete_the_square vertex (3,-4)");
        auto q = S8_Purana::solve_quadratic(Rational(1), Rational(-6), Rational(5));
        bool roots = q.root1.has_value() && q.root2.has_value() &&
                     ((*q.root1 == 1 && *q.root2 == 5) || (*q.root1 == 5 && *q.root2 == 1));
        check(roots && q.discriminant == 16, "S8 solve_quadratic roots {1,5}, disc 16");
        auto irr = S8_Purana::solve_quadratic(Rational(1), Rational(0), Rational(-2));
        check(!irr.exact, "S8 x^2-2 flagged as not exactly rational");
    }

    // S9 Calana Kalanabhyam -- calculus on polynomials.
    {
        std::vector<Rational> p{Rational(5), Rational(3), Rational(2)};  // 5 + 3x + 2x^2
        auto d = S9_Calana::differentiate(p);                            // 3 + 4x
        check(d.size() == 2 && d[0] == 3 && d[1] == 4, "S9 differentiate");
        auto in = S9_Calana::integrate(d);                               // 3x + 2x^2 (+C=0)
        check(in.size() == 3 && in[0] == 0 && in[1] == 3 && in[2] == 2, "S9 integrate");
        check(S9_Calana::evaluate(p, Rational(2)) == ref_poly_eval(p, Rational(2)),
              "S9 evaluate matches direct");
        auto cp = S9_Calana::find_critical_points(p);                    // 3+4x=0 -> -3/4
        check(cp.size() == 1 && cp[0] == Rational(-3, 4), "S9 critical point -3/4");
    }

    // S10 Yavadunam -- squaring by deficiency.
    {
        bool ok = true;
        for (const char* s : {"98", "9997", "103", "9", "1000001"}) {
            BigInt n(s);
            ok = ok && S10_Yavadunam::square(n).square == n * n;
        }
        check(ok, "S10 square == n*n on 5 values");
        check(S10_Yavadunam::square(BigInt(97), BigInt(100)).square == BigInt(9409),
              "S10 97^2 base 100 == 9409");
    }

    // S11 Vyashti Samashti -- part and whole.
    {
        std::vector<Rational> v{Rational(4), Rational(8), Rational(12)};
        // factor_common states the distributive law: k*sum(v) == sum(k*v).
        auto f = S11_Vyashti::factor_common(v, Rational(4));
        Rational direct = 0;
        for (const auto& x : v) direct += Rational(4) * x;
        check(f.sum_of_parts == 24 && f.total == direct,
              "S11 factor_common: 4*sum(v) == sum(4*v)");
        // total_energy is E = mc^2 with c^2 exact, not a plain sum.
        check(S11_Vyashti::total_energy(v)
                  == Rational(BigInt("89875517873681764")) * Rational(24),
              "S11 total_energy == c^2 * total mass");
        check(S11_Vyashti::gcd_multiple({BigInt(12), BigInt(18), BigInt(24)}) == 6,
              "S11 gcd(12,18,24) == 6");
    }

    // S12 Sesanyankena Caramena -- divisibility by digit rules.
    {
        bool ok = true;
        for (long n = -50; n <= 200; ++n) {
            BigInt b(n);
            ok = ok && S12_Sesanyankena::divisible_by_2(b)  == (n % 2 == 0);
            ok = ok && S12_Sesanyankena::divisible_by_3(b)  == (n % 3 == 0);
            ok = ok && S12_Sesanyankena::divisible_by_4(b)  == (n % 4 == 0);
            ok = ok && S12_Sesanyankena::divisible_by_5(b)  == (n % 5 == 0);
            ok = ok && S12_Sesanyankena::divisible_by_8(b)  == (n % 8 == 0);
            ok = ok && S12_Sesanyankena::divisible_by_9(b)  == (n % 9 == 0);
            ok = ok && S12_Sesanyankena::divisible_by_11(b) == (n % 11 == 0);
        }
        check(ok, "S12 seven divisibility rules over -50..200 (1757 checks)");
        // 7^128 mod 13, computed by repeated squaring here as the reference.
        BigInt ref = 1, base = 7;
        for (int i = 0; i < 128; ++i) ref = (ref * base) % 13;
        check(S12_Sesanyankena::mod_pow(BigInt(7), BigInt(128), BigInt(13)) == ref,
              "S12 mod_pow(7,128,13)");
    }

    // S13 Sopantyadvayamantyam -- continued fractions.
    {
        // [3;7,15,1] -> 355/113, the classic pi convergent.
        std::vector<BigInt> cf{BigInt(3), BigInt(7), BigInt(15), BigInt(1)};
        auto c = S13_Sopantya::nth_convergent(cf, 3);
        check(c.numerator == 355 && c.denominator == 113, "S13 [3;7,15,1] == 355/113");
        // [3;7,15,1] and [3;7,16] are the two forms of the same rational;
        // to_continued_fraction returns the canonical one, so the round-trip
        // is asserted on the VALUE rather than on one chosen spelling.
        auto back = S13_Sopantya::to_continued_fraction(Rational(BigInt(355), BigInt(113)));
        auto rebuilt = S13_Sopantya::nth_convergent(back, back.size() - 1);
        check(rebuilt.numerator == 355 && rebuilt.denominator == 113,
              "S13 355/113 continued fraction round-trips by value");
        // all-ones continued fraction gives consecutive Fibonacci ratios.
        auto g = S13_Sopantya::golden_ratio_convergent(9);
        check(g.numerator == 89 && g.denominator == 55, "S13 golden convergent 89/55");
    }

    // S14 Ekanyunena Purvena -- multiply by a string of nines.
    {
        bool ok = true;
        for (const char* s : {"4567", "7", "999", "123456"})
            for (size_t k = 1; k <= 6; ++k) {
                BigInt n(s), nines = util::power(BigInt(10), k) - 1;
                ok = ok && S14_Ekanyunena::multiply_by_nines(n, k).product == n * nines;
            }
        check(ok, "S14 multiply_by_nines == n*(10^k - 1), 24 cases");
        check(S14_Ekanyunena::multiply_near_power(BigInt(76), BigInt(98)) == BigInt(76) * 98,
              "S14 multiply_near_power 76*98");
    }

    // S15 Gunita Samuccayah -- the product of the sums.
    {
        auto d = S15_Gunitasamuccaya::verify_distributive(
            Rational(1), Rational(2), Rational(3), Rational(4));
        check(d.left_side == 21 && d.verified, "S15 (1+2)(3+4)==21 verified");
        // (x-1)(x-2) = x^2 - 3x + 2, roots 1 and 2.
        std::vector<Rational> poly{Rational(2), Rational(-3), Rational(1)};
        auto v = S15_Gunitasamuccaya::verify_roots(poly, {Rational(1), Rational(2)});
        check(v.sum_verified && v.product_verified, "S15 verify_roots x^2-3x+2");
    }

    // S16 Gunaka Samuccayah -- the factors of the sum.
    {
        auto f = S16_Gunakasamuccaya::prime_factorize(BigInt(360));  // 2^3 * 3^2 * 5
        BigInt rebuilt = 1;
        for (const auto& pf : f) rebuilt *= util::power(pf.prime, pf.exponent);
        check(rebuilt == 360 && f.size() == 3, "S16 prime_factorize(360) rebuilds to 360");
        check(S16_Gunakasamuccaya::verify_gcd(BigInt(12), BigInt(18), BigInt(6)),
              "S16 verify_gcd(12,18,6)");
        check(S16_Gunakasamuccaya::verify_lcm(BigInt(12), BigInt(18), BigInt(36)),
              "S16 verify_lcm(12,18,36)");
        check(S16_Gunakasamuccaya::verify_gcd_lcm_identity(BigInt(12), BigInt(18)),
              "S16 gcd*lcm == a*b");
    }

    // US1 Anurupyena -- proportionality.
    {
        check(US1_Anurupyena::scale(Rational(3), Rational(4)) == 12, "US1 scale 3*4");
        check(US1_Anurupyena::lerp(Rational(0), Rational(10), Rational(1, 4)) == Rational(5, 2),
              "US1 lerp(0,10,1/4) == 5/2");
        auto p = US1_Anurupyena::divide_proportionally(Rational(100), Rational(2), Rational(3));
        check(p.first == 40 && p.second == 60, "US1 100 split 2:3 -> 40/60");
    }

    // US2 Shishyate Sheshasamjnah -- cycle detection in x -> a*x + b mod m.
    {
        auto c = US2_Shishyate::detect_linear_cycle(BigInt(1), BigInt(1), BigInt(7), BigInt(0));
        check(c.has_cycle && c.cycle_length == 7, "US2 x+1 mod 7 has cycle length 7");
        auto d = US2_Shishyate::detect_linear_cycle(BigInt(0), BigInt(3), BigInt(10), BigInt(5));
        check(d.has_cycle && d.cycle_length == 1, "US2 constant map cycles at length 1");
    }

    // US3 Adyamadyenantyamantyena -- first by first, last by last.
    {
        std::vector<Rational> v{Rational(2), Rational(5), Rational(9)};
        auto e = US3_Adyam::apply_endpoints(v, [](const Rational& r) { return r * r; });
        check(e.first_result == 4 && e.last_result == 81, "US3 endpoints squared -> 4, 81");
        auto b = US3_Adyam::check_bounds(v);
        check(b.min_bound == 2 && b.max_bound == 9 && b.sorted_ascending,
              "US3 check_bounds on ascending input");
    }

    // US4 Kevalaih Saptakam Gunyat -- the sevens.
    {
        bool ok = true;
        for (long n : {0L, 1L, 13L, 143L, 999999L}) {
            BigInt b(n);
            ok = ok && US4_Kevalaih::multiply_by_7(b) == b * 7;
            ok = ok && US4_Kevalaih::multiply_by_7_shift(b) == b * 7;
        }
        check(ok, "US4 both multiply_by_7 routes == n*7");
        // multiply_by_complement(n,k) is 10n - (10-k)n, which is n*k -- the
        // complement route to an ordinary small multiplication, not n*(10^k-2).
        bool okc = true;
        for (long n : {0L, 7L, 76L, 12345L}) for (int k = 1; k <= 9; ++k)
            okc = okc && US4_Kevalaih::multiply_by_complement(BigInt(n), k) == BigInt(n) * k;
        check(okc, "US4 multiply_by_complement(n,k) == n*k, 36 cases");
    }

    // US5 Vestanam -- osculation.
    {
        auto o = US5_Vestanam::find_positive_osculator(BigInt(19));
        check(o.has_value() && *o == 2, "US5 osculator of 19 is 2");
        bool ok = true;
        for (long n = 19; n <= 19 * 40; n += 19)
            ok = ok && US5_Vestanam::divisibility_by_osculation(BigInt(n), BigInt(19), BigInt(2), true);
        check(ok, "US5 osculation confirms 40 multiples of 19");
        check(!US5_Vestanam::divisibility_by_osculation(BigInt(20), BigInt(19), BigInt(2), true),
              "US5 osculation rejects 20");
    }

    // US6 / US7 Yavadunam variants -- squaring near a base.
    {
        bool ok6 = true, ok7 = true;
        for (const char* s : {"98", "103", "9997", "12"}) {
            BigInt n(s);
            ok6 = ok6 && US6_Yavadunam_Squared::square_by_deficiency(n).result == n * n;
            ok7 = ok7 && US7_Yavadunam_Extended::square_extended(n, BigInt(100)).result == n * n;
        }
        check(ok6, "US6 square_by_deficiency == n*n on 4 values");
        check(ok7, "US7 square_extended base 100 == n*n on 4 values");
    }

    // US8 Antyayordashake'pi -- last digits summing to ten.
    {
        auto r = US8_Antyayor::multiply_sum_to_ten(BigInt(43), BigInt(47));  // 3+7=10, same tens
        check(r.applicable && r.product == 43 * 47, "US8 43*47 applicable and correct");
        auto q = US8_Antyayor::multiply_sum_to_ten(BigInt(43), BigInt(48));
        check(!q.applicable, "US8 43*48 correctly reported inapplicable");
    }

    // US9 Antyayoreva -- last digits only.
    {
        bool ok = true;
        for (long a = 0; a < 40; ++a) for (long b = 0; b < 40; ++b)
            ok = ok && US9_Antyayoreva::last_digit_of_product(BigInt(a), BigInt(b)) == (a * b) % 10;
        check(ok, "US9 last_digit_of_product over 1600 pairs");
        // 7^222 mod 10 -- cycle 7,9,3,1, so 222 mod 4 == 2 -> 9.
        check(US9_Antyayoreva::last_digit_of_power(BigInt(7), BigInt(222)) == 9,
              "US9 last digit of 7^222 is 9");
    }

    // US10 Samuccayagunitah -- the sum multiplied.
    {
        std::vector<Rational> v{Rational(1), Rational(2), Rational(3)};
        check(US10_Samuccayagunitah::multiply_sum(v, Rational(10)) == 60, "US10 10*(1+2+3)==60");
        auto b = US10_Samuccayagunitah::multiply_batch(v, Rational(3));
        check(b.size() == 3 && b[0] == 3 && b[1] == 6 && b[2] == 9, "US10 multiply_batch");
        std::vector<Rational> w{Rational(4), Rational(5), Rational(6)};
        check(US10_Samuccayagunitah::dot_product(v, w) == 32, "US10 dot_product == 32");
    }

    // US11 Lopanasthapanabhyam -- by elimination and retention.
    {
        std::vector<std::vector<Rational>> A{
            {Rational(1), Rational(1)}, {Rational(2), Rational(3)}};
        std::vector<Rational> b{Rational(3), Rational(8)};
        auto e = US11_Lopanasthapana::eliminate_variable(A, b, 0);
        check(e.reduced_matrix.size() == 1 && e.eliminated_variable == 0,
              "US11 eliminating one variable leaves one row");
    }

    // US12 Vilokanam -- by observation.
    {
        auto s = US12_Vilokanam::check_perfect_square(BigInt(144));
        check(s.type == US12_Vilokanam::PatternType::PERFECT_SQUARE && s.parameters[0] == 12,
              "US12 144 is 12 squared");
        auto n = US12_Vilokanam::check_perfect_square(BigInt(145));
        check(n.type == US12_Vilokanam::PatternType::NONE, "US12 145 is not a perfect square");
        auto ap = US12_Vilokanam::check_arithmetic_progression(
            {Rational(2), Rational(5), Rational(8), Rational(11)});
        check(ap.type == US12_Vilokanam::PatternType::ARITHMETIC_PROGRESSION && ap.parameters[1] == 3,
              "US12 AP detected with common difference 3");
        auto gp = US12_Vilokanam::check_geometric_progression(
            {Rational(3), Rational(6), Rational(12), Rational(24)});
        check(gp.type == US12_Vilokanam::PatternType::GEOMETRIC_PROGRESSION && gp.parameters[1] == 2,
              "US12 GP detected with common ratio 2");
    }

    // US13 -- the two summation sutras taken together.
    {
        std::vector<Rational> a{Rational(1), Rational(2)}, b{Rational(3), Rational(4)};
        auto c = US13_Gunitasamuccaya_Samuccayagunitah::verify_consistency(a, b);
        check(c.product_of_sums == 21, "US13 (1+2)(3+4) == 21");
        check(US13_Gunitasamuccaya_Samuccayagunitah::verify_polynomial_product(
                  a, b, ref_poly_mul(a, b)), "US13 verify_polynomial_product accepts the truth");
        check(!US13_Gunitasamuccaya_Samuccayagunitah::verify_polynomial_product(
                  a, b, {Rational(1), Rational(1), Rational(1)}),
              "US13 verify_polynomial_product rejects a wrong product");
    }
}

// ------------------------------------------- part two: what combinations compute

static void part_two() {
    std::printf("\n=== PART TWO: combinations -- two sutras, one quantity ===\n");

    // Four independent multiplication sutras must agree with each other and
    // with the operator, on numbers where all four apply.
    {
        BigInt a("43"), b("47");   // last digits sum to ten, both near 45
        BigInt truth = a * b;
        bool ok = S3_Urdhva::multiply(a, b).product == truth
               && S2_Nikhilam::multiply_extended(a, b) == truth
               && US8_Antyayor::multiply_sum_to_ten(a, b).product == truth;
        check(ok, "S3 + S2 + US8 agree on 43*47");
        BigInt c("76");
        check(US4_Kevalaih::multiply_by_complement(c, 7) == S3_Urdhva::multiply(c, BigInt(7)).product
              && US4_Kevalaih::multiply_by_7(c) == S3_Urdhva::multiply(c, BigInt(7)).product,
              "US4 complement route and multiply_by_7 both == S3 Urdhva on 76*7");
    }

    // Three squaring sutras against each other.
    {
        bool ok = true;
        for (const char* s : {"98", "103", "9997"}) {
            BigInt n(s);
            ok = ok && S10_Yavadunam::square(n).square
                    == US6_Yavadunam_Squared::square_by_deficiency(n).result;
            ok = ok && S10_Yavadunam::square(n).square
                    == US7_Yavadunam_Extended::square_extended(n, BigInt(100)).result;
        }
        check(ok, "S10 + US6 + US7 agree on three squarings");
    }

    // Completing the square (S8) and calculus (S9) reach the vertex by wholly
    // different routes: algebraic rearrangement versus setting f'(x) = 0.
    {
        bool ok = true;
        struct { int a, b, c; } qs[] = {{1,-6,5}, {2,4,-6}, {3,-12,7}, {1,0,-9}};
        for (auto& q : qs) {
            auto cs = S8_Purana::complete_the_square(Rational(q.a), Rational(q.b), Rational(q.c));
            auto cp = S9_Calana::find_critical_points(
                {Rational(q.c), Rational(q.b), Rational(q.a)});
            ok = ok && cp.size() == 1 && cp[0] == cs.h;
        }
        check(ok, "S8 vertex == S9 critical point on 4 quadratics");
    }

    // Two independent divisibility routes: digit rules (S12) versus osculation
    // (US5). They share no code path.
    {
        bool ok = true;
        auto osc = US5_Vestanam::find_positive_osculator(BigInt(11));
        for (long n = 1; n <= 400; ++n)
            ok = ok && S12_Sesanyankena::divisible_by_11(BigInt(n))
                    == US5_Vestanam::divisibility_by_osculation(BigInt(n), BigInt(11), *osc, true);
        check(ok, "S12 digit rule == US5 osculation for 11, over 400 values");
    }

    // Divisibility (S12) must predict exactly when division (S4) leaves nothing.
    {
        bool ok = true;
        for (long n = 0; n <= 300; ++n) {
            auto d = S4_Paravartya::divide(BigInt(n), BigInt(9));
            ok = ok && ((d.remainder == 0) == S12_Sesanyankena::divisible_by_9(BigInt(n)));
        }
        check(ok, "S4 zero remainder <=> S12 divisible_by_9, over 301 values");
    }

    // Factorisation (S16) must reproduce the gcd that S11 computes directly.
    {
        bool ok = true;
        struct { long a, b; } ps[] = {{12,18}, {360,84}, {17,5}, {100,75}, {64,48}};
        for (auto& p : ps) {
            auto fa = S16_Gunakasamuccaya::prime_factorize(BigInt(p.a));
            auto fb = S16_Gunakasamuccaya::prime_factorize(BigInt(p.b));
            BigInt g = 1;
            for (const auto& x : fa) for (const auto& y : fb)
                if (x.prime == y.prime)
                    g *= util::power(x.prime, x.exponent < y.exponent ? x.exponent : y.exponent);
            ok = ok && g == S11_Vyashti::gcd_multiple({BigInt(p.a), BigInt(p.b)})
                    && g == ref_gcd(BigInt(p.a), BigInt(p.b));
        }
        check(ok, "S16 factorisation gcd == S11 gcd_multiple == reference, 5 pairs");
    }

    // Polynomial multiply (S3) then divide (S4) must return the original.
    {
        std::vector<Rational> p{Rational(1), Rational(2), Rational(3)};
        std::vector<Rational> q{Rational(-1), Rational(1)};
        auto prod = S3_Urdhva::polynomial_multiply(p, q);
        auto back = S4_Paravartya::polynomial_divide(prod, q);
        bool rem0 = true;
        for (const auto& c : back.remainder) rem0 = rem0 && c == 0;
        check(back.quotient == p && rem0, "S3 multiply then S4 divide round-trips");
    }

    // Calculus round-trip: integrating a derivative restores the polynomial up
    // to the constant the derivative discarded.
    {
        std::vector<Rational> p{Rational(7), Rational(3), Rational(2), Rational(5)};
        auto back = S9_Calana::integrate(S9_Calana::differentiate(p));
        bool ok = back.size() == p.size() && back[0] == 0;
        for (size_t i = 1; ok && i < p.size(); ++i) ok = back[i] == p[i];
        check(ok, "S9 integrate(differentiate(p)) == p with constant dropped");
    }

    // Last-digit shortcut (US9) must match the full product (S3) every time.
    {
        bool ok = true;
        for (long a = 90; a < 130; ++a) for (long b = 90; b < 130; ++b) {
            BigInt full = S3_Urdhva::multiply(BigInt(a), BigInt(b)).product;
            ok = ok && US9_Antyayoreva::last_digit_of_product(BigInt(a), BigInt(b)) == full % 10;
        }
        check(ok, "US9 last digit == S3 full product mod 10, 1600 pairs");
    }

    // Continued fractions (S13) and proportion (US1): a convergent scaled by its
    // own denominator must return the numerator.
    {
        auto c = S13_Sopantya::nth_convergent(
            {BigInt(3), BigInt(7), BigInt(15), BigInt(1)}, 3);
        Rational conv(c.numerator, c.denominator);
        check(US1_Anurupyena::scale(conv, Rational(c.denominator)) == Rational(c.numerator),
              "S13 convergent scaled by US1 recovers the numerator");
    }

    // The two summation sutras (S15, US13) state the same identity.
    {
        bool ok = true;
        for (int a = 1; a <= 5; ++a) for (int b = 1; b <= 5; ++b) {
            auto d = S15_Gunitasamuccaya::verify_distributive(
                Rational(a), Rational(b), Rational(a + 1), Rational(b + 1));
            auto c = US13_Gunitasamuccaya_Samuccayagunitah::verify_consistency(
                {Rational(a), Rational(b)}, {Rational(a + 1), Rational(b + 1)});
            ok = ok && d.left_side == c.product_of_sums;
        }
        check(ok, "S15 product-of-sums == US13 product_of_sums, 25 pairs");
    }

    // Recurring decimal (S1) reconstructed by division (S4): the period of 1/d
    // is the smallest k with 10^k == 1 mod d, which S12's mod_pow can confirm.
    {
        bool ok = true;
        for (long d : {7L, 11L, 13L, 19L, 21L}) {
            auto r = S1_Ekadhikena::divide_recurring(BigInt(1), BigInt(d), 200);
            BigInt k(static_cast<long>(r.recurring.size()));
            ok = ok && S12_Sesanyankena::mod_pow(BigInt(10), k, BigInt(d)) == 1;
        }
        check(ok, "S1 period length k satisfies 10^k == 1 mod d, via S12 mod_pow, 5 denominators");
    }
}

int main() {
    part_one();
    part_two();

    std::printf("\n  %d passed, %d failed\n", passed, failed);
    for (const auto& f : failures) std::printf("  FAILED: %s\n", f.c_str());

    auto self = tests::run_all_tests();
    size_t sp = 0;
    for (const auto& r : self) if (r.passed) ++sp;
    std::printf("  (the header's own suite, for reference: %zu/%zu)\n", sp, self.size());
    return failed == 0 ? 0 : 1;
}
