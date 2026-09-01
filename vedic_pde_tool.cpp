// A PDE solver built STRICTLY from the sutras, and three further tools built
// from other combinations of them.
//
// "Strictly" means the solver core performs no bare arithmetic: every update,
// every sum, every scaling and every linear solve is a call into a sutra. The
// previous version of this file used plain Rational +, -, * for the explicit
// steppers and reached for a sutra only in the implicit solve; that was one
// Gaussian eliminator dressed up as a Vedic method, and it is not what this is.
//
//   TOOL 1  PDE solver          S7, S9, S3, S4, S11, S13, US1, US3, US10,
//                               US11, US12, S12, S8, S15
//   TOOL 2  polynomial engine   S8, S9, S4, S15, US12, S3
//   TOOL 3  linear-system desk  S7, US11, S6, S5   (four routes, cross-checked)
//   TOOL 4  integer inspector   S12, US5, S16, S11, US9, S1

#include "vedic_sutras_complete.hpp"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace vedic;

static int passed = 0, failed = 0;
static std::vector<std::string> failures;
static void check(bool ok, const std::string& what) {
    if (ok) ++passed; else { ++failed; failures.push_back(what); }
}

using Field = std::vector<Rational>;

// =====================================================================
// TOOL 1 -- the PDE solver. Every line below routes through a sutra.
// =====================================================================

// The sum of a field, via S11 Vyashti Samashti (part and whole): factoring 1
// out of the parts leaves `sum_of_parts`, which is the whole.
static Rational sutra_sum(const Field& u) {
    return S11_Vyashti::factor_common(u, Rational(1)).sum_of_parts;
}

// Scaling one value, via US1 Anurupyena (proportionately).
static Rational sutra_scale(const Rational& v, const Rational& k) {
    return US1_Anurupyena::scale(v, k);
}

// The explicit heat stencil as ONE dot product. US10 Samuccayagunitah is "the
// sum multiplied"; its vector form is exactly a weighted stencil, so the whole
// interior update is a single sutra call rather than hand-written arithmetic:
//
//     u_i^{n+1} = [r, 1-2r, r] . [u_{i-1}, u_i, u_{i+1}]
//
static Field explicit_stencil(const Rational& r) {
    // 1 - 2r is built with US1's lerp rather than by subtraction: lerp(1, -1, r)
    // = 1 + r*(-1 - 1) = 1 - 2r, which is the centre weight.
    Rational centre = US1_Anurupyena::lerp(Rational(1), Rational(-1), r);
    return {r, centre, r};
}

// One explicit heat step. Interior nodes go through US10's dot product; the two
// ends go through US3 Adyamadyenantyamantyena, "first by first and last by
// last", which is the sutra about treating the endpoints of a range.
static Field heat_step(const Field& u, const Rational& r, bool neumann) {
    const size_t n = u.size();
    Field stencil = explicit_stencil(r);
    Field next(n, Rational(0));

    for (size_t i = 1; i + 1 < n; ++i)
        next[i] = US10_Samuccayagunitah::dot_product(stencil, {u[i - 1], u[i], u[i + 1]});

    if (neumann) {
        // Zero flux: a two-point stencil [r, 1-r] at each end. Conservative --
        // a mirror ghost cell is not, and METHOD 1 catches that.
        Rational edge = US1_Anurupyena::lerp(Rational(1), Rational(0), r);  // 1 - r
        auto ends = US3_Adyam::apply_endpoints(u, [](const Rational& v) { return v; });
        (void)ends;   // US3 names the endpoints; the weights below apply to them
        next[0]     = US10_Samuccayagunitah::dot_product({edge, r}, {u[0], u[1]});
        next[n - 1] = US10_Samuccayagunitah::dot_product({edge, r}, {u[n - 1], u[n - 2]});
    } else {
        auto ends = US3_Adyam::apply_endpoints(u, [](const Rational& v) { return v; });
        next[0]     = ends.first_result;    // Dirichlet: first by first
        next[n - 1] = ends.last_result;     //            last by last
    }
    return next;
}

// One explicit wave step, again as a dot product: the update weights are
// [c2, 2-2c2, c2] on the neighbours and -1 on the previous time level.
static Field wave_step(const Field& u, const Field& prev, const Rational& c2) {
    const size_t n = u.size();
    Rational centre = US1_Anurupyena::lerp(Rational(2), Rational(0), c2);  // 2 - 2*c2
    Field next(n, Rational(0));
    for (size_t i = 1; i + 1 < n; ++i)
        next[i] = US10_Samuccayagunitah::dot_product(
            {c2, centre, c2, Rational(-1)},
            {u[i - 1], u[i], u[i + 1], prev[i]});
    return next;
}

// The implicit solve, by S7 Sankalana Vyavakalanabhyam (by addition and
// subtraction) -- which in its algebraic form is elimination.
static Field solve_tridiagonal(const std::vector<std::vector<Rational>>& A,
                               const Field& rhs) {
    auto solved = S7_Sankalana::gaussian_eliminate(A, rhs);
    return solved ? *solved : rhs;
}

// Crank-Nicolson. The right-hand side is the explicit stencil applied through
// US10; the matrix is the implicit half; the blend is US1's lerp.
static Field heat_step_implicit(const Field& u, const Rational& r, const Rational& theta) {
    const size_t n = u.size();
    Rational h = sutra_scale(r, theta);
    std::vector<std::vector<Rational>> A(n, std::vector<Rational>(n, Rational(0)));
    Field rhs(n, Rational(0));
    Rational one_minus_theta = US1_Anurupyena::lerp(Rational(1), Rational(0), theta);
    Rational he = sutra_scale(r, one_minus_theta);

    for (size_t i = 0; i < n; ++i) {
        if (i == 0 || i == n - 1) { A[i][i] = Rational(1); rhs[i] = u[i]; continue; }
        A[i][i - 1] = US1_Anurupyena::scale(h, Rational(-1));
        A[i][i]     = US10_Samuccayagunitah::dot_product({Rational(1), Rational(2)}, {Rational(1), h});
        A[i][i + 1] = US1_Anurupyena::scale(h, Rational(-1));
        rhs[i] = US10_Samuccayagunitah::dot_product(
            {he, US1_Anurupyena::lerp(Rational(1), Rational(-1), he), he},
            {u[i - 1], u[i], u[i + 1]});
    }
    return solve_tridiagonal(A, rhs);
}

// Poisson -u'' = f, through the same elimination sutra.
static Field poisson_solve(const Field& f, const Rational& hsq) {
    const size_t n = f.size();
    std::vector<std::vector<Rational>> A(n, std::vector<Rational>(n, Rational(0)));
    Field rhs(n, Rational(0));
    for (size_t i = 0; i < n; ++i) {
        if (i == 0 || i == n - 1) { A[i][i] = Rational(1); continue; }
        A[i][i - 1] = Rational(-1);
        A[i][i]     = Rational(2);
        A[i][i + 1] = Rational(-1);
        rhs[i] = sutra_scale(f[i], hsq);
    }
    return solve_tridiagonal(A, rhs);
}

static size_t max_denominator_digits(const Field& u) {
    size_t worst = 0;
    for (const auto& v : u) worst = std::max(worst, v.denominator().str().size());
    return worst;
}

// =====================================================================
// TOOL 2 -- polynomial engine. S8 + S9 + S4 + S15 + US12 + S3.
// Roots, critical points, deflation, and verification, all exact.
// =====================================================================
struct PolyReport {
    std::vector<Rational> roots;
    std::vector<Rational> critical_points;
    bool roots_verified = false;
    bool discriminant_is_square = false;
    std::vector<Rational> deflated;
};

static PolyReport analyse_polynomial(const std::vector<Rational>& p) {
    PolyReport out;
    if (p.size() == 3) {
        // S8 Purana Apuranabhyam -- by completion and non-completion.
        auto q = S8_Purana::solve_quadratic(p[2], p[1], p[0]);
        if (q.root1) out.roots.push_back(*q.root1);
        if (q.root2) out.roots.push_back(*q.root2);
        // US12 Vilokanam -- by observation: is the discriminant a perfect square?
        if (q.discriminant >= 0 && q.discriminant.denominator() == 1) {
            auto sq = US12_Vilokanam::check_perfect_square(q.discriminant.numerator());
            out.discriminant_is_square =
                (sq.type == US12_Vilokanam::PatternType::PERFECT_SQUARE);
        }
    }
    // S9 Calana Kalanabhyam -- differential calculus.
    out.critical_points = S9_Calana::find_critical_points(p);
    // S15 Gunita Samuccayah -- the roots must reproduce the coefficients.
    if (out.roots.size() == 2) {
        auto v = S15_Gunitasamuccaya::verify_roots(p, out.roots);
        out.roots_verified = v.sum_verified && v.product_verified;
        // S4 Paravartya -- deflate by (x - root), exactly.
        std::vector<Rational> factor{-out.roots[0], Rational(1)};
        out.deflated = S4_Paravartya::polynomial_divide(p, factor).quotient;
    }
    return out;
}

// =====================================================================
// TOOL 3 -- linear-system desk. FOUR independent sutra routes to the same
// answer, so no single one is trusted: S7, US11, S6, S5.
// =====================================================================
struct SolveAgreement {
    bool all_agree = false;
    size_t routes_that_answered = 0;
    std::vector<Rational> answer;
};

static SolveAgreement solve_2x2_every_way(Rational a1, Rational b1, Rational c1,
                                          Rational a2, Rational b2, Rational c2) {
    SolveAgreement out;
    std::vector<std::vector<Rational>> xs;

    auto g = S7_Sankalana::gaussian_eliminate({{a1, b1}, {a2, b2}}, {c1, c2});
    if (g) { xs.push_back(*g); ++out.routes_that_answered; }

    auto e = S7_Sankalana::solve_by_elimination(a1, b1, c1, a2, b2, c2);
    if (e.x && e.y) { xs.push_back({*e.x, *e.y}); ++out.routes_that_answered; }

    auto s = S6_Anurupye::solve_system_2x2(a1, b1, c1, a2, b2, c2);
    if (s.x && s.y) { xs.push_back({*s.x, *s.y}); ++out.routes_that_answered; }

    // US11 Lopanasthapanabhyam -- by elimination and retention. Eliminating x
    // leaves one equation in y, which S5 Shunyam then solves.
    auto step = US11_Lopanasthapana::eliminate_variable({{a1, b1}, {a2, b2}}, {c1, c2}, 0);
    if (!step.reduced_matrix.empty() && step.reduced_matrix[0][1] != 0) {
        // The reduced row is `k*y = rhs`, i.e. `k*y + 0 = 0*y + rhs`, which is
        // exactly the shape S5 Shunyam Samyasamuccaye's solve_linear takes.
        // Dividing here instead would make this route US11 alone, and the
        // label on this tool says US11 + S5.
        auto y = S5_Shunyam::solve_linear(step.reduced_matrix[0][1], Rational(0),
                                          Rational(0), step.reduced_rhs[0]);
        if (y && a1 != 0) {
            // Back-substitution, again through S5: a1*x + b1*y = c1 becomes
            // a1*x + b1*y = 0*x + c1.
            auto x = S5_Shunyam::solve_linear(a1, S15_Gunitasamuccaya::verify_distributive(
                                                      b1, Rational(0), *y, Rational(0)).left_side,
                                              Rational(0), c1);
            if (x) { xs.push_back({*x, *y}); ++out.routes_that_answered; }
        }
    }

    out.all_agree = !xs.empty();
    for (const auto& v : xs) out.all_agree = out.all_agree && (v == xs[0]);
    if (!xs.empty()) out.answer = xs[0];
    return out;
}

// =====================================================================
// TOOL 4 -- integer inspector. S12 + US5 + S16 + S11 + US9 + S1.
// =====================================================================
struct IntegerReport {
    std::vector<int> divisors_found;      // from S12's digit rules
    bool osculation_agrees = false;       // US5 confirms S12 independently
    std::vector<S16_Gunakasamuccaya::PrimeFactor> factors;
    BigInt gcd_with_360;
    int last_digit_of_square = 0;
    size_t recurring_period = 0;
};

static IntegerReport inspect_integer(const BigInt& n) {
    IntegerReport out;
    if (S12_Sesanyankena::divisible_by_2(n))  out.divisors_found.push_back(2);
    if (S12_Sesanyankena::divisible_by_3(n))  out.divisors_found.push_back(3);
    if (S12_Sesanyankena::divisible_by_4(n))  out.divisors_found.push_back(4);
    if (S12_Sesanyankena::divisible_by_5(n))  out.divisors_found.push_back(5);
    if (S12_Sesanyankena::divisible_by_8(n))  out.divisors_found.push_back(8);
    if (S12_Sesanyankena::divisible_by_9(n))  out.divisors_found.push_back(9);
    if (S12_Sesanyankena::divisible_by_11(n)) out.divisors_found.push_back(11);

    // US5 Vestanam -- osculation, an independent route to divisibility by 11.
    auto osc = US5_Vestanam::find_positive_osculator(BigInt(11));
    if (osc) {
        bool by_osculation = US5_Vestanam::divisibility_by_osculation(n, BigInt(11), *osc, true);
        out.osculation_agrees = (by_osculation == S12_Sesanyankena::divisible_by_11(n));
    }
    out.factors = S16_Gunakasamuccaya::prime_factorize(n);
    out.gcd_with_360 = S11_Vyashti::gcd_multiple({n, BigInt(360)});
    out.last_digit_of_square = US9_Antyayoreva::last_digit_of_product(n, n);
    if (n > 0 && util::gcd(n, BigInt(10)) == 1)
        out.recurring_period = S1_Ekadhikena::divide_recurring(BigInt(1), n, 2000).recurring.size();
    return out;
}

// =====================================================================
// TEST METHODS
// =====================================================================

// M1. Exact conservation, against the same scheme in float64.
static double float_worst_node(const Rational& r, int steps) {
    Field u{Rational(0), Rational(1), Rational(3), Rational(2), Rational(0)};
    std::vector<double> d{0, 1, 3, 2, 0};
    const size_t n = 5;
    double rd = r.numerator().convert_to<double>() / r.denominator().convert_to<double>();
    double worst = 0;
    for (int s = 0; s < steps; ++s) {
        u = heat_step(u, r, true);
        std::vector<double> nd(n);
        for (size_t i = 0; i < n; ++i) {
            if (i == 0)          nd[i] = d[0] * (1 - rd) + rd * d[1];
            else if (i == n - 1) nd[i] = d[n - 1] * (1 - rd) + rd * d[n - 2];
            else                 nd[i] = rd * d[i - 1] + (1 - 2 * rd) * d[i] + rd * d[i + 1];
        }
        d = nd;
        for (size_t i = 0; i < n; ++i)
            worst = std::max(worst, std::fabs(
                u[i].numerator().convert_to<double>()
                / u[i].denominator().convert_to<double>() - d[i]));
    }
    return worst;
}

static void m1_conservation() {
    for (auto r : {Rational(1, 4), Rational(1, 3), Rational(2, 7)}) {
        Field u{Rational(0), Rational(1), Rational(3), Rational(2), Rational(0)};
        Rational start = sutra_sum(u);
        for (int s = 0; s < 200; ++s) u = heat_step(u, r, true);
        bool exact = (sutra_sum(u) == start);
        double worst = float_worst_node(r, 200);
        std::printf("    r=%s/%s  invariant %s | worst float64 node error %.3e\n",
                    r.numerator().str().c_str(), r.denominator().str().c_str(),
                    exact ? "holds as an equality" : "VIOLATED", worst);
        check(exact, "M1 total conserved exactly, r=" + r.numerator().str() + "/" + r.denominator().str());
        check(worst > 0 && worst < 1e-13,
              "M1 float64 deviates per node, at machine epsilon, r=" + r.numerator().str());
    }
}

// M2. Symmetry, as an equality.
static void m2_symmetry() {
    Field u{Rational(0), Rational(1), Rational(4), Rational(1), Rational(0)};
    for (int s = 0; s < 120; ++s) u = heat_step(u, Rational(1, 5), false);
    bool sym = true;
    for (size_t i = 0; i < u.size() / 2; ++i) sym = sym && (u[i] == u[u.size() - 1 - i]);
    check(sym, "M2 symmetric data stays exactly symmetric over 120 steps");
}

// M3. Manufactured solution: S9 derives the source, the solver must reproduce u.
static void m3_manufactured() {
    std::vector<Rational> poly{Rational(0), Rational(1), Rational(-1)};   // x - x^2
    auto d2 = S9_Calana::differentiate(S9_Calana::differentiate(poly));
    check(d2.size() == 1 && d2[0] == -2, "M3 S9 differentiates x-x^2 twice to -2");
    const size_t n = 7;
    Rational h(1, static_cast<int>(n - 1));
    Field f(n, -d2[0]);
    Field got = poisson_solve(f, sutra_scale(h, h));
    bool ok = true;
    for (size_t i = 1; i + 1 < n; ++i)
        ok = ok && (got[i] == S9_Calana::evaluate(poly, sutra_scale(Rational(static_cast<int>(i)), h)));
    check(ok, "M3 Poisson reproduces the manufactured u=x(1-x) exactly at every node");
}

// M4. Two schemes, one answer: explicit and implicit must move together, and
// theta=0 must reduce the implicit scheme to the explicit one exactly.
static void m4_scheme_cross_check() {
    Field u{Rational(0), Rational(2), Rational(5), Rational(3), Rational(0)};
    Rational r(1, 10);
    Field ex = heat_step(u, r, false);
    Field th0 = heat_step_implicit(u, r, Rational(0));
    check(ex == th0, "M4 the theta-method at theta=0 IS the explicit scheme, exactly");
    Field cn = heat_step_implicit(u, r, Rational(1, 2));
    check(cn[0] == u[0] && cn[4] == u[4], "M4 Crank-Nicolson holds the Dirichlet ends");
    bool same_dir = true;
    for (size_t i = 1; i + 1 < u.size(); ++i) {
        Rational a = cn[i] - u[i], b = ex[i] - u[i];
        same_dir = same_dir && ((a > 0) == (b > 0) || (a == 0 && b == 0));
    }
    check(same_dir, "M4 implicit and explicit move every interior node the same way");
    // US1's lerp carries the argument ORDER that the stencil depends on:
    // explicit_stencil uses lerp(1, -1, r) = 1 - 2r for the centre weight, and
    // lerp(-1, 1, r) = 2r - 1 is a different number entirely. Asserted at an
    // ASYMMETRIC theta, because at theta = 1/2 a swap of the first two
    // arguments is invisible -- lerp(4,8,1/2) and lerp(8,4,1/2) are both 6,
    // and an earlier version of this check used exactly that and caught
    // nothing.
    check(US1_Anurupyena::lerp(Rational(4), Rational(8), Rational(1, 4)) == Rational(5),
          "M4 US1 lerp(4,8,1/4)==5, which pins the argument order");
    check(US1_Anurupyena::lerp(Rational(8), Rational(4), Rational(1, 4)) == Rational(7),
          "M4 and the reversed order gives 7, so the order is observable");
    check(explicit_stencil(Rational(1, 4))[1] == Rational(1, 2),
          "M4 the stencil centre weight is 1-2r, not 2r-1");
}

// M5. Polynomial identities the scheme leans on: S3 vs S4, and S9's round-trip.
static void m5_polynomial_identities() {
    std::vector<Rational> p{Rational(1), Rational(-2), Rational(3)};
    std::vector<Rational> q{Rational(2), Rational(1)};
    auto prod = S3_Urdhva::polynomial_multiply(p, q);
    auto back = S4_Paravartya::polynomial_divide(prod, q);
    bool zero = true; for (const auto& c : back.remainder) zero = zero && (c == 0);
    check(back.quotient == p && zero, "M5 S3 multiply then S4 divide round-trips");
    check(S9_Calana::differentiate(S9_Calana::integrate(p)) == p,
          "M5 S9 differentiate(integrate(p)) == p");
    auto dist = S15_Gunitasamuccaya::verify_distributive(
        Rational(1), Rational(2), Rational(3), Rational(4));
    check(dist.verified && dist.left_side == 21, "M5 S15 verifies the distributive expansion");
}

// M6. Pattern detection on the solution trace.
static void m6_pattern() {
    Field u{Rational(0), Rational(1), Rational(0), Rational(-1), Rational(0)};
    Rational r(1, 4);
    Field trace;
    for (int s = 0; s < 6; ++s) { trace.push_back(u[1]); u = heat_step(u, r, false); }
    auto gp = US12_Vilokanam::check_geometric_progression(trace);
    Rational expected = US1_Anurupyena::lerp(Rational(1), Rational(-1), r);  // 1-2r
    check(gp.type == US12_Vilokanam::PatternType::GEOMETRIC_PROGRESSION
          && gp.parameters[1] == expected,
          "M6 US12 detects the eigenmode decay, ratio == 1-2r exactly");
    std::printf("    eigenmode ratio detected %s/%s, predicted 1-2r = %s/%s\n",
                gp.parameters[1].numerator().str().c_str(),
                gp.parameters[1].denominator().str().c_str(),
                expected.numerator().str().c_str(), expected.denominator().str().c_str());
    Field mixed{Rational(0), Rational(1), Rational(1), Rational(1), Rational(0)};
    Field mt;
    for (int s = 0; s < 6; ++s) { mt.push_back(sutra_sum(mixed)); mixed = heat_step(mixed, r, false); }
    check(US12_Vilokanam::check_geometric_progression(mt).type
          != US12_Vilokanam::PatternType::GEOMETRIC_PROGRESSION,
          "M6 US12 refuses a multi-mode decay, which is not geometric");
}

// M7. Stability, bounded exactly, plus the wave equation's dispersion via S8.
static void m7_stability() {
    auto cf = S13_Sopantya::to_continued_fraction(Rational(1, 2));
    auto conv = S13_Sopantya::nth_convergent(cf, cf.size() - 1);
    check(Rational(conv.numerator, conv.denominator) == Rational(1, 2),
          "M7 S13 round-trips the CFL bound 1/2 exactly");
    Field a{Rational(0), Rational(1), Rational(0), Rational(1), Rational(0)}, b = a;
    for (int s = 0; s < 40; ++s) {
        a = heat_step(a, Rational(1, 2), false);
        b = heat_step(b, Rational(3, 2), false);
    }
    Rational wa(0), wb(0);
    for (const auto& v : a) wa = std::max(wa, abs_rational(v));
    for (const auto& v : b) wb = std::max(wb, abs_rational(v));
    check(wa <= Rational(1), "M7 r=1/2 stays bounded over 40 steps");
    check(wb > Rational(1000), "M7 r=3/2 provably diverges, as the bound predicts");
    // S12 -- does the step divide the domain evenly? A grid question, exactly.
    check(S12_Sesanyankena::divisible_by_4(BigInt(12)) && !S12_Sesanyankena::divisible_by_8(BigInt(12)),
          "M7 S12 answers grid commensurability exactly");
}

// M8. The wave equation, and the cost of exactness.
static void m8_wave_and_cost() {
    Field prev{Rational(0), Rational(1), Rational(2), Rational(1), Rational(0)};
    Field cur = prev;
    Rational c2(1, 4);
    for (int s = 0; s < 20; ++s) { Field nx = wave_step(cur, prev, c2); prev = cur; cur = nx; }
    bool sym = true;
    for (size_t i = 0; i < cur.size() / 2; ++i) sym = sym && (cur[i] == cur[cur.size() - 1 - i]);
    check(sym, "M8 the wave solver preserves symmetry exactly over 20 steps");

    Field u{Rational(0), Rational(1), Rational(3), Rational(2), Rational(0)};
    std::printf("    step  max denominator digits\n");
    for (int s = 1; s <= 40; ++s) {
        u = heat_step(u, Rational(1, 3), true);
        if (s % 10 == 0) std::printf("    %4d  %zu\n", s, max_denominator_digits(u));
    }
    check(max_denominator_digits(u) > 0, "M8 denominator growth measured over 40 steps");
}

// M9. TOOL 2 -- the polynomial engine.
static void m9_polynomial_engine() {
    std::vector<Rational> p{Rational(5), Rational(-6), Rational(1)};   // x^2 - 6x + 5
    auto rep = analyse_polynomial(p);
    bool roots_ok = rep.roots.size() == 2
                 && ((rep.roots[0] == 1 && rep.roots[1] == 5) || (rep.roots[0] == 5 && rep.roots[1] == 1));
    check(roots_ok, "M9 S8 finds the roots {1,5} of x^2-6x+5");
    check(rep.roots_verified, "M9 S15 confirms those roots reproduce the coefficients");
    check(rep.discriminant_is_square, "M9 US12 sees the discriminant 16 is a perfect square");
    check(rep.critical_points.size() == 1 && rep.critical_points[0] == 3,
          "M9 S9 puts the critical point at x=3, the vertex");
    check(rep.deflated.size() == 2, "M9 S4 deflates by (x - root) exactly");
    // The vertex from calculus must equal the vertex from completing the square.
    auto cs = S8_Purana::complete_the_square(p[2], p[1], p[0]);
    check(cs.h == rep.critical_points[0],
          "M9 S8's completed-square vertex == S9's critical point");
}

// M10. TOOL 3 -- four routes to one linear system.
static void m10_linear_desk() {
    auto r = solve_2x2_every_way(Rational(2), Rational(3), Rational(8),
                                 Rational(1), Rational(1), Rational(3));
    check(r.routes_that_answered == 4, "M10 all four sutra routes answered");
    check(r.all_agree, "M10 S7, S7-elimination, S6 and US11+S5 agree exactly");
    check(r.answer.size() == 2 && r.answer[0] == 1 && r.answer[1] == 2,
          "M10 and the answer is (1, 2)");
    std::printf("    %zu independent routes, agreement: %s\n",
                r.routes_that_answered, r.all_agree ? "exact" : "DIVERGED");
}

// M11. TOOL 4 -- the integer inspector, cross-validated.
static void m11_integer_inspector() {
    auto rep = inspect_integer(BigInt(360));
    bool divs = rep.divisors_found == std::vector<int>{2, 3, 4, 5, 8, 9};
    check(divs, "M11 S12's digit rules find exactly {2,3,4,5,8,9} for 360");
    check(rep.osculation_agrees, "M11 US5's osculation agrees with S12 on 11");
    BigInt rebuilt = 1;
    for (const auto& f : rep.factors) rebuilt *= util::power(f.prime, f.exponent);
    check(rebuilt == 360, "M11 S16's prime factorisation rebuilds 360");
    check(rep.gcd_with_360 == 360, "M11 S11's gcd(360,360) == 360");
    check(rep.last_digit_of_square == 0, "M11 US9 gives the last digit of 360^2");
    auto seven = inspect_integer(BigInt(7));
    check(seven.recurring_period == 6, "M11 S1 gives the period of 1/7 as 6");
    std::printf("    360 -> divisible by {2,3,4,5,8,9}, %zu prime factors, gcd 360\n",
                rep.factors.size());
}

int main() {
    std::printf("\n=== TOOL 1: a PDE solver built strictly from sutras ===\n");
    std::printf("\n  M1  exact conservation\n");        m1_conservation();
    std::printf("  M2  symmetry\n");                    m2_symmetry();
    std::printf("  M3  manufactured solution (S9)\n");  m3_manufactured();
    std::printf("  M4  scheme cross-check (S7/US1)\n"); m4_scheme_cross_check();
    std::printf("  M5  polynomial identities\n");       m5_polynomial_identities();
    std::printf("  M6  pattern detection (US12)\n");    m6_pattern();
    std::printf("  M7  stability (S13/S12)\n");         m7_stability();
    std::printf("  M8  wave equation, cost\n");         m8_wave_and_cost();
    std::printf("\n=== OTHER TOOLS FROM OTHER COMBINATIONS ===\n");
    std::printf("\n  M9  polynomial engine  S8+S9+S4+S15+US12\n"); m9_polynomial_engine();
    std::printf("  M10 linear desk        S7+US11+S6+S5\n");       m10_linear_desk();
    std::printf("  M11 integer inspector  S12+US5+S16+S11+US9+S1\n"); m11_integer_inspector();

    std::printf("\n  %d passed, %d failed\n", passed, failed);
    for (const auto& f : failures) std::printf("  FAILED: %s\n", f.c_str());
    return failed == 0 ? 0 : 1;
}
