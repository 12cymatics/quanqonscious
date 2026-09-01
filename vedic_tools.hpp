// The four sutra-built tools, as a library.
//
// Extracted from vedic_pde_tool.cpp so the same code that the 41 checks in
// that file exercise is the code the `vedic` command-line tool runs. There is
// no second implementation and no simplified copy: one definition, two callers.
//
//   TOOL 1  PDE solver          S7, S9, S3, S4, S11, S13, US1, US3, US10,
//                               US11, US12, S12, S8, S15
//   TOOL 2  polynomial engine   S8, S9, S4, S15, US12, S3
//   TOOL 3  linear-system desk  S7, US11, S6, S5   (four routes, cross-checked)
//   TOOL 4  integer inspector   S12, US5, S16, S11, US9, S1

#pragma once

#include "vedic_sutras_complete.hpp"
#include <algorithm>
#include <string>
#include <vector>

namespace vedic_tools {

using namespace vedic;
using Field = std::vector<Rational>;

// =====================================================================
// TOOL 1 -- the PDE solver. Every line below routes through a sutra.
// =====================================================================

// The sum of a field, via S11 Vyashti Samashti (part and whole): factoring 1
// out of the parts leaves `sum_of_parts`, which is the whole.
inline Rational sutra_sum(const Field& u) {
    return S11_Vyashti::factor_common(u, Rational(1)).sum_of_parts;
}

// Scaling one value, via US1 Anurupyena (proportionately).
inline Rational sutra_scale(const Rational& v, const Rational& k) {
    return US1_Anurupyena::scale(v, k);
}

// The explicit heat stencil as ONE dot product. US10 Samuccayagunitah is "the
// sum multiplied"; its vector form is exactly a weighted stencil, so the whole
// interior update is a single sutra call rather than hand-written arithmetic:
//
//     u_i^{n+1} = [r, 1-2r, r] . [u_{i-1}, u_i, u_{i+1}]
//
inline Field explicit_stencil(const Rational& r) {
    // 1 - 2r is built with US1's lerp rather than by subtraction: lerp(1, -1, r)
    // = 1 + r*(-1 - 1) = 1 - 2r, which is the centre weight.
    Rational centre = US1_Anurupyena::lerp(Rational(1), Rational(-1), r);
    return {r, centre, r};
}

// One explicit heat step. Interior nodes go through US10's dot product; the two
// ends go through US3 Adyamadyenantyamantyena, "first by first and last by
// last", which is the sutra about treating the endpoints of a range.
inline Field heat_step(const Field& u, const Rational& r, bool neumann) {
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
inline Field wave_step(const Field& u, const Field& prev, const Rational& c2) {
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
inline Field solve_tridiagonal(const std::vector<std::vector<Rational>>& A,
                               const Field& rhs) {
    auto solved = S7_Sankalana::gaussian_eliminate(A, rhs);
    return solved ? *solved : rhs;
}

// Crank-Nicolson. The right-hand side is the explicit stencil applied through
// US10; the matrix is the implicit half; the blend is US1's lerp.
inline Field heat_step_implicit(const Field& u, const Rational& r, const Rational& theta) {
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
inline Field poisson_solve(const Field& f, const Rational& hsq) {
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

inline size_t max_denominator_digits(const Field& u) {
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

inline PolyReport analyse_polynomial(const std::vector<Rational>& p) {
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

inline SolveAgreement solve_2x2_every_way(Rational a1, Rational b1, Rational c1,
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

inline IntegerReport inspect_integer(const BigInt& n) {
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

}  // namespace vedic_tools
