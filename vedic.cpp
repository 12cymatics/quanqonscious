// `vedic` -- the command line for the four sutra-built tools.
//
// Everything here is exact. Inputs are integers or ratios written a/b; there
// is no float path in, through, or out, so a result printed as 1/3 is 1/3 and
// not 0.333333. Where a method has no exact answer it says so and stops
// rather than returning a rounded stand-in.
//
//   vedic heat     --init 0,0,1,0,0 --r 1/4 --steps 20 [--neumann] [--theta 1/2]
//   vedic wave     --init 0,0,1,0,0 --c2 1/4 --steps 20
//   vedic poisson  --f 1,1,1,1,1 --h 1/4
//   vedic poly     --coeffs 6,-5,1
//   vedic solve    --system 2,3,8,5,-1,3
//   vedic int      --n 360
//
// The tool bodies live in vedic_tools.hpp and are the same definitions the 41
// checks in vedic_pde_tool.cpp exercise.

#include "vedic_tools.hpp"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>

using namespace vedic;
using namespace vedic_tools;

// ------------------------------------------------------------------ parsing
static Rational parse_rational(const std::string& s) {
    if (s.empty()) throw std::runtime_error("empty number");
    if (s.find('.') != std::string::npos)
        throw std::runtime_error(
            "'" + s + "' is a decimal. This tool is exact; write it as a ratio "
            "(0.25 is 1/4). Refusing rather than rounding.");
    size_t slash = s.find('/');
    if (slash == std::string::npos) return Rational(BigInt(s));
    BigInt num(s.substr(0, slash)), den(s.substr(slash + 1));
    if (den == 0) throw std::runtime_error("'" + s + "' has a zero denominator");
    return Rational(num, den);
}

static Field parse_field(const std::string& s) {
    Field out;
    size_t i = 0;
    while (i <= s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        std::string tok = s.substr(i, j - i);
        if (!tok.empty()) out.push_back(parse_rational(tok));
        if (j == s.size()) break;
        i = j + 1;
    }
    if (out.empty()) throw std::runtime_error("no values parsed from '" + s + "'");
    return out;
}

static std::string show(const Rational& r) {
    std::string n = r.numerator().str();
    if (r.denominator() == 1) return n;
    return n + "/" + r.denominator().str();
}

static void print_field(const Field& u, const char* label) {
    std::printf("  %-8s", label);
    for (size_t i = 0; i < u.size(); ++i)
        std::printf("%s%s", show(u[i]).c_str(), i + 1 < u.size() ? "  " : "");
    std::printf("\n");
}

// ------------------------------------------------------------------- flags
struct Args {
    std::map<std::string, std::string> kv;
    bool has(const char* k) const { return kv.count(k) != 0; }
    std::string get(const char* k, const char* fallback = nullptr) const {
        auto it = kv.find(k);
        if (it != kv.end()) return it->second;
        if (fallback) return fallback;
        throw std::runtime_error(std::string("missing required flag --") + k);
    }
};

static Args parse_args(int argc, char** argv, int from) {
    Args a;
    for (int i = from; i < argc; ++i) {
        std::string t = argv[i];
        if (t.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument '" + t + "'");
        std::string key = t.substr(2);
        if (key == "neumann") { a.kv[key] = "1"; continue; }
        if (i + 1 >= argc) throw std::runtime_error("--" + key + " needs a value");
        a.kv[key] = argv[++i];
    }
    return a;
}

// -------------------------------------------------------------- TOOL 1: PDE
static int cmd_heat(const Args& a) {
    Field u = parse_field(a.get("init"));
    Rational r = parse_rational(a.get("r"));
    long steps = std::stol(a.get("steps", "1"));
    bool neumann = a.has("neumann");
    Rational theta = parse_rational(a.get("theta", "0"));

    if (u.size() < 3) throw std::runtime_error("--init needs at least 3 nodes");
    if (steps < 0) throw std::runtime_error("--steps must not be negative");

    std::printf("HEAT  u_t = u_xx   r = %s   %ld steps   %s   theta = %s\n",
                show(r).c_str(), steps,
                neumann ? "Neumann (zero flux)" : "Dirichlet (ends held)",
                show(theta).c_str());
    std::printf("      via US10 dot-product stencil, US1 lerp weights, US3 endpoints%s\n\n",
                theta == 0 ? "" : ", S7 elimination for the implicit half");

    // The explicit scheme is unstable above r = 1/2 -- say so, do not silently
    // clamp r or switch scheme.
    if (theta == 0 && r > Rational(1, 2))
        std::printf("  ! r > 1/2: the explicit scheme is unstable here. Running it anyway,\n"
                    "    exactly, so you can watch it diverge. --theta 1/2 is unconditionally stable.\n\n");

    Rational total0 = sutra_sum(u);
    print_field(u, "t=0");
    for (long k = 1; k <= steps; ++k) {
        u = (theta == 0) ? heat_step(u, r, neumann) : heat_step_implicit(u, r, theta);
        if (steps <= 12 || k == steps) {
            char lab[32]; std::snprintf(lab, sizeof lab, "t=%ld", k);
            print_field(u, lab);
        }
    }
    Rational total1 = sutra_sum(u);
    std::printf("\n  total    %s -> %s   (%s)\n", show(total0).c_str(), show(total1).c_str(),
                total0 == total1 ? "conserved exactly"
                                 : "not conserved -- expected under Dirichlet ends");
    std::printf("  widest denominator: %zu digits\n", max_denominator_digits(u));
    return 0;
}

static int cmd_wave(const Args& a) {
    Field u = parse_field(a.get("init"));
    Rational c2 = parse_rational(a.get("c2"));
    long steps = std::stol(a.get("steps", "1"));
    if (u.size() < 3) throw std::runtime_error("--init needs at least 3 nodes");

    std::printf("WAVE  u_tt = c^2 u_xx   c^2 = %s   %ld steps\n", show(c2).c_str(), steps);
    std::printf("      via US10 dot product over [u_{i-1}, u_i, u_{i+1}, u^{n-1}_i]\n\n");

    Field prev = u;
    print_field(u, "t=0");
    for (long k = 1; k <= steps; ++k) {
        Field next = wave_step(u, prev, c2);
        prev = u; u = next;
        if (steps <= 12 || k == steps) {
            char lab[32]; std::snprintf(lab, sizeof lab, "t=%ld", k);
            print_field(u, lab);
        }
    }
    std::printf("\n  widest denominator: %zu digits\n", max_denominator_digits(u));
    return 0;
}

static int cmd_poisson(const Args& a) {
    Field f = parse_field(a.get("f"));
    Rational h = parse_rational(a.get("h"));
    if (f.size() < 3) throw std::runtime_error("--f needs at least 3 nodes");

    Rational hsq = US10_Samuccayagunitah::dot_product({h}, {h});   // h^2, via US10
    std::printf("POISSON  -u'' = f   h = %s   h^2 = %s   %zu nodes\n",
                show(h).c_str(), show(hsq).c_str(), f.size());
    std::printf("         via S7 Sankalana elimination on the tridiagonal system\n\n");

    Field u = poisson_solve(f, hsq);
    print_field(f, "f");
    print_field(u, "u");
    std::printf("\n  ends held at 0; interior solved exactly, no iteration and no tolerance\n");
    return 0;
}

// ------------------------------------------------------- TOOL 2: polynomial
static int cmd_poly(const Args& a) {
    Field p = parse_field(a.get("coeffs"));
    std::printf("POLYNOMIAL  coefficients low to high: ");
    for (size_t i = 0; i < p.size(); ++i) std::printf("%s ", show(p[i]).c_str());
    std::printf("\n            via S8 roots, S9 derivative, S15 verification, S4 deflation, US12 observation\n\n");

    PolyReport rep = analyse_polynomial(p);

    if (p.size() == 3) {
        if (rep.roots.size() == 2) {
            std::printf("  roots            %s, %s\n",
                        show(rep.roots[0]).c_str(), show(rep.roots[1]).c_str());
            std::printf("  discriminant     %s\n",
                        rep.discriminant_is_square ? "a perfect square (US12 by observation)"
                                                   : "not a perfect square");
            std::printf("  S15 cross-check  %s\n",
                        rep.roots_verified ? "roots reproduce both coefficients"
                                           : "MISMATCH -- do not trust these roots");
            std::printf("  deflated by (x - %s)  ", show(rep.roots[0]).c_str());
            for (const auto& c : rep.deflated) std::printf("%s ", show(c).c_str());
            std::printf("\n");
        } else {
            std::printf("  roots            REFUSED -- the discriminant is not a rational square,\n"
                        "                   so the roots are irrational and cannot be written exactly.\n"
                        "                   No decimal is offered in their place.\n");
        }
    } else {
        std::printf("  roots            S8 handles quadratics; this is degree %zu\n", p.size() - 1);
    }
    std::printf("  critical points  ");
    if (rep.critical_points.empty()) std::printf("none rational\n");
    else { for (const auto& c : rep.critical_points) std::printf("%s ", show(c).c_str()); std::printf("\n"); }

    auto d = S9_Calana::differentiate(p);
    std::printf("  derivative       ");
    for (const auto& c : d) std::printf("%s ", show(c).c_str());
    std::printf("\n");
    return 0;
}

// ------------------------------------------------------ TOOL 3: linear desk
static int cmd_solve(const Args& a) {
    Field s = parse_field(a.get("system"));
    if (s.size() != 6) throw std::runtime_error("--system needs exactly 6 values: a1,b1,c1,a2,b2,c2");

    std::printf("LINEAR DESK   %s x + %s y = %s\n              %s x + %s y = %s\n",
                show(s[0]).c_str(), show(s[1]).c_str(), show(s[2]).c_str(),
                show(s[3]).c_str(), show(s[4]).c_str(), show(s[5]).c_str());
    std::printf("              four independent sutra routes: S7 elimination, S7 by addition\n"
                "              and subtraction, S6 ratio, US11 retention + S5 zero-sum\n\n");

    SolveAgreement r = solve_2x2_every_way(s[0], s[1], s[2], s[3], s[4], s[5]);
    std::printf("  routes that answered   %zu of 4\n", r.routes_that_answered);
    if (r.answer.empty()) {
        Rational det = s[0] * s[4] - s[3] * s[1];
        std::printf("  determinant            %s\n", show(det).c_str());
        std::printf("  answer                 REFUSED -- the system is singular. No route\n"
                    "                         approximates it; a least-squares fit here would\n"
                    "                         be a different question's answer.\n");
        return 2;
    }
    std::printf("  agreement              %s\n",
                r.all_agree ? "all routes identical" : "ROUTES DISAGREE -- do not trust this");
    std::printf("  x = %s\n  y = %s\n", show(r.answer[0]).c_str(), show(r.answer[1]).c_str());

    Rational e1 = s[0] * r.answer[0] + s[1] * r.answer[1];
    Rational e2 = s[3] * r.answer[0] + s[4] * r.answer[1];
    std::printf("  substituted back       %s = %s, %s = %s  (%s)\n",
                show(e1).c_str(), show(s[2]).c_str(), show(e2).c_str(), show(s[5]).c_str(),
                (e1 == s[2] && e2 == s[5]) ? "exact" : "MISMATCH");
    return r.all_agree ? 0 : 3;
}

// -------------------------------------------------- TOOL 4: integer inspector
static int cmd_int(const Args& a) {
    BigInt n(a.get("n"));
    std::printf("INTEGER  %s\n", n.str().c_str());
    std::printf("         via S12 digit rules, US5 osculation, S16 factorisation,\n"
                "         S11 gcd, US9 last digit, S1 recurring period\n\n");

    IntegerReport rep = inspect_integer(n);

    std::printf("  divisible by     ");
    if (rep.divisors_found.empty()) std::printf("none of {2,3,4,5,8,9,11}");
    else for (size_t i = 0; i < rep.divisors_found.size(); ++i)
        std::printf("%d%s", rep.divisors_found[i], i + 1 < rep.divisors_found.size() ? ", " : "");
    std::printf("\n");

    std::printf("  US5 osculation   %s S12 on divisibility by 11\n",
                rep.osculation_agrees ? "agrees with" : "DISAGREES with");

    std::printf("  prime factors    ");
    if (rep.factors.empty()) std::printf("(none -- n is 0 or 1)");
    else for (size_t i = 0; i < rep.factors.size(); ++i)
        std::printf("%s^%zu%s", rep.factors[i].prime.str().c_str(), rep.factors[i].exponent,
                    i + 1 < rep.factors.size() ? " * " : "");
    std::printf("\n");

    std::printf("  gcd with 360     %s\n", rep.gcd_with_360.str().c_str());
    std::printf("  last digit of n^2  %d\n", rep.last_digit_of_square);
    if (rep.recurring_period)
        std::printf("  1/n recurs with a period of %zu digits\n", rep.recurring_period);
    else
        std::printf("  1/n terminates or n shares a factor with 10 -- no pure recurring period\n");
    return 0;
}

// -------------------------------------------------------------------- main
static void usage() {
    std::printf(
"vedic -- four tools built strictly from the Vedic sutras. Exact rationals only.\n"
"\n"
"  vedic heat     --init 0,0,1,0,0 --r 1/4 --steps 20 [--neumann] [--theta 1/2]\n"
"  vedic wave     --init 0,0,1,0,0 --c2 1/4 --steps 20\n"
"  vedic poisson  --f 0,1,1,1,0 --h 1/4\n"
"  vedic poly     --coeffs 6,-5,1                 (low to high: 6 - 5x + x^2)\n"
"  vedic solve    --system 2,3,8,5,-1,3           (a1,b1,c1,a2,b2,c2)\n"
"  vedic int      --n 360\n"
"\n"
"Numbers are integers or ratios written a/b. Decimals are refused, not rounded.\n");
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string cmd = argv[1];
    if (cmd == "-h" || cmd == "--help" || cmd == "help") { usage(); return 0; }
    try {
        Args a = parse_args(argc, argv, 2);
        if (cmd == "heat")    return cmd_heat(a);
        if (cmd == "wave")    return cmd_wave(a);
        if (cmd == "poisson") return cmd_poisson(a);
        if (cmd == "poly")    return cmd_poly(a);
        if (cmd == "solve")   return cmd_solve(a);
        if (cmd == "int")     return cmd_int(a);
        std::fprintf(stderr, "unknown command '%s'\n\n", cmd.c_str());
        usage();
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "vedic: %s\n", e.what());
        return 1;
    }
}
