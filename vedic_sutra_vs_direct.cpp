// Sutra vs non-sutra: speed and primitive-operation efficiency.
//
// Both arms compute the SAME value on the SAME type (boost cpp_int), and every
// pair is verified equal before any timing is reported. A ratio between two
// arms that disagree is meaningless, which is how the speedups this repository
// used to publish came to be wrong in the same direction every time.
//
// The efficiency half counts single-digit multiplications, which is the claim
// the Vedic literature actually makes -- not wall clock. The counting versions
// are checked against the header's own output on every input, so a count is
// only reported for an algorithm demonstrably computing the right answer.

#include "vedic_sutras_complete.hpp"
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace vedic;
using Clock = std::chrono::steady_clock;

static volatile unsigned long long SINK = 0;
static void sink(const BigInt& v) { SINK += static_cast<unsigned long long>(v % 1000003); }

template <typename F>
static double bench(F f, int iters) {
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) f();
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
}

// ---------------------------------------------------------------- efficiency

// Single-digit multiplications used by schoolbook (long) multiplication.
static BigInt schoolbook_counted(const BigInt& a, const BigInt& b, long& digit_mults) {
    std::vector<int> da = util::to_digits(a), db = util::to_digits(b);
    std::vector<int> acc(da.size() + db.size(), 0);
    for (size_t i = 0; i < da.size(); ++i)
        for (size_t j = 0; j < db.size(); ++j) {
            acc[i + j + 1] += da[i] * db[j];   // one single-digit multiplication
            ++digit_mults;
        }
    for (size_t k = acc.size(); k-- > 1;) { acc[k - 1] += acc[k] / 10; acc[k] %= 10; }
    return util::from_digits(acc);
}

// Urdhva Tiryagbhyam: the same partial products, gathered by column instead of
// by row. The count is therefore identical -- the sutra reorders the work, it
// does not remove any of it.
static BigInt urdhva_counted(const BigInt& a, const BigInt& b, long& digit_mults) {
    std::vector<int> da = util::to_digits(a), db = util::to_digits(b);
    size_t n = da.size(), m = db.size();
    std::vector<BigInt> col(n + m, 0);
    for (size_t s = 0; s < n + m - 1; ++s)
        for (size_t i = 0; i < n; ++i) {
            size_t j = s - i;
            if (i <= s && j < m) { col[s] += BigInt(da[i]) * db[j]; ++digit_mults; }
        }
    BigInt out = 0, place = 1;
    for (size_t s = n + m - 1; s-- > 0;) { out += col[s] * place; place *= 10; }
    return out;
}

// Nikhilam: when both operands sit near a common base the whole product is one
// small multiplication of the two deficiencies plus one cross addition.
static BigInt nikhilam_counted(const BigInt& a, const BigInt& b, const BigInt& base,
                               long& digit_mults) {
    BigInt da = base - a, db = base - b;
    long inner = 0;
    BigInt right = schoolbook_counted(da < 0 ? -da : da, db < 0 ? -db : db, inner);
    digit_mults += inner;                       // only the deficiencies are multiplied
    if ((da < 0) != (db < 0)) right = -right;
    return (a - db) * base + right;             // scaling by the base is a shift
}

// ---------------------------------------------------------------------- main

static void row(const char* name, double sut, double dir, bool ok) {
    std::printf("  %-34s %10.1f %10.1f   %7.2fx   %s\n", name, sut, dir,
                dir > 0 ? sut / dir : 0.0, ok ? "match" : "MISMATCH");
}

int main() {
    const int IT = 20000;
    std::printf("\n=== SPEED: sutra vs direct, same type, verified equal ===\n");
    std::printf("  %-34s %10s %10s   %8s\n", "operation", "sutra ns", "direct ns", "ratio");

    {   // Nikhilam, both operands near 10^4 -- the sutra's own best case.
        BigInt a("9998"), b("9997"), base("10000");
        bool ok = S2_Nikhilam::multiply(a, b, base).product == a * b;
        double s = bench([&]{ sink(S2_Nikhilam::multiply(a, b, base).product); }, IT);
        double d = bench([&]{ sink(a * b); }, IT);
        row("Nikhilam multiply (near base)", s, d, ok);
    }
    {   // Urdhva, general multiplication.
        BigInt a("847362514"), b("293847561");
        bool ok = S3_Urdhva::multiply(a, b).product == a * b;
        double s = bench([&]{ sink(S3_Urdhva::multiply(a, b).product); }, IT);
        double d = bench([&]{ sink(a * b); }, IT);
        row("Urdhva multiply (9x9 digits)", s, d, ok);
    }
    {   // Yavadunam squaring near a base.
        BigInt n("9997");
        bool ok = S10_Yavadunam::square(n).square == n * n;
        double s = bench([&]{ sink(S10_Yavadunam::square(n).square); }, IT);
        double d = bench([&]{ sink(n * n); }, IT);
        row("Yavadunam square (near base)", s, d, ok);
    }
    {   // Ekanyunena: multiply by a string of nines.
        BigInt n("4567"); size_t k = 4;
        BigInt nines = util::power(BigInt(10), k) - 1;
        bool ok = S14_Ekanyunena::multiply_by_nines(n, k).product == n * nines;
        double s = bench([&]{ sink(S14_Ekanyunena::multiply_by_nines(n, k).product); }, IT);
        double d = bench([&]{ sink(n * nines); }, IT);
        row("Ekanyunena multiply by nines", s, d, ok);
    }
    {   // Paravartya integer division.
        BigInt p("987654321"), q("1234");
        auto r = S4_Paravartya::divide(p, q);
        bool ok = (r.quotient == p / q) && (r.remainder == p % q);
        double s = bench([&]{ sink(S4_Paravartya::divide(p, q).quotient); }, IT);
        double d = bench([&]{ sink(p / q); }, IT);
        row("Paravartya divide", s, d, ok);
    }

    std::printf("\n=== EFFICIENCY: single-digit multiplications ===\n");
    std::printf("  %-34s %10s %10s   %8s\n", "operation", "sutra", "schoolbook", "ratio");
    struct Case { const char* name; const char* a; const char* b; const char* base; };
    Case cases[] = {
        {"2x2 digits, near base 100",     "98",         "97",         "100"},
        {"4x4 digits, near base 10000",   "9998",       "9997",       "10000"},
        {"4x4 digits, NOT near a base",   "4736",       "2938",       ""},
        {"9x9 digits, general",           "847362514",  "293847561",  ""},
    };
    for (const auto& c : cases) {
        BigInt a(c.a), b(c.b);
        long school = 0, urdhva = 0;
        BigInt r_school = schoolbook_counted(a, b, school);
        BigInt r_urdhva = urdhva_counted(a, b, urdhva);
        bool ok = (r_school == a * b) && (r_urdhva == a * b);
        if (c.base[0]) {
            long nik = 0;
            BigInt r_nik = nikhilam_counted(a, b, BigInt(c.base), nik);
            ok = ok && (r_nik == a * b);
            std::printf("  %-34s %10ld %10ld   %7.2fx   %s   (Urdhva %ld)\n",
                        c.name, nik, school, school ? double(nik) / school : 0.0,
                        ok ? "match" : "MISMATCH", urdhva);
        } else {
            std::printf("  %-34s %10ld %10ld   %7.2fx   %s\n",
                        c.name, urdhva, school, school ? double(urdhva) / school : 0.0,
                        ok ? "match" : "MISMATCH");
        }
    }

    std::printf("\n  dead-code sink: %llu (%zu digits, nonzero=%d)\n",
                SINK, std::to_string(SINK).size(), SINK != 0 ? 1 : 0);
    auto results = tests::run_all_tests();
    size_t passed = 0;
    for (const auto& r : results) if (r.passed) ++passed;
    std::printf("\n  header self-tests: %zu/%zu pass\n", passed, results.size());
    return 0;
}
