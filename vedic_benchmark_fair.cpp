// Fair head-to-head: each Vedic routine vs the standard BigInt operator.
//
// Why this exists
// ---------------
// VEDIC_SUTRAS_AUTHENTIC_COMPLETE.md published a "Speedup vs. Standard" table
// -- S2 2x, S3 1-1.5x, S10 3x, S14 4x, US8 5x -- and nothing measured it.
// Every entry is wrong, and wrong in the same direction: on this header's own
// implementations these routines are 3x to 170x SLOWER than the operator they
// are compared against, and for S3 the gap widens with operand size rather
// than closing (~2,600x at 256 digits).
//
// The repository already had vedic_benchmark.cpp, which timed S2 and printed
// both numbers without forming a ratio -- 74 ns/op against 10 ns/op, visible
// on screen and never read. It also guarded only the STANDARD arm with
// `volatile` while leaving the Vedic arm's result unconsumed, so the
// optimiser was free to delete work on one side of the comparison only. That
// bias favoured the Vedic arm, and it still lost.
//
// Here both arms accumulate into the same sink, so neither can be eliminated,
// and both pay the same accumulate cost.
//
// What this does NOT show
// -----------------------
// The classical claim is about digit operations done by a person, and the same
// document states it plainly two pages earlier: "~50% fewer digit operations
// vs. standard multiplication -- Practical for manual calculation." That may
// well hold. What does not hold is the unqualified reading of the table as a
// claim about software, which is how it reads sitting under a heading called
// "Performance Characteristics" in a repository of code.
//
// These implementations also return heavyweight structs (NikhilamResult
// carries six BigInts; UrdhvaResult carries a cross_products vector), so a
// speed-tuned version would beat these numbers. It would not close a 2,600x
// gap: a quadratic digit loop against hardware limb multiplication with
// Karatsuba above it is the dominant fact, not struct overhead.
//
//   g++ -std=c++17 -O2 -I. -o vedic_benchmark_fair vedic_benchmark_fair.cpp
#include "vedic_sutras_complete.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace vedic;
using namespace std::chrono;

static BigInt SINK = 0;

template <typename F>
double bench(F f, int iters) {
    for (int i = 0; i < iters / 10; ++i) SINK += f();       // warm up
    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) SINK += f();
    auto t1 = high_resolution_clock::now();
    return duration_cast<nanoseconds>(t1 - t0).count() / double(iters);
}

static void row(const char* name, const char* claim, double v, double s) {
    std::cout << std::left << std::setw(26) << name
              << std::setw(10) << claim
              << std::right << std::setw(9) << std::fixed << std::setprecision(1) << v
              << std::setw(11) << s
              << std::setw(12) << std::setprecision(2) << (s / v) << "x\n";
}

int main() {
    const int N = 200000;
    std::cout << std::left << std::setw(26) << "sutra" << std::setw(10) << "claimed"
              << std::right << std::setw(9) << "vedic" << std::setw(11) << "standard"
              << std::setw(13) << "measured\n";
    std::cout << std::string(69, '-') << "\n";

    { BigInt a(9998), b(9997), base(10000);
      double v = bench([&]{ return S2_Nikhilam::multiply(a,b,base).product; }, N);
      double s = bench([&]{ return a * b; }, N);
      row("S2 Nikhilam", "2x", v, s); }

    { BigInt a(123456), b(654321);
      double v = bench([&]{ return S3_Urdhva::multiply(a,b).product; }, N);
      double s = bench([&]{ return a * b; }, N);
      row("S3 Urdhva", "1-1.5x", v, s); }

    { BigInt n(9998), base(10000);
      double v = bench([&]{ return S10_Yavadunam::square(n,base).square; }, N);
      double s = bench([&]{ return n * n; }, N);
      row("S10 Yavadunam (square)", "3x", v, s); }

    { BigInt n(456);
      double v = bench([&]{ return S14_Ekanyunena::multiply_by_nines(n,3).product; }, N);
      BigInt nines(999);
      double s = bench([&]{ return n * nines; }, N);
      row("S14 Ekanyunena", "4x", v, s); }

    { BigInt a(43), b(47);
      double v = bench([&]{ return US8_Antyayor::multiply_sum_to_ten(a,b).product; }, N);
      double s = bench([&]{ return a * b; }, N);
      row("US8 Antyayor", "5x", v, s); }

    std::cout << "\n(measured = standard/vedic; >1 means Vedic is faster)\n";
    std::cout << "sink " << (SINK % 1000) << "\n";
}
