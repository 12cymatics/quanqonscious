#include "dyadic128.hpp"

static const int CAP = 8192;
static D128 A[CAP], B[CAP], O[CAP];
static i32 C[CAP];

extern "C" {
D128 *bufA(){ return A; }
D128 *bufB(){ return B; }
D128 *bufO(){ return O; }
i32 *bufC(){ return C; }
int capacity(){ return CAP; }
void opMul(int n){ for (int i = 0; i < n; i++) O[i] = dMul(A[i], B[i]); }
void opAdd(int n){ for (int i = 0; i < n; i++) O[i] = dAdd(A[i], B[i]); }
void opSub(int n){ for (int i = 0; i < n; i++) O[i] = dSub(A[i], B[i]); }
void opDiv(int n){ for (int i = 0; i < n; i++) O[i] = dDiv(A[i], B[i]); }
void opSqrt(int n){ for (int i = 0; i < n; i++) O[i] = dSqrt(A[i]); }
void opCmp(int n){ for (int i = 0; i < n; i++) C[i] = dCmp(A[i], B[i]); }
void opFromI64(int n, i64 base, i64 step){ for (int i = 0; i < n; i++) O[i] = dFromI64(base + step*i); }
void opScale2(int n, int k){ for (int i = 0; i < n; i++) O[i] = dScale2(A[i], k); }
}
