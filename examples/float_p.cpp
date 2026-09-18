#include <cfloat>
#include <cmath>
#include <cstdio>

int main() {
    const int mantissa_bits = FLT_MANT_DIG; // 24 = 23 显式位 + 1 隐含位
    printf("FLT_EPSILON (1.0 的 ulp) = %.9g  %a\n", FLT_EPSILON, FLT_EPSILON);
    printf("连续整数精度上界 2^%d = %.0f  (ulp=1)\n", mantissa_bits - 1,
           std::ldexp(1.0f, mantissa_bits - 1));
    printf("x+1==x 精度边界   2^%d = %.0f  (ulp=2)\n\n", mantissa_bits,
           std::ldexp(1.0f, mantissa_bits));

    printf("%2s %12s %14s %18s %12s %s\n", "n", "x=2^n", "ulp(x)", "next(x)",
           "x+1", "x+1==x");
    for (int n = 0; n <= mantissa_bits + 1; n++) {
        float x = std::ldexp(1.0f, n);
        float nxt = std::nextafter(x, INFINITY);
        float ulp = nxt - x;
        float xp1 = x + 1.0f;
        printf("%2d %12.0f %14.9g %18.9g %12.0f %s\n", n, x, ulp, nxt, xp1,
               (xp1 == x) ? "yes" : "no");
    }
}
