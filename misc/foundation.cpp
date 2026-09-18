#include <cmath>
#include <cstdio>

int main(int argc, char *argv[]) {
    printf("%d\n", 0.1F + 0.2F == 0.3F);
    printf("%f\n", 1e10F + 1.0F - 1e10F);
    printf("%f\n", 1e10F + (1.0F - 1e10F));
    printf("%f\n", std::exp(89.0F));
    printf("%f\n", std::exp(89.0F) / std::exp(89.0F));
    return 0;
}