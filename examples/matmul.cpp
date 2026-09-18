#include <cstddef>
#include <iostream>

int main(int argc, char *argv[]) {
    constexpr size_t M = 4;
    constexpr size_t N = 2;
    constexpr size_t K = 3;

    float A[M][K] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    float B[K][N] = {{1, 2}, {3, 4}, {5, 6}};
    float C[M][N] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};

    // Caculation Times = M*N*K*2
    // 2 : += and *
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                C[i][j] = A[i][k] * B[k][j];
            }
        }
    }
    std::cout << "C = " << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}