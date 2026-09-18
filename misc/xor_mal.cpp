#include <algorithm>
#include <chrono>
#include <cstdio>

// 网络结构: 2 -> 2 -> 1
constexpr int kSamples = 4;  // 4 组输入一起算, 相当于 batch
constexpr int kInput = 2;
constexpr int kHidden = 2;
constexpr int kOutput = 1;

// 把 4 组输入摆成矩阵: 行是样本, 列是特征
constexpr int kX[kSamples][kInput] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
constexpr int kTarget[kSamples][kOutput] = {{0}, {1}, {1}, {0}};

int relu(int x) {
    return std::max(0, x);
}

// 一层神经元: Y = relu(X @ W + B)
//   x: [M][K]  M 个样本, 每个样本 K 个输入
//   w: [K][N]  w[k][j] = 第 k 个输入连到第 j 个神经元的权重
//   b: [1][N]  一行偏置, 广播给 M 个样本
//   y: [M][N]  M 个样本各自的 N 个神经元输出
template <int M, int K, int N>
void neuron(const int (&x)[M][K], const int (&w)[K][N], const int (&b)[1][N],
            int (&y)[M][N]) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int acc = b[0][j];
            for (int k = 0; k < K; k++) {
                acc += x[i][k] * w[k][j];
            }
            y[i][j] = relu(acc);
        }
    }
}

int main(int argc, char *argv[]) {
    constexpr int kLo = -2;
    constexpr int kHi = 2;

    int w1[kInput][kHidden];   // 输入层 -> 隐藏层
    int b1[1][kHidden];
    int w2[kHidden][kOutput];  // 隐藏层 -> 输出层
    int b2[1][kOutput];

    int h[kSamples][kHidden];  // 隐藏层输出
    int y[kSamples][kOutput];  // 网络输出

    int solutions = 0;
    long long hidden_calls = 0;
    auto start = std::chrono::steady_clock::now();

    // 外层 6 重: 隐藏层参数 w1 / b1
    for (w1[0][0] = kLo; w1[0][0] <= kHi; w1[0][0]++) {
        for (w1[1][0] = kLo; w1[1][0] <= kHi; w1[1][0]++) {
            for (b1[0][0] = kLo; b1[0][0] <= kHi; b1[0][0]++) {
                for (w1[0][1] = kLo; w1[0][1] <= kHi; w1[0][1]++) {
                    for (w1[1][1] = kLo; w1[1][1] <= kHi; w1[1][1]++) {
                        for (b1[0][1] = kLo; b1[0][1] <= kHi; b1[0][1]++) {
                            // 隐藏层只和 w1/b1 有关, 提到这里算一次, 4 个样本一起出结果
                            neuron(kX, w1, b1, h);
                            hidden_calls++;

                            // 内层 3 重: 输出层参数 w2 / b2, 直接复用上面的 h
                            for (w2[0][0] = kLo; w2[0][0] <= kHi; w2[0][0]++) {
                                for (w2[1][0] = kLo; w2[1][0] <= kHi; w2[1][0]++) {
                                    for (b2[0][0] = kLo; b2[0][0] <= kHi; b2[0][0]++) {
                                        neuron(h, w2, b2, y);

                                        bool ok = true;
                                        for (int i = 0; i < kSamples; i++) {
                                            if (y[i][0] != kTarget[i][0]) {
                                                ok = false;
                                                break;
                                            }
                                        }
                                        if (!ok) {
                                            continue;
                                        }
                                        solutions++;
                                        printf(
                                            "n1_w1=%d, n1_w2=%d, n1_b=%d, n2_w1=%d, n2_w2=%d, "
                                            "n2_b=%d, n3_w1=%d, n3_w2=%d, n3_b=%d\n",
                                            w1[0][0], w1[1][0], b1[0][0], w1[0][1], w1[1][1],
                                            b1[0][1], w2[0][0], w2[1][0], b2[0][0]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - start)
                  .count();
    printf("solutions=%d, hidden_calls=%lld, time=%lldms\n", solutions, hidden_calls,
           static_cast<long long>(ms));
    return 0;
}
