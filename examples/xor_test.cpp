#include <algorithm>
#include <cstdio>

int relu(int x) {
    return std::max(0, x);
}

int neuron(int x1, int x2, int w1, int w2, int b) {
    return relu(x1 * w1 + x2 * w2 + b);
}

int xor_function(int x1, int x2) {
    return x1 ^ x2;
}

int main(int argc, char *argv[]) {

    constexpr int kLo = -2;
    constexpr int kHi = 2;

    int n1_w1 = 0, n1_w2 = 0, n1_b = 0;
    int n2_w1 = 0, n2_w2 = 0, n2_b = 0;
    int n3_w1 = 0, n3_w2 = 0, n3_b = 0;

    for (n1_w1 = kLo; n1_w1 <= kHi; n1_w1++) {
        for (n1_w2 = kLo; n1_w2 <= kHi; n1_w2++) {
            for (n1_b = kLo; n1_b <= kHi; n1_b++) {
                for (n2_w1 = kLo; n2_w1 <= kHi; n2_w1++) {
                    for (n2_w2 = kLo; n2_w2 <= kHi; n2_w2++) {
                        for (n2_b = kLo; n2_b <= kHi; n2_b++) {
                            for (n3_w1 = kLo; n3_w1 <= kHi; n3_w1++) {
                                for (n3_w2 = kLo; n3_w2 <= kHi; n3_w2++) {
                                    for (n3_b = kLo; n3_b <= kHi; n3_b++) {

                                        bool flag_00 = false;
                                        bool flag_01 = false;
                                        bool flag_10 = false;
                                        bool flag_11 = false;

                                        int h1_00 = neuron(0, 0, n1_w1, n1_w2, n1_b);
                                        int h2_00 = neuron(0, 0, n2_w1, n2_w2, n2_b);
                                        int n3_00 = neuron(h1_00, h2_00, n3_w1, n3_w2, n3_b);

                                        if (n3_00 == xor_function(0, 0)) {
                                            flag_00 = true;
                                        }

                                        int h1_01 = neuron(0, 1, n1_w1, n1_w2, n1_b);
                                        int h2_01 = neuron(0, 1, n2_w1, n2_w2, n2_b);
                                        int n3_01 = neuron(h1_01, h2_01, n3_w1, n3_w2, n3_b);

                                        if (n3_01 == xor_function(0, 1)) {
                                            flag_01 = true;
                                        }

                                        int h1_10 = neuron(1, 0, n1_w1, n1_w2, n1_b);
                                        int h2_10 = neuron(1, 0, n2_w1, n2_w2, n2_b);
                                        int n3_10 = neuron(h1_10, h2_10, n3_w1, n3_w2, n3_b);

                                        if (n3_10 == xor_function(1, 0)) {
                                            flag_10 = true;
                                        }

                                        int h1_11 = neuron(1, 1, n1_w1, n1_w2, n1_b);
                                        int h2_11 = neuron(1, 1, n2_w1, n2_w2, n2_b);
                                        int n3_11 = neuron(h1_11, h2_11, n3_w1, n3_w2, n3_b);

                                        if (n3_11 == xor_function(1, 1)) {
                                            flag_11 = true;
                                        }

                                        if (flag_00 && flag_01 && flag_10 && flag_11) {
                                            printf(
                                                "n1_w1=%d, n1_w2=%d, n1_b=%d, n2_w1=%d, n2_w2=%d, "
                                                "n2_b=%d, n3_w1=%d, n3_w2=%d, n3_b=%d\n",
                                                n1_w1, n1_w2, n1_b, n2_w1, n2_w2, n2_b, n3_w1,
                                                n3_w2, n3_b);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}