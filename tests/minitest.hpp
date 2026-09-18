#pragma once
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <string>
#include <vector>
namespace minitest {

struct test_case {
    std::string name;
    std::function<void()> func;
};

struct failure : std::exception {
    std::string msg;
    explicit failure(std::string msg) : msg(std::move(msg)) {}
    const char *what() const noexcept override {
        return msg.c_str();
    }
};

class case_manager {
  public:
    static case_manager &instance() {
        static case_manager manager;
        return manager;
    }

    void add_test_case(const test_case &test_case) {
        cases_.push_back(test_case);
    }

    const std::vector<test_case> &cases() const {
        return cases_;
    }

    void run_all() {
        int failed = 0;
        for (const auto &c : cases_) {
            try {
                c.func();
                std::printf("  \033[32mPASS\033[0m  %s\n", c.name.c_str());
            } catch (const std::exception &e) {
                std::printf("  \033[31mFAIL\033[0m  %s\n        %s\n", c.name.c_str(), e.what());
                ++failed;
            }
        }
        std::printf("\n%zu 个用例，\033[31m%d 个失败\033[0m\n", cases_.size(), failed);
    }
    case_manager(const case_manager &) = delete;
    case_manager &operator=(const case_manager &) = delete;
    case_manager(case_manager &&) = delete;
    case_manager &operator=(case_manager &&) = delete;

  private:
    case_manager() = default;

    std::vector<test_case> cases_;
};

} // namespace minitest

#define TEST(name)                                                                                 \
    static void test_##name();                                                                     \
    namespace {                                                                                    \
    struct registrar_##name {                                                                      \
        registrar_##name() {                                                                       \
            minitest::case_manager::instance().add_test_case({#name, test_##name});                \
        }                                                                                          \
    } registered_case_##name;                                                                      \
    }                                                                                              \
    static void test_##name()

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            throw minitest::failure("CHECK failed: " + std::string(#cond));                        \
        }                                                                                          \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                                      \
    do {                                                                                           \
        const double va = (a), vb = (b);                                                           \
        if (std::abs(va - vb) > (eps)) {                                                           \
            throw ::minitest::failure("CHECK_NEAR 失败: " #a "=" + std::to_string(va) +            \
                                      " vs " #b "=" + std::to_string(vb) + " @ " + __FILE__ +      \
                                      ":" + std::to_string(__LINE__));                             \
        }                                                                                          \
    } while (0)

#define CHECK_THROWS(expr)                                                                         \
    do {                                                                                           \
        bool thrown = false;                                                                       \
        try {                                                                                      \
            (void)(expr);                                                                          \
        } catch (...) {                                                                            \
            thrown = true;                                                                         \
        }                                                                                          \
        if (!thrown) {                                                                             \
            throw ::minitest::failure("期望 " #expr " 抛异常，但它没有 @ " +                       \
                                      std::string(__FILE__) + ":" + std::to_string(__LINE__));     \
        }                                                                                          \
    } while (0)
