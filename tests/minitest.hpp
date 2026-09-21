#pragma once
#include <cmath>
#include <cstdio>
#include <exception>
#include <format>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif
namespace minitest {

inline bool use_color() {
#if defined(_WIN32)
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(STDOUT_FILENO) != 0;
#endif
}

inline const char *ansi_green() {
    return use_color() ? "\033[32m" : "";
}
inline const char *ansi_red() {
    return use_color() ? "\033[31m" : "";
}
inline const char *ansi_cyan() {
    return use_color() ? "\033[36m" : "";
}
inline const char *ansi_reset() {
    return use_color() ? "\033[0m" : "";
}

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

    void list() const {
        for (const auto &c : cases_) {
            std::puts(c.name.c_str());
        }
    }

    // filters 为空时跑全部；否则只跑名称精确匹配的用例。返回失败个数。
    int run(const std::vector<std::string> &filters = {}) const {
        int failed = 0;
        int ran = 0;
        for (const auto &c : cases_) {
            if (!filters.empty()) {
                bool selected = false;
                for (const auto &f : filters) {
                    if (c.name == f) {
                        selected = true;
                        break;
                    }
                }
                if (!selected) {
                    continue;
                }
            }
            ++ran;
            std::printf("\n  %s\n", c.name.c_str());
            try {
                c.func();
                std::printf("  %sPASS%s  %s\n", ansi_green(), ansi_reset(), c.name.c_str());
            } catch (const std::exception &e) {
                std::printf("  %sFAIL%s  %s\n        %s\n", ansi_red(), ansi_reset(),
                            c.name.c_str(), e.what());
                ++failed;
            }
        }
        if (!filters.empty() && ran == 0) {
            std::printf("没有匹配的用例:");
            for (const auto &f : filters) {
                std::printf(" %s", f.c_str());
            }
            std::printf("\n可用用例:\n");
            list();
            return 1;
        }
        std::printf("\n%d 个用例，%s%d 个失败%s\n", ran, ansi_red(), failed, ansi_reset());
        return failed;
    }

    int run_all() const {
        return run({});
    }
    case_manager(const case_manager &) = delete;
    case_manager &operator=(const case_manager &) = delete;
    case_manager(case_manager &&) = delete;
    case_manager &operator=(case_manager &&) = delete;

  private:
    case_manager() = default;

    std::vector<test_case> cases_;
};

inline void report_check_pass(const std::string &detail) {
    std::printf("        %sPASS%s  %s\n", ansi_green(), ansi_reset(), detail.c_str());
}

inline void report_print(const std::string &loc, std::string text) {
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
    const char *c = ansi_cyan();
    const char *r = ansi_reset();
    if (text.find('\n') == std::string::npos) {
        std::printf("        %sINFO%s  %s @ %s\n", c, r, text.c_str(), loc.c_str());
        return;
    }
    std::printf("        %sINFO%s  @ %s\n", c, r, loc.c_str());
    std::size_t start = 0;
    while (start < text.size()) {
        const auto end = text.find('\n', start);
        const std::string line =
            text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        std::printf("          %s\n", line.c_str());
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
}

// PRINT 不能写成 std::format(__VA_ARGS__)：宏参数对 clang 不是
// consteval 常量。做成函数对象后，格式串是真正的调用实参。
struct printer {
    std::string loc;

    template <typename... Args>
    void operator()(std::format_string<Args...> fmt, Args &&...args) const {
        report_print(loc, std::format(fmt, std::forward<Args>(args)...));
    }
};

template <typename T>
void print_expr_at(const std::string &loc, const char *expr, const T &value) {
    std::string body = std::vformat("{}", std::make_format_args(value));
    while (!body.empty() && (body.back() == '\n' || body.back() == '\r')) {
        body.pop_back();
    }
    if (body.find('\n') == std::string::npos) {
        report_print(loc, std::string(expr) + " = " + body);
    } else {
        report_print(loc, std::string(expr) + " =\n" + body);
    }
}

inline std::string at(const char *file, int line) {
    return std::string(file) + ":" + std::to_string(line);
}

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
            throw ::minitest::failure("CHECK failed: " + std::string(#cond) + " @ " +              \
                                      ::minitest::at(__FILE__, __LINE__));                         \
        }                                                                                          \
        ::minitest::report_check_pass(std::string("CHECK(" #cond ") @ ") +                         \
                                      ::minitest::at(__FILE__, __LINE__));                         \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                                      \
    do {                                                                                           \
        const double va = (a), vb = (b);                                                           \
        const std::string loc = ::minitest::at(__FILE__, __LINE__);                                \
        if (std::abs(va - vb) > (eps)) {                                                           \
            throw ::minitest::failure("CHECK_NEAR 失败: " #a "=" + std::to_string(va) +            \
                                      " vs " #b "=" + std::to_string(vb) + " @ " + loc);           \
        }                                                                                          \
        ::minitest::report_check_pass("CHECK_NEAR(" #a ", " #b "): " + std::to_string(va) +        \
                                      " vs " + std::to_string(vb) + " @ " + loc);                  \
    } while (0)

#define CHECK_THROWS(expr)                                                                         \
    do {                                                                                           \
        bool thrown = false;                                                                       \
        try {                                                                                      \
            (void)(expr);                                                                          \
        } catch (...) {                                                                            \
            thrown = true;                                                                         \
        }                                                                                          \
        const std::string loc = ::minitest::at(__FILE__, __LINE__);                                \
        if (!thrown) {                                                                             \
            throw ::minitest::failure("期望 " #expr " 抛异常，但它没有 @ " + loc);                 \
        }                                                                                          \
        ::minitest::report_check_pass(std::string("CHECK_THROWS(" #expr ") @ ") + loc);            \
    } while (0)

#define PRINT ::minitest::printer{::minitest::at(__FILE__, __LINE__)}

#define PRINT_EXPR(...)                                                                            \
    do {                                                                                           \
        ::minitest::print_expr_at(::minitest::at(__FILE__, __LINE__), #__VA_ARGS__,                 \
                                  (__VA_ARGS__));                                                  \
    } while (0)
