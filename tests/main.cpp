#include "minitest.hpp"
#include <iostream>
#include <string>
#include <vector>

static void print_usage(const char *argv0) {
    std::printf("用法:\n");
    std::printf("  %s                 运行全部 TEST()\n", argv0);
    std::printf("  %s <name> [...]    运行指定 TEST(name)\n", argv0);
    std::printf("  %s --list          列出全部用例名\n", argv0);
}

int main(int argc, char **argv) {
    auto &mgr = minitest::case_manager::instance();
    std::vector<std::string> filters;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--list" || arg == "-l") {
            mgr.list();
            return 0;
        }
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (!arg.empty() && arg[0] == '-') {
            std::fprintf(stderr, "未知参数: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 2;
        }
        filters.push_back(arg);
    }

    if (filters.empty()) {
        std::cout << "运行测试用例:" << std::endl;
    } else {
        std::cout << "运行测试用例:";
        for (const auto &name : filters) {
            std::cout << ' ' << name;
        }
        std::cout << std::endl;
    }

    return mgr.run(filters) == 0 ? 0 : 1;
}
