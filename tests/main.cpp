#include "minitest.hpp"
#include <iostream>

int main(int argc, char *argv[]) {
    std::cout << "运行测试用例:!" << std::endl;
    minitest::case_manager::instance().run_all();
    return 0;
}