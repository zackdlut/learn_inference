#include "inference/graph.hpp"
#include "inference/mlp.hpp"
#include "inference/model_file.hpp"
#include "inference/op.hpp"
#include "inference/ops.hpp"

#include <algorithm>
#include <format>
#include <iostream>
namespace
{

template <class... Args> void println(std::format_string<Args...> fmt, Args &&...args)
{
    std::cout << std::format(fmt, std::forward<Args>(args)...) << '\n';
}
const char *resolve_dir(const char *env_name, const char *compiled_default)
{
    const char *from_env = std::getenv(env_name);
    return from_env != nullptr ? from_env : compiled_default;
}

void section(const char *title)
{
    std::cout << std::format("\n\033[36m===== {} =====\033[0m\n", title);
}

} // namespace

using inference::Executor;
using inference::Graph;
using inference::Tensor;
using inference::ValueId;
int main(int argc, char *argv[])
{
    inference::register_builtin_ops();
    const char *infer_model_dir = resolve_dir("INFER_MODEL_DIR", INFER_MODEL_DIR);
    // -------------------------------------------------------------------------
    section("1. 已注册的算子");
    auto names = inference::OpRegistry::instance().list();
    std::sort(names.begin(), names.end());
    std::string line = "   ";
    for (const auto &n : names)
        line += " " + n;
    println("{}", line);
    println("");
    println("这些算子谁都没在「某个列表」里登记过。每个算子的 .cpp 里写一句");
    println("REGISTER_OP(XxxOp)，靠全局对象的构造函数在 main() 之前自动注册。");
    // -------------------------------------------------------------------------
    section("2. 用图重建阶段 3 的 MLP");
    const std::string model_path = std::string(infer_model_dir) + "/mnist_mlp.inference";
    auto mf = inference::ModelFile::load(model_path);
    inference::Graph g;
    const ValueId x = g.add_input("x", {1, 784});
    const ValueId w1 = g.add_weight("fc1.w", mf.get_tensor("fc1.weight"));
    const ValueId b1 = g.add_weight("fc1.b", mf.get_tensor("fc1.bias"));
    const ValueId w2 = g.add_weight("fc2.w", mf.get_tensor("fc2.weight"));
    const ValueId b2 = g.add_weight("fc2.b", mf.get_tensor("fc2.bias"));

    const ValueId h1 = g.add_node("linear", "fc1", {x, w1, b1});
    const ValueId a1 = g.add_node("relu", "act1", {h1});
    const ValueId h2 = g.add_node("linear", "fc2", {a1, w2, b2});
    const ValueId p = g.add_node("softmax", "prob", {h2});
    g.mark_output(p);
    g.finalize(); // 拓扑排序 + 形状推断 + 使用计数
    std::cout << g.to_string();
    println("");
    println("注意所有中间结果的形状都填好了 —— finalize() 做了形状推断，");
    println("一次实际计算都没跑，就知道每步输出多大。这是内存规划的前提。");
    // -------------------------------------------------------------------------
    const std::string activation_path =
        std::string(infer_model_dir) + "/mnist_activation.inference";
    const inference::ModelFile activation_file = inference::ModelFile::load(activation_path);
    const inference::Tensor input_activation = activation_file.get_tensor("input");
    const inference::Tensor z1_activation = activation_file.get_tensor("z1");
    const inference::Tensor h1_activation = activation_file.get_tensor("h1");
    const inference::Tensor z2_activation = activation_file.get_tensor("z2");
    const inference::Tensor prob_activation = activation_file.get_tensor("prob");

    inference::Executor exec(g);
    const std::vector<Tensor> out = exec.run({{"x", input_activation}});
    const inference::MLP hardcoded = inference::MLP::load(model_path);
    const Tensor want = hardcoded.forward_trace(input_activation).prob;

    println("    图执行结果   vs  硬编码版:  最大误差 {:.3e}",
            static_cast<double>(inference::ops::max_abs_diff(out[0], want)));
    println("    图执行结果   vs  Python  :  最大误差 {:.3e}",
            static_cast<double>(
                inference::ops::max_abs_diff(out[0], activation_file.get_tensor("prob"))));
    println("    预测数字: {}", inference::ops::argmax(out[0]));
    println("");
    println("    执行了 {} 个节点，中间结果峰值内存 {} 字节", exec.stats().nodes_run,
            exec.stats().peak_bytes);
}