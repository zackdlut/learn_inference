#include "inference/graph.hpp"
#include "inference/op.hpp"
#include "inference/tensor.hpp"

#include <chrono>
#include <deque>
#include <format>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace inference
{
ValueId Graph::new_value(std::string name, Shape shape, Value::Kind kind)
{
    ValueId id = static_cast<ValueId>(values_.size());
    values_.push_back(
        Value{.id = id, .name = std::move(name), .shape = std::move(shape), .kind = kind});
    return id;
}

ValueId Graph::add_input(std::string name, Shape shape)
{
    if (finalized_)
    {
        throw std::runtime_error("Graph is finalized");
    }
    return new_value(std::move(name), std::move(shape), Value::Kind::Input);
}

ValueId Graph::add_weight(std::string name, Tensor tensor)
{
    if (finalized_)
    {
        throw std::runtime_error("Graph is finalized");
    }
    Shape shape = tensor.shape();
    const ValueId id = new_value(std::move(name), std::move(shape), Value::Kind::Weight);
    weights_.emplace(id, std::move(tensor));
    return id;
}

ValueId Graph::add_node(std::string type, std::string name, std::vector<ValueId> inputs,
                        Attributes attributes)
{
    if (finalized_)
    {
        throw std::runtime_error("Graph is finalized");
    }
    if (!OpRegistry::instance().has(type))
    {
        OpRegistry::instance().get(type);
    }
    for (ValueId in : inputs)
    {
        if (in < 0 || in >= static_cast<ValueId>(values_.size()))
        {
            throw std::runtime_error("Invalid input value id");
        }
    }
    const auto node_id = static_cast<int>(nodes_.size());
    const ValueId out = new_value(name + ".out", Shape{}, Value::Kind::Intermediate);

    nodes_.push_back(Node{.id = node_id,
                          .type = std::move(type),
                          .name = std::move(name),
                          .inputs = std::move(inputs),
                          .output = out,
                          .attributes = std::move(attributes)});
    return out;
}

void Graph::mark_output(ValueId id)
{
    if (id < 0 || id >= static_cast<ValueId>(values_.size()))
    {
        throw std::runtime_error("Invalid value id");
    }
    outputs_.push_back(id);
}

std::vector<int> Graph::topo_order() const
{
    // 初始化 ready 集合, indegree = 0 : 所有的Inputs 和  Weight
    std::unordered_set<ValueId> ready;
    for (const Value &v : values_)
    {
        if (v.kind != Value::Kind::Intermediate)
        {
            ready.insert(v.id);
        }
    }
    // 记录每个Compute Node 的 入度
    std::vector<int> indegree(nodes_.size(), 0);
    // 记录每个ValueId 的 等待者
    std::unordered_map<ValueId, std::vector<int>> waiters;
    // 遍历所有Node, 计算每个Node 的 入度 和 等待者
    for (const Node &n : nodes_)
    {
        for (ValueId in : n.inputs)
        {
            // 检查每个Node 的每个input, 如果不在ready 中，则将indegree 加1，并将给阶段加入Value
            // 的等待队列
            if (!ready.contains(in))
            {
                ++indegree[n.id];
                waiters[in].push_back(n.id);
            }
        }
    }
    // Khan算法初始化 待处理队列, 将入度为0的Node 加入队列
    std::deque<int> q;
    for (const Node &n : nodes_)
    {
        if (indegree[n.id] == 0)
        {
            q.push_back(n.id);
        }
    }

    // 记录拓扑排序结果
    std::vector<int> order;
    order.reserve(nodes_.size());
    while (!q.empty())
    {
        // 从队列中取出 待处理Node
        const int id = q.front();
        q.pop_front();
        order.push_back(id);

        // 执行 node, 参数 Intermediate ValueId
        const ValueId produced = nodes_[id].output;
        // 检查 该ValueId 是否有等待者, 有则将等待者入度减1，如果入度为0，则将等待者加入队列
        if (auto it = waiters.find(produced); it != waiters.end())
        {
            // 遍历 该ValueId 的等待者
            for (int w : it->second)
            {
                if (--indegree[w] == 0)
                {
                    q.push_back(w);
                }
            }
        }
    }

    if (order.size() != nodes_.size())
    {
        throw std::runtime_error("Graph has a cycle");
    }
    return order;
}

void Graph::finalize()
{
    if (finalized_)
        return;
    if (outputs_.empty())
    {
        throw std::runtime_error("Graph has no outputs");
    }
    order_ = topo_order();
    const OpRegistry &reg = OpRegistry::instance();
    // 遍历 拓扑排序结果, 验证每个Node 的输入 和 输出
    for (int nid : order_)
    {
        Node &node = nodes_[nid];
        // 验证 Node 的输入 和 输出 是否匹配
        const Op &op = reg.get(node.type);
        if (const int want = op.num_inputs(); want > 0 && node.inputs.size() != want)
        {
            throw std::runtime_error("Node " + node.name + " has " +
                                     std::to_string(node.inputs.size()) + " inputs, but " +
                                     op.type() + " expects " + std::to_string(want));
        }
        // 遍历当前 Compute Node 的所有输入  将shape放入vector 中
        std::vector<Shape> input_shapes;
        input_shapes.reserve(node.inputs.size());
        for (ValueId in : node.inputs)
        {
            input_shapes.push_back(values_[in].shape);
        }
        // 调用 Op 的 inference_shape 方法，根据当前 Compute Node 的输入 shape 和 attributes
        // 推导出输出 shape， 放入 Output Value 的 shape 中
        try
        {
            values_[node.output].shape = op.inference_shape(input_shapes, node.attributes);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Node " + node.name + " failed to infer shape: " + e.what());
        }
    }
    // 遍历所有 Node, 计算每个 Input Value 的 使用者数量
    for (const Node &node : nodes_)
    {
        for (ValueId input : node.inputs)
            ++uses_[input];
    }
    // 遍历所有 Output Value, 计算每个 Output Value 的使用者数量
    for (ValueId output : outputs_)
        ++uses_[output];
    finalized_ = true;
}
// Get the value by id
const Value &Graph::value(ValueId id) const
{
    if (id < 0 || id >= static_cast<ValueId>(values_.size()))
    {
        throw std::runtime_error("Invalid value id");
    }
    return values_[id];
}

int Graph::consumer_count(ValueId id) const noexcept
{
    if (auto it = uses_.find(id); it != uses_.end())
    {
        return it->second;
    }
    return 0;
}

std::string Graph::to_string() const
{
    auto value_tag = [](ValueId id)
    {
        return std::format("v{}", id);
    };

    auto append_named_value = [&](std::string &out, const Value &value)
    {
        out += std::format("    {:<5}{:<16}{}\n", value_tag(value.id), value.name,
                           inference::to_string(value.shape));
    };

    std::string s =
        std::format("Graph({} 个节点, {} 个 Value)\n", nodes_.size(), values_.size());

    s += "  输入:\n";
    for (const auto &value : values_)
    {
        if (value.kind == Value::Kind::Input)
            append_named_value(s, value);
    }

    s += "  权重:\n";
    for (const auto &value : values_)
    {
        if (value.kind == Value::Kind::Weight)
            append_named_value(s, value);
    }

    std::vector<const Node *> ordered_nodes;
    ordered_nodes.reserve(nodes_.size());
    if (finalized_ && !order_.empty())
    {
        for (const int node_id : order_)
            ordered_nodes.push_back(&nodes_[node_id]);
    }
    else
    {
        for (const auto &node : nodes_)
            ordered_nodes.push_back(&node);
    }

    auto format_args = [](const Node &node)
    {
        std::string ins;
        for (std::size_t i = 0; i < node.inputs.size(); ++i)
        {
            if (i > 0)
                ins += ", ";
            ins += std::format("v{}", node.inputs[i]);
        }
        return std::format("({})", ins);
    };

    std::size_t max_args_width = 0;
    for (const Node *node : ordered_nodes)
        max_args_width = std::max(max_args_width, format_args(*node).size());

    s += "  计算:\n";
    for (const Node *node : ordered_nodes)
    {
        const std::string args = format_args(*node);
        const std::string pad(args.size() < max_args_width ? max_args_width - args.size() : 0, ' ');
        s += std::format("    {:<5}= {:<10}{}{}  {}\n", value_tag(node->output), node->type, args,
                         pad, inference::to_string(values_[node->output].shape));
    }

    s += "  输出: ";
    for (std::size_t i = 0; i < outputs_.size(); ++i)
    {
        if (i > 0)
            s += ", ";
        s += std::format("v{}", outputs_[i]);
    }
    s += '\n';
    return s;
}

std::vector<Tensor> Executor::run(const std::unordered_map<std::string, Tensor> &feeds)
{
    const Graph &g = *g_;
    // 检查 Graph 是否确认
    if (!g.finalized())
    {
        throw std::runtime_error("Graph is not finalized");
    }

    stats_ = Stats{};
    using Clock = std::chrono::steady_clock;
    const auto ms_since = [](Clock::time_point t0)
    { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); };
    const Clock::time_point run_t0 = timing_ ? Clock::now() : Clock::time_point{};
    if (timing_)
        stats_.node_ms.assign(g.nodes().size(), 0.0);
    std::unordered_map<ValueId, Tensor> live;
    std::unordered_map<ValueId, int> remaining;

    // 检查Graph 所需要的 Inputs 是否都提供了,如果提供了存放在live 中，否则报错
    for (const auto &v : g.values())
    {
        if (v.kind == Value::Kind::Input)
        {
            // 检查Graph 中的Input 在 inputs 中存不存在，不存在则报错
            auto it = feeds.find(v.name);
            if (it == feeds.end())
            {
                std::string names;
                for (const auto &[name, _] : feeds)
                {
                    if (!names.empty())
                        names += ", ";
                    names += name;
                }
                throw std::runtime_error("Missing input: " + v.name + " (required: " + names + ")");
            }
            // 检查Graph 中的Input 的 shape 与 inputs 中的 shape 不匹配，则报错
            if (v.shape != it->second.shape())
            {
                throw std::runtime_error("Input " + v.name + " has shape " +
                                         to_string(it->second.shape()) + " but expected " +
                                         to_string(v.shape));
            }
            live.emplace(v.id, it->second);
        }
    }
    // 将 Graph 中的 Weight 存放在 live 中
    for (const auto &[id, tensor] : g.weights())
    {
        live.emplace(id, tensor);
    }
    // 将 Graph 中的中间结果存放在remaining 中，
    // 并统计当前有几个使用者(多少个计算节点使用这个中间结果)
    for (const auto &v : g.values())
    {
        if (v.kind == Value::Kind::Intermediate)
        {
            remaining.emplace(v.id, g.consumer_count(v.id));
        }
    }

    const OpRegistry &reg = OpRegistry::instance();
    std::size_t alive_bytes = 0;
    for (int nid : g.exec_order())
    {
        const auto &node = g.nodes()[nid];
        // 初始化 OpContext
        OpContext context;
        context.inputs.reserve(node.inputs.size());
        for (ValueId in : node.inputs)
        {
            if (auto it = live.find(in); it != live.end())
            {
                context.inputs.push_back(&it->second);
            }
            else
            {
                throw std::runtime_error("Missing input: v" + std::to_string(in));
            }
        }
        context.attributes = &node.attributes;
        context.kv_cache = kv_cache_;
        // 执行 Op 获得 output Tensor
        const Op &op = reg.get(node.type);
        const Clock::time_point node_t0 = timing_ ? Clock::now() : Clock::time_point{};
        Tensor output = op.compute(&context);
        if (timing_)
        {
            const double dt = ms_since(node_t0);
            stats_.node_ms[static_cast<std::size_t>(nid)] = dt;
            stats_.type_ms[node.type] += dt;
        }
        // 检查 计算出的output Tensor Shape 是否和output.shape 一致
        if (output.shape() != g.value(node.output).shape)
        {
            throw std::runtime_error("Output shape mismatch: " + std::to_string(node.output) +
                                     " has shape " + to_string(output.shape()) +
                                     " but expected " + to_string(g.value(node.output).shape));
        }

        const auto nbytes = output.numel() * sizeof(float);
        stats_.total_bytes += nbytes;
        alive_bytes += nbytes;
        live.emplace(node.output, std::move(output));
        ++stats_.nodes_run;

        // 如果启用 reuse，则清理已经不需要的输入 Tensor
        if (reuse_)
        {
            for (ValueId in : node.inputs)
            {
                auto it = remaining.find(in);
                if (it == remaining.end())
                    continue;
                if (--it->second == 0)
                {
                    auto lit = live.find(in);
                    if (lit != live.end())
                    {
                        alive_bytes -= lit->second.numel() * sizeof(float);
                        live.erase(lit);
                    }
                }
            }
        }
        // 更新峰值内存使用量
        stats_.peak_bytes = std::max(stats_.peak_bytes, alive_bytes);
    }
    std::vector<Tensor> result;
    result.reserve(g.outputs().size());
    for (ValueId id : g.outputs())
    {
        auto it = live.find(id);
        if (it == live.end())
        {
            throw std::runtime_error("Output v" + std::to_string(id) + " not found");
        }
        result.push_back(it->second);
    }
    if (timing_)
        stats_.run_ms = ms_since(run_t0);
    return result;
}

} // namespace inference