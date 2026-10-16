#pragma once

#include "inference/op.hpp"
#include "inference/tensor.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace inference
{
using ValueId = int;
struct Value
{
    ValueId id = -1;
    std::string name;
    Shape shape;
    enum class Kind
    {
        Input,
        Weight,
        Intermediate
    };
    Kind kind = Kind::Intermediate;
};
struct Node
{
    int id = -1;
    std::string type;
    std::string name;
    std::vector<ValueId> inputs;
    ValueId output;
    Attributes attributes;
};

class Graph
{
  public:
    ValueId add_input(std::string name, Shape shape);
    ValueId add_weight(std::string name, Tensor tensor);

    ValueId add_node(std::string type, std::string name, std::vector<ValueId> inputs,
                     Attributes attributes = {});

    void mark_output(ValueId id);
    void finalize();

    bool finalized() const noexcept
    {
        return finalized_;
    }
    const Value &value(ValueId id) const;

    const std::vector<Value> &values() const noexcept
    {
        return values_;
    }
    const std::vector<Node> &nodes() const noexcept
    {
        return nodes_;
    }
    const std::unordered_map<ValueId, Tensor> &weights() const noexcept
    {
        return weights_;
    }
    const std::vector<ValueId> &outputs() const noexcept
    {
        return outputs_;
    }

    std::span<const int> exec_order() const noexcept
    {
        return order_;
    }

    int consumer_count(ValueId id) const noexcept;

    std::string to_string() const;

  private:
    std::vector<Value> values_;
    std::vector<Node> nodes_;
    std::unordered_map<ValueId, Tensor> weights_;
    std::vector<ValueId> outputs_;

    std::vector<int> order_;
    std::unordered_map<ValueId, int> uses_;
    bool finalized_ = false;

    ValueId new_value(std::string name, Shape shape, Value::Kind kind);
    std::vector<int> topo_order() const;
};

class Executor
{
  public:
    explicit Executor(const Graph &g) : g_(&g) {}
    std::vector<Tensor> run(const std::unordered_map<std::string, Tensor> &inputs);

    void set_reuse(bool reuse) noexcept
    {
        reuse_ = reuse;
    }

    struct Stats
    {
        std::size_t peak_bytes = 0;
        std::size_t total_bytes = 0;
        int nodes_run = 0;
    };
    const Stats &stats() const noexcept
    {
        return stats_;
    }

  private:
    const Graph *g_;
    bool reuse_ = true;
    Stats stats_;
};

} // namespace inference