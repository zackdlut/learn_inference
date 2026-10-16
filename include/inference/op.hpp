#pragma once

#include "inference/tensor.hpp"
#include <memory>
#include <unordered_map>
#include <variant>

namespace inference
{

using Attribute = std::variant<std::int64_t, float, std::string>;
class Attributes
{
  public:
    bool has(const std::string &key) const
    {
        return kv_map.contains(key);
    }
    Attributes &set(std::string key, Attribute value)
    {
        kv_map.insert_or_assign(std::move(key), std::move(value));
        return *this;
    }
    template <typename T> T get(const std::string &key) const;
    template <typename T> T get(const std::string &key, const T default_value) const
    {
        return kv_map.contains(key) ? get<T>(key) : default_value;
    }
    std::string to_string() const;

  private:
    std::unordered_map<std::string, Attribute> kv_map;
};

struct OpContext
{
    std::vector<const Tensor *> inputs;
    const Attributes *attributes = nullptr;
};

class Op
{
  public:
    virtual ~Op() = default;
    virtual std::string type() const = 0;
    virtual int num_inputs() const = 0;
    virtual Shape inference_shape(std::span<const Shape> input_shapes,
                                  const Attributes &attributes) const = 0;
    virtual Tensor compute(const OpContext *context) const = 0;
};

class OpRegistry
{
  public:
    static OpRegistry &instance();
    void add(std::unique_ptr<Op> op);
    bool has(const std::string &type) const;
    const Op &get(const std::string &type) const;
    std::vector<std::string> list() const;
    std::string to_string() const;

  private:
    std::unordered_map<std::string, std::unique_ptr<Op>> ops_map;
};

namespace detail
{
struct OpRegistryManager
{
    explicit OpRegistryManager(std::unique_ptr<Op> op)
    {
        OpRegistry::instance().add(std::move(op));
    }
};
} // namespace detail

#define REGISTER_OP(op_class)                                                                      \
    namespace                                                                                      \
    {                                                                                              \
    const ::inference::detail::OpRegistryManager registry_manager_##op_class{                      \
        std::make_unique<op_class>()};                                                             \
    }

void register_builtin_ops();

} // namespace inference