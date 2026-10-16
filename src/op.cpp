#include "inference/op.hpp"
#include <string>
#include <variant>

namespace inference
{

template <typename T> T Attributes::get(const std::string &key) const
{
    auto it = kv_map.find(key);
    if (it == kv_map.end())
    {
        std::string keys;
        for (const auto &[k, v] : kv_map)
            keys += " " + k;
        throw std::runtime_error("Attribute not found: " + key + "\nAvailable attributes: " + keys);
    }
    if (!std::holds_alternative<T>(it->second))
    {
        throw std::runtime_error("Attribute type mismatch: " + key + "\nExpected type: " +
                                 typeid(T).name() + "\nActual type: " + typeid(it->second).name());
    }
    return std::get<T>(it->second);
}
// 显式实例化：只允许这三种类型，别的类型在链接期就会报错
template std::int64_t Attributes::get<std::int64_t>(const std::string &) const;
template float Attributes::get<float>(const std::string &) const;
template std::string Attributes::get<std::string>(const std::string &) const;

namespace
{

std::string attribute_to_string(std::int64_t value)
{
    return std::to_string(value);
}
std::string attribute_to_string(float value)
{
    return std::to_string(value);
}
std::string attribute_to_string(const std::string &value)
{
    return value;
}
std::string attribute_to_string(const Attribute &value)
{
    return std::visit([](const auto &v) { return attribute_to_string(v); }, value);
}

} // namespace
std::string Attributes::to_string() const
{
    std::string info;
    for (const auto &[key, value] : kv_map)
    {
        info += key + ": " + attribute_to_string(value) + "\n";
    }
    return info;
}

OpRegistry &OpRegistry::instance()
{
    static OpRegistry instance;
    return instance;
}

void OpRegistry::add(std::unique_ptr<Op> op)
{
    std::string type = op->type();
    if (ops_map.contains(type))
    {
        throw std::runtime_error("Op already registered: " + op->type());
    }
    ops_map.emplace(type, std::move(op));
}

bool OpRegistry::has(const std::string &type) const
{
    return ops_map.contains(type);
}

const Op &OpRegistry::get(const std::string &type) const
{
    auto it = ops_map.find(type);
    if (it == ops_map.end())
    {
        std::string registered_ops_list;
        for (const auto &[type, op] : ops_map)
        {
            registered_ops_list += type + ", ";
        }
        throw std::runtime_error("Op not found: " + type +
                                 "\nRegistered ops: " + registered_ops_list);
    }
    return *it->second;
}

std::vector<std::string> OpRegistry::list() const
{
    std::vector<std::string> types;
    for (const auto &[type, op] : ops_map)
    {
        types.push_back(type);
    }
    return types;
}

std::string OpRegistry::to_string() const
{
    std::string info;
    for (const auto &[type, op] : ops_map)
    {
        info += type + "\n";
    }
    return info;
}
} // namespace inference