#pragma once
#include "Node.hpp"
#include "Order.hpp"

namespace QTree {

/// Placeholder
template <typename T>
class Placeholder : public TreeNode<T> {
    std::shared_ptr<NodeOrder<T>> m_order;
    std::vector<bool> m_path;
public:
    Placeholder(const std::vector<bool>& path, std::shared_ptr<NodeOrder<T>> order) : m_path(path), m_order(order) {}

    Eigen::VectorXf evaluate(const Eigen::VectorXf& X) const override {
        throw std::runtime_error("Tried to evaluate a placeholder, that cannot work.");
    }

    Eigen::VectorXf evaluate(const Eigen::VectorXf& X, std::function<void(std::any&)> func) override {
        throw std::runtime_error("Tried to evaluate a placeholder, that cannot work.");
    }
};

}