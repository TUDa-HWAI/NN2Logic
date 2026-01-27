#pragma once

#include "Placeholder.hpp"

namespace QTree {

template <typename T, typename I = double>
struct Work {
    std::shared_ptr<TreeDecision<T>> node;
    std::vector<bool> path;
    std::shared_ptr<NodeOrder<T>> order;
    std::list<Eigen::Matrix<I,Eigen::Dynamic,1>> simData;  // we are simulating

    Work(size_t nIdx, std::shared_ptr<NodeOrder<T>> order, const std::vector<bool>& path, std::list<Eigen::Matrix<I,Eigen::Dynamic,1>>&& simData)
        : path(path), order(order), simData(std::move(simData)) {

        this->path.reserve(order->maxDepth());
        node = std::make_shared<TreeDecision<T>>(order->getWB(nIdx), order->getScale(nIdx), order->layerIdx, nIdx);
    }

    Work(const std::vector<bool>& path, std::shared_ptr<TreeDecision<T>> node, std::list<Eigen::Matrix<I,Eigen::Dynamic,1>>&& simData)
        : node(node), path(path), order(nullptr), simData(std::move(simData)) {}

    Work(const std::vector<bool>& path, std::shared_ptr<TreeDecision<T>> node, std::shared_ptr<NodeOrder<T>> order,
        std::list<Eigen::Matrix<I,Eigen::Dynamic,1>>&& simData) : node(node), path(path), order(order), simData(std::move(simData)) {}

    [[nodiscard]] bool isDone() const {
        return order == nullptr;
    }
};
}