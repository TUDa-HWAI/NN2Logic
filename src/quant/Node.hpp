#pragma once
#include <Eigen/Dense>
#include <cassert>
#include <vector>
#include <any>

namespace QTree {

template <typename T> class TreeDecision;
template <typename T> class TreeLeaf;


template <typename T>
class TreeNode {
protected:
    std::weak_ptr<TreeDecision<T>> m_parent;
    std::optional<bool> m_parentDecision = {};

    TreeNode(const TreeNode<T>& other) : m_parent(other.m_parent), m_parentDecision(other.m_parentDecision) {}
    TreeNode() {}
public:
    std::any m_visitorMem;

    virtual Eigen::VectorXf evaluate(const Eigen::VectorXf& inp) const = 0;

    virtual Eigen::VectorXf evaluate(const Eigen::VectorXf& inp, std::function<void(std::any&)> func) = 0;

    [[nodiscard]] bool parentDecision() const {
        return m_parentDecision.value();
    }

    void setParent(std::shared_ptr<TreeDecision<T>> parent, bool parentDec) {
        assert(parent != nullptr);
        m_parent = parent;
        m_parentDecision = parentDec;
    }

    std::shared_ptr<TreeDecision<T>> getParent() const {
        return m_parent.lock();
    }

    void replace(std::shared_ptr<TreeNode<T>> other) {
        assert(m_parentDecision.has_value());
        assert(!m_parent.expired());

        auto parent = m_parent.lock();
        parent->setChild(other, m_parentDecision.value());
        m_parent.reset();  // remove parent from self
        m_parentDecision.reset();
    }
};
}