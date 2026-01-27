#pragma once
#include "Node.hpp"
#include "SEvaluator.hpp"

namespace QTree {

/// Class that represents a decision in the tree.
template <typename T>
class TreeDecision : public TreeNode<T>, public std::enable_shared_from_this<TreeDecision<T>> {
    std::shared_ptr<TreeNode<T>> m_true;  /// Child if true path
    std::shared_ptr<TreeNode<T>> m_false;   /// Child if false path

public:
    std::shared_ptr<SEvaluator<T>> m_evalFunc;

    const size_t neuronIdx;
    const size_t layerIdx;

    TreeDecision(std::shared_ptr<std::pair<const Eigen::Matrix<T,Eigen::Dynamic,1>,T>> decisionWeight, double scale, size_t layerIdx, size_t neuronIdx) 
        : neuronIdx(neuronIdx), layerIdx(layerIdx), m_true(nullptr), m_false(nullptr) {
        m_evalFunc = std::make_shared<SEvaluator<T>>(decisionWeight, scale, 127);
    }

    Eigen::VectorXf evaluate(const Eigen::VectorXf& X) const override {
        return X;  // FIXME:
    }

    Eigen::VectorXf evaluate(const Eigen::VectorXf& X, std::function<void(std::any&)> func) override {
        return X;  // FIXME:
    }

    /// set a child of this decision.
    void setChild(std::shared_ptr<TreeNode<T>> child, bool decision) {
        if (decision) m_true = child;
        else m_false = child;

        if (child != nullptr) child->setParent(std::enable_shared_from_this<TreeDecision<T>>::shared_from_this(), decision);
    }

    /// return if the decisions is constant.
    [[nodiscard]] bool isConst() const {
        assert(m_true || m_false);
        return m_true == nullptr || m_false == nullptr;
    }

    [[nodiscard]] std::list<std::shared_ptr<const TreeDecision<T>>> getIIS() const {
        assert(isConst());
        return m_evalFunc->getIIS();
    }

    /// returns a list of pairs that include a decision node and the decision it has made.
    [[nodiscard]] std::list<std::pair<bool,std::shared_ptr<TreeDecision<T>>>> decisionPath() const {
        std::list<std::pair<bool,std::shared_ptr<TreeDecision>>> toRet;

        auto child = std::enable_shared_from_this<TreeDecision<T>>::shared_from_this();
        auto parent = std::static_pointer_cast<TreeDecision<T>>(child->m_parent.lock());

        while (parent != nullptr) {
            toRet.emplace_front(child->m_parentDecision.value(), parent);

            child = parent;
            parent = std::static_pointer_cast<TreeDecision<T>>(child->m_parent.lock());
        }

        return toRet;
    }

    /// use gurobi to determine if a certain decision is possible given the previous decisions.
    bool decisionPossible(bool dec, const Encoder* encoder) {
        return m_evalFunc->checkPathPossible(std::enable_shared_from_this<TreeDecision<T>>::shared_from_this(), dec, encoder);
    }

    /// returns the child on the true path. could be a nullptr if const decision.
    std::shared_ptr<TreeNode<T>> trueChild() {
        return m_true;
    }

    /// returns the child on the false path. could be a nullptr if const decision.
    std::shared_ptr<TreeNode<T>> falseChild() {
        return m_false;
    }

    /// checks the ReLU condition
    bool decide(const Eigen::Matrix<double, Eigen::Dynamic,1>& sample) const {
        return m_evalFunc->decide(sample);
    }
};
}