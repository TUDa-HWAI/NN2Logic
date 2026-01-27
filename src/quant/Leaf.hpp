#pragma once
#include "Node.hpp"

namespace QTree {

/// Leaves
template <typename T>
class TreeLeaf : public TreeNode<T> {
protected:
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> m_weight;
    Eigen::Matrix<T,Eigen::Dynamic,1> m_bias;

    explicit TreeLeaf(const TreeLeaf<T>& other) : TreeNode<T>(other), m_weight(other.m_weight), m_bias(other.m_bias) {}
    TreeLeaf() : TreeNode<T>(), m_weight() {}
public:
    TreeLeaf(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> weight, Eigen::Matrix<T,Eigen::Dynamic,1> bias) : m_weight(weight), m_bias(bias) {}

    virtual Eigen::VectorXf evaluate(const Eigen::VectorXf& inp) const override {
        assert(false);
        return m_weight.template cast<float>() * inp + m_bias.template cast<float>();
    }

    Eigen::VectorXf evaluate(const Eigen::VectorXf& X, std::function<void(std::any&)> func) override {
        func(TreeNode<T>::m_visitorMem);
        return evaluate(X);
    }

    [[nodiscard]] std::pair<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>,Eigen::Matrix<T,Eigen::Dynamic,1>> getWeightB() const {
        return {m_weight,m_bias};
    }
};

}