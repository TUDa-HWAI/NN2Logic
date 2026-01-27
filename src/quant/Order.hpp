#pragma once

#include "Exchange.hpp"

#include <algorithm>
#include <memory>
#include <list>
#include <optional>
#include <Eigen/Dense>
#include <vector>
#include <mutex>


namespace QTree {

template <typename T>
class NodeOrder final {
    typedef std::shared_ptr<std::pair<const Eigen::Matrix<T,Eigen::Dynamic,1>,T>> WBPtr_t;

    /// Internal tree structure to keep track of available candidates on a given path
    struct Candidate final {
        std::optional<std::list<size_t>> m_cand;
        std::mutex m_lock;

        std::unique_ptr<Candidate> m_tPath = nullptr;
        std::unique_ptr<Candidate> m_fPath = nullptr;

        std::optional<size_t> m_idx = {};

        explicit Candidate(const std::list<size_t>& candidates) : m_cand(candidates) {}

        void fix(size_t idx) {
            std::lock_guard<std::mutex> l(m_lock);
            m_cand->remove(idx);
            m_idx = idx;

            m_tPath = std::make_unique<Candidate>(m_cand.value());
            m_fPath = std::make_unique<Candidate>(m_cand.value());
            m_cand.reset();
        }
    };

    std::unique_ptr<Candidate> m_root;

    /// holds weights and bias for each neuron
    std::vector<WBPtr_t> m_weightBias;
    //Eigen::Matrix<float,Eigen::Dynamic,1> m_scales;
    std::vector<QTree::FixedPoint> m_scales;

public:
    const size_t layerIdx;

    NodeOrder(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& weight, const Eigen::Matrix<T,Eigen::Dynamic,1>& bias,
        const /*Eigen::Matrix<float,Eigen::Dynamic,1>*/ std::vector<QTree::FixedPoint>& scales, size_t layerIdx) : layerIdx(layerIdx), m_scales(scales) {
        assert(weight.rows() == bias.size());
        m_weightBias.reserve(weight.rows());
        std::list<size_t> idxs;

        for (size_t i = 0; i < weight.rows(); ++i) {
            idxs.push_back(i);  // save idx
            m_weightBias.push_back(std::make_shared<std::pair<const Eigen::Matrix<T,Eigen::Dynamic,1>,T>>(weight.row(i), bias(i)));  // create shared weight
        }

        m_root = std::make_unique<Candidate>(idxs);
    }

    /// get next neuron index for a certain path
    /// order is in tree construction
    std::optional<size_t> next(const std::vector<bool>& path) {
        Candidate* ptr = m_root.get();
        for (bool dec : path) {
            if (dec) {
                ptr = ptr->m_tPath.get();
            } else {
                ptr = ptr->m_fPath.get();
            }
        }

        // Leaf or next layer
        if (ptr->m_cand->empty()) return {};

        // get and fix index
        size_t idx = ptr->m_cand->front();
        ptr->fix(idx);
        return idx;
    }

    [[nodiscard]] double getScale(size_t idx) const {
        //return m_scales(idx);
        return m_scales[idx];
    }


    [[nodiscard]] WBPtr_t getWB(size_t idx) const {
        return m_weightBias.at(idx);
    }

    [[nodiscard]] size_t maxDepth() const {
        return m_weightBias.size();
    }
};






}