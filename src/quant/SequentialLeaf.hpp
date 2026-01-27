#pragma once
#include "Worker.hpp"
#include "Leaf.hpp"
#include "Path.hpp"
#include "ConstraintTracker.hpp"
#include "SimplePool.hpp"
#include "MultiProgress.hpp"
#include "SEvaluator.hpp"
#include <memory>

namespace QTree {

template <typename T> class Layer;



template <typename T>
class SequentialLeaf final {
    std::list<path::Path> m_paths;
    std::list<std::shared_ptr<TreeLeaf<T>>> m_leaves;


    /// takes a node and checks if it evaluates to true or false
    typedef std::list<Eigen::Matrix<double,Eigen::Dynamic,1>> Samples_t;
    static std::pair<Samples_t,Samples_t> splitSimData(const Samples_t& simData, std::shared_ptr<TreeDecision<T>> node) {
        Samples_t tSplit, fSplit;

        for (const auto& X : simData) {
            bool dec = node->decide(X);
            if (dec) tSplit.emplace_front(X);
            else fSplit.emplace_front(X);
        }

        assert(!tSplit.empty() || !fSplit.empty());
        return std::make_pair(tSplit, fSplit);
    }


    /**
     * Build an equation for a neuron of a final layer. Does scaling but does not have saturation/relu.
     * @param neuronIdx index of the neuron in the final layer
     * @param vars Input variables
     * @return gurobi linear expression
     */
    static GRBLinExpr equ(const Layer<T>& layer, size_t neuronIdx, const std::vector<GRBVar>& vars) {
        assert(vars.size() == layer.weight.cols());

        // Build equation
        GRBLinExpr expr(layer.bias(neuronIdx));
        for (size_t i = 0; i < vars.size(); ++i) {
            expr += layer.weight(neuronIdx, i) * vars.at(i);
        }

        // scaling
        expr *= layer.requant.scales.at(neuronIdx);

        return expr;
    }

    /**
     * Uses Gurobi to check if a given Leaf is constant.
     * @param leaf Leaf to check
     * @param enc Encoder object
     * @return path::Leaf object with const-ness and IIS information
     */
    static path::Leaf constChecker(std::shared_ptr<TreeLeaf<T>> leaf, const Layer<T>& layer, const Encoder* enc) {
        // obtain gurobi model
        auto model = enc->getBaseModel();

        // add all decisions
        ConstraintTracker<TreeDecision<T>> constrTracker;
        auto decisionPath = leaf->getParent()->decisionPath();
        decisionPath.emplace_back(leaf->parentDecision(), leaf->getParent());
        
        SEvaluator<T>::addPriorDecisions(model, constrTracker, decisionPath);

        // cheaty way to get index
        auto layerIdx = model.layerVars.rbegin()->first;
        size_t numNeurons = model.layerVars.at(layerIdx).size();
        auto vars = model.layerInputVars(layerIdx);

        // for each class
        // - possible that it is bigger than all the other classes?
        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> weight;
        Eigen::Matrix<T,Eigen::Dynamic,1> bias; 
        std::tie(weight, bias) = leaf->getWeightB();

        std::vector<bool> isPossible(numNeurons, true);
        std::set<std::pair<size_t,size_t>> iisNodes;

        for (size_t i = 0; i < numNeurons; ++i) {  // TODO: bound to -127/127?
            model.grb->addConstr(model.neuronVar(layerIdx, i), GRB_EQUAL, equ(layer, i, vars));
        }


        for (size_t i = 0; i < numNeurons; ++i) {  
            std::vector<GRBConstr> constr;
            auto neuron = equ(layer, i, vars);

            // get other
            for (size_t ii = 0; ii < numNeurons; ++ii) {
                if (i == ii) continue;

                constr.push_back(model.grb->addConstr(
                    model.neuronVar(layerIdx, i),
                    GRB_GREATER_EQUAL,
                    model.neuronVar(layerIdx, ii)));
            }

            model.grb->optimize();
            const auto status = model.grb->get(GRB_IntAttr_Status);
            if (status == GRB_TIME_LIMIT) {
                // treat as not const for now
                fmt::print("timeout for leaf\n");  // FIXME
            } else if (status == GRB_INFEASIBLE) {
                // this neuron will always be dominated
                isPossible.at(i) = false;
                
                // obtain iis
                model.grb->computeIIS();
                for (auto dec : constrTracker.getIIS()) {
                    iisNodes.emplace(dec->layerIdx, dec->neuronIdx);
                }

                if (numNeurons == 2) {  // it is a binary leaf!
                    // TODO: replace node in tree
                } else {
                    throw std::runtime_error("Non-binary leaves are not implemented yet.");
                }
            } else if (status != GRB_OPTIMAL && status != GRB_SOLUTION_LIMIT) {
                throw std::runtime_error("Unhandled Gurobi status " + std::to_string(status));
            }

            // remove constraints
            for (const auto& c : constr) {
                model.grb->remove(c);
            }

            model.grb->reset(1);  // discard solution status
        }

        //assert(std::any_of(isPossible.begin(), isPossible.end(), [](bool b) { return b; }));
        if(std::none_of(isPossible.begin(), isPossible.end(), [](bool b) { return b; })) {
            throw std::runtime_error("No leaf class possible");
        }

        return path::Leaf(isPossible, iisNodes);
    }






public:
    SequentialLeaf(const std::list<Work<T>>& work, const Layer<T>& layer, const Encoder* enc, MultiProgress& mp) {
        size_t numSamples = 0;
        for (const auto& w : work) {
            numSamples += w.simData.size();
        }

        std::list<std::pair<std::shared_ptr<TreeLeaf<T>>,size_t>> worklist;

        auto addToWorklist = [&](std::shared_ptr<TreeDecision<T>> node, const bool dec, const size_t visitFreq) {
            auto toRepl = dec ? node->trueChild() : node->falseChild();
            if (toRepl == nullptr || visitFreq == 0) return;

            auto leaf = std::make_shared<TreeLeaf<T>>(layer.weight, layer.bias);
            toRepl->replace(leaf);

            // save leaf
            m_leaves.push_front(leaf);

            // add to worklist
            worklist.emplace_front(leaf, visitFreq);
        };


        for (const auto& w : work) {
            assert(w.node->getParent() != nullptr);
            auto [tSplit, fSplit] = /*Creator<T,int>::*/splitSimData(w.simData, w.node);

            addToWorklist(w.node, true, tSplit.size());
            addToWorklist(w.node, false, fSplit.size());
        }


        // FIXME check constness of leaves
        std::mutex mut;
        SimplePool<std::pair<std::shared_ptr<TreeLeaf<T>>,size_t>> sp(worklist, "Leaves", mp);
        sp.setNumThreads(std::max(1, std::div(std::thread::hardware_concurrency()-1, 32).quot));  // in theory, gurobi should use 32 threads?

        sp.run([&](decltype(sp)* sp, Encoder encoder) {
            std::optional<std::pair<std::shared_ptr<TreeLeaf<T>>,size_t>> work;
            while ((work = sp->popIfPossible())) {
                std::shared_ptr<TreeLeaf<T>> leaf;
                size_t visitFreq;
                std::tie(leaf, visitFreq) = work.value();

                // create leaf
                path::Leaf l = /*LeafCrator<T>::*/constChecker(leaf, layer, &encoder);

                // decisions
                auto p = leaf->getParent()->decisionPath();
                p.emplace_back(leaf->parentDecision(), leaf->getParent());

                std::vector<path::Decision> decisions;
                decisions.reserve(p.size());
                std::transform(p.begin(), p.end(), std::back_inserter(decisions), [](const auto& pair) {
                    bool dec; std::shared_ptr<TreeDecision<T>> node;
                    std::tie(dec, node) = pair;

                    std::set<std::pair<size_t,size_t>> iisNodes;
                    if (node->isConst()) {
                        for (auto iisNode : node->m_evalFunc->getIIS()) {
                            iisNodes.emplace(iisNode->layerIdx, iisNode->neuronIdx);
                        }
                    }

                    return path::Decision{
                        node->layerIdx,
                        node->neuronIdx,
                        dec,
                        false,
                        node->isConst(),
                        iisNodes
                    };
                });

                {
                    std::lock_guard<std::mutex> lock(mut);
                    m_paths.push_front(path::Path{decisions, l, visitFreq});
                }
            }
        }, *enc);


        // sort paths
        m_paths.sort([](const auto& a, const auto& b) -> bool {
            return a.visitFreq > b.visitFreq;
        });

        size_t totalVisitFreq = 0;
        for (const auto& p : m_paths) {
            totalVisitFreq += p.visitFreq;
        }

        assert(totalVisitFreq == numSamples);
    }

    std::list<path::Path> getPaths() const {
        return m_paths;
    }

    std::list<std::shared_ptr<TreeLeaf<T>>> getLeaves() const {
        return m_leaves;
    }
};
}