#pragma once

#include "MultiProgress.hpp"
#include "SimplePool.hpp"
#include "SequentialLeaf.hpp"
#include "Decision.hpp"
#include <fstream>
#include <fmt/os.h>


namespace QTree {

/*
 * 1. Find all paths
 * 2. Check leaves for constness
 * 3. Check constness for decisions that lead to a const leaf
 */
template <typename T, typename I>
class SequentialCreator final {
    typedef std::list<Eigen::Matrix<double,Eigen::Dynamic,1>> Samples_t;

    std::shared_ptr<TreeDecision<T>> m_root;
    std::list<std::shared_ptr<TreeLeaf<T>>> m_leaves;


    /// takes a node and checks if it evaluates to true or false
    static std::pair<Samples_t,Samples_t> splitSimData(const Samples_t& simData, std::shared_ptr<TreeDecision<T>> node) {
        Samples_t tSplit, fSplit;

        for (const auto& X : simData) {
            bool dec = node->decide(X);
            if (dec) tSplit.emplace_front(X);
            else fSplit.emplace_front(X);
        }

        assert(tSplit.size() + fSplit.size() == simData.size());
        return std::make_pair(tSplit, fSplit);
    }


    /// constructs either true or false decison of a single decision node in a path
    static std::optional<Work<T>> handleBranch(std::shared_ptr<TreeDecision<T>> parent, std::shared_ptr<NodeOrder<T>> order, std::vector<bool> path, bool dec,
        Samples_t simData) {

        path.push_back(dec);
        if (simData.empty()) {  // construct a placeholder!
            auto placeholder = std::make_shared<Placeholder<T>>(path, order);
            parent->setChild(placeholder, dec);
            return {};
        }

        auto nIdx = order->next(path);

        if (nIdx.has_value()) {  // decision
            Work<T> w(nIdx.value(), order, path, std::move(simData));
            parent->setChild(w.node, dec);
            return w;
        
        } else {  // layer is done
            parent->setChild(std::make_shared<Placeholder<T>>(std::vector<bool>{}, nullptr), dec);
            return Work<T>({}, parent, std::move(simData));
        }
    }


    /// process a given path and put new side-paths into worklist for other threads to work on
    static std::list<Work<T>> processWork(Work<T> start, MultiProgress::Handle& progress) {
        std::list<Work<T>> kindaLeaves;
        std::queue<Work<T>> worklist;
        worklist.push(start);

        // some bookkeeping
        auto processBranch = [&](const std::optional<Work<T>>& nextWork, bool dec, const Samples_t& samples) -> void {
            auto w = handleBranch(nextWork->node, nextWork->order, nextWork->path, dec, samples);

            if (!w.has_value()) return;  // constructed a placeholder on intermediate level

            if (w->isDone()) {  // is actually a leaf (placeholder at this point)
                kindaLeaves.push_front(w.value());
                progress.tick(w->simData.size());

            } else {  // put into worklist
                worklist.push(w.value());
            }
        };

        // main loop
        while (!worklist.empty()) {
            // obtain work
            std::optional<Work<T>> nextWork = worklist.front();
            worklist.pop();

            assert(!nextWork->isDone());  // check for work not to be a leaf

            // split simulation data
            auto [tSplit, fSplit] = splitSimData(nextWork->simData, nextWork->node);

            processBranch(nextWork, true, tSplit);
            processBranch(nextWork, false, fSplit);
        }

        return kindaLeaves;
    }


    static Samples_t processSamplesLayer(const Layer<T>& layer, const Samples_t& samples) {
        assert(layer.relu);

        Samples_t res;
        auto weight = layer.weight.template cast<double>();
        auto bias = layer.bias.template cast<double>();
        auto scaler = layer.requant.scales;//. template cast<codegen::FixedPoint>();
        Eigen::VectorXd scalerVec(scaler.size());
        for (size_t i = 0; i < scaler.size(); ++i) {
            scalerVec[i] = scaler[i];
        }

        std::transform(samples.begin(), samples.end(), std::front_inserter(res), [&](const auto& row) {
            Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> im = weight * row + bias;

            im = im.cwiseProduct(scalerVec);  // scaling for requant
            im = im.cwiseMin(127.0);  // saturation
            im = im.cwiseMax(0.0);  // relu

            return im;
        });

        return res;
    }


    std::list<path::Path> m_paths;  // FIXME
    std::vector<Layer<T>> m_layers;

public:
    //SequentialCreator(const std::vector<Layer<T>>& layers, Encoder* encoder, const Eigen::Matrix<I,Eigen::Dynamic,Eigen::Dynamic>& data) /*: m_layers(layers)*/ {
    SequentialCreator(const std::vector<Layer<T>>& layers, Encoder* encoder, const Eigen::Ref<const Eigen::Matrix<I,Eigen::Dynamic,Eigen::Dynamic>> data) : m_layers(layers) {        
        assert(layers.size() > 1);
        assert(layers.back().relu == false);

        // create samples for simulation
        Samples_t simData;
        auto& rowwise = data.rowwise();
        std::transform(rowwise.begin(), rowwise.end(), std::front_inserter(simData), [](const auto& row) -> Eigen::Matrix<double,Eigen::Dynamic,1> {
            return row.template cast<double>();
        });
        const size_t numSamples = simData.size();
        
        // handle multiple progress bars
        MultiProgress barManager;

        // ends of pathes
        std::list<Work<T>> kindaLeaves;

        // create tree from all but the last layers
        for (size_t i = 0; i < layers.size()-1; ++i) {
            assert(layers.at(i).relu);

            // ensure that encoder has the variables for the layer
            encoder->createLayerVars(i, layers.at(i).bias.size());
            encoder->update();

            // progressbar and worklist
            auto progBar = barManager.create(fmt::format("Layer {}", i), numSamples);
            std::list<Work<T>> worklist;  // here we store our work

            // convenience
            auto replEnd = [&](auto end, bool path, Samples_t& simData) -> void {
                if (simData.empty()) return;  // placeholder stays

                auto node = path ? end->trueChild() : end->falseChild();
                assert(node != nullptr);  // because we do const checking in a later step

                simData = processSamplesLayer(layers.at(i-1), simData);
                auto nOrder = std::make_shared<NodeOrder<T>>(layers.at(i).weight, layers.at(i).bias, layers.at(i).requant.scales, i);

                Work<T> w(nOrder->next({}).value(), nOrder, {}, std::move(simData));  // FIXME
                node->replace(w.node);
                worklist.emplace_front(w);
            };


            if (i == 0) {  // first layer
                // Node ordering object
                auto nOrder = std::make_shared<NodeOrder<T>>(layers.at(i).weight, layers.at(i).bias, layers.at(i).requant.scales, i);

                Work<T> w(nOrder->next({}).value(), nOrder, {}, std::move(simData));
                m_root = w.node;  // put root into Worklist
                worklist.emplace_front(w);

            } else {  // following layer
                // process work pieces one by one
                for (auto& leaf : kindaLeaves) {
                    // split simulation data
                    auto [tSplit, fSplit] = splitSimData(leaf.simData, leaf.node);

                    replEnd(leaf.node, true, tSplit);
                    replEnd(leaf.node, false, fSplit);
                }
                kindaLeaves.clear();  // clear list
            }

            // actual processing
            for (const auto& w : worklist) {
                auto kl = processWork(w, progBar);
                kindaLeaves.insert(kindaLeaves.begin(), kl.begin(), kl.end());
            }
        }

        // check that we did not loose anything
        assert(numSamples == std::accumulate(kindaLeaves.begin(), kindaLeaves.end(), size_t(0), [](const size_t v, const auto& w) -> size_t {
            return v + w.simData.size();
        }));


        /*
         * Create Leaves
         */
        encoder->createLayerVars(layers.size()-1, layers.back().bias.size());  // tell model about layer size
        encoder->update();

        SequentialLeaf<T> lc(kindaLeaves, layers.back(), encoder, barManager);
 
        m_paths = lc.getPaths(); 
        m_leaves = lc.getLeaves();


        /*
         * Check Constness of decisions
         */
        std::set<std::pair<std::shared_ptr<TreeDecision<T>>,bool>> candidates;
        std::map<std::shared_ptr<TreeDecision<T>>,std::list<path::Decision*>> toFix;

        for (auto& p : m_paths) {  // m_leaves -> path::Leaf
            if (!p.leaf.isConst()) continue;  // do not care if not const

            // for each const leaf
            // get the actual tree path
            // check each decision if one child is a placeholder
            // if so put decision into candidate set

            auto& decisions = p.decisions;
            std::shared_ptr<TreeDecision<T>> r = m_root;
            
            for (size_t i = 0; i < decisions.size(); ++i) {
                assert(decisions[i].layerIdx == r->layerIdx && decisions[i].neuronIdx == r->neuronIdx);

                auto toCheck = decisions[i].decision ? r->falseChild() : r->trueChild();
                if (std::dynamic_pointer_cast<Placeholder<T>>(toCheck) != nullptr) {  // check if candidate
                    candidates.emplace(r, !decisions[i].decision);
                    toFix[r].push_front(&decisions[i]);
                }

                // next one to traverse
                if (i < decisions.size()-1)  // last one will have a leaf..
                    r = std::static_pointer_cast<TreeDecision<T>>(decisions[i].decision ? r->trueChild() : r->falseChild());
            }
        }

        // actual const checking of the candidates
        SimplePool<std::pair<std::shared_ptr<TreeDecision<T>>,bool>> sp(candidates, "Decision Constness", barManager);
        sp.setNumThreads(std::max(1, std::div(std::thread::hardware_concurrency()-1, 32).quot));  // in theory, gurobi should use 32 threads?
        sp.run([](decltype(sp)* sp, Encoder enc){
            std::optional<std::pair<std::shared_ptr<TreeDecision<T>>,bool>> work;
            while ((work = sp->popIfPossible())) {
                const auto& [c, dec] = work.value();
                if (!c->decisionPossible(dec, &enc)) {
                    c->setChild(nullptr, dec);
                }
            }
        }, *encoder);


        // fixing paths
        for (const auto& [ptr, decs] : toFix) {
            if (!ptr->isConst()) continue;

            // iis
            std::set<std::pair<size_t,size_t>> iis;
            for (const auto node : ptr->getIIS()) {
                iis.emplace(node->layerIdx, node->neuronIdx);
            }

            for (auto d : decs) {
                d->isConst = true;
                d->iisNodes = iis;
            }
        }


        // terminate bars
        //barManager.mark_as_completed();
    }

    // https://stackoverflow.com/a/2072890
    static bool ends_with(const std::string& value, const std::string& ending) {
        if (ending.size() > value.size()) return false;
        return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
    }

    void print(const std::string& filename) const {
        if (ends_with(filename, ".json")) {
            // construct json
            nlohmann::json j;
            j["layers"] = m_layers;
            j["paths"] = m_paths;

            // write json to file
            std::ofstream dat_out(filename);
            dat_out << j;
            dat_out.close();

            return;
        }

        assert(ends_with(filename, ".txt"));
        auto out = fmt::output_file(filename);
        for (const auto& p : m_paths) {
            out.print("{}\n", p);
        }
    }

    nlohmann::json toJson() const {
        nlohmann::json j;
        j["layers"] = m_layers;
        j["paths"] = m_paths;

        return j;
    }

    std::list<path::Path> getPaths() const {
        return m_paths;
    }
};
}