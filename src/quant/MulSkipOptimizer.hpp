#pragma once

#include <queue>

#include "Path.hpp"
#include "SimulatedAnnealing.hpp"

namespace QTree {

//template <bool SIMPLE>
class MulSkipOptimizer final {
    std::vector<Layer<int>> layers;
    std::vector<size_t> layerSizes;
    std::vector<std::pair<path::Path,path::Path::PathBitset>> samples;
    std::map<std::pair<size_t,size_t>,size_t> numMuls;  // how many multiplications with the result of that neuron happen

    std::map<std::pair<size_t,size_t>,std::set<std::pair<size_t,size_t>>> totalRequiredNeurons;

    static std::vector<path::Decision> mergeDecisions(const std::vector<path::Decision>& a, const std::vector<path::Decision>& b) {
        assert(!(a.empty() && b.empty()));
        
        // combine in one vector
        std::vector<path::Decision> buf(a.begin(), a.end());
        buf.reserve(a.size()+b.size());
        buf.insert(buf.end(), b.begin(), b.end());

        assert(buf.size() == a.size() + b.size());

        // sort
        std::sort(buf.begin(), buf.end(), [](const auto& a, const auto& b) -> bool {
            return std::tie(a.layerIdx, a.neuronIdx) < std::tie(b.layerIdx, b.neuronIdx);
        });

        return buf;
    }


    std::set<std::pair<size_t,size_t>> detReqNeurons(const std::pair<size_t,size_t>& idx) const {
        // determine all neurons that need to be calculated
        std::set<std::pair<size_t,size_t>> reqNeurons;
        std::queue<std::pair<size_t,size_t>> work;

        work.push(idx);
        reqNeurons.insert(idx);

        while (!work.empty()) {
            auto w = work.front();
            work.pop();

            if (w.first == 0) continue;  // already in 1st layer

            auto [weight, bias, fp] = layers.at(w.first).getNeuron(w.second);
            for (size_t i = 0; i < weight.size(); ++i) {
                if (weight[i] == 0) continue;

                auto nextIdx = std::make_pair(w.first-1, i);

                if (reqNeurons.find(nextIdx) != reqNeurons.end()) continue;

                reqNeurons.insert(nextIdx);
                work.push(nextIdx);
            }
        }

        return reqNeurons;
    }

    bool m_simple;
    bool m_skipOnlyGood = true;

    // good matches, bad matches, skipped muls
    std::tuple<size_t,size_t,size_t,size_t> pathPerformance(const std::vector<path::Decision>& decisions, const std::vector<bool>& possClasses) const {
        std::set<std::pair<size_t,size_t>> reqNeurons;
        for (const auto& d : decisions) {
            const auto& buf = totalRequiredNeurons.at(d.idx());
            reqNeurons.insert(buf.begin(), buf.end());
        }
        
        size_t good = 0, bad = 0, skipped = 0, affectedPaths = 0;
        path::Path::PathBitset pb(layerSizes, decisions);

        // iterate over all samples
        for (const auto& [p, b] : samples) {
            if (!pb.overlaps(b)) continue;  // no overlap, dismiss

            affectedPaths += 1;

            if (p.leaf.possClasses() == possClasses) {  // count overlap
                good += p.visitFreq;
            } else {
                bad += p.visitFreq;
            }

            // determine skipped muls
            if (p.leaf.possClasses() == possClasses || !m_skipOnlyGood) {
                for (const auto& d : p.decisions) {
                    if ((m_simple || d.decision) && reqNeurons.find(d.idx()) == reqNeurons.end()) {  // FIXME
                        skipped += p.visitFreq * numMuls.at(d.idx());
                    }
                }
            }
        }

        return std::make_tuple(good, bad, skipped, affectedPaths);
    }

    int m_scoreFacBad = 1000;
    int m_scoreFacGood = 100;

    double scoreFunc(const std::vector<path::Decision>& decisions, const std::vector<bool>& possClasses) const {
        size_t good, bad, skipped, affectedPaths;
        std::tie(good, bad, skipped, affectedPaths) = pathPerformance(decisions, possClasses);

        //return skipped - 1000*double(bad) + 100*double(good);
        return skipped - double(m_scoreFacBad * bad) + double(m_scoreFacGood * good);
    }

public:
    MulSkipOptimizer(const std::vector<Layer<int>> layers, const std::vector<path::Path>& samples, bool simple = false) : layers(layers), m_simple(simple) {
        // determine layer sizes for bitsets
        for (const auto& l : layers) {
            layerSizes.push_back(l.numNeurons());
        }
        
        // build PathBitset
        std::transform(samples.begin(), samples.end(), std::back_inserter(this->samples), [&](const auto& p) {
            return std::make_pair(p, p.asBitset(layerSizes));
        });

        // determine how many multiplications happen with the result of a neuron
        for (size_t i = 1; i < layers.size(); ++i) {
            for (size_t n = 0; n < layers[i].numNeurons(); ++n) {
                auto [weight, bias, fp] = layers[i].getNeuron(n);

                for (size_t wIdx = 0; wIdx < weight.size(); ++wIdx) {
                    if (weight[wIdx] == 0) continue;
                    numMuls[{i-1, wIdx}] += 1;
                }
            }
        }

        // determine dependencies of neurons
        for (size_t i = 0; i < layers.size(); ++i) {
            for (size_t ii = 0; ii < layers.at(i).numNeurons(); ++ii) {
                auto idx = std::make_pair(i, ii);
                totalRequiredNeurons[idx] = detReqNeurons(idx);
            }
        }
    }

    void setFacBad(int fac) {
        m_scoreFacBad = fac;
    }

    void setFacGood(int fac) {
        m_scoreFacGood = fac;
    }

    void setSkipOnlyGood(bool onlyGood) {
        m_skipOnlyGood = onlyGood;
    }

    struct OptPath {
        path::Path p;
        size_t goodMatches;
        size_t badMatches;
        size_t skippedMuls;
        double score;

        [[nodiscard]] operator path::Path() const {
            return p;
        }
    };


    OptPath process(const path::Path& candidate) const {
        assert(candidate.leaf.isConst());

        // check which iis decisions are not const
        std::vector<path::Decision> iisConst, iisNonConst;
        std::set<std::pair<size_t,size_t>> leafIis = candidate.leaf.iisIDxs();
        for (const auto& d : candidate.decisions) {
            if (leafIis.find(d.idx()) == leafIis.end()) continue;  // decision not relevant

            if (d.isConst) iisConst.push_back(path::Decision{
                .layerIdx = d.layerIdx,
                .neuronIdx = d.neuronIdx,
                .decision = d.decision,
                .iisMember = true,
                .isConst = false
            });
            else iisNonConst.push_back(d);
        }

        std::vector<path::Decision> bestSolution;
        double score;

        if (iisConst.empty()) {  // cannot use this for further optimizations
            bestSolution = iisNonConst;
            score = scoreFunc(bestSolution, candidate.leaf.possClasses());

        } else {
            // Simulated Annealing part
            std::vector<bool> state(iisConst.size(), true);
            QTree::SimulatedAnnealing sa(iisConst, state);
            sa.run(9999999.0, [](size_t a, size_t b) { return 0.9; }, [&](const auto& c) -> double {
                return scoreFunc(mergeDecisions(c, iisNonConst), candidate.leaf.possClasses());  // FIXME, needs to be merged
            });

            bestSolution = mergeDecisions(sa.bestSolution(), iisNonConst);
            score = sa.bestScore();
        }

        // get performance data
        size_t good, bad, skipped;
        std::tie(good, bad, skipped, std::ignore) = pathPerformance(bestSolution, candidate.leaf.possClasses());

        // build new path
        path::Path result{
            .decisions = bestSolution,
            .leaf = candidate.leaf,  // FIXME
            .visitFreq = good
        };

        return OptPath{
            .p = result,
            .goodMatches = good,
            .badMatches = bad,
            .skippedMuls = skipped,
            .score = score
        };
    }
};
}