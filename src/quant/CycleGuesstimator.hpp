#pragma once
#include "codegen/PathScheduler.hpp"
#include "Exchange.hpp"
#include <functional>


namespace QTree {

class CycleGuesstimator final {
    const std::vector<Layer<int>> m_layers;
    const std::vector<path::Path> m_paths;


    static std::map<std::pair<size_t,size_t>,size_t> countNonZero(const std::vector<path::Path>& paths) {
        std::map<std::pair<size_t,size_t>,size_t> buf;

        for (const auto& p : paths) {
            for (const auto& d : p.decisions) {
                if (d.decision) {
                    buf[d.idx()] += p.visitFreq;
                }
            }
        }

        return buf;
    }

/*
    template <typename F>
    static void removePaths(std::vector<path::Path>& paths, F&& pred) {
        auto it = std::remove_if(paths.begin(), paths.end(), pred);
        paths.erase(it, paths.end());
    }
*/
    template <typename F>
    static std::vector<path::Path> extractPaths(std::vector<path::Path>& paths, F&& pred) {
        std::vector<path::Path> extr, keep;

        for (const auto& p : paths) {
            if (pred(p)) extr.push_back(p);
            else keep.push_back(p);
        }

        std::swap(paths, keep);
        return extr;


        /*
        const auto initialSize = paths.size();
        auto it = std::partition(paths.begin(), paths.end(), std::not_fn(pred));
        std::vector<path::Path> toRet(it, paths.end());
        paths.erase(it, paths.end());

        assert(paths.size() + toRet.size() == initialSize);

        return toRet;
        */
    }


    static size_t visitFreq(const std::vector<path::Path>& paths) {
        size_t toRet = 0;
        for (const auto& p : paths) {
            toRet += p.visitFreq;
        }

        return toRet;
    }

    std::map<size_t,size_t> l1Trues;
    size_t totalSamples = 0;

    // count true decisions per layer
    static std::map<size_t,size_t> layerTrues(const std::vector<path::Path>& paths, const size_t layerIdx = 1) {
        if (paths.empty()) fmt::print("layerIdx {}\n", layerIdx);
        assert(!paths.empty());
        std::map<size_t,size_t> toRet;

        for (const auto& p : paths) {
            for (const auto& d : p.decisions) {
                if (d.layerIdx == layerIdx) {
                    toRet[d.neuronIdx] += d.decision ? p.visitFreq : 0;  // makes sure element exists in map
                }
            }
        }

        assert(!toRet.empty());

        return toRet;
    }

public:
    CycleGuesstimator(const std::vector<Layer<int>>& layers, const std::vector<path::Path>& paths)
        : m_layers(layers), m_paths(paths) {
            assert(layers.size() == 3);
            l1Trues = layerTrues(paths, 0);
            for (const auto& p : paths) {
                totalSamples += p.visitFreq;
            }
        }


    static bool decisionsMayOverlap(const std::vector<path::Decision>& a, const std::vector<path::Decision>& b) {
        // convert to map for easier lookup
        std::map<std::pair<size_t,size_t>,bool> mA;
        for (const auto& d : a) mA[d.idx()] = d.decision;

        // compare b against a
        for (const auto& d : b) {
            auto it = mA.find(d.idx());
            if (it != mA.end() && d.decision != it->second) {  // share decision with different decision value
                return false;
            }
        }

        return true;
    }


    [[nodiscard]] double guess(const codegen::ConstSchedule& schedule) const {
        assert(schedule.size() > 0);
        double cycles = 0;
        std::vector<path::Path> paths(m_paths.begin(), m_paths.end());  // copy paths vector

        // We do not consider compuatation of first layer
        bool firstBatch = true;
        size_t idx = 0;
        for (const auto& [neurons, checks] : schedule.bulk()) {
            //fmt::print("{:03d}]num Paths: {}\n", idx++, paths.size());

            const size_t total = visitFreq(paths);
            auto l1Trues = layerTrues(paths, 0);
            auto l2Trues = layerTrues(paths, 1);

            if (firstBatch) {  // first batch gets special treatment
                
                for (const auto& n : neurons) {
                    for (size_t i = 0; i < l1Trues.size(); ++i) {
                        if (m_layers.front().weight.coeff(n, i) != 0) {
                            cycles += l1Trues.at(i);
                        }
                    }
                }

                firstBatch = false;
            } else {  // not first batch
                std::vector<bool> wasUsed(m_layers.front().bias.size(), false);
                
                // what i want to determine:
                // - for every multiplication of a neuron in this batch i need to load the input data (but only once)
                // - then i need to determine the relu 

                for (const auto& n : neurons) {
                    for (size_t i = 0; i < l1Trues.size(); ++i) {
                        if (m_layers.front().weight.coeff(n, i) != 0) {
                            cycles += l1Trues.at(i);

                            if (!wasUsed.at(i)) {  // we have to load this
                                wasUsed[i] = true;

                                // add cost of `if (inp > 0)`
                                auto num = l2Trues.at(n);
                                assert(total >= num);
                                cycles += total;    // cost of loading variable
                                cycles += 3*(total-num);  // cost of < 0
                            }
                        }
                    }
                }
            }

            // ReLU cost
            if (!checks.empty()) {
                for (const auto& n : neurons) {
                    cycles += total + 3*(total-l2Trues.at(n));  // load var + penalty for < 0
                }
            }

            // still need to handle paths of special treatment
            for (const auto& c : checks) {
                // determine which paths match
                auto matches = extractPaths(paths, [&](const auto& p) -> bool {
                    return decisionsMayOverlap(p.decisions, c.decisions);
                });

                if (false && visitFreq(matches) > c.visitFreq) {  // can happen if classes are violated!
                    fmt::print("  {}  \n", c);
                    fmt::print("{} - {}\n", matches.size(), paths.size());
                    fmt::print("{} - {}; {}\n", visitFreq(matches), visitFreq(paths), c.visitFreq);
                    for (const auto& buf : matches) {
                        fmt::print("> {}\n", buf);
                    }
                    assert(false);
                }

                // add cost of check
                cycles += visitFreq(matches) * 3;  // cost of matches
                cycles += visitFreq(paths) * 3;    // cost of not-matching
            }
        }

        // ReLU + Requant of 2nd Layer for last layer
        auto notTree = layerTrues(paths, 1);
        auto notOnTree = visitFreq(paths);
        for (size_t i = 0; i < m_layers.at(1).bias.size(); ++i) {  // FIXME
            cycles += notOnTree + 3*(notOnTree-notTree.at(i));
        }

        
        // multiplications of last layer
        for (const auto& row : m_layers.back().weight.rowwise()) {
            for (size_t i = 0; i < row.size(); ++i) {
                if (i != 0) {
                    cycles += notTree.at(i);
                }
            }

            // requantization
            // FIXME
        }
        
        return cycles / totalSamples;
    }
};
}