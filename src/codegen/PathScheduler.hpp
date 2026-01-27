#pragma once

#include "Path.hpp"
#include "quant/VarFac.hpp"
//#include "helper.hpp"
#include <list>
#include <gurobi_c++.h>
#include <algorithm>
#include <memory>
#include <limits>
#include <map>
#include <fmt/format.h>

namespace codegen {

class ConstSchedule final {
public:
    //typedef std::set<path::Path,std::greater<path::Path>> PSet_t;
    typedef std::list<path::Path> PSet_t;
private:
    std::vector<std::pair<size_t,PSet_t>> ts;

    ConstSchedule(decltype(ts)::const_iterator itb, decltype(ts)::const_iterator ite) : ts(itb, ite) {}

    size_t m_numPaths = 0;

public:
    ConstSchedule() {}
    ConstSchedule(const ConstSchedule& other) : ts(other.ts), m_numPaths(other.m_numPaths) {}


    [[nodiscard]] bool empty() const {
        if (ts.empty()) {
            assert(m_numPaths == 0);
        }
        return ts.empty();
    }


    void addNeuron(size_t idx) {
        ts.emplace_back(idx, PSet_t{});
    }

    void addPath(const path::Path& path) {
        assert(!ts.empty());
        assert(!path.decisions.empty());
        //ts.back().second.insert(path);
        ts.back().second.push_back(path);
        
        m_numPaths += 1;

        // reduce decisions to non-const iis
        // auto reduced = path.reduce();
        // reduced.decisions = reduced.nonConst();

        //ts.back().second.insert(reduced);
    }

    decltype(ts)::const_iterator begin() const {
        return ts.cbegin();
    }

    decltype(ts)::const_iterator end() const {
        return ts.cend();
    }

    [[nodiscard]] std::pair<ConstSchedule,ConstSchedule> split() const {
        // check if split is actually possible
        if (!ts.back().second.empty()) {
            return std::make_pair(ConstSchedule(ts.cbegin(), ts.cend()), ConstSchedule{});  // basically returns a copy of itself
        }

        int pos = ts.size()-1;
        for (; pos >= 0; --pos) {
            if (!ts.at(pos).second.empty()) break;
        }
        auto it = std::next(ts.begin(), pos+1);
        //fmt::print("First pos after checks: {}\n", pos+1);  // 16


        return std::make_pair(ConstSchedule(ts.begin(), it), ConstSchedule(it, ts.end()));
    }

    [[nodiscard]] size_t size() const {
        return ts.size();
    }

    [[nodiscard]] std::vector<std::pair<std::set<size_t>,PSet_t>> bulk() const {
        std::vector<std::pair<std::set<size_t>,PSet_t>> toRet;

        for (const auto& [neuron, checks] : ts) {
            if (toRet.empty() || !toRet.back().second.empty()) {
                toRet.emplace_back();
            }

            toRet.back().first.insert(neuron);
            toRet.back().second = checks;
        }

        return toRet;
    }

    void print() const {
        size_t i = 0;
        for (const auto& [n, checks] : ts) {
            fmt::print("{:02d}) {}\n", i++, n);
            for (const auto& c : checks) {
                fmt::print("  - {}\n", c);
            }
        }
    }

    [[nodiscard]] size_t numPaths() const {
        return m_numPaths;
    }

    [[nodiscard]] size_t numNeurons() const {
        return ts.size();
    }

    [[nodiscard]] std::vector<path::Path> getPaths() const {
        std::vector<path::Path> toRet;

        for (const auto& [a, b] : ts) {
            toRet.insert(toRet.end(), b.begin(), b.end());
        }

        return toRet;
    }
};

}