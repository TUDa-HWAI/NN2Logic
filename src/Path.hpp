#pragma once

#include <memory>
#include <vector>
#include <set>
#include <cassert>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <boost/dynamic_bitset.hpp>
#include <bitset>

namespace path {

struct Decision final {
    size_t layerIdx;
    size_t neuronIdx;
    bool decision;
    bool iisMember;
    bool isConst;

    std::set<std::pair<size_t,size_t>> iisNodes = {};

    std::pair<size_t,size_t> idx() const {
        return std::make_pair(layerIdx, neuronIdx);
    }

    friend bool operator==(const Decision& a, const Decision& b) {
        return std::tie(a.layerIdx, a.neuronIdx, a.decision, a.iisMember, a.isConst, a.iisNodes) ==
            std::tie(b.layerIdx, b.neuronIdx, b.decision, b.iisMember, b.isConst, b.iisNodes);
    }
};

struct DecisionSorter final {
    bool operator()(const Decision& lhs, const Decision& rhs) const {
        if (lhs.layerIdx != rhs.layerIdx) return lhs.layerIdx < rhs.layerIdx;
        if (lhs.neuronIdx != rhs.neuronIdx) return lhs.neuronIdx < rhs.neuronIdx;

        if (lhs.decision != rhs.decision) return lhs.decision;

        // makes no sense otherwise
        assert(lhs.iisMember == rhs.iisMember);
        assert(lhs.isConst == rhs.isConst);

        return false;
    };
};


class Leaf {
    std::vector<bool> m_isPossible;
    std::set<std::pair<size_t,size_t>> iisNodes;
public:
    Leaf(const std::vector<bool> &posMax, const std::set<std::pair<size_t,size_t>>& iisNodes = {})
        : m_isPossible(posMax), iisNodes(iisNodes) {}

    Leaf() {}

    [[nodiscard]] bool isConst() const {
        return std::count(m_isPossible.begin(), m_isPossible.end(), true) == 1;
    }

    [[nodiscard]] bool isPossible(size_t idx) const {
        return m_isPossible[idx];
    }

    [[nodiscard]] std::vector<Decision> iisDecisions(const std::vector<Decision>& decisions) const {
        assert(isConst());

        std::vector<Decision> toRet;
        toRet.reserve(decisions.size());

        std::copy_if(decisions.begin(), decisions.end(), std::back_inserter(toRet), [&](const auto& d) -> bool {
            return iisNodes.find(d.idx()) != iisNodes.end();
        });

        assert(!toRet.empty());
        return toRet;
    }

    std::set<std::pair<size_t,size_t>> iisIDxs() const {
        return iisNodes;
    }

    std::vector<bool> possClasses() const {
        return m_isPossible;
    }
};


struct Path final {
    std::vector<Decision> decisions;
    Leaf leaf;
    size_t visitFreq = 0;


    friend bool operator<(const Path& lhs, const Path& rhs) {
        if (lhs.visitFreq == rhs.visitFreq) return lhs.decisions.size() < rhs.decisions.size();
        return lhs.visitFreq < rhs.visitFreq;
    }

    friend bool operator>(const Path& lhs, const Path& rhs) {
        return !(lhs < rhs);
    }

    std::optional<Decision> getDecision(size_t layerIdx, size_t neuronIdx) const {
        auto it = std::find_if(decisions.begin(), decisions.end(), [&](const auto& d) -> bool {
            return d.layerIdx == layerIdx && d.neuronIdx == neuronIdx;
        });

        if (it == decisions.end()) return {};
        return *it;
    }

    std::vector<Decision> getDecisions(const std::set<std::pair<size_t,size_t>>& idxs) const {
        std::vector<Decision> toRet;
        for (const auto& i : idxs) {
            toRet.push_back(getDecision(i.first, i.second).value());
        }
        return toRet;
    }

    std::vector<Decision> iisDecisions() const {
        return leaf.iisDecisions(decisions);
/*
        assert(!decisions.empty());

        std::vector<Decision> toRet;
        std::copy_if(decisions.begin(), decisions.end(), std::back_inserter(toRet), [](const auto& d) {
            return d.iisMember;
        });

        assert(!toRet.empty());

        return toRet;
*/
    }

    std::vector<Decision> nonConst() const {
        std::vector<Decision> toRet;

        std::copy_if(decisions.begin(), decisions.end(), std::back_inserter(toRet), [](const auto& d) {
            return !d.isConst;
        });

        return toRet;
    }

    Path reduce() const {
        std::vector<Decision> iisDec;
        std::copy_if(decisions.begin(), decisions.end(), std::back_inserter(iisDec), [](const auto& d) -> bool {
            return d.iisMember;
        });

        return Path{
            .decisions = iisDec,
            .leaf = leaf,
            .visitFreq = visitFreq
        };
    }

    class PathBitset final {  /// ffuuu, layer sind unterschiedlich groß und es gibt keinen single-int identifier für decisions. brauchen auch layer sizes
        //boost::dynamic_bitset<> m_vals;
        //boost::dynamic_bitset<> m_mask;
        std::bitset<64> m_vals;
        std::bitset<64> m_mask;
        const std::vector<size_t> m_layerSizes;

    public:
        explicit PathBitset(const std::vector<size_t> layerSizes) : m_layerSizes(layerSizes) {
            assert(!layerSizes.empty());
            size_t total = 0;
            for (size_t t : layerSizes) {
                total += t;
            }
            assert(total <= 64);
            //m_vals = boost::dynamic_bitset<>(total);
            //m_mask = boost::dynamic_bitset<>(total);
        }

        bool overlaps(const PathBitset& other) const {
            return ((m_vals ^ other.m_vals) & m_mask & other.m_mask).none();

        /*
            // mask is & of masks
            auto mask = m_mask & other.m_mask;

            // we care about differences in those masked bits -> xor
            auto buf = (m_vals & mask) ^ (other.m_vals & mask);

            // if there is a difference (a bit is 1): no overlap
            // no bit is 1: overlap
            return buf.none();
        */
        }

        void set(const std::pair<size_t,size_t>& pos, bool val) {
            // determine position in bitset
            size_t bitPos = pos.second;
            for (size_t i = 0; i < pos.first; ++i) {
                bitPos += m_layerSizes.at(i);
            }

            // set val
            m_vals[bitPos] = val;
            m_mask[bitPos] = true;
        }

        explicit PathBitset(const std::vector<size_t> layerSizes, const std::vector<Decision>& decisions) : PathBitset(layerSizes) {
            for (const auto& d : decisions) {
                set(d.idx(), d.decision);
            }
        }
    };


    PathBitset asBitset(const std::vector<size_t> layerSizes) const {
        return PathBitset(layerSizes, decisions);
    }

};

// JSON converters
void to_json(nlohmann::json& j, const Path& p);  /// Path <-> JSON
void from_json(const nlohmann::json& j, Path& p);

void to_json(nlohmann::json& j, const Decision& p);  /// Decisions <-> JSON
void from_json(const nlohmann::json& j, Decision& d);

void to_json(nlohmann::json& j, const Leaf& l);  /// Leaf <-> JSON
void from_json(const nlohmann::json& j, Leaf& l);
}


// format a decision
template <> struct fmt::formatter<path::Decision>: formatter<std::string_view> {
    auto format(const path::Decision& d, format_context& ctx) const {
        char c = ' ';
        if (d.iisMember && d.isConst) c = '#';
        else if (d.iisMember) c = 'i';
        else if (d.isConst) c = 'c';

        return formatter<string_view>::format(
            fmt::format("({},{:2d},{}{})", d.layerIdx, d.neuronIdx, d.decision ? 'T' : 'F', c), ctx);
    }
};


// takes care of output formatting of a path
template <> struct fmt::formatter<path::Path>: formatter<std::string_view> {
    auto format(const path::Path& p, format_context& ctx) const {
        auto leaf = p.leaf.isConst() ? fmt::format("c {:d}", fmt::join(p.leaf.possClasses(), "")) : "";
        auto end = fmt::format("{} -> {} {}", fmt::join(p.decisions, "-"), p.visitFreq, leaf);
        return formatter<string_view>::format(end, ctx);
    }
};