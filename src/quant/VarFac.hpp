#pragma once

#include <map>
#include <string>
#include <fmt/format.h>
#include <gurobi_c++.h>

template <typename> struct is_pair: std::false_type {};
template <typename ...T> struct is_pair<std::pair<T...>>: std::true_type {};


// Creates binary variables
template <typename T>
class GrbVarFac final {  // for binary variables
    const std::string prefix;
    GRBModel* model;
    std::map<T,GRBVar> store;

public:
    GrbVarFac(const std::string& prefix, GRBModel* model) : prefix(prefix), model(model) {}

    GRBVar get(T idx) {
        // check if variable exists
        auto it = store.find(idx);
        if (it != store.end()) return it->second;

        // create var
        std::string name;
        if constexpr (is_pair<T>::value) {
            name = fmt::format("{}{}_{}", prefix, idx.first, idx.second);
        } else {
            static_assert(std::is_scalar<T>::value);
            name = fmt::format("{}{}", prefix, idx);
        }
         
        GRBVar var = model->addVar(0, 1, 0, GRB_BINARY, name);
        
        // store and return
        store[idx] = var;

        return var;
    }

    GRBVar at(T idx) const {
        return store.at(idx);
    }

    std::set<T> idxs() const {
        std::set<T> toRet;

        for (const auto& [k, v] : store) {
            toRet.insert(k);
        }

        return toRet;
    }

    std::vector<GRBVar> get(const std::vector<T>& idxs) {
        std::vector<GRBVar> toRet;
        toRet.reserve(idxs.size());

        for (T i : idxs) {
            toRet.push_back(get(i));
        }

        return toRet;
    }

    // check if a certain index exists
    [[nodiscard]] bool contains(T idx) const {
        return store.find(idx) != store.end();
    }

    [[nodiscard]] std::vector<T> solution() const {
        std::vector<T> toRet;

        for (const auto& [k, v] : store) {
            if (v.get(GRB_DoubleAttr_X) == 1) {
                toRet.push_back(k);
            }
        }

        return toRet;
    }

    [[nodiscard]] size_t size() const {
        return store.size();
    }

    /// obtain solution for a variable
    [[nodiscard]] bool value(T idx) const {
        return store.at(idx).get(GRB_DoubleAttr_X) == 1;
    }

    /// remove variable from model
    void remove(T idx) {
        model->remove(store.at(idx));
        store.erase(idx);
    }


    /// make it iterable
    typename std::map<T,GRBVar>::const_iterator begin() const {
        return store.cbegin();
    }

    /// make it iterable
    typename std::map<T,GRBVar>::const_iterator end() const {
        return store.cend();
    }
};