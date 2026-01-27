#pragma once

#include <map>
#include <set>
#include <gurobi_c++.h>


namespace QTree {

template <typename T>
class ConstraintTracker final {
    struct CStore final {
        std::vector<GRBConstr> m_reg;
        std::vector<GRBGenConstr> m_gen;

        template <decltype(GRB_IntAttr_IISConstr) Attr, typename C>
        static bool any_of(const std::vector<C>& vec) {
            for (const auto& c : vec) {
                if (c.get(Attr) == 1) return true;
            }
            return false;
        }

        bool isIIS() const {
            if (any_of<GRB_IntAttr_IISConstr>(m_reg)) return true;
            if (any_of<GRB_IntAttr_IISGenConstr>(m_gen)) return true;

            return false;
        }
    };

    std::map<std::shared_ptr<T>, CStore> store;

public:
    void add(std::shared_ptr<T> dec, const GRBConstr& constr) {
        store[dec].m_reg.push_back(constr);
    }

    void add(std::shared_ptr<T> dec, const std::vector<GRBConstr>& constrs) {
        std::copy(constrs.begin(), constrs.end(), std::back_inserter(store[dec].m_reg));
    }

    void add(std::shared_ptr<T> dec, const GRBGenConstr& constr) {
        store[dec].m_gen.push_back(constr);
    }

    bool getIIS(std::shared_ptr<T> dec) const {
        return store.at(dec).isIIS();
    }

    std::list<std::shared_ptr<T>> getIIS() const {
        std::set<std::shared_ptr<T>> toRet;

        for (const auto& [k, v] : store) {
            if (v.isIIS()) {
                toRet.insert(k);
            }
        }

        return std::list<std::shared_ptr<T>>(toRet.begin(), toRet.end());
    }
};
}