#pragma once

#include "Path.hpp"

#include <list>
#include <map>
#include <vector>
#include <queue>
#include <fmt/format.h>
#include <gurobi_c++.h>
#include "VarFac.hpp"

namespace QTree {

using Idx_t = std::pair<size_t,size_t>;


class IISMarker final {
    class ConstDecTracker final {
        const GRBVar cVar;
        const GRBVar dVar;
        std::vector<std::pair<size_t,std::vector<GRBVar>>> collection;

    public:
        ConstDecTracker(const GRBVar& cVar, const GRBVar& dVar) : cVar(cVar), dVar(dVar) {}

        void add(size_t leaf, const std::vector<GRBVar>& iis, bool value) {
            collection.emplace_back(leaf, iis);
        }

        void addToModel(GRBModel& model, const GrbVarFac<size_t> lFac) const;

        [[nodiscard]] bool isConst() const {
            return cVar.get(GRB_DoubleAttr_X) == 1;
        }
    };

    std::list<path::Path> m_paths;

    void savePath(const std::vector<path::Path>& paths, const std::vector<size_t>& chosen, const std::vector<Idx_t>& consts,
        const GrbVarFac<Idx_t>& dFac, const GrbVarFac<Idx_t>& iisFac, GRBModel& model);


    void processConst(const std::vector<path::Path>& paths) {
        // parameters
        double alpha = 1.0, beta = 1.0;

        // Model and Variable factories
        GRBModel model(GRBEnv(false));
        model.set(GRB_IntParam_OutputFlag, 0);

        GrbVarFac<Idx_t> cFac("c", &model);
        GrbVarFac<size_t> lFac("l", &model);
        GrbVarFac<Idx_t> dFac("d", &model);
        GrbVarFac<Idx_t> iisFac("iis", &model);

        std::map<Idx_t, GRBVar> inverted;  // inverted d-vars


        // convert a vec of decisions to a vector of gurobi vars
        auto decToGrb = [&dFac,&model,&inverted](const std::vector<path::Decision>& decs) -> std::vector<GRBVar> {
            std::vector<GRBVar> toRet;
            toRet.reserve(decs.size());

            for (const auto& d : decs) {
                auto var = dFac.get(d.idx());

                if (d.decision) {
                    toRet.push_back(var);
                } else {  // need to invert it
                    auto it = inverted.find(d.idx());
                    if (it == inverted.end()) {  // does not exist
                        auto tmp = model.addVar(0, 1, 0, GRB_BINARY, fmt::format("nd{}_{}", d.layerIdx, d.neuronIdx));
                        model.addConstr(tmp, GRB_EQUAL, 1-var);
                        inverted[d.idx()] = tmp;

                        toRet.push_back(tmp);
                    } else {  // exists
                        toRet.push_back(it->second);
                    }
                }
            }

            return toRet;
        };

        // objective
        GRBQuadExpr obj;

        // store for const decision conditions
        std::map<std::pair<size_t,size_t>,ConstDecTracker> constDecCond;

        for (size_t i = 0; i < paths.size(); ++i) {
            auto leaf = lFac.get(i);

            // add complete path as part of potential iis check
            {
                auto whole = decToGrb(paths[i].decisions);
                for (size_t ii = 0; ii < whole.size(); ++ii) {
                    auto idx = paths[i].decisions[ii].idx();
                    model.addQConstr(leaf, GRB_LESS_EQUAL, iisFac.get(idx) * whole[ii]);
                    //model.addConstr(iisFac.get(idx) * whole[ii] - leaf, GRB_GREATER_EQUAL, 0);
                }
            }

            // add leaf IIS
            // convert iis decisions to gurobi
            std::vector<path::Decision> iis = paths[i].leaf.iisDecisions(paths[i].decisions);
            auto grbDec = decToGrb(iis);

            // create AND constraint
            auto tmp = model.addVar(0, 1, 0, GRB_BINARY, fmt::format("l{}iis", i));
            model.addGenConstrAnd(tmp, grbDec.data(), grbDec.size());

            // leaf only possible if AND constraint holds
            model.addConstr(leaf, GRB_LESS_EQUAL, tmp);

            // add leaf to objective function
            obj += alpha * paths[i].visitFreq * leaf;

            // handle constness of decisions
            std::queue<path::Decision> work;
            for (const auto& d : iis) {  // add const iis decisions to worklist, regular ones are already handled
                // transient iis for consts
                if (d.isConst) {
                    work.push(d);
                }

                // add iis constraint for leaf
                model.addConstr(iisFac.get(d.idx()), GRB_GREATER_EQUAL, leaf);
            }

            // process worklist of iis-consts
            std::set<Idx_t> used;
            while (!work.empty()) {
                const auto idx = work.front().idx();

                used.insert(idx);  // mark used
                auto var = cFac.get(idx);  // get gurobi var

                // add to objective
                obj += beta * leaf * paths[i].visitFreq * var;

                // handle iis constraints
                auto iis = paths[i].getDecisions(work.front().iisNodes);  // get iis decisions

                // tracking for gurobi magic
                {
                    auto elem = constDecCond.try_emplace(idx, var, dFac.get(idx)).first;
                    elem->second.add(i, decToGrb(iis), work.front().decision);
                }

                // remove first item
                work.pop();

                // add const decisions to worklist
                for (const auto& d : iis) {
                    // if const, not in first layer, and not already processed
                    if (d.isConst && d.layerIdx > 0 && used.find(d.idx()) == used.end()) {
                        work.push(d);
                    }

                    // add iis constraint for const
                    model.addQConstr(iisFac.get(idx), GRB_GREATER_EQUAL, leaf * var);  // ensures that decision is marked iis if leaf and const are chosen
                }
            }
        }

        // insert const decision stuff
        for (const auto& [k, v] : constDecCond) {
            v.addToModel(model, lFac);
        }

        // add objective
        model.setObjective(obj, GRB_MAXIMIZE);

        double objVal = 0;

        do {
            // solve
            model.optimize();

            // get objective value
            objVal = model.get(GRB_DoubleAttr_ObjVal);

            // print solution
/*
            fmt::print("Objective: {}\n", objVal);
            fmt::print("Leaves: {}\n", fmt::join(lFac.solution(), ","));
            fmt::print("Consts: {} of {}\n", fmt::join(cFac.solution(), ", "), cFac.size());

            fmt::print("Paths:\n");
            for (size_t s : lFac.solution()) {
                fmt::print(" {:02d}) {}\n", s, paths.at(s));
            } 
*/

            // save path
            savePath(paths, lFac.solution(), cFac.solution(), dFac, iisFac, model);

            // ban solution
            for (const auto& s : lFac.solution()) {
                //model.addConstr(lFac.at(s), GRB_EQUAL, 0);
                lFac.remove(s);
            }
            model.reset();
        } while (objVal >= 100);  // FIXME: threshold
    }



public:
    explicit IISMarker(const std::list<path::Path>& paths) {
        // Sort by possible classes
        std::map<std::vector<bool>,std::list<path::Path>> byClasses;
        for (const auto& p : paths) {
            byClasses[p.leaf.possClasses()].emplace_front(p);
        }

        // do your thing
        for (const auto& [k, v] : byClasses) {
            fmt::print("Processing: {:d}\n", fmt::join(k, ""));

            size_t count = std::count(k.begin(), k.end(), true);
            if (count < k.size()) {  // const leaf
                std::vector<path::Path> vec(v.begin(), v.end());
                processConst(vec);
            } else {
                // TODO
                fmt::print("skipping non-const leaves");
            }
        }

        // sort paths
        m_paths.sort([](const auto& a, const auto& b) -> bool {
            if (a.visitFreq == b.visitFreq) return a.decisions.size() < b.decisions.size();
            return a.visitFreq > b.visitFreq;
        });
    }

    [[nodiscard]] std::list<path::Path> getPaths() const {
        return m_paths;
    }
};

}