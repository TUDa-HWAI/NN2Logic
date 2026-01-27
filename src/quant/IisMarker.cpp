#include "IisMarker.hpp"
#include <map>
#include <gurobi_c++.h>


void QTree::IISMarker::ConstDecTracker::addToModel(GRBModel& model, const GrbVarFac<size_t> lFac) const {
    std::set<size_t> otherLeaves = lFac.idxs();
    GRBLinExpr ensLeaf;

    for (const auto& p : collection) {
        otherLeaves.erase(p.first);

        // to ensure one of the leaves is chosen
        auto leaf = lFac.at(p.first);
        ensLeaf += leaf;

        // construct linear expression that ensures iis conditions hold if relevant leaf is true 
        // (sum vars - len(vars)*leaf) >= 0

        GRBLinExpr expr;
        std::vector<double> ones(p.second.size(), 1);

        expr.addTerms(ones.data(), p.second.data(), ones.size());
        expr -= ones.size() * leaf;

        // add indicator constraint
        model.addGenConstrIndicator(cVar, 1, expr, GRB_GREATER_EQUAL, 0);
    }

    // ensure that only leaves are active where decision could be const
    GRBLinExpr sum;
    for (const auto& ol : otherLeaves) {
        sum += lFac.at(ol);
    }
    model.addGenConstrIndicator(cVar, 1, sum, GRB_EQUAL, 0);

    // ensure leaf
    //model.addConstr(cVar - ensLeaf, GRB_LESS_EQUAL, 0);  // FIXME: this one is certainly wrong
    //model.addGenConstrIndicator(cVar, 1, ensLeaf, GRB_EQUAL, 0);
}


void QTree::IISMarker::savePath(const std::vector<path::Path>& paths, const std::vector<size_t>& chosen, const std::vector<Idx_t>& consts,
    const GrbVarFac<QTree::Idx_t>& dFac, const GrbVarFac<QTree::Idx_t>& iisFac, GRBModel& model) {

    if (chosen.size() > 1) {
        fmt::print("Found one larger than 1!\n");
    }

    std::set<Idx_t> constIdx(consts.begin(), consts.end());

    // obtain the relevant paths
    std::vector<path::Path> relevant;
    relevant.reserve(chosen.size());
    for (size_t idx : chosen) {
        relevant.push_back(paths.at(idx));
    }

    // ensure sanity
    assert(std::all_of(relevant.begin(), relevant.end(), [&](const auto& p) -> bool {  // possible classes are equal
        return relevant.front().leaf.possClasses() == p.leaf.possClasses();
    }));

    // start constructing new path
    std::map<Idx_t,path::Decision> nPath;

    // populate map using first path
    for (const auto& d : relevant.front().decisions) {
        nPath[d.idx()] = path::Decision{
            .layerIdx = d.layerIdx,
            .neuronIdx = d.neuronIdx,
            .decision = dFac.value(d.idx()),
            .iisMember = iisFac.value(d.idx()),
            .isConst = constIdx.find(d.idx()) != constIdx.end()
        };
    }

    // iterate over all paths
    size_t visitFreq = 0;
    for (const auto& p : relevant) {
        visitFreq += p.visitFreq;

        // check path
        for (const auto& d : p.decisions) {
            if (d.layerIdx == 0) continue;

            if (nPath.at(d.idx()).isConst) {
                assert(d.isConst);  // sanity
                assert(d.decision == nPath[d.idx()].decision);
            }
        }
    }

    // copy only iis
    std::vector<path::Decision> decisions;
    std::map<Idx_t,path::Decision> iisPath;
    for (const auto& [i, d] : nPath) {
        if (d.iisMember && !d.isConst) {
            iisPath.emplace(i, d);
            decisions.push_back(d);
        }
    }

    // ensure that decisions match
    for (const auto& p : relevant) {
        for (const auto& d : p.decisions) {
            auto it = iisPath.find(d.idx());
            if (it != iisPath.end() && it->second.decision != d.decision) {
                model.write("fail.lp");
                fmt::print("Conflicting decision value for {}\n", d.idx());
                assert(false);
                //assert(it->second.decision == d.decision);
            }
        }
    }

    // construct final path
    m_paths.push_back(path::Path{
        .decisions = decisions,
        .leaf = relevant.front().leaf,
        .visitFreq = visitFreq
    });
}