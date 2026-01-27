#pragma once

#include <vector>
#include <optional>
#include <gurobi_c++.h>
#include <fmt/format.h>
#include <limits>

#include "Encoder.hpp"
#include "ConstraintTracker.hpp"

namespace QTree {

template <typename T> class TreeDecision;

template <typename T>
class SEvaluator final {
    typedef Eigen::Matrix<T,Eigen::Dynamic,1> Vec_t;

    std::shared_ptr<std::pair<const Vec_t, T>> m_weightBias;  /// weight and bias

    double m_scale;  /// scaling factor for weight/output

    std::optional<T> m_lowerLimit;  /// lower limit for saturation (not implemented)
    std::optional<T> m_upperLimit;  /// upper limit for saturation


    std::list<std::weak_ptr<const TreeDecision<T>>> m_iis;  /// predecessing decisions that are responsible for constness

public:
    enum class GrbStatus {
        UNSAT, SAT, TIMEOUT
    };
    
    static GrbStatus modelUnsat(GRBModel& model) {
        model.optimize();
        const auto status = model.get(GRB_IntAttr_Status);

        if (status == GRB_OPTIMAL || status == GRB_SOLUTION_LIMIT) return GrbStatus::SAT;
        if (status == GRB_INFEASIBLE) return GrbStatus::UNSAT;
        if (status == GRB_TIME_LIMIT) return GrbStatus::TIMEOUT;
        
        throw std::runtime_error("Unhandled Gurobi status " + std::to_string(status));
    }


    SEvaluator(std::shared_ptr<std::pair<const Vec_t,T>> weightBias, double scale, std::optional<T> upperLimit = {}, std::optional<T> lowerLimit = {})
        : m_weightBias(weightBias), m_scale(scale), m_lowerLimit(lowerLimit), m_upperLimit(upperLimit) {}


    std::optional<T> upperLimit() const {
        return m_upperLimit;
    }

    GRBLinExpr baseEqu(const std::vector<GRBVar>& vars) const {
        assert(vars.size() == m_weightBias->first.size());

        // Build equation
        GRBLinExpr expr(m_weightBias->second);
        for (size_t i = 0; i < vars.size(); ++i) {
            expr += m_weightBias->first(i) * vars.at(i);
        }

        // scaling
        expr *= m_scale;

        return expr;
    }

/*
    std::vector<GRBConstr> satEqu(const GRBVar& dst, const GRBLinExpr& expr, GRBModel* model) const {
        std::vector<GRBConstr> constrs;
        if (!m_upperLimit.has_value()) return constrs;
        assert(!m_lowerLimit.has_value());

        auto addConstr = [&model, &constrs](auto lhs, auto type, auto rhs) {
            constrs.push_back(model->addConstr(lhs, type, rhs));
        };

        auto C = m_upperLimit.value();
        
        // Create Vars
        auto x_1 = *model->addVars(1, GRB_CONTINUOUS);
        addConstr(x_1, GRB_EQUAL, expr);

        double M = 10e5;
        double eps = 10e-3;


        auto S_1 = model->addVars(1, GRB_BINARY);
        auto S_2 = model->addVars(3, GRB_BINARY);
        auto S_3 = model->addVars(1, GRB_BINARY);
        auto S_4 = model->addVars(3, GRB_BINARY);

        // x_1 >= C <=> S_1
        addConstr(x_1 - M*S_1[0], GRB_GREATER_EQUAL, C - M);
        addConstr(x_1 - M*S_1[0], GRB_LESS_EQUAL,    C - eps);

        // (dst = x_1) <=> S_2
        addConstr(dst - x_1 + M * S_2[1],   GRB_GREATER_EQUAL, eps);
        addConstr(dst - x_1 + M * S_2[1],   GRB_LESS_EQUAL,    M);
        addConstr(dst - x_1 - M * S_2[2],   GRB_GREATER_EQUAL, -M);
        addConstr(dst - x_1 - M * S_2[2],   GRB_LESS_EQUAL,    -eps);
        addConstr(-S_2[1] + S_2[0],         GRB_LESS_EQUAL,    0);
        addConstr(-S_2[2] + S_2[0],         GRB_LESS_EQUAL,    0);
        addConstr(S_2[1] + S_2[2] - S_2[0], GRB_LESS_EQUAL,    1);

        // (x_1 < C) <=> S_3
        addConstr(x_1 + M * S_3[0], GRB_GREATER_EQUAL, C);
        addConstr(x_1 + M * S_3[0], GRB_LESS_EQUAL,    M + C - eps);

        // (dst=C) <=> S_4
        addConstr(dst + M * S_4[1],         GRB_GREATER_EQUAL, C + eps);
        addConstr(dst + M * S_4[1],         GRB_LESS_EQUAL,    M + C);
        addConstr(dst - M * S_4[2],         GRB_GREATER_EQUAL, C - M);
        addConstr(dst - M * S_4[2],         GRB_LESS_EQUAL,    C - eps);
        addConstr(-S_4[1] + S_4[0],         GRB_LESS_EQUAL,    1);
        addConstr(-S_4[2] + S_4[0],         GRB_LESS_EQUAL,    1);
        addConstr(S_4[1] + S_4[2] - S_4[0], GRB_LESS_EQUAL,    1);

        // Boolean ORs
        addConstr(S_1[0] + S_2[0], GRB_GREATER_EQUAL, 1);
        addConstr(S_3[0] + S_4[0], GRB_GREATER_EQUAL, 1);

        return constrs;
    }
*/

    static void addPriorDecisions(Encoder::Model& model, ConstraintTracker<TreeDecision<T>>& constrTracker,
        const std::list<std::pair<bool,std::shared_ptr<TreeDecision<T>>>>& decisionPath) {

        for (const auto& [dec, node] : decisionPath) {
            auto nVar = model.neuronVar(node->layerIdx, node->neuronIdx);
            GRBLinExpr expr = node->m_evalFunc->baseEqu(model.layerInputVars(node->layerIdx));

            if (dec) {  // positive, we need saturation and scaling   
                auto uL = node->m_evalFunc->upperLimit();
                if (uL.has_value()) {   // saturation
                    auto tmp = *model.grb->addVars(1, GRB_CONTINUOUS);
                    constrTracker.add(node, model.grb->addConstr(tmp, GRB_EQUAL, expr));
                    constrTracker.add(node, model.grb->addGenConstrMin(nVar, &tmp, 1, uL.value()));
                } else {
                    constrTracker.add(node, model.grb->addConstr(nVar, GRB_EQUAL, expr));
                }

                constrTracker.add(node, model.grb->addConstr(-expr, GRB_LESS_EQUAL, 0.0));

            } else {  // negative -> 0 ; only works because last layer will never do a ReLU!!
                constrTracker.add(node, model.grb->addConstr(expr, GRB_LESS_EQUAL, 0.0));
                constrTracker.add(node, model.grb->addConstr(nVar, GRB_EQUAL, 0.0));
            }
        }
    }

    /// get list of nodes that are part of the IIS
    std::list<std::shared_ptr<const TreeDecision<T>>> getIIS() const {
        std::list<std::shared_ptr<const TreeDecision<T>>> nodes;
        std::transform(m_iis.begin(), m_iis.end(), std::front_inserter(nodes), [](auto n) {
            return n.lock();
        });
        //assert(!nodes.empty());
        return nodes;
    }

    /// use gurobi to determine if a certain decision is possible given the provided path
    bool checkPathPossible(std::shared_ptr<const TreeDecision<T>> owner, bool dec, const Encoder* encoder) {
        assert(m_iis.empty());
        auto model = encoder->getBaseModel();

        // add equations of previous decisions to solver
        ConstraintTracker<TreeDecision<T>> constrTracker;
        addPriorDecisions(model, constrTracker, owner->decisionPath());

        // get own equation and add as constraint
        model.grb->addConstr(baseEqu(model.layerInputVars(owner->layerIdx)), (dec ? GRB_GREATER_EQUAL : GRB_LESS_EQUAL), 0.0);

        // check if solution exists
        if (modelUnsat(*model.grb) == GrbStatus::UNSAT) {
            model.grb->computeIIS();
            auto iis = constrTracker.getIIS();
            m_iis = std::list<std::weak_ptr<const TreeDecision<T>>>(iis.begin(), iis.end());

            return false;
        }

        return true;
    }

    /// checks the ReLU condition
    bool decide(const Eigen::Matrix<double, Eigen::Dynamic,1>& sample) const {
        assert(m_weightBias->first.size() == sample.size());
        
        double res = m_weightBias->second;
        for (size_t i = 0; i < sample.size(); ++i) {
            res += m_weightBias->first(i) * sample(i);
        }

        return res > 0.0;
    }
};
}