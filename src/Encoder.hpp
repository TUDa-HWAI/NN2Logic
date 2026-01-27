#pragma once

#include <string>
#include <gurobi_c++.h>
#include <algorithm>
#include <memory>
#include <limits>
#include <map>
#include <fmt/format.h>
#include <set>
#include <boost/dynamic_bitset.hpp>
#include <list>

class Encoder final {
    std::vector<std::string> m_varNames;
    std::map<size_t,std::vector<std::string>> m_layerVarNames;
    std::map<std::string,double> m_binaryScaler;

    // Bookkeeping for binaries
    std::set<std::string> m_binaryNames;
    std::vector<std::pair<std::set<std::string>,bool>> m_binaryGroups;  // bool -> strict

    // Gurobi stuff
    GRBEnv m_env = GRBEnv(true);
    std::unique_ptr<GRBModel> m_model;

    void initEnv() {
        // make env quiet
        m_env.set(GRB_IntParam_OutputFlag, 0);
        m_env.start();
    }

    void setModelParams() {
        //m_model.set(GRB_IntParam_OutputFlag, 0);  // make gurobi quiet
        //m_model->set(GRB_IntParam_ConcurrentMIP, 1);
        m_model->set(GRB_IntParam_MIPFocus, 1);  // focus on finding feasible solutions quickly
        m_model->set(GRB_IntParam_SolutionLimit, 1);  // stop after first feasible solution
        //m_model->set(GRB_IntParam_Threads, 2);
        //m_model.set(GRB_IntParam_IISMethod, 1);

        m_model->set(GRB_DoubleParam_TimeLimit, 3600.0);  // set a timeout of 30s
        //m_model.set(GRB_DoubleParam_PerturbValue, 0.002);  // default is  	0.0002
    }


    // generate all possible one hot encodings for a given vector size
    static std::list<boost::dynamic_bitset<>> oneHotPermutations(size_t num, bool strict) {
        assert(num > 0 || !strict);
        std::list<boost::dynamic_bitset<>> toRet;
        
        if (!strict) {
            toRet.emplace_front(num);
        }

        for (size_t i = 0; i < num; ++i) {
            toRet.emplace_front(num);
            toRet.front()[i] = true;
        }

        return toRet;
    }


public:
    Encoder(void) {
        initEnv();
        m_model = std::make_unique<GRBModel>(m_env);
        setModelParams();
    }

    // Copy constructor, as env cannot be shared among threads. Therefore new env for each Encoder instance
    Encoder(const Encoder& other) : m_varNames(other.m_varNames), m_layerVarNames(other.m_layerVarNames),
                                    m_binaryNames(other.m_binaryNames), m_binaryGroups(other.m_binaryGroups), m_binaryScaler(other.m_binaryScaler) {
        initEnv();
        m_model = std::make_unique<GRBModel>(*other.m_model, m_env);
        setModelParams();
    }

    struct Model final {
        std::unique_ptr<GRBModel> grb;
        std::vector<GRBVar> vars;
        std::map<size_t,std::vector<GRBVar>> layerVars;

        std::vector<GRBVar> layerInputVars(size_t layerIdx) const {
            if (layerIdx == 0) return vars;
            return layerVars.at(layerIdx-1);
        }

        GRBVar neuronVar(size_t layerIdx, size_t neuronIdx) const {
            return layerVars.at(layerIdx).at(neuronIdx);
        }

        // Binary nightmares
        struct BinaryGroup final {
            std::vector<GRBVar> members;
            const bool strict;

            [[nodiscard]] size_t size() const {
                return members.size();
            }
        };

        std::vector<BinaryGroup> binaries;

        [[nodiscard]] bool containsBinaries() const {
            return !binaries.empty();
        }

        struct BinaryPermutation final {
            std::vector<GRBVar> members;
            boost::dynamic_bitset<> values;

            std::vector<GRBConstr> constraints;

            void apply(GRBModel* model) {
                assert(constraints.empty());

                for (size_t i = 0; i < members.size(); ++i) {
                    constraints.push_back(model->addConstr(members[i], GRB_EQUAL, values[i] ? 1.0 : 0.0));
                }
            }

            void release(GRBModel* model) {
                for (auto& c : constraints) {
                    model->remove(c);
                }
                constraints.clear();
            }

            void append(const std::vector<GRBVar>& members, const boost::dynamic_bitset<>& values) {
                this->members.insert(this->members.end(), members.begin(), members.end());
                this->values.reserve(this->members.size());

                for (size_t i = 0; i < values.size(); ++i) {  // dynamic_bitset has no neater way apparently?
                    this->values.push_back(values[i]);
                }

                assert(this->members.size() == this->values.size());
            }
        };

        [[nodiscard]] std::list<BinaryPermutation> binaryInputPermutations() const {
            std::list<BinaryPermutation> toRet{BinaryPermutation{}};  // put an empty one in there
            
            for (const auto& bg : binaries) {
                // save current state of to ret as starting point
                std::list<BinaryPermutation> buf;
                std::swap(toRet, buf);

                // get permutations
                for (const auto& ohp : oneHotPermutations(bg.size(), bg.strict)) {
                    std::list<BinaryPermutation> im(buf.begin(), buf.end());  // copy list
                    for (auto& bp : im) {
                        bp.append(bg.members, ohp);  // append current thing
                    }
                    toRet.insert(toRet.begin(), im.begin(), im.end());  // put all those back into toRet
                }
            }
            
            return toRet;
        }
    };

    void registerInt(const std::string& name, int upperLimit = 127) {
        m_varNames.push_back(name);
         m_model->addVar(0.0, upperLimit, 0.0, GRB_INTEGER, name);
    }

    void registerScalar(const std::string& name, double upperLimit = 1.0) {
        m_varNames.push_back(name);
        m_model->addVar(0.0, upperLimit, 0.0, GRB_CONTINUOUS, name);
    }

    void createLayerVars(size_t layerIdx, size_t numNeurons) {
        for (size_t i = 0; i < numNeurons; ++i) {
            std::string name = fmt::format("layer_{}_neuron_{}", layerIdx, i);
            m_layerVarNames[layerIdx].push_back(name);
            m_model->addVar(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max(), 0.0, GRB_CONTINUOUS, name);
        }
    }

    void registerBinary(const std::string& name, double upperLimit = 1.0) {
        m_varNames.push_back(name);
        m_model->addVar(0.0, 1.0, 0.0, GRB_BINARY, name);
        if (upperLimit != 1.0) {
            m_binaryScaler[name] = upperLimit;
        }

        m_binaryNames.insert(name);
    }

    void markBinariesOneHot(const std::vector<std::string>& names, bool strict = false) {
        m_model->update();
        GRBLinExpr expr = 0;
        for (const auto& n : names) {  // no quicksum in c++ api
            expr += m_model->getVarByName(n);
        }

        m_model->addConstr(expr, strict ? GRB_EQUAL : GRB_LESS_EQUAL, 1.0);  // FIXME

        // save binary group
        m_binaryGroups.emplace_back(std::piecewise_construct, std::forward_as_tuple(names.begin(), names.end()), std::forward_as_tuple(strict));  // sorcery
    }

    // we are done creating the model, now presolve
    void update() {
        m_model->update();
        m_model->tune();
        //m_model.presolve();  // ?
    }

    // get BaseModel and variables in creation order
    [[nodiscard]] Model getBaseModel() const {
        // create new gurobi model and transfer variables based on names
        auto model = std::make_unique<GRBModel>(*m_model);
        std::vector<GRBVar> vars;
        vars.reserve(m_varNames.size());

        std::map<size_t,std::vector<GRBVar>> layerVars;

        auto varTransform = [&](const auto& name) -> GRBVar {
            auto var = model->getVarByName(name);

            auto it = m_binaryScaler.find(name);
            if (it != m_binaryScaler.end()) {
                auto nVar = model->addVar(0.0, it->second, 0.0, GRB_CONTINUOUS);
                model->addConstr(nVar, GRB_EQUAL, it->second * var);
                return nVar;
            }

            return var;
        };

        std::transform(m_varNames.cbegin(), m_varNames.cend(), std::back_inserter(vars), varTransform);

        for (const auto& [idx, names] : m_layerVarNames) {
            std::transform(names.cbegin(), names.cend(), std::back_inserter(layerVars[idx]), varTransform);
        }
        
        model->update();

        // bookkeeping for binaries
        std::vector<Model::BinaryGroup> binaries;
        for (const auto& [names, strict] : m_binaryGroups) {  // groups first
            std::vector<GRBVar> members;
            members.reserve(names.size());

            std::transform(names.begin(), names.end(), std::back_inserter(members), varTransform);

            binaries.push_back(Model::BinaryGroup{ .members = members, .strict = strict });
        }

        for (const auto& name : m_binaryNames) {
            // check if handled by group
            if (std::any_of(m_binaryGroups.begin(), m_binaryGroups.end(), [&](const auto& group) -> bool {
                return group.first.find(name) != group.first.end();
            })) continue;

            // binary var is not in a group
            binaries.push_back(Model::BinaryGroup{ .members = {varTransform(name)}, .strict = false });
        }

        model->update();

        return Model{ .grb = std::move(model), .vars = vars, .layerVars = layerVars, .binaries = binaries };
    }
};