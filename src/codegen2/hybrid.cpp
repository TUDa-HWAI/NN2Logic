#include "hybrid.hpp"

#include <fmt/format.h>
#include <fmt/os.h>
#include <optional>
#include <queue>
#include <sstream>

#include "visitors.hpp"

std::set<std::pair<size_t,size_t>> codegen2::HybridGen::obtainRequired(std::pair<size_t,size_t> inp) {
    if (isScheduled[inp]) return {};  // neuron has been scheduled already
        
    // our return
    std::set<std::pair<size_t,size_t>> toRet;
    toRet.insert(inp);
    
    std::queue<std::pair<size_t,size_t>> worklist;
    worklist.push(inp);

    while (!worklist.empty()) {
        // get relevant neuron from stack
        auto work = worklist.front();
        auto [weights, bias, req] = layers.at(work.first).getNeuron(work.second);
        worklist.pop();

        // if current neuron is in first layer we can skip as there are no predecessors
        if (work.first == 0) continue;

        // figure out which predecessor neurons are relevant
        for (size_t i = 0; i < weights.size(); ++i) {
            if (weights[i] == 0) continue;

            auto potCand = std::make_pair(inp.first-1, i);
            if (!isScheduled[potCand] && toRet.find(potCand) == toRet.end()) {
                toRet.insert(potCand);
                worklist.push(potCand);
            }
        }
    }

    return toRet;
}



std::pair<codegen2::HybridGen::CodeIt_t,codegen2::HybridGen::CodeIt_t> codegen2::HybridGen::getRange(codegen2::HybridGen::CodeIt_t vBegin,
    codegen2::HybridGen::CodeIt_t vEnd, const std::string& varname) {

    auto posA = std::find_if(vBegin, vEnd, [&](auto ptr) -> bool {
        auto im = std::dynamic_pointer_cast<codegen::c::Assignment>(ptr);
        if (im == nullptr) return false;

        auto ari = im->getArith();

        if (ari.has_value()) return ari->uses(varname);

        return false;
    });

    auto posB = std::find_if(posA, vEnd, [&](auto ptr) -> bool {
        auto im = std::dynamic_pointer_cast<codegen::c::Assignment>(ptr);
        if (im == nullptr) return true;

        auto ari = im->getArith();

        if (ari.has_value()) return !ari->uses(varname);

        return true;
    });

    return std::make_pair(posA, posB);
}



void codegen2::HybridGen::rangeToIf(const std::string& varname, size_t delta) {
    auto posA = code.begin();
    auto posB = code.begin();

    while (posA != code.end()) {
        std::tie(posA, posB) = getRange(posB, code.end(), varname);
        if (std::distance(posA, posB) < delta) continue;

        auto nIt = std::next(posA);  // increment by one

        std::vector<std::shared_ptr<codegen::c::Statement>> nextBlock(posA, posB);
        code.erase(nIt, posB);
        *posA = std::make_shared<codegen::c::Ifelse>(codegen::c::Comparison{
            .variable = varname,
            .type = codegen::c::Comparison::GT,
            .value = 0
        }, codegen::c::Block(nextBlock));

        // ?
        posB = nIt;
    }
}



codegen2::HybridGen::HybridGen(const std::vector<QTree::Layer<int>>& layers) : layers(layers) {
    // create Neuron IR
    size_t layerIdx = 0;
    for (const auto& l : layers) {
        for (size_t i = 0; i < l.numNeurons(); ++i) {
            auto [w, b, r] = l.getNeuron(i);

            if (l.relu) {  // relu
                neurons[{layerIdx, i}] = std::make_shared<IRNeuronReLu>(layerIdx, i, w, b, r);
            } else {
                neurons[{layerIdx, i}] = std::make_shared<IRNeuron>(layerIdx, i, w, b, r);
            }
        }

        ++layerIdx;
    }
}



void codegen2::HybridGen::addPath(const path::Path& p) {
    assert(p.decisions.size() <= 32);
    assert(!m_isFinal);

    /*
        * Get Stuff that needs to be scheduled
        * add tree checks
        * insert exit
        */

    // check what needs to be scheduled
    std::set<std::pair<size_t,size_t>> reqN;
    for (const auto& d : p.decisions) {
        auto req = obtainRequired(d.idx());
        reqN.insert(req.begin(), req.end());
    }

    // obtain neurons
    std::vector<std::shared_ptr<IRNeuron>> relNeurons;
    std::transform(reqN.begin(), reqN.end(), std::back_inserter(relNeurons), [&](auto idx) {
        isScheduled[idx] = true;  // mark as scheduled
        return neurons.at(idx);
    });

    // move multiplications into relu
    std::vector<Instr> remMul;
    for (auto n : relNeurons) {
        for (const auto& mul : n->popMuls()) {
            assert(mul.type == Instr::MULADD);

            auto it = std::find_if(relNeurons.begin(), relNeurons.end(), [&](auto blub) -> bool {
                return blub->var() == mul.opB;
            });

            if (it == relNeurons.end()) {  // do not handle the creating neuron atm
                remMul.push_back(mul);
            } else {
                std::dynamic_pointer_cast<IRNeuronReLu>(*it)->addToIf(mul);
            }
        }
    }

    // sort the multiplications
    auto mulSortCmp = [](const auto& a, const auto& b) {
        if (a.opB == b.opB) return a.dst < b.dst;
        if (a.opB.length() == b.opB.length()) return a.opB < b.opB;
        return a.opB.length() < b.opB.length();
    };
    std::sort(remMul.begin(), remMul.end(), mulSortCmp);


    // add tree checks
    auto treeVar = fmt::format("onTree{}", treeVars.size());
    treeVars.push_back(treeVar);

    std::map<std::string,Instr> treeTests;

    size_t testPos = 0;
    int64_t cmp = 0;
    for (const auto& d : p.decisions) {
        treeTests.emplace(neurons.at(d.idx())->var(), Instr(Instr::SET_BIT, treeVar, std::to_string(testPos)));

        cmp <<= 1;
        if (d.decision) cmp += 1;

        testPos += 1;
    }

    // create code for neurons
    decltype(code) section(remMul.begin(), remMul.end());
    for (auto rn : relNeurons) {
        section.push_back(rn->instantiateIfElse());
    }


    // path exit
    codegen::c::Block exitBlock;

    size_t outIdx = 0;
    for (const auto x : p.leaf.possClasses()) {
        exitBlock.create<codegen::c::Assignment>(fmt::format("out[{}]", outIdx++), int32_t(127)*x);
    }

    exitBlock.create<codegen::c::RawStatement>("return;\n");

    auto treeCheck = std::make_shared<codegen::c::Ifelse>(codegen::c::Comparison{
        .variable = treeVar,
        .type = codegen::c::Comparison::EQ,
        .value = cmp
    }, exitBlock);

    section.push_back(treeCheck);


    // add section to global code
    code.insert(code.end(), section.begin(), section.end());


    // finally insert the treechecks into relus
    for (const auto& [var, check] : treeTests) {
        ReLUFinder rf(var);
        auto it = std::find_if(code.begin(), code.end(), rf);
        assert(it != code.end());

        auto ptr = std::dynamic_pointer_cast<codegen::c::Ifelse>(*it);
        assert(ptr != nullptr);

        ptr->trueBlock.add(check);
    }
}



void codegen2::HybridGen::finalize() {
    assert(!m_isFinal);
    m_isFinal = true;

    // schedule remaining neurons
    std::set<std::pair<size_t,size_t>> remN;
    for (const auto& [i, n] : neurons) {
        if (!isScheduled[i]) {
            remN.insert(i);
        }
    }
    std::vector<std::shared_ptr<IRNeuron>> remNeurons;
    std::transform(remN.begin(), remN.end(), std::back_inserter(remNeurons), [&](auto idx) {
        return neurons.at(idx);
    });

    std::vector<Instr> remMul;
    for (auto n : remNeurons) {
        for (const auto& mul : n->popMuls()) {
            assert(mul.type == Instr::MULADD);

            auto it = std::find_if(remNeurons.begin(), remNeurons.end(), [&](auto blub) -> bool {
                return blub->var() == mul.opB;
            });

            if (it == remNeurons.end()) {  // do not handle the creating neuron atm
                remMul.push_back(mul);
            } else {
                std::dynamic_pointer_cast<IRNeuronReLu>(*it)->addToIf(mul);
            }
        }
    }

    // sort the multiplications
    auto mulSortCmp = [](const auto& a, const auto& b) {
        if (a.opB == b.opB) return a.dst < b.dst;
        if (a.opB.length() == b.opB.length()) return a.opB < b.opB;
        return a.opB.length() < b.opB.length();
    };
    std::sort(remMul.begin(), remMul.end(), mulSortCmp);


    // create code for neurons
    decltype(code) section(remMul.begin(), remMul.end());
    for (auto rn : remNeurons) {
        section.push_back(rn->instantiateIfElse());
    }

    // insert section code
    code.insert(code.end(), section.begin(), section.end());

    // add variable declarations and initializations
    for (auto [idx, neuron] : neurons) {
        VarUseFinder vf(neuron->var());
        auto pos = std::find_if(code.begin(), code.end(), vf);
        assert(pos != code.end());

        auto toIns = neuron->getInit();
        code.insert(pos, toIns.begin(), toIns.end());
    }

    // add if GT 0 regions
    // TODO future work: might also work nicely for inputs
    for (auto [idx, neuron] : neurons) {
        rangeToIf(neuron->var());
    }


    // add tree vars
    for (const auto& v : treeVars) {
        VarUseFinder vf(v);
        auto pos = std::find_if(code.begin(), code.end(), vf);
        assert(pos != code.end());
        code.insert(pos, std::make_shared<codegen::c::RawStatement>(fmt::format("uint32_t {} = 0;\n", v)));
    }


    // FIXME: move output neurons to out
    code.push_back(std::make_shared<codegen::c::RawStatement>("out[0] = x_2_00;\n"));
    code.push_back(std::make_shared<codegen::c::RawStatement>("out[1] = x_2_01;\n"));
    //conti.create<codegen::c::RawStatement>("out[0] = x_2_00;\n");
    //conti.create<codegen::c::RawStatement>("out[1] = x_2_01;\n");
}


void codegen2::HybridGen::renderToFile(const std::vector<std::shared_ptr<codegen::c::Statement>>& code, const std::string& filename) const {
    codegen::c::Function func{
        .name = "network",
        .retType = "void",
        .params = {*inp, *out},
        .members = code
    };

    // print to file
    auto file = fmt::output_file(filename);
    file.print("#include \"network.h\"\n\n");

    file.print(func.print());
}


void codegen2::HybridGen::render(const std::string& filename) const {
    assert(m_isFinal);
    renderToFile(code, filename);
}



class TreeRemovalVis : public codegen::c::CVisitor {
    std::set<std::string> m_relevant;
    std::shared_ptr<codegen::c::Statement> buf;

    // check if given name matches a tree variable name
    bool aNameMatches(const std::string& name) const {
        return std::any_of(m_relevant.begin(), m_relevant.end(), [&](const auto& str) -> bool {
            return str == name;
        });
    }


    bool condContainsVar(std::variant<codegen::c::Comparison, std::string> c) const {
        if (std::holds_alternative<codegen::c::Comparison>(c)) {
            return aNameMatches(std::get<codegen::c::Comparison>(c).variable);
        }

        auto s = std::get<std::string>(c);
        for (const auto& name : m_relevant) {
            if (s.find(name) != std::string::npos) {
                return true;
            }
        }

        return false;
    }


    codegen::c::Block filterBlock(codegen::c::Block block) {
        codegen::c::Block toRet;

        for (auto s : block) {
            s->accept(this);
            if (buf != nullptr) toRet.add(buf);
        }

        return toRet;
    }


public:
    explicit TreeRemovalVis(const std::set<std::string>& relevant) : m_relevant(relevant) {}

    std::vector<std::shared_ptr<codegen::c::Statement>> filter(const std::vector<std::shared_ptr<codegen::c::Statement>>& code) {
        std::vector<std::shared_ptr<codegen::c::Statement>> toRet;
        toRet.reserve(code.size());

        for (auto s : code) {
            s->accept(this);
            if (buf != nullptr) toRet.push_back(buf);
        }

        return toRet;
    }

    void use(codegen::c::RawStatement* s) override {
        for (const auto& name : m_relevant) {
            if (s->raw.find(name) != std::string::npos) {
                buf = nullptr;
                return;
            }
        }

        buf = std::make_shared<codegen::c::RawStatement>(*s);
    }

    void use(codegen::c::Blank* s) override {
        buf = std::make_shared<codegen::c::Blank>(*s);
    }

    void use(codegen::c::Comment* s) override {
        buf = std::make_shared<codegen::c::Comment>(*s);
    }

    void use(codegen::c::VarDecl* s) override {
        buf = aNameMatches(s->access()) ? nullptr : std::make_shared<codegen::c::VarDecl>(*s);
    }

    void use(codegen::c::ArrayDecl* s) override {
        buf = aNameMatches(s->getName()) ? nullptr : std::make_shared<codegen::c::ArrayDecl>(*s);
    }

    void use(codegen::c::Block* s) override {
        assert(false);
    }

    void use(codegen::c::Assignment* s) override {
        buf = aNameMatches(s->getDst()) ? nullptr : std::make_shared<codegen::c::Assignment>(*s);
    }

    void use(codegen::c::Ifelse* s) override {
        // if condition contains var -> discard
        for (const auto& c : s->condition()) {
            if (condContainsVar(c)) {
                buf = nullptr;
                return;
            }
        }
        
        // filter blocks
        auto trueBlock = filterBlock(s->trueBlock);
        auto falseBlock = filterBlock(s->falseBlock);

        buf = std::make_shared<codegen::c::Ifelse>(s->condition(), trueBlock, falseBlock);
    }
    
    void use(codegen::c::Ret* s) override {
        buf = std::make_shared<codegen::c::Ret>(*s);
    }
};


void codegen2::HybridGen::renderReference(const std::string& filename) const {
    assert(m_isFinal);
    std::set<std::string> relevant(treeVars.begin(), treeVars.end());

    TreeRemovalVis trv(relevant);
    auto code = trv.filter(this->code);

    renderToFile(code, filename);
}




bool mulVarCmp(const std::string& a, const std::string& b) {
    const auto parseVarname = [](const std::string& inp) -> std::vector<size_t> {        
        if (inp[0] == 'x') {
            std::vector<size_t> toRet;
            size_t current = 0;
            bool inProgress = false;

            for (const auto s : inp) {
                if (std::isdigit(s)) {
                    current = current * 10 + (s - '0');
                    inProgress = true;
                } else if (inProgress) {
                    inProgress = false;
                    toRet.push_back(current);
                    current = 0;
                }
            }

            if (inProgress) toRet.push_back(current);

            return toRet;
        }

        size_t buf = std::stol(inp.substr(4, inp.size()-5));
        return {buf};
    };

    auto varA = parseVarname(a);
    auto varB = parseVarname(b);

    if (varA.size() != varB.size()) return varA.size() < varB.size();

    if (varA.size() == 1) return varA.front() < varB.front();
    else {
        assert(varA.size() == 2);
        return std::tie(varA[0], varA[1]) < std::tie(varB[0], varB[1]);
    }
}


bool codegen2::HybridGenSimple::mulSortCmp(const codegen2::Instr& a, const codegen2::Instr&b) {
    //assert(a.)

    if (a.opB == b.opB) return mulVarCmp(a.dst, b.dst);
    return mulVarCmp(a.opB, b.opB);
}


void codegen2::placeIfElse(std::vector<std::shared_ptr<codegen::c::Statement>>& code, std::shared_ptr<codegen::c::Ifelse> ifelse, const std::string& var) {
    assert(!code.empty());
    
    // reverse iterators and std::distance don't cut it
    auto findLastPos = [&](auto& vis) {
        auto it = std::prev(code.end());
        while (it != code.begin() && !vis(*it)) {
            it = std::prev(it);
        }
        return it;
    };
    
    // find last assignment to var
    codegen2::VarAssignFinder vaf(var);
    auto it = findLastPos(vaf);
    assert(it != code.begin());

    // determine input variable that was used in this block (they are sorted)
    auto conv = std::dynamic_pointer_cast<codegen::c::Assignment>(*it);
    assert(conv != nullptr);
    auto arith = conv->getArith();
    auto inpVar = arith->getVar();
    assert(inpVar.has_value());

    // find next position where inpVar is not used
    codegen2::VarUseFinder vuf(inpVar.value());
    while (it != code.end() && vuf(*it)) {
        it = std::next(it);
    }

    // actually insert ifelse
    code.insert(it, ifelse);  
}



void codegen2::HybridGenSimple::addPath(const path::Path& p) {
    assert(!m_isFinal);
    assert(p.leaf.isConst());
    //assert(p.decisions.size() <= 32);

    /*
     * Get Stuff that needs to be scheduled
     * add tree checks
     * insert exit
     */

    // check what needs to be scheduled
    std::set<std::pair<size_t,size_t>> reqN;
    for (const auto& d : p.decisions) {
        auto req = obtainRequired(d.idx());
        reqN.insert(req.begin(), req.end());
    }

    // obtain neurons
    std::vector<std::shared_ptr<IRNeuron>> relNeurons;
    std::transform(reqN.begin(), reqN.end(), std::back_inserter(relNeurons), [&](auto idx) {
        isScheduled[idx] = true;  // mark as scheduled
        return neurons.at(idx);
    });

    // extract multiplications
    std::vector<Instr> remMul;
    for (auto n : relNeurons) {
        for (const auto& mul : n->popMuls()) {
            assert(mul.type == Instr::MULADD);
            remMul.push_back(mul);
        }
    }

    // sort the multiplications
    std::sort(remMul.begin(), remMul.end(), mulSortCmp);


    // add tree checks
    auto nextTreeVar = [&]() {
        treeVars.push_back(fmt::format("onTree{}", treeVars.size()));
    };
    nextTreeVar();

    codegen::c::Comparisons cmps(true);
    auto addComparison = [&](int64_t value) {
        cmps.add(codegen::c::Comparison{
            .variable = treeVars.back(),
            .type = codegen::c::Comparison::EQ,
            .value = value
        });
    };

    std::map<std::string,Instr> treeTests;

    size_t testPos = 0;
    int64_t cmp = 0;
    for (const auto& d : p.decisions) {
        if (testPos == 32) {
            addComparison(cmp);
            nextTreeVar();
            testPos = 0;
            cmp = 0;
        }

        treeTests.emplace(neurons.at(d.idx())->var(), Instr(Instr::SET_BIT, treeVars.back(), std::to_string(testPos)));

        cmp <<= 1;
        if (d.decision) cmp += 1;

        testPos += 1;
    }

    addComparison(cmp);

    // create code for neurons
    decltype(code) section(remMul.begin(), remMul.end());
    std::for_each(relNeurons.rbegin(), relNeurons.rend(), [&](auto rn) {
        placeIfElse(section, rn->instantiateIfElse(), rn->var());
    });


    // path exit
    codegen::c::Block exitBlock;

    size_t outIdx = 0;
    for (const auto x : p.leaf.possClasses()) {
        exitBlock.create<codegen::c::Assignment>(fmt::format("out[{}]", outIdx++), int32_t(127)*x);
    }

    exitBlock.create<codegen::c::RawStatement>("return;\n");

    auto treeCheck = std::make_shared<codegen::c::Ifelse>(/*codegen::c::Comparison{
        .variable = treeVars.back(),
        .type = codegen::c::Comparison::EQ,
        .value = cmp
    }*/cmps, exitBlock);

    section.push_back(treeCheck);


    // add section to global code
    code.insert(code.end(), section.begin(), section.end());


    // finally insert the treechecks into relus
    for (const auto& [var, check] : treeTests) {
        ReLUFinder rf(var);
        auto it = std::find_if(code.begin(), code.end(), rf);
        assert(it != code.end());

        auto ptr = std::dynamic_pointer_cast<codegen::c::Ifelse>(*it);
        assert(ptr != nullptr);

        ptr->trueBlock.add(check);
    }
}



void codegen2::HybridGenSimple::finalize() {  // FIXME
    assert(!m_isFinal);
    m_isFinal = true;
    // schedule remaining neurons
    std::set<std::pair<size_t,size_t>> remN;
    for (const auto& [i, n] : neurons) {
        if (!isScheduled[i]) {
            remN.insert(i);
        }
    }
    std::vector<std::shared_ptr<IRNeuron>> remNeurons;
    std::transform(remN.begin(), remN.end(), std::back_inserter(remNeurons), [&](auto idx) {
        return neurons.at(idx);
    });

    std::vector<Instr> remMul;
    for (auto n : remNeurons) {
        for (const auto& mul : n->popMuls()) {
            assert(mul.type == Instr::MULADD);
            remMul.push_back(mul);
        }
    }

    // sort the multiplications
    std::sort(remMul.begin(), remMul.end(), mulSortCmp);


    // create code for neurons
    decltype(code) section(remMul.begin(), remMul.end());
    std::for_each(remNeurons.rbegin(), remNeurons.rend(), [&](auto rn) {
        placeIfElse(section, rn->instantiateIfElse(), rn->var());
    });

    // insert section code
    code.insert(code.end(), section.begin(), section.end());

    // add variable declarations and initializations
    for (auto [idx, neuron] : neurons) {
        VarUseFinder vf(neuron->var());
        auto pos = std::find_if(code.begin(), code.end(), vf);
        assert(pos != code.end());

        auto toIns = neuron->getInit();
        code.insert(pos, toIns.begin(), toIns.end());
    }


    // add tree vars
    for (const auto& v : treeVars) {
        VarUseFinder vf(v);
        auto pos = std::find_if(code.begin(), code.end(), vf);
        assert(pos != code.end());
        code.insert(pos, std::make_shared<codegen::c::RawStatement>(fmt::format("uint32_t {} = 0;\n", v)));
    }


    // FIXME: move output neurons to out
    code.push_back(std::make_shared<codegen::c::RawStatement>("out[0] = x_2_00;\n"));
    code.push_back(std::make_shared<codegen::c::RawStatement>("out[1] = x_2_01;\n"));
}