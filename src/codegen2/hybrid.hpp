#pragma once

#include <map>
#include "Path.hpp"
#include "quant/Exchange.hpp"
#include "IRmodel.hpp"
#include "codegen/c.hpp"


namespace codegen2 {

class HybridGen {
public:
    typedef std::vector<std::shared_ptr<codegen::c::Statement>>::iterator CodeIt_t;

protected:
    std::vector<QTree::Layer<int>> layers;
    std::map<std::pair<size_t,size_t>,std::shared_ptr<IRNeuron>> neurons;
    std::map<std::pair<size_t,size_t>,bool> isScheduled;  // defaults to 0 -> false

    // in and outputs
    std::shared_ptr<codegen::c::VarDecl> inp = std::make_shared<codegen::c::VarDecl>("const uint8_t*", "inp");
    std::shared_ptr<codegen::c::VarDecl> out = std::make_shared<codegen::c::VarDecl>("int8_t*", "out");

    // keep track of path variables
    std::vector<std::string> treeVars;

    // c code
    std::vector<std::shared_ptr<codegen::c::Statement>> code;

    // check if finalized
    bool m_isFinal = false;

    // get neurons that need to be scheduled
    std::set<std::pair<size_t,size_t>> obtainRequired(std::pair<size_t,size_t> inp);

    // find continous blocks of variable usage
    static std::pair<CodeIt_t,CodeIt_t> getRange(CodeIt_t vBegin, CodeIt_t vEnd, const std::string& varname);

    // finds continous blocks of multiplications that use varname and puts them into an if > 0, given they are at least delta long
    void rangeToIf(const std::string& varname, size_t delta = 5);

    void renderToFile(const std::vector<std::shared_ptr<codegen::c::Statement>>& code, const std::string& filename) const;
public:
    explicit HybridGen(const std::vector<QTree::Layer<int>>& layers);

    // add a path and generate its code
    void addPath(const path::Path& p);

    // schedule the remainder of the NN and move to outputs
    void finalize();

    // puts the code into a function and generates the file
    void render(const std::string& filename) const;

    // remove tree stuff and render as reference into given file
    void renderReference(const std::string& filename) const;
};


class HybridGenSimple : public HybridGen {
    static bool mulSortCmp(const Instr& a, const Instr& b);

public:
    using HybridGen::HybridGen;

    void addPath(const path::Path& p);

    void finalize();
};


void placeIfElse(std::vector<std::shared_ptr<codegen::c::Statement>>& code, std::shared_ptr<codegen::c::Ifelse> ifelse, const std::string& var);

}