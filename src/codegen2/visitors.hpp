#pragma once

#include "codegen/c.hpp"


namespace codegen2 {

class VarUseFinder : public codegen::c::CVisitor {
    std::string varName;
    bool found;

public:
    explicit VarUseFinder(const std::string& varName) : varName(varName) {}

    ~VarUseFinder() {}

    bool operator()(std::shared_ptr<codegen::c::Statement> s) {
        found = false;
        s->accept(this);
        return found;
    }

    void use(codegen::c::RawStatement* s) override {
        found = s->raw.find(varName) != std::string::npos;
    }
    
    void use(codegen::c::Blank* s) override {}
    
    void use(codegen::c::Comment* s) override {}
    
    void use(codegen::c::VarDecl* s) override {
        found = s->access() == varName;
    }
    
    void use(codegen::c::ArrayDecl* s) override {
        found = s->getName() == varName;
    }
    
    void use(codegen::c::Block* s) override {
        for (const auto& ptr : *s) {
            ptr->accept(this);

            if (found) return;
        }
    }
    
    void use(codegen::c::Assignment* s) override {
        found = s->getDst() == varName;

        auto arith = s->getArith();
        if (arith.has_value() && !found) {
            found = arith->uses(varName);  // FIXME
        }
    }
    
    void use(codegen::c::Ifelse* s) override {

        if (!found) use(&(s->trueBlock));
        if (!found) use(&(s->falseBlock));
    }
    
    void use(codegen::c::Ret* s) override {
        if (std::holds_alternative<std::string>(s->toRet)) {
            found = std::get<std::string>(s->toRet) == varName;
        }
    }
};

class VarAssignFinder : public codegen::c::CVisitor {
    std::string varName;
    bool found;

public:
    explicit VarAssignFinder(const std::string& varName) : varName(varName) {}

    ~VarAssignFinder() {}

    bool operator()(std::shared_ptr<codegen::c::Statement> s) {
        found = false;
        s->accept(this);
        return found;
    }

    void use(codegen::c::RawStatement* s) override {
        found = s->raw.find(varName) != std::string::npos;
    }
    
    void use(codegen::c::Blank* s) override {}
    
    void use(codegen::c::Comment* s) override {}
    
    void use(codegen::c::VarDecl* s) override {
        found = s->access() == varName;
    }
    
    void use(codegen::c::ArrayDecl* s) override {
        found = s->getName() == varName;
    }
    
    void use(codegen::c::Block* s) override {
        for (const auto& ptr : *s) {
            ptr->accept(this);

            if (found) return;
        }
    }
    
    void use(codegen::c::Assignment* s) override {
        found = s->getDst() == varName;
    }
    
    void use(codegen::c::Ifelse* s) override {
        if (!found) use(&(s->trueBlock));
        if (!found) use(&(s->falseBlock));
    }
    
    void use(codegen::c::Ret* s) override {}
};

class ReLUFinder : public codegen::c::CVisitor {
    std::string varName;
    bool found;

    bool condContains(std::variant<codegen::c::Comparison, std::string> c) const {
        if (std::holds_alternative<codegen::c::Comparison>(c)) {
            return std::get<codegen::c::Comparison>(c).variable == varName;
        }

        auto s = std::get<std::string>(c);
        return s.find(varName) != std::string::npos;
    }
public:
    explicit ReLUFinder(const std::string& varName) : varName(varName) {}

    ~ReLUFinder() {}

    bool operator()(std::shared_ptr<codegen::c::Statement> s) {
        found = false;
        s->accept(this);
        return found;
    }

    void use(codegen::c::Ifelse* s) override {
        for (const auto& c : s->condition()) {
            if (condContains(c)) {
                found = true;
                return;
            }
        }
    }
};

}
