#include "c.hpp"
#include <fmt/os.h>
#include <sstream>

using namespace codegen::c;

struct CommonVariantVis {
    std::string operator()(int32_t val) {
        return std::to_string(val);
    }

    std::string operator()(const std::string& str) {
        return str;
    }

    std::string operator()(const Arith& a) {
        return a.print();
    }

    std::string operator()(const Comparisons& a) {
        return a.print();
    }

    std::string operator()(bool b) {
        return b ? "true" : "false";
    }

    std::string operator()(std::monostate) {
        return "";
    }
};


std::string Ret::print(size_t offset) const {
    return fmt::format("{}return {};\n", alignment(offset), std::visit(CommonVariantVis{}, toRet));
}


std::string Comparison::print() const {
    std::string cmpOp;
    switch (type) {
        case EQ: cmpOp = "=="; break;
        case GT: cmpOp = ">"; break;
        case LT: cmpOp = "<"; break;
        case GE: cmpOp = ">="; break;
        case LE: cmpOp = "<="; break;
    };

    return fmt::format("{} {} {}", variable, cmpOp, value);
}


std::string Blank::print(size_t) const {
    std::string toRet;
    toRet.resize(n, '\n');
    return toRet;
}


std::string Comment::print(size_t offset) const {
    std::string toRet;
    std::string line;
    std::stringstream stream(comment);
    bool first = true;
    const auto align = alignment(offset);

    while(std::getline(stream, line, '\n')) {
        fmt::format_to(std::back_inserter(toRet), "{}{}* {}\n", align, first ? '/' : ' ', line);
        first = false;
    }

    toRet.pop_back();  // remove last linebreak
    return fmt::format("{} */\n", toRet);
}


std::string Arith::print() const {
    if (op == INV) {
        return fmt::format("~({})", std::get<1>(a)->print());
    }

    std::string sOp;
    switch(op) {
        case MUL: sOp = "*"; break;
        case ADD: sOp = "+"; break;
        case RSHFT: sOp = ">>"; break;
        default: assert(false);
    }

    // check if we work on a variable or another arith
    std::string left;
    if (std::holds_alternative<std::string>(a)) {
        left = std::get<std::string>(a);
    } else {
        left = fmt::format("({})", std::get<1>(a)->print());
    }

    return fmt::format("{} {} {}", left, sOp, b);
}


std::string Function::print() const {
    std::stringstream ss;

    ss << fmt::format("{} {}(", retType, name);
    for (size_t i = 0; i < params.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << params[i].print();
    }

    ss << ") {{\n";

    for (const auto& m : members) {
        ss << m->print(1);
    }

    ss << "}}";

    return ss.str();
}


std::string Block::print(size_t offset) const {
    // if a block only contains another block: just print the inner one
    if (members.size() == 1 && std::dynamic_pointer_cast<Block>(members.front()) != nullptr) {
        return members.front()->print(offset);
    }

    std::string toRet = "{{\n";

    for (const auto& m : members) {
        toRet += m->print(offset+1);
    }

    toRet += alignment(offset) + "}}";
    return toRet;
}


std::string Assignment::print(size_t offset) const {
    std::string op = append ? "+=" : "=";
    return fmt::format("{}{} {} {};\n", alignment(offset), var, op, std::visit(CommonVariantVis{}, rhs));
}


std::string Ifelse::print(size_t offset) const {
    assert(!trueBlock.empty());
    std::string toRet = fmt::format("{}if ({}) {}", alignment(offset), cmps.print(), trueBlock.print(offset));

    // handle else
    if (falseBlock.empty()) toRet += "\n";
    else {
        toRet += fmt::format(" else {}\n", falseBlock.print(offset));
    }

    return toRet;
}


/*
 * Visitor Stuff
 */

void codegen::c::RawStatement::accept(codegen::c::CVisitor* vis) {
    vis->use(this);
}

void codegen::c::Blank::accept(codegen::c::CVisitor* vis) {
    vis->use(this);
}

void codegen::c::VarDecl::accept(codegen::c::CVisitor* vis) {
    vis->use(this);
}

void codegen::c::ArrayDecl::accept(codegen::c::CVisitor* vis) {
    vis->use(this);
}

void codegen::c::Block::accept(codegen::c::CVisitor* vis) {
    vis->use(this);
}

void codegen::c::Assignment::accept(codegen::c::CVisitor* vis) {
    vis->use(this);
}

void codegen::c::Ifelse::accept(codegen::c::CVisitor* vis) {
    vis->use(this);
}

void codegen::c::Ret::accept(codegen::c::CVisitor* vis) {
    vis->use(this);
}