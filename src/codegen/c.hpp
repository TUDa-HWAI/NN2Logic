#pragma once
#include <string>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <variant>
#include <type_traits>
#include <cassert>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <vector>
#include <functional>
#include <optional>

namespace codegen::c {

class CVisitor;


struct Comparison {
    std::string variable;
    enum {
        EQ, GT, LT, GE, LE
    } type;
    int64_t value;

    std::string print() const;
};


struct Statement {
    virtual std::string print(size_t offset) const = 0;

    static std::string alignment(size_t offset) {
        return fmt::format("{: >{}}", "", offset * 4);
    }

    virtual ~Statement() {}

    virtual void accept(CVisitor*) = 0;
};


class RawStatement : public Statement {
public:
    const std::string raw;
    const bool indent;

    RawStatement(const std::string& s, bool indent = true) : raw(s), indent(indent) {}

    std::string print(size_t offset) const override {
        if (!indent) offset = 0;
        return fmt::format("{}{}", alignment(offset), raw);
    }

    void accept(CVisitor*) override;
};


class Blank : public Statement {
    size_t n;

public:
    explicit Blank(size_t n = 1) : n(n) {}
    std::string print(size_t) const override;

    void accept(CVisitor*) override;
};


class Comment : public Statement {
    std::string comment;

public:
    explicit Comment(const std::string& comment) : comment(comment) {}

    std::string print(size_t offset) const override;

    void accept(CVisitor*) override {}
};


class VarDecl : public Statement {
    std::string type;
    std::string name;

public:
    VarDecl(const std::string& type, const std::string& name) : type(type), name(name) {}

    std::string print() const {
        return fmt::format("{} {}", type, name);
    }

    std::string print(size_t offset) const override {
        return fmt::format("{}{} {};\n", alignment(offset), type, name);
    }

    std::string access() const {
        return name;
    }

    std::string accessAt(size_t idx) const {  // keeping this for pointer access
        return fmt::format("{}[{}]", name, idx);
    }

    void accept(CVisitor*) override;
};


class ArrayDecl : public Statement {
    std::string type;
    std::string name;
    size_t size;

public:
    ArrayDecl(const std::string& type, const std::string& name, size_t size) : type(type), name(name), size(size) {}

    std::string print(size_t offset) const override {
        return fmt::format("{}{} {}[{}];\n", alignment(offset), type, name, size);
    }

    std::string accessAt(size_t idx) const {
        assert(idx < size);
        return fmt::format("{}[{}]", name, idx);
    }

    std::string getName() const {
        return name;
    }

    void accept(CVisitor*) override;
};



/**
 * Logical Combination of several instances of type Comparison.
 */
class Comparisons {
    std::vector<std::variant<Comparison, std::string>> m_members;
    bool m_isAnd;
public:
    Comparisons(bool isAnd = true) : m_members(), m_isAnd(isAnd) {}
    Comparisons(const std::vector<Comparison> &cs, bool isAnd = true) : m_members(cs.begin(), cs.end()), m_isAnd(isAnd) {}
    Comparisons(const Comparison& comp, bool isAnd = true) : m_members({comp}), m_isAnd(isAnd) {}
    Comparisons(const std::string& s, bool isAnd = true) : m_members({s}), m_isAnd(isAnd) {}

    std::vector<std::variant<Comparison, std::string>>::const_iterator begin() const {
        return m_members.begin();
    }

    std::vector<std::variant<Comparison, std::string>>::const_iterator end() const {
        return m_members.end();
    }

    std::string print() const {
        assert(!m_members.empty());

        std::vector<std::string> asStr;
        asStr.reserve(m_members.size());

        for (const auto& c : m_members) {
            if (const std::string* p = std::get_if<std::string>(&c)) {
                asStr.push_back(*p);
            } else {
                asStr.push_back(fmt::format("({})", std::get<Comparison>(c).print()));
            }
        }

        if (asStr.size() == 1) return asStr.front();

        std::string joiner = m_isAnd ? " && " : " || ";
        return fmt::format("({})", fmt::join(asStr, joiner));
    }

    void add(Comparison comp) {
        m_members.push_back(comp);
    }

    void add(const Comparisons& other) {
        m_members.push_back(other.print());
    }

    void add (const std::string& str) {
        m_members.push_back(str);
    }
};


class Arith {
public:
    enum op_t {
        MUL, ADD, RSHFT, INV
    };
private:
    std::variant<std::string,std::shared_ptr<Arith>> a;
    int32_t b;
    op_t op;

public:
    Arith(const std::string& var, op_t op, int32_t val) : a(var), b(val), op(op) {}
    Arith(std::shared_ptr<Arith> var, op_t op, int32_t val) : a(var), b(val), op(op) {}

    std::string print() const;

    [[nodiscard]] bool uses(const std::string& var) const {
        if (std::holds_alternative<std::string>(a)) return std::get<std::string>(a) == var;
        return false;
    }

    [[nodiscard]] std::optional<std::string> getVar() const {
        if (std::holds_alternative<std::string>(a)) {
            return std::get<std::string>(a);
        }
        return std::nullopt;
    }
};


struct Function {
    std::string name;
    std::string retType;
    std::vector<VarDecl> params;

    std::vector<std::shared_ptr<Statement>> members;

    std::string print() const;
};


class Block : public Statement {
    std::vector<std::shared_ptr<Statement>> members;

public:
    explicit Block(std::initializer_list<std::shared_ptr<Statement>> l) : members(l) {}
    explicit Block(const std::vector<std::shared_ptr<Statement>>& v) : members(v) {}
    explicit Block(std::shared_ptr<Statement> s) : members({s}) {}
    Block() : members() {}

    std::string print(size_t offset) const override;

    std::vector<std::shared_ptr<Statement>>::const_iterator begin() const {
        return members.cbegin();
    }

    std::vector<std::shared_ptr<Statement>>::const_iterator end() const {
        return members.cend();
    }

    bool empty() const {
        return members.empty();
    }

    void add(std::shared_ptr<Statement> s) {
        members.push_back(s);
    }

    void add(const std::vector<std::shared_ptr<Statement>>& s) {
        if (s.size() == 1) {
            members.push_back(s.front());
        } else {
            members.insert(members.end(), s.begin(), s.end());
        }
    }

    template <class T, typename... Args>
    std::shared_ptr<T> create(Args... args) {
        auto ptr = std::make_shared<T>(args...);
        members.push_back(ptr);
        return ptr;
    }

    std::vector<std::shared_ptr<Statement>> get_if(std::function<bool(std::shared_ptr<Statement>)>& func) const {
        std::vector<std::shared_ptr<Statement>> toRet;
        std::copy_if(members.begin(), members.end(), std::back_inserter(toRet), func);
        return toRet;
    }

    void accept(CVisitor*) override;
};


class Assignment : public Statement {
    std::string var;
    std::variant<Arith,int32_t,std::string,bool,Comparisons> rhs;
    bool append;

public:
    Assignment(const std::string& var, const decltype(rhs)& rhs, bool append = false) : var(var), rhs(rhs), append(append) {}

    std::string print(size_t offset) const override;

    std::optional<Arith> getArith() const {
        if (std::holds_alternative<Arith>(rhs)) {
            return std::get<Arith>(rhs);
        }

        return {};
    }

    [[nodiscard]] std::string getDst() const {
        return var;
    }

    void accept(CVisitor*) override;
};


class Ifelse : public Statement {
    Comparisons cmps;

public:
    Block trueBlock;
    Block falseBlock;

    Ifelse(const Comparisons& cmps, const Block& trueBlock, const Block& falseBlock = Block()) : cmps(cmps), trueBlock(trueBlock), falseBlock(falseBlock) {}
    Ifelse(const Comparison& cmp, const Block& trueBlock, const Block& falseBlock = Block()) : cmps(cmp), trueBlock(trueBlock), falseBlock(falseBlock) {}

    std::string print(size_t offset) const override;

    void accept(CVisitor*) override;

    const Comparisons& condition() const {
        return cmps;
    }
};


class Ret : public Statement {
public:
    const std::variant<std::monostate,int32_t, std::string> toRet;

    Ret() : toRet(std::monostate{}) {}
    explicit Ret(const std::string& var) : toRet(var) {}
    explicit Ret(int32_t val) : toRet(val) {}

    std::string print(size_t offset) const override;

    void accept(CVisitor*) override;
};


class CVisitor {

public:
    virtual ~CVisitor() {};

    virtual void use(RawStatement* s) {}
    virtual void use(Blank* s) {}
    virtual void use(Comment* s) {}
    virtual void use(VarDecl* s) {}
    virtual void use(ArrayDecl* s) {}
    virtual void use(Block* s) {}
    virtual void use(Assignment* s) {}
    virtual void use(Ifelse* s) {}
    virtual void use(Ret* s) {}
};

}