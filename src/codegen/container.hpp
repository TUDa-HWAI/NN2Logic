#pragma once

#include "c.hpp"
//#include "model.hpp"
#include <vector>


namespace codegen::c {
class Container final {
    std::vector<std::shared_ptr<Statement>> m_statements;

public:
    void add(std::shared_ptr<Statement> s) {
        m_statements.push_back(s);
    }

    void add(const std::vector<std::shared_ptr<Statement>>& s) {
        if (s.size() == 1) {
            add(s.front());
        } else {
            m_statements.insert(m_statements.end(), s.begin(), s.end());
        }
    }

    void add(const Container& c) {
        add(c.m_statements);
    }
/*
    void add(std::shared_ptr<BuildingBlock> bb) {
        add(bb->create());
    }
*/
    template<class InputIt>
    void add(InputIt first, InputIt last) {
        m_statements.insert(m_statements.end(), first, last);
    }

    std::vector<std::shared_ptr<Statement>> get() const {
        return m_statements;
    }

    template <class T, typename... Args>
    std::shared_ptr<T> create(Args... args) {
        auto ptr = std::make_shared<T>(args...);
        m_statements.push_back(ptr);
        return ptr;
    }
};
}