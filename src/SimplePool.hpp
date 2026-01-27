#pragma once

#include <queue>
#include <memory>
#include <mutex>
#include <cassert>
#include <optional>
#include <thread>

#include "MultiProgress.hpp"

template <typename T>
class SimplePool final {
    MultiProgress::Handle m_progress;
    std::queue<T> m_work;
    std::mutex m_lock;
    size_t m_numThreads;

public:
    template <typename I>
    SimplePool(const I& cont, const std::string& label, MultiProgress& mp) : m_progress(mp.create(label, cont.size())) {
        m_numThreads = std::min<size_t>(cont.size(), std::thread::hardware_concurrency());

        for (const auto& c : cont) {
            m_work.push(c);
        }
    }

    void setNumThreads(size_t num) {
        assert(num > 0);
        m_numThreads = num;
    }

    class Token final : public std::optional<T> {
        SimplePool<T>* m_sp = nullptr;

    public:
        Token(const T& t, SimplePool<T>* s) : std::optional<T>(t), m_sp(s) {}
        Token(void) : std::optional<T>(std::nullopt) {}

        ~Token() {
            if (m_sp) {
                std::lock_guard<std::mutex> l(m_sp->m_lock);
                m_sp->m_progress.tick();
            }
        }
    };

    friend Token;

    [[nodiscard]] Token popIfPossible() {
        std::lock_guard<std::mutex> l(m_lock);
        if (m_work.empty()) return Token();

        T toRet = m_work.front();
        m_work.pop();

        return Token(toRet, this);
    }

    template <typename F, typename... Args>
    void run(F&& func, Args... args) {
        std::vector<std::thread> threads;

        for (size_t i = 1; i < m_numThreads; ++i) {
            threads.push_back(std::thread(func, this, args...));
        }

        func(this, args...);  // execute one in this thread

        for (auto& t : threads) {
            t.join();
        }

        m_progress.mark_as_completed();
    }


    [[nodiscard]] bool empty() {
        std::lock_guard<std::mutex> l(m_lock);
        return m_work.empty();
    }
};