#pragma once

#include <indicators/progress_bar.hpp>
#include <queue>
#include <memory>
#include <mutex>
#include <cassert>
#include <optional>
#include <thread>

template
<typename T>
class IndicatorWorklist final {
    std::queue<T> m_work;
    std::unique_ptr<indicators::ProgressBar> m_bar;
    size_t m_val = 0;
    size_t m_active = 0;
    std::mutex m_lock;
    const size_t m_numTotal;
    size_t m_updRing = 0;
    const size_t m_updIntervall;

    void displayProgress() {
        m_bar->set_option(indicators::option::PostfixText{std::to_string(m_val) + '/' + std::to_string(m_numTotal)});
        m_bar->set_progress(m_val);
    }

public:
    IndicatorWorklist(size_t numElements, const std::string& label) : m_numTotal(numElements),
        m_updIntervall(std::min<size_t>(numElements / 10000, 10000)) {
        m_bar = std::make_unique<indicators::ProgressBar>(
            indicators::option::MaxProgress{numElements},
            indicators::option::BarWidth{50},
            indicators::option::PrefixText{label},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
            indicators::option::ShowPercentage(true)
        );
    }

    ~IndicatorWorklist() {
        displayProgress();
        m_bar->mark_as_completed();
    }

    void push(const T& work) {
        std::lock_guard<std::mutex> l(m_lock);
        m_work.push(work);
    }

    [[nodiscard]] std::optional<T> popIfPossible() {
        std::lock_guard<std::mutex> l(m_lock);
        if (m_work.empty()) return {};

        T toRet = m_work.front();
        m_work.pop();
        m_active += 1;
        return toRet;
    }

    [[nodiscard]] bool empty() {
        std::lock_guard<std::mutex> l(m_lock);
        return m_work.empty();
    }

    void release() {
        std::lock_guard<std::mutex> l(m_lock);
        assert(m_active > 0);
        m_active -= 1;
    }

    [[nodiscard]] bool active() {
        std::lock_guard<std::mutex> l(m_lock);
        return m_active > 0 || !m_work.empty();
    }

    void update(size_t step = 1) {
        std::lock_guard<std::mutex> l(m_lock);
        m_val += step;
        if (m_updRing++ > m_updIntervall) {
            displayProgress();
            m_updRing = 0;
        }
    }
};