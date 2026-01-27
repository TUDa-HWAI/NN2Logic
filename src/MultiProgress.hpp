#pragma once
#include <string>
#include <vector>
#include <indicators/progress_bar.hpp>
#include <indicators/dynamic_progress.hpp>

class MultiProgress final {

    indicators::DynamicProgress<indicators::ProgressBar> m_manager{};
    std::vector<size_t> m_progress;
    std::vector<size_t> m_max;


    void tick(size_t idx, size_t amount) {
        m_progress[idx] += amount;
        m_manager[idx].set_progress(m_progress[idx]);
        m_manager[idx].set_option(indicators::option::PostfixText{std::to_string(m_progress[idx]) + '/' + std::to_string(m_max[idx])});
    }

public:
    MultiProgress() {
        m_manager.set_option(indicators::option::HideBarWhenComplete{false});
    }


    struct Handle final {
        MultiProgress* const m_ptr;
        const size_t m_idx;

        void tick(size_t amount = 1) {
            m_ptr->tick(m_idx, amount);
        }

        void mark_as_completed() {
            m_ptr->m_manager[m_idx].mark_as_completed();
        }
    };

    friend Handle;

    Handle create(const std::string& label, size_t numElements = 100) {
        // create actual bar
        auto bar = std::make_unique<indicators::ProgressBar>(
            indicators::option::MaxProgress{numElements},
            indicators::option::BarWidth{50},
            indicators::option::PrefixText{label},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
            indicators::option::ShowPercentage(true)
        );

        auto idx = m_manager.push_back(std::move(bar));

        // create progress saver
        m_progress.push_back(0);
        assert(m_progress.size() == idx + 1);

        m_max.push_back(numElements);
        assert(m_max.size() == m_progress.size());

        // return handle
        return Handle{
            .m_ptr = this,
            .m_idx = idx
        };
    }

    void mark_as_completed() {
        for (size_t i = 0; i < m_max.size(); ++i) {
            m_manager[i].mark_as_completed();
        }
    }
};