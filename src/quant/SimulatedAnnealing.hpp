#pragma once

#include "Path.hpp"
#include <thread>
#include <future>
#include <algorithm>
#include <random>

namespace QTree {

class SimulatedAnnealing final {
    std::random_device m_rd;  // a seed source for the random number engine
    std::mt19937 m_gen; // mersenne_twister_engine
    std::uniform_int_distribution<> m_distrib;

    const std::vector<path::Decision> m_positions;
    std::vector<bool> m_state;

    std::vector<bool> step() {  // toggle random bit
        auto pos = m_distrib(m_gen);
        std::vector<bool> buf = m_state;
        buf[pos] = !buf[pos];  // FIXME
        return buf;
    }

    static double getProbability(double difference, double temperature) {
        return exp(-1*difference/temperature);
    }

    std::vector<path::Decision> getDecisions(const std::vector<bool>& state) const {
        assert(state.size() == m_positions.size());
        std::vector<path::Decision> buf;

        for (size_t i = 0; i < state.size(); ++i) {
            if (state[i]) buf.push_back(m_positions[i]);
        }

        return buf;
    }

    double score = 0;
    double best_score = -1;
    size_t ts;
    std::vector<bool> best_state;
    void newScore(double score, const std::vector<bool>& state, size_t ts, bool forceBest = false) {
        if (score > best_score || forceBest) {
            best_score = score;
            best_state = state;
        }

        this->score = score;
        m_state = state;
        this->ts = ts;
    }
public:
    SimulatedAnnealing(const std::vector<path::Decision>& positions, const std::vector<bool>& initialState)
        : m_gen(m_rd()), m_distrib(0, positions.size()-1), m_positions(positions), m_state(initialState) {
            assert(!m_state.empty());
            assert(!m_positions.empty());
        }

    explicit SimulatedAnnealing(const std::vector<path::Decision>& positions)
        : m_gen(m_rd()), m_distrib(0, positions.size()-1), m_positions(positions), m_state(positions.size(), true) {
            assert(!m_state.empty());
            assert(!m_positions.empty());
        }

/*
    void reset() {
        m_state = std::vector<bool>(m_state.size(), true);
    }
*/

    template <typename FC, typename F>
    void run(double temperature, FC&& coolingRate, F&& scoreFunc) {
        // randomness stuff
        std::uniform_real_distribution<double> proba(0, 1);

        newScore(scoreFunc(getDecisions(m_state)), m_state, 0, true);
        //fmt::print("Initial score {}\n", score);

        size_t acceptedChanges = 0;

        for (size_t i = 0; i < 1000 && temperature > 0 && i - ts < 100; ++i) {
            auto candidate = step();

            // check if new candidate is better than old score or accepted by probability
            double candScore = scoreFunc(getDecisions(candidate));
            if ((candScore >= score) || (getProbability(std::abs(candScore - score), temperature) > proba(m_gen))) {
                newScore(candScore, candidate, i);
                acceptedChanges += 1;

                //fmt::print("{:03d}) new Score: {}; {:d}\n", i, score, fmt::join(m_state, ""));
            }

            // adjust temperature
            temperature *= coolingRate(i, acceptedChanges);
        }
    }

    // return best solution
    [[nodiscard]] std::vector<path::Decision> bestSolution() const {
        //assert(best_score >= 0 && !best_state.empty());
        assert(!best_state.empty());  // FIXME
        return getDecisions(best_state);
    }

    [[nodiscard]] double bestScore() const {
        return best_score;
    }
};


template <typename FC, typename F>
std::vector<path::Decision> SApool(const std::vector<path::Decision>& positions, double temperature, FC&& coolingRate, F&& scoreFunc) {
    // random stuff for filling initial states
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 1);

    std::vector<bool> state(positions.size(), true);

    std::vector<std::future<std::pair<double,std::vector<path::Decision>>>> futures;
    for (size_t i = 1; i < NUM_CPU; ++i) {  // create parallel work
        futures.push_back(std::async([&]() {
            std::vector<bool> tState(state.begin(), state.end());  // FIXME toggle random bits?
            SimulatedAnnealing sa(positions, tState);
            sa.run(temperature, coolingRate, scoreFunc);
            return std::make_pair(sa.bestScore(), sa.bestSolution());
        }));

        // new state vec
        std::generate(state.begin(), state.end(), [&]() -> bool {
            return dist(gen);
        });
    }

    // use this thread as well
    SimulatedAnnealing sa(positions, state);
    sa.run(temperature, coolingRate, scoreFunc);
    
    // find best solution  across pool
    auto bestScore = sa.bestScore();
    auto bestSolution = sa.bestSolution();

    for (auto& f : futures) {
        f.wait();
        auto buf = f.get();

        if (buf.first > bestScore) {
            std::tie(bestScore, bestSolution) = buf;
        }
    }

    return bestSolution;
}

}