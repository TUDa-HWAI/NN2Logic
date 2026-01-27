#pragma once
#include <optional>


namespace QTree {

template
<typename T, bool MAX = true, typename S = double>
class ScoreKeeper final {

    std::optional<T> best = {};
    S score;
public:
    void addScore(const S score, const T& solution) {
        if (!best.has_value() || (MAX && score > this->score) || (!MAX && score < this->score)) {
            this->score = score;
            best = solution;
        }
    }

    [[nodiscard]] T getBestSolution() const {
        assert(best.has_value());
        return best.value();
    }

    [[nodiscard]] S getBestScore() const {
        assert(best.has_value());
        return score;
    }

    [[nodiscard]] std::pair<S,T> getBest() const {
        return std::make_pair(score, best.value());
    }

};
}