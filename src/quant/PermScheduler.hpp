#pragma once

#include <vector>
#include <map>
#include <set>
#include <mutex>
#include <thread>
#include <execution>

#include "Path.hpp"
#include "ScoreKeeper.hpp"
#include "MultiProgress.hpp"
#include "SimplePool.hpp"



namespace QTree {

template<bool MAX = false>
class PermScheduler final {
    const size_t numDecisions;
    const std::vector<path::Path> forced;
    const std::vector<path::Path> niceToHave;

    ScoreKeeper<codegen::ConstSchedule,MAX> score;

    codegen::ConstSchedule orderToSched(const std::vector<path::Path>& paths) const {
        codegen::ConstSchedule toRet;
        std::set<size_t> used;

        auto addIfRequired = [&](size_t idx) -> void {
            assert(idx < numDecisions);
            if (used.find(idx) == used.end()) {
                toRet.addNeuron(idx);
                used.insert(idx);
            }
        };

        for (const auto& p : paths) {
            for (const auto& d : p.decisions) {
                if (d.layerIdx == 1)
                    addIfRequired(d.neuronIdx);
            }

            toRet.addPath(p);
        }


        for (size_t i = 0; i < numDecisions; ++i) {
            addIfRequired(i);
        }

        assert(toRet.numPaths() == paths.size());

        return toRet;
    }

    static size_t factorial(size_t i) {
        if (i == 0) return 1;
        return i * factorial(i-1);
    }


    static size_t numSchedules(const size_t n) {
        size_t toRet = 0;

        for (size_t i = 1; i <= n; ++i) {
            toRet += factorial(i);
        }

        return n * toRet;
    }


    // https://en.wikipedia.org/wiki/Heap%27s_algorithm
    template <typename T>
    static std::list<std::vector<T>> heaps(std::vector<T> A) {
        assert(!A.empty());
        const size_t N = A.size();

        std::list<std::vector<T>> toRet;
        toRet.emplace_front(A);
        
        std::vector<size_t> c(A.size(), 0);

        size_t i = 1;  // i acts similarly to a stack pointer
        while (i < N) {
            if (c[i] < i) {
                if (i % 2 == 0) {  // is even
                    std::swap(A[0], A[i]);
                } else {
                    std::swap(A[c[i]], A[i]);
                }

                toRet.emplace_front(A);

                // Swap has occurred ending the while-loop. Simulate the increment of the while-loop counter
                c[i] += 1;
                // Simulate recursive call reaching the base case by bringing the pointer to the base case analog in the array
                i = 1;
            } else {
                // Calling generate(i+1, A) has ended as the while-loop terminated. Reset the state and simulate popping the stack by incrementing the pointer.
                c[i] = 0;
                i += 1;
            }
        }

        assert(factorial(N) == toRet.size());

        return toRet;
    }


    template <typename T>
    static std::vector<T> wrapAroundCopy(const std::vector<T>& inp, const size_t from, const size_t n) {
        std::vector<T> toRet;
        toRet.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            toRet.push_back(inp.at((from + i) % inp.size()));
        }

        return toRet;
    }

public:
    PermScheduler(size_t numDecisions, const std::vector<path::Path>& forced, const std::vector<path::Path>& niceToHave)
        : numDecisions(numDecisions), forced(forced), niceToHave(niceToHave) {}


    template <typename F>
    void run(MultiProgress& barManager, F&& scoreFunc, size_t maxNum = 0, size_t minNum = 1) {
        //assert(forced.size() == 1);

        assert(minNum <= niceToHave.size());
        assert(maxNum <= niceToHave.size());

        if (maxNum == 0) {
            maxNum = std::min<size_t>(2, niceToHave.size());
        } else {
            maxNum = std::min<size_t>(maxNum, niceToHave.size());
        }

        assert(minNum >= 1);

        //assert(minNum <= maxNum);
        minNum = std::min<size_t>(minNum, maxNum);
        std::vector<std::vector<path::Path>> work;
        if (!forced.empty()) work.emplace_back(forced);  // test forced paths alone

        for (size_t n = minNum; n <= maxNum; ++n) {
            for (size_t i = 0; i < niceToHave.size(); ++i) {
                auto opt = wrapAroundCopy(niceToHave, i, n);
                assert(opt.size() == n);
                for (const auto& perm : heaps(opt)) {  // check all possible permutations
                    if (!forced.empty()) work.emplace_back(forced);
                    std::copy(perm.begin(), perm.end(), std::back_inserter(work.back()));
                }
            }
        }

        std::mutex mut;
        SimplePool<std::vector<path::Path>> sp(work, "PermScheduler", barManager);
        sp.run([&](decltype(sp)* sp){
            std::optional<std::vector<path::Path>> work;
            while ((work = sp->popIfPossible())) {
                auto sched = orderToSched(work.value());
                auto score = scoreFunc(sched);
                {
                    const std::lock_guard<std::mutex> lock(mut);
                    this->score.addScore(score, sched);
                }
            }
        });
    }

    [[nodiscard]] codegen::ConstSchedule bestSolution() const {
        return score.getBestSolution();
    }

    [[nodiscard]] double bestScore() const {
        return score.getBestScore();
    }

};
}