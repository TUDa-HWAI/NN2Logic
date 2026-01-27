#include "test.hpp"

#include <fmt/core.h>
#include <fstream>
#include <map>
#include "Path.hpp"
#include "quant/Exchange.hpp"
#include "IRmodel.hpp"
#include "codegen/container.hpp"
#include <iostream>
#include "quant/MulSkipOptimizer.hpp"
#include "SimplePool.hpp"
#include "quant/ScoreKeeper.hpp"
#include "codegen2/hybrid.hpp"
#include "sim/CompilerAdaptor.hpp"
#include "sim/Runner.hpp"
#include "renderPlain.hpp"
#include "quant/CycleGuesstimator.hpp"
#include "quant/PermScheduler.hpp"

#include "hybrid.hpp"



struct CmpDecVecs {
    bool operator()(const std::vector<path::Decision>& a, const std::vector<path::Decision>& b) const {
        if (a.size() != b.size()) return a.size() < b.size();

        for (size_t i = 0; i < a.size(); ++i) {
            const auto& aD = a.at(i);
            const auto& bD = b.at(i);

            if (std::tie(aD.layerIdx, aD.neuronIdx, aD.decision) != std::tie(bD.layerIdx, bD.neuronIdx, bD.decision)) {
                return std::tie(aD.layerIdx, aD.neuronIdx, aD.decision) < std::tie(bD.layerIdx, bD.neuronIdx, bD.decision);
            }
        }

        return false;  // if we are here the vecs are equal, therefore none of them is less than the other
    }
};


size_t sumPathVisitFreq(const std::vector<path::Path>& paths) {
    return std::accumulate(paths.begin(), paths.end(), 0, [](size_t acc, const auto& p) {
        return p.visitFreq + acc;
    });
}


void codegen2::loadFromJson(const std::string& filename) {
    // load paths
    std::ifstream ifs(filename);
    nlohmann::json jf = nlohmann::json::parse(ifs);
    ifs.close();

    std::vector<path::Path> paths;
    jf.at("paths").get_to(paths);

    std::vector<QTree::Layer<int>> layers;
    jf.at("layers").get_to(layers);
    assert(layers.size() == 3);

    std::vector<std::vector<int>> dataset;
    std::vector<int> targets;
    jf.at("data").get_to(dataset);
    jf.at("targets").get_to(targets);


    // only use reduced const paths
    std::vector<path::Path> constPaths;
    std::map<std::vector<path::Decision>,std::vector<path::Path>,CmpDecVecs> relPaths;
    for (const auto& p : paths) {
        if (p.leaf.isConst()) {
            auto reduced = p.iisDecisions();
            relPaths[reduced].push_back(p);
        }
    }

    // now merge those paths
    for (const auto& [r, v] : relPaths) {
        constPaths.push_back(path::Path{
            .decisions = r,
            .leaf = v.front().leaf,
            .visitFreq = sumPathVisitFreq(v)
        });
    }

    assert(!constPaths.empty());  // if we do select a subset for testing, make sure there are actual candidates


    // handle multiple progress bars
    MultiProgress barManager;
    std::vector<QTree::MulSkipOptimizer::OptPath> scores;
    size_t idCounter = 0;
    std::mutex lock;

    // Initial optimization Stage
    {
        QTree::MulSkipOptimizer opt(layers, paths, true);
        SimplePool<path::Path> sp(constPaths, "Optimizer lvl 0", barManager);
        sp.setNumThreads(std::max(8, int(std::thread::hardware_concurrency())-2));

        sp.run([&](auto* sp) {
            SimplePool<path::Path>::Token work;
            while ((work = sp->popIfPossible())) {
                auto result = opt.process(work.value());
                std::lock_guard<std::mutex> l(lock);
                scores.push_back(result);
            }
        });
    }

    // sort results
    std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) -> bool {
        return std::tie(a.score, a.skippedMuls, a.goodMatches, b.badMatches) > std::tie(b.score, b.skippedMuls, b.goodMatches, a.badMatches);
    });
    assert(scores.front().score >= scores.back().score);

    // select best N
    QTree::ScoreKeeper<codegen::ConstSchedule, false, double> score;  // false -> use minimum, true -> max
    const size_t topNum = std::min<size_t>(scores.size(), 5);
    for (size_t i = 0; i < topNum; ++i) {
        const auto firstPath = scores[i].p;

        // use firstPath as precondition to filter samples
        std::vector<path::Path> samples;
        std::copy_if(paths.begin(), paths.end(), std::back_inserter(samples), [&](const auto& p) -> bool {
            return !QTree::CycleGuesstimator::decisionsMayOverlap(firstPath.decisions, p.decisions);
        });

        // use firstPath as precondition to filter constPaths
        std::vector<path::Path> filConstPaths;
        std::copy_if(constPaths.begin(), constPaths.end(), std::back_inserter(filConstPaths), [&](const auto& p) -> bool {
            return !QTree::CycleGuesstimator::decisionsMayOverlap(firstPath.decisions, p.decisions);
        });

        // optimize remaining paths
        QTree::MulSkipOptimizer opt(layers, samples, true);
        SimplePool<path::Path> sp(constPaths, fmt::format("Optimizer lvl 1 ({}/{})", i+1, topNum), barManager);
        sp.setNumThreads(std::max(8, int(std::thread::hardware_concurrency())-2));

        decltype(scores) localScores;

        sp.run([&](auto* sp) {
            SimplePool<path::Path>::Token work;
            while ((work = sp->popIfPossible())) {
                auto result = opt.process(work.value());
                std::lock_guard<std::mutex> l(lock);
                localScores.push_back(result);
            }
        });

        std::sort(localScores.begin(), localScores.end(), [](const auto& a, const auto& b) -> bool {
            return std::tie(a.score, a.skippedMuls, a.goodMatches, b.badMatches) > std::tie(b.score, b.skippedMuls, b.goodMatches, a.badMatches);
        });

        // select best 5
        std::vector<path::Path> candidates;
        for (size_t i = 0; i < std::min<size_t>(localScores.size(), 5); ++i) {
            candidates.push_back(localScores[i].p);
        }
        assert(!candidates.empty());


        // use permutation scheduler
        QTree::PermScheduler psch(layers.at(1).numNeurons(), {firstPath}, candidates);
                
        std::map<std::thread::id,QSim::CompilerAdaptor> compi;
        psch.run(barManager, [&](const codegen::ConstSchedule& schedule) -> double {  // FIXME
            size_t id = 0;
            decltype(compi)::iterator ca = compi.end();  // not sure how threadsafe this is?

            {  // stuff that needs protection by mutex
                std::lock_guard<std::mutex> l(lock);
                id = idCounter++;
                ca = compi.try_emplace(std::this_thread::get_id(), dataset, targets).first;
            }

            // render and simulate
            HybridGenSimple hg(layers);
            assert(schedule.numPaths() > 0);
            for (const auto& p : schedule.getPaths()) {
                hg.addPath(p);
            }
            hg.finalize();

            QSim::Runner runner(id, hg);
            return runner.run(ca->second);
        }, 3);  // limit to 3 paths, reduces runtime significantly

        score.addScore(psch.bestScore(), psch.bestSolution());
    }

    // write c files
    auto blah = score.getBestSolution();

    HybridGenSimple hg(layers);
    for (const auto& p : blah.getPaths()) {
        hg.addPath(p);
    }
    hg.finalize();
    hg.render("hybrid-simple.c");

    hg.renderReference("ref-from-hybrid-simple.c");

    // reference implementation
    renderNetwork(layers, "reference-simple.c");
}