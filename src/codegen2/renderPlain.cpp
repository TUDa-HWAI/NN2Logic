#include "renderPlain.hpp"
#include "visitors.hpp"
#include "IRmodel.hpp"
#include "hybrid.hpp"
#include <fmt/os.h>


//void codegen2::renderFromJsonFile(const std::string& jsonFile, const std::string& filename) {}


void codegen2::renderNetwork(const std::vector<QTree::Layer<int>>& layers, const std::string& filename) {
    // in and outputs
    std::shared_ptr<codegen::c::VarDecl> inp = std::make_shared<codegen::c::VarDecl>("const uint8_t*", "inp");
    std::shared_ptr<codegen::c::VarDecl> out = std::make_shared<codegen::c::VarDecl>("int8_t*", "out");

    // comparison function to sort multiplications based on input variable
    auto mulSortCmp = [](const auto& a, const auto& b) {
        assert(a.type == Instr::MULADD && b.type == Instr::MULADD);
        if (a.opB == b.opB) return a.dst < b.dst;
        if (a.opB.length() == b.opB.length()) return a.opB < b.opB;
        return a.opB.length() < b.opB.length();
    };

    /*
     * add and sort muls of first layer
     * add relus +req
     * add and sort muls of second layer
     * add relus +req
     * add and sort muls of third layer
     * add move to output +req
     * add all the variable initializations to latest possible point?
     */
    codegen::c::Container cont;
    std::vector<std::shared_ptr<IRNeuron>> allNeurons;

    size_t layerIdx = 0;
    for (const auto& l : layers) {
        std::vector<std::shared_ptr<IRNeuron>> neurons;
        std::vector<Instr> layerMuls;

        for (size_t i = 0; i < l.numNeurons(); ++i) {
            // create neuron
            auto [w, b, r] = l.getNeuron(i);
            std::shared_ptr<IRNeuron> n = nullptr;

            if (l.relu) {  // relu
                n = std::make_shared<IRNeuronReLu>(layerIdx, i, w, b, r);
            } else {
                n = std::make_shared<IRNeuron>(layerIdx, i, w, b, r);
                assert(layerIdx == layers.size()-1);  // ensures only last layer has no ReLU activation function
            }

            // save neuron
            allNeurons.push_back(n);
            neurons.push_back(n);

            // extract multiplications
            auto muls = n->popMuls();
            layerMuls.insert(layerMuls.end(), muls.begin(), muls.end());
        }

        // sort muls and add generate code
        std::sort(layerMuls.begin(), layerMuls.end(), mulSortCmp);
        // using the iterator way because in theory i did implement a conversion function from Instr to Statement
        std::vector<std::shared_ptr<codegen::c::Statement>> layerCode(layerMuls.begin(), layerMuls.end());

        // add variable declarations and initializations
        for (auto n : neurons) {
            VarUseFinder vf(n->var());
            auto pos = std::find_if(layerCode.begin(), layerCode.end(), vf);
            assert(pos != layerCode.end());

            auto toIns = n->getInit();
            layerCode.insert(pos, toIns.begin(), toIns.end());
        }

        // add if/else
        /*
        for (auto n : neurons) {
            layerCode.push_back(n->instantiateIfElse());
        }
        */
        std::for_each(neurons.rbegin(), neurons.rend(), [&](auto n) {
            placeIfElse(layerCode, n->instantiateIfElse(), n->var());
        });

        // last layer -> move outputs
        if (l.relu) {
            // FIXME
        }

        // put layerCode into Container
        cont.create<codegen::c::Comment>(fmt::format("--- layer {} ---", layerIdx));
        cont.add(layerCode);


        ++layerIdx;
    }


    // create output FIXME
    cont.create<codegen::c::RawStatement>("out[0] = x_2_00;\n");
    cont.create<codegen::c::RawStatement>("out[1] = x_2_01;\n");


    // write to file
    codegen::c::Function func{
        .name = "network",
        .retType = "void",
        .params = {*inp, *out},
        .members = cont.get()
    };

    // print to file
    auto file = fmt::output_file(filename);
    file.print("#include \"network.h\"\n\n");

    file.print(func.print());
}