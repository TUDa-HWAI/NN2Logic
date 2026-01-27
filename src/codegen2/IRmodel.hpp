#pragma once

#include <vector>
#include <variant>
#include "quant/Exchange.hpp"
#include "codegen/c.hpp"
#include "codegen/container.hpp"

#include <iostream>


namespace codegen2 {

struct IRInput final {
    size_t index;
};


struct Instr {
    enum type_t {
        VARDECL, ASSIGN, MULADD, COND_GT_ZERO, MULTIPLY, LSHIFT, EXIT, COND_GT, SET_BIT
    } type;

    std::string dst;
    std::string opA;
    std::string opB;

    Instr(type_t t, const std::string& dst, const std::string& opA, const std::string& opB = "")
        : type(t), dst(dst), opA(opA), opB(opB) {}


    operator std::shared_ptr<codegen::c::Statement>() const {  // FIXME
        if (type == VARDECL) return std::make_shared<codegen::c::VarDecl>("int32_t", dst);
        if (type == ASSIGN) return std::make_shared<codegen::c::Assignment>(dst, opA);
        if (type == MULADD) return std::make_shared<codegen::c::Assignment>(dst, codegen::c::Arith(
            opB, codegen::c::Arith::MUL, std::stoi(opA) // FIXME: stoi is kinda ugly here 
        ), true);

        if (type == SET_BIT) return std::make_shared<codegen::c::RawStatement>(fmt::format("{} |= 0b{:b};\n", dst, size_t(1) << std::stoi(opA)));
        
        assert(false);
        return nullptr;
    }
};


class IRNeuron {
protected:
    Instr varInit;
    std::vector<Instr> muls;
    QTree::FixedPoint requant;


    static std::shared_ptr<codegen::c::Ifelse> saturate(const std::string& var, int32_t val) {  // FIXME
        assert(val != 0);
        
        codegen::c::Comparison cmp{
                .variable = var,
                .type = val < 0 ? codegen::c::Comparison::LT : codegen::c::Comparison::GT,
                .value = val
            };

        return std::make_shared<codegen::c::Ifelse>(cmp,
            codegen::c::Block(std::make_shared<codegen::c::Assignment>(var, val))
        );
    }
public:
    const size_t layerIdx;
    const size_t neuronIdx;

    [[nodiscard]] std::string var() const {
        return fmt::format("x_{}_{:02d}", layerIdx, neuronIdx);
    }

    IRNeuron(size_t layerIdx, size_t neuronIdx, const QTree::Layer<int>::Vec_t weights, int bias, QTree::FixedPoint requant)
        : layerIdx(layerIdx), neuronIdx(neuronIdx), varInit(Instr::ASSIGN, var(), std::to_string(bias)), requant(requant) {

        varInit.dst = var();  // FIXME this is ugly

        std::string formatString = layerIdx == 0 ? "inp[{}]" : fmt::format("x_{}_{{:02d}}", layerIdx-1);

        // build weights
        for (size_t i = 0; i < weights.size(); ++i) {
            if (weights[i] == 0) continue;

            muls.emplace_back(Instr::MULADD, var(), std::to_string(weights[i]), fmt::format(formatString, i));
        }
    }

    virtual ~IRNeuron() {}

    std::vector<std::shared_ptr<codegen::c::Statement>> getInit() const {
        codegen::c::Container toRet;

        toRet.create<codegen::c::RawStatement>(fmt::format("int32_t {} = {};\n", var(), varInit.opA));

        return toRet.get();
    }


    [[nodiscard]] std::pair<size_t,size_t> idx() const {
        return std::make_pair(layerIdx, neuronIdx);
    }

    std::vector<Instr> extractMulIf(std::function<bool(const Instr&)> func) {
        std::vector<Instr> keep, ret;

        for (const auto& i : muls) {
            if (func(i)) ret.push_back(i);
            else keep.push_back(i);
        }

        muls = keep;

        return ret;
    }

    std::vector<Instr> popMuls() {
        std::vector<Instr> toRet;
        std::swap(toRet, muls);
        return toRet;
    }

    virtual std::shared_ptr<codegen::c::Ifelse> instantiateIfElse() const {        
        /*
         if > 0
           mul and shift
           if > 127
             -> 127
         else
           invert
           mul and shift
          if < -127
            -> -127
         */

        auto blockPos = codegen::c::Block{};
        blockPos.create<codegen::c::RawStatement>(fmt::format("{} = ({} * {}) >> {};\n", var(), var(), requant.value, requant.shift));
        blockPos.add(saturate(var(), 127));

        auto blockNeg = codegen::c::Block{};
        blockNeg.create<codegen::c::RawStatement>(fmt::format("{} = ~((~({} * {})) >> {});\n", var(), var(), requant.value, requant.shift));
        blockNeg.add(saturate(var(), -127));

        auto ifelse = std::make_shared<codegen::c::Ifelse>(
            codegen::c::Comparison{
                .variable = var(),
                .type = codegen::c::Comparison::GT,
                .value = 0
            }, blockPos, blockNeg
        );

        return ifelse;
    }
};


class IRNeuronReLu : public IRNeuron {

    std::vector<Instr> blockIf, blockElse;


    static std::vector<std::shared_ptr<codegen::c::Statement>> convInstr(const std::vector<Instr>& instr) {
        return std::vector<std::shared_ptr<codegen::c::Statement>>(instr.begin(), instr.end());
    }
public:
    using IRNeuron::IRNeuron;

    void addToIf(Instr a) {
        blockIf.push_back(a);
    }

    void addToElse(Instr a) {
        blockElse.push_back(a);
    }

    std::shared_ptr<codegen::c::Ifelse> instantiateIfElse() const override {
        auto blockPos = codegen::c::Block{};
        blockPos.create<codegen::c::RawStatement>(fmt::format("{} = ({} * {}) >> {};\n", var(), var(), requant.value, requant.shift));
        blockPos.add(saturate(var(), 127));
        blockPos.add(convInstr(blockIf));

        auto blockNeg = codegen::c::Block{};
        blockNeg.create<codegen::c::RawStatement>(fmt::format("{} = 0;\n", var()));  // FIXME
        blockNeg.add(convInstr(blockElse));

        auto ifelse = std::make_shared<codegen::c::Ifelse>(
            codegen::c::Comparison{
                .variable = var(),
                .type = codegen::c::Comparison::GT,
                .value = 0
            }, blockPos, blockNeg
        );

        return ifelse;
    }
};


}