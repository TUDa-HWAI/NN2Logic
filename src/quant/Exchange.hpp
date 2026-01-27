#pragma once

#include <nlohmann/json.hpp>
#include <Eigen/Dense>


namespace QTree {


struct FixedPoint final {
    size_t value;
    size_t shift;

    operator double() const {
        return ((double) value) / (1 << shift);
    }
};


template <typename T>
struct Scaler final {
    //Eigen::VectorXf scales;
    std::vector<FixedPoint> scales;
    T upperLimit;
    T lowerLimit = 0;
};


template <typename T>
struct Layer final {
    typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> Matrix_t;
    typedef Eigen::Matrix<T,Eigen::Dynamic,1> Vec_t;

    Matrix_t weight;
    Vec_t bias;
    bool relu;
    Scaler<T> requant;

    operator std::tuple<Matrix_t,Vec_t,std::vector<FixedPoint>>() const {
        //return ((double) value) / (1 << shift);
        return std::make_tuple(weight, bias, requant.scales);
    }

    [[nodiscard]] size_t numNeurons() const {
        assert(bias.size() == weight.rows());
        return bias.size();
    }

    [[nodiscard]] std::tuple<Vec_t,T,FixedPoint> getNeuron(size_t idx) const {
        assert(idx < numNeurons());

        return std::make_tuple(weight.row(idx), bias[idx], requant.scales.at(idx));
    }

    [[nodiscard]] size_t numInputs() const {
        assert(bias.size() == weight.rows());
        return weight.cols();
    }
};


void to_json(nlohmann::json& j, const FixedPoint& fp);
void from_json(const nlohmann::json& j, FixedPoint& fp);


template <typename T>
void to_json(nlohmann::json& j, const Scaler<T>& s) {
    j = nlohmann::json{
        {"upperLimit", s.upperLimit},
        {"scales", s.scales}
    };
}

template <typename T>
void from_json(const nlohmann::json& j, Scaler<T>& s) {
    j.at("upperLimit").get_to(s.upperLimit);
    
    //std::vector<float> scales;
    j.at("scales").get_to(s.scales);
    // s.scales = Eigen::VectorXf(scales.size());
    // auto it = s.scales.begin();
    // for (T w : scales) {
    //     //s.scales << w;
    //     *it = w;
    //     ++it;
    // }
}

template <typename T>
void to_json(nlohmann::json& j, const Layer<T>& l) {
    // helper
    auto m2j = [](const auto& m) {
        nlohmann::json a;
        for (const auto& r : m.rowwise()) {
            a.push_back(r);
        }

        return a;
    };

    j = nlohmann::json{
        {"weight", m2j(l.weight)}, 
        {"bias", l.bias},
        {"relu", l.relu},
        {"requant", l.requant}
    };
}

template <typename T>
void from_json(const nlohmann::json& j, Layer<T>& l) {
    j.at("relu").get_to(l.relu);
    j.at("requant").get_to(l.requant);

    // parse bias
    std::vector<T> bias;
    j.at("bias").get_to(bias);
    l.bias = typename Layer<T>::Vec_t(bias.size());

{
    auto it = l.bias.begin();
    for (T w : bias) {
        *it = w;
        ++it;
    }
}

    // parse weight
    std::vector<std::vector<T>> weight;
    j.at("weight").get_to(weight);
    l.weight = typename Layer<T>::Matrix_t(weight.size(), weight.front().size());

    for (size_t i = 0; i < weight.size(); ++i) {
        for (size_t ii = 0; ii < weight.front().size(); ++ii) {
            l.weight(i, ii) = weight.at(i).at(ii);
        }
    }
/*
{
    auto it = l.weight.reshaped().begin();
    for (const auto& row : weight) {
        for (T w : row) {
            *it = w;
            ++it;
        }
    }
}
*/
}
}