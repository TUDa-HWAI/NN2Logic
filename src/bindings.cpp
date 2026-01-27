#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11_json/pybind11_json.hpp>

#include <gurobi_c++.h>

#include "Encoder.hpp"
#include "quant/SequentialCreator.hpp"
#include "codegen2/test.hpp"

PYBIND11_MODULE(nn2logic, m) {
    pybind11::class_<QTree::FixedPoint>(m, "FixedPoint")
        .def(pybind11::init<size_t,size_t>());

    m.def("QHybridCreator", &codegen2::loadFromJson);

    // Encoder
    pybind11::class_<Encoder>(m, "InputEncoder")
        .def(pybind11::init())
        .def("registerScalar", &Encoder::registerScalar)
        .def("registerBinary", &Encoder::registerBinary)
        .def("registerInt", &Encoder::registerInt)
        .def("update", &Encoder::update)
        .def("markBinariesOneHot", &Encoder::markBinariesOneHot);

    // Scaler
    pybind11::class_<QTree::Scaler<int>>(m, "QScales")
        .def(pybind11::init<std::vector<QTree::FixedPoint>&,int>())
        .def_readwrite("scales", &QTree::Scaler<int>::scales)
        .def_readwrite("upperLimit", &QTree::Scaler<int>::upperLimit)
        .def_readwrite("lowerLimit", &QTree::Scaler<int>::lowerLimit);

    // Layer Descriptor
    pybind11::class_<QTree::Layer<int>>(m, "QLayer")
        .def(pybind11::init<const Eigen::Ref<const Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic>>,const Eigen::Ref<const Eigen::Matrix<int,Eigen::Dynamic,1>>,bool,const QTree::Scaler<int>&>())
        .def_readwrite("weight", &QTree::Layer<int>::weight)
        .def_readwrite("bias", &QTree::Layer<int>::bias)
        .def_readwrite("relu", &QTree::Layer<int>::relu)
        .def_readwrite("requant", &QTree::Layer<int>::requant);

    // Quantized Sequential Tree Creator Class
    pybind11::class_<QTree::SequentialCreator<int,int>>(m, "QTreeBuilder")
        .def(pybind11::init<const std::vector<QTree::Layer<int>>&, Encoder*, const Eigen::Ref<const Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic>>>())
        .def("getPaths", &QTree::SequentialCreator<int,int>::getPaths)
        .def("print", &QTree::SequentialCreator<int,int>::print)
        .def("toJson", &QTree::SequentialCreator<int,int>::toJson);


    // handle Gurobi exceptions
    pybind11::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const GRBException &e) {
            throw std::runtime_error("Gurobi Error code = " + std::to_string(e.getErrorCode()) + "\n" + e.getMessage());
        }
    });
}