#pragma once

#include <vector>
#include <tuple>
#include "quant/Exchange.hpp"

namespace codegen2 {

void renderNetwork(const std::vector<QTree::Layer<int>>& layers, const std::string& filename);

}