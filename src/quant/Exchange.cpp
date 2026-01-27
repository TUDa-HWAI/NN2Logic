#include "Exchange.hpp"

void QTree::to_json(nlohmann::json& j, const QTree::FixedPoint& fp) {
    j = nlohmann::json{
        {"value", fp.value},
        {"shift", fp.shift}
    };
}

void QTree::from_json(const nlohmann::json& j, QTree::FixedPoint& fp) {
    j.at("value").get_to(fp.value);
    j.at("shift").get_to(fp.shift);
}