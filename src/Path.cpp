#include "Path.hpp"


/*
 * Path
 */
void path::to_json(nlohmann::json& j, const path::Path& p) {
    j = nlohmann::json{
        {"visitFreq", p.visitFreq},
        {"decisions", p.decisions},
        {"leaf", p.leaf}
    };
}

void path::from_json(const nlohmann::json& j, path::Path& p) {
    j.at("visitFreq").get_to(p.visitFreq);
    j.at("decisions").get_to(p.decisions);
    //j.at("leaf").get_to(p.leaf);
    //auto test = j.at("leaf").template get<path::Leaf>();

    p.leaf = path::Leaf(
        j.at("leaf").at("possClasses").template get<std::vector<bool>>(),
        j.at("leaf").at("iis").template get<std::set<std::pair<size_t,size_t>>>()
    );
}


/*
 * Decision
 */
void path::to_json(nlohmann::json& j, const path::Decision& d) {
    j = nlohmann::json{ 
        {"layerIdx", d.layerIdx},
        {"neuronIdx", d.neuronIdx},
        {"decision", d.decision},
        {"isConst", d.isConst},
        {"iis", d.iisNodes}
    };
}

void path::from_json(const nlohmann::json& j, path::Decision& d) {
    j.at("layerIdx").get_to(d.layerIdx);
    j.at("neuronIdx").get_to(d.neuronIdx);
    j.at("decision").get_to(d.decision);
    j.at("isConst").get_to(d.isConst);
    j.at("iis").get_to(d.iisNodes);
}


/*
 * Leaf
 */
void path::to_json(nlohmann::json& j, const path::Leaf& l) {
    j = nlohmann::json{
        {"isConst", l.isConst()},
        {"possClasses", l.possClasses()},
        {"iis", l.iisIDxs()}
    };
}

// void from_json(const nlohmann::json& j, path::Leaf& l) {


//     //std::vector<bool> possClasses;
//     //j.at("possClasses").get_to(possClasses);
//     //fmt::print("possClasses {}\n", fmt::join(possClasses, ""));
//     //l = path::Leaf(j.at("possClasses"), j.at("iis"));
// }