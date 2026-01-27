#pragma once

#include <filesystem>
#include <fmt/format.h>
#include "CompilerAdaptor.hpp"

namespace QSim {

class Runner final {
    const size_t m_id;
    std::filesystem::path m_srcFile;
    const bool m_isHybrid;


    void genFilePath() {
        // create filebase
        std::filesystem::path tmp("./gen");
        std::filesystem::create_directory(tmp);
        assert(std::filesystem::exists(tmp));

        m_srcFile = tmp / fmt::format("{}_{:06d}.c", m_isHybrid ? "hybrid" : "network", m_id);
    }


public:
    template <typename H>
    Runner(size_t id, const H& generator) : m_id(id), m_isHybrid(true) {
        genFilePath();
        generator.render(m_srcFile.string());
        assert(std::filesystem::exists(m_srcFile));
    }


    // TODO return runtime and accuracy
    size_t run(CompilerAdaptor& ca) const {
        // Add hybrid implementation and compile
        ca.addFile(m_srcFile.string(), m_isHybrid ? "hybrid.c" : "network.c");
        assert(ca.compile());

        // execute
        auto res = executeCommand(fmt::format("/opt/riscv/bin/spike --isa=RV32IMAC_ZICSR {}", ca.get().string()));
        assert(res.status >= 0);


        // parse spike output
        std::stringstream ss(res.stdout);

        size_t count; char r;
        size_t total = 0;
        size_t iteration = 0;
        while(!ss.eof()) {
            assert(iteration++ < 100000);

            count = 0;
            r = ' ';
            ss >> count >> r;

            if (count == 0) continue;

            if (r != 'T' && r != 'F') {
                //std::cout << res.stdout << std::endl;
                std::cout << "expected T or F but got \"" << r << "\"; " << count << std::endl;
                throw std::runtime_error("could not parse output of network\n");
            }
            total += count;
        }

        // remove files
        ca.removeFile("test.elf");
        ca.removeFile(m_isHybrid ? "hybrid.c" : "network.c");
        ca.removeFile(m_isHybrid ? "hybrid.o" : "network.o");

        return total;
    }
};
}