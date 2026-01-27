#pragma once

#include <filesystem>
#include <string>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <fmt/format.h>
#include <fmt/os.h>
#include <iostream>
#include <vector>
#include <thread>
#include <fmt/ranges.h>


namespace QSim {


struct ExecResult final {
    const int status;
    const std::string stdout;
};


ExecResult executeCommand(const std::string& cmd);


class CompilerAdaptor final {
    std::filesystem::path m_dir;

public:
    CompilerAdaptor() {
        auto tmp_dir = std::filesystem::temp_directory_path();
        m_dir = tmp_dir / "LeTree_Sim" / std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));

        if (std::filesystem::exists(m_dir)) {  // check if directory already exists
            std::filesystem::remove_all(m_dir);
        }

        std::filesystem::create_directories(m_dir);  // create directory

        // initialize with templates
        // TODO check if local exists and fall back to BUILD_DIR version
        std::filesystem::copy(std::filesystem::path(BUILD_DIR) / "template", m_dir, std::filesystem::copy_options::recursive);
    }

    // pass training set
    CompilerAdaptor(const std::vector<std::vector<int>>& data, const std::vector<int>& target) : CompilerAdaptor() {
        assert(data.size() == target.size());
        
        // intermediate strings
        std::vector<std::string> im;
        im.reserve(data.size());
        for (const auto& sample : data) {
            im.push_back(fmt::format("{{ {} }}", fmt::join(sample, ", ")));
        }

        // create file
        auto filePath = m_dir / "dataset.h";
        auto out = fmt::output_file(filePath.string());

        // write actual file
        out.print(
            "#ifndef H_DATASET\n"
            "#define H_DATASET\n"

            "#include <stdint.h>\n"
            "#include <stdbool.h>\n"

            "const size_t datasetSize = {0};\n"

            "const uint8_t dataset[{0}][{1}] = {{\n"
            "\t{2}\n"
            "}};"

            "const uint8_t target[] = {{ {3} }};\n"

            "#endif",
                data.size(), data.front().size(),
                fmt::join(im, ",\n\t"), fmt::join(target, ", ")
            );
    }

    ~CompilerAdaptor() {
        std::filesystem::remove_all(m_dir);
    }

    void addFile(const std::filesystem::path& src, const std::string& dst = "") const;

    bool compile() const {
        //auto res = executeCommand(fmt::format("/bin/bash -c 'cd \"{}\";PATH=$PATH:/opt/riscv/bin make &> compilelog.txt'", m_dir.string()));
        auto res = executeCommand(fmt::format("/bin/bash -c 'cd \"{}\";PATH=$PATH:/opt/riscv/bin make &> /dev/null'", m_dir.string()));
        
        if (res.status < 0)
            throw std::runtime_error(res.stdout);


        return res.status >= 0;
    }


    std::filesystem::path get() const {
        auto f = m_dir / "test.elf";
        if (!std::filesystem::exists(f)) {
            fmt::print("tried using {} but did not exist\n", f.string());
            assert(false);
        }
        return f;
    }

    bool removeFile(const std::string& name) const {
        return std::filesystem::remove(m_dir / name);
    }
};
}