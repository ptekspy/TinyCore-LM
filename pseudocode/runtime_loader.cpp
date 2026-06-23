#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// TCMDL native runtime spike placeholder.
// Do not optimise before Python architecture stabilises.

struct TensorSection {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t offset;
    uint64_t length;
};

struct TCMDLManifest {
    std::string format;
    std::string format_version;
    std::string architecture;
    std::vector<TensorSection> sections;
};

class TCMDLLoader {
public:
    explicit TCMDLLoader(const std::string& path) : path_(path) {}

    bool open() {
        std::ifstream file(path_, std::ios::binary);
        if (!file) return false;
        // TODO: parse magic, header length, JSON manifest, section table.
        return true;
    }

private:
    std::string path_;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: tinycore-inspect <model.tcmdl>\n";
        return 1;
    }
    TCMDLLoader loader(argv[1]);
    if (!loader.open()) {
        std::cerr << "failed to open model\n";
        return 1;
    }
    std::cout << "TCMDL loader stub OK\n";
    return 0;
}
