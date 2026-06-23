#include "native_runtime.hpp"

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace tinycore::native;

std::string json_escape(const std::string& text) {
    std::ostringstream escaped;
    for (unsigned char character : text) {
        switch (character) {
            case '\\':
                escaped << "\\\\";
                break;
            case '"':
                escaped << "\\\"";
                break;
            case '\n':
                escaped << "\\n";
                break;
            case '\r':
                escaped << "\\r";
                break;
            case '\t':
                escaped << "\\t";
                break;
            default:
                if (character < 0x20) {
                    escaped << "\\u";
                    escaped << "00";
                    const char* digits = "0123456789abcdef";
                    escaped << digits[(character >> 4) & 0x0F] << digits[character & 0x0F];
                } else {
                    escaped << static_cast<char>(character);
                }
                break;
        }
    }
    return escaped.str();
}

void print_token_array(const std::vector<std::uint64_t>& tokens) {
    std::cout << "[";
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index > 0) {
            std::cout << ",";
        }
        std::cout << tokens[index];
    }
    std::cout << "]";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 7) {
        std::cerr << "usage: tinycore-generate <artifact-dir|model.tcmdl> [prompt] [max_new_tokens] [temperature] [top_k] [seed]\n";
        return 2;
    }

    try {
        const std::string prompt = argc >= 3 ? argv[2] : "TinyCore";
        const std::uint64_t max_new_tokens = argc >= 4 ? static_cast<std::uint64_t>(std::stoull(argv[3])) : 8;
        const double temperature = argc >= 5 ? std::stod(argv[4]) : 0.0;
        const std::uint64_t top_k = argc >= 6 ? static_cast<std::uint64_t>(std::stoull(argv[5])) : 0;
        const std::uint64_t seed = argc >= 7 ? static_cast<std::uint64_t>(std::stoull(argv[6])) : 1337;
        const NativeRuntime runtime = load_native_runtime(argv[1]);
        const NativeGenerationResult generated =
            generate_with_options(
                runtime,
                encode_ascii_prompt(prompt),
                NativeGenerationOptions{
                    .max_new_tokens = max_new_tokens,
                    .temperature = temperature,
                    .top_k = top_k,
                    .seed = seed,
                }
            );

        std::cout << "{\"text\":\"" << json_escape(generated.text) << "\",";
        std::cout << "\"tokens\":";
        print_token_array(generated.tokens);
        std::cout << ",\"new_tokens\":";
        print_token_array(generated.new_tokens);
        std::cout << ",\"generation\":{\"temperature\":" << temperature << ",\"top_k\":" << top_k << ",\"seed\":" << seed << "}";
        std::cout << ",\"runtime\":\"native\",";
        std::cout << "\"model\":{\"architecture\":\"" << json_escape(extract_string(runtime.manifest_json, "architecture")) << "\",";
        std::cout << "\"model_name\":\"" << json_escape(extract_string(runtime.manifest_json, "model_name")) << "\",";
        std::cout << "\"run_id\":\"" << json_escape(extract_string(runtime.manifest_json, "run_id")) << "\"}}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }

    return 0;
}
