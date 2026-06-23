#include "../src/native_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace tinycore::native;

const TensorIndexEntry& require_tensor(const NativeRuntime& runtime, const std::string& name) {
    const TensorIndexEntry* tensor = find_tensor_entry(runtime.tensors, name);
    if (tensor == nullptr) {
        throw std::runtime_error("missing tensor in test: " + name);
    }
    return *tensor;
}

void require_true(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double actual, double expected, double tolerance, const std::string& message) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) + ", got " + std::to_string(actual)
        );
    }
}

std::string artifact_path(const std::string& suffix) {
    return std::string(TINYCORE_REPO_ROOT) + "/" + suffix;
}

void test_tensor_index_and_bytes(const NativeRuntime& runtime) {
    require_true(extract_string(runtime.manifest_json, "architecture") == "tinycore_recurrent_v0", "architecture mismatch");
    require_true(runtime.tensors.size() == 40, "unexpected tensor count");

    const TensorIndexEntry& token_embedding = require_tensor(runtime, "token_emb.weight");
    require_true(token_embedding.dtype == "float32", "token embedding dtype mismatch");
    require_true(token_embedding.shape.size() == 2, "token embedding rank mismatch");
    require_true(token_embedding.shape[1] == 32, "token embedding width mismatch");

    const FloatTensorSummary summary =
        summarize_float32_tensor(runtime.tensor_data, token_embedding.offset, token_embedding.length, 8);
    require_true(summary.values == 4096, "token embedding value count mismatch");
    require_close(summary.sum_first_values, -2.12113, 1e-5, "token embedding first values sum mismatch");
}

void test_composed_q_projection(const NativeRuntime& runtime) {
    const TensorIndexEntry& token_embedding = require_tensor(runtime, "token_emb.weight");
    const TensorIndexEntry& position_embedding = require_tensor(runtime, "pos_emb.weight");
    const TensorIndexEntry& norm1_weight = require_tensor(runtime, "block.norm1.weight");
    const TensorIndexEntry& q_basis = require_tensor(runtime, "block.q.basis");
    const TensorIndexEntry& q_coeff = require_tensor(runtime, "block.q.coeff");
    const TensorIndexEntry& q_u = require_tensor(runtime, "block.q.u");
    const TensorIndexEntry& q_v = require_tensor(runtime, "block.q.v");

    std::vector<double> x = read_float32_row(runtime.tensor_data, token_embedding, 84);
    const std::vector<double> pos = read_float32_row(runtime.tensor_data, position_embedding, 0);
    for (std::size_t index = 0; index < x.size(); ++index) {
        x[index] += pos[index];
    }
    const std::vector<double> normalized =
        rms_norm(x, read_float32_vector(runtime.tensor_data, norm1_weight), 1e-6);
    const std::vector<double> q_output =
        composed_linear_matvec(runtime.tensor_data, normalized, q_basis, q_coeff, &q_u, &q_v, 0);

    require_true(q_output.size() == 32, "q projection width mismatch");
    require_close(q_output.front(), 0.291857, 1e-5, "q projection first value mismatch");
    require_close(sum_first(q_output, 8), 1.45013, 1e-5, "q projection first values sum mismatch");
}

void test_greedy_generation(const NativeRuntime& runtime) {
    const NativeGenerationResult generated = generate_greedy(runtime, encode_ascii_prompt("TinyCore"), 8);
    const std::vector<std::uint64_t> expected_new_tokens = {101, 101, 101, 114, 114, 101, 101, 32};

    require_true(generated.text == "TinyCoreeeerree ", "generated text mismatch");
    require_true(generated.new_tokens == expected_new_tokens, "generated token mismatch");
    require_true(generated.new_token_logits.size() == expected_new_tokens.size(), "generated logits count mismatch");
    require_close(generated.new_token_logits.front(), 6.27124, 1e-5, "first generated logit mismatch");
}

void test_top_k_one_matches_greedy(const NativeRuntime& runtime) {
    const NativeGenerationResult greedy = generate_greedy(runtime, encode_ascii_prompt("TinyCore"), 8);
    const NativeGenerationResult sampled = generate_with_options(
        runtime,
        encode_ascii_prompt("TinyCore"),
        NativeGenerationOptions{
            .max_new_tokens = 8,
            .temperature = 0.8,
            .top_k = 1,
            .seed = 2026,
        }
    );

    require_true(sampled.text == greedy.text, "top_k=1 text should match greedy");
    require_true(sampled.new_tokens == greedy.new_tokens, "top_k=1 tokens should match greedy");
}

void test_seeded_top_k_sampling_stays_in_top_k(const NativeRuntime& runtime) {
    const std::uint64_t top_k = 4;
    const NativeGenerationResult sampled = generate_with_options(
        runtime,
        encode_ascii_prompt("TinyCore"),
        NativeGenerationOptions{
            .max_new_tokens = 8,
            .temperature = 0.8,
            .top_k = top_k,
            .seed = 2026,
        }
    );

    require_true(sampled.new_tokens.size() == 8, "sampled token count mismatch");
    std::vector<std::uint64_t> prefix = encode_ascii_prompt("TinyCore");
    for (std::size_t step = 0; step < sampled.new_tokens.size(); ++step) {
        const std::vector<double> logits = native_logits_for(runtime, prefix);
        std::vector<std::pair<double, std::uint64_t>> ranked;
        ranked.reserve(logits.size());
        for (std::uint64_t token = 0; token < logits.size(); ++token) {
            ranked.push_back({logits[static_cast<std::size_t>(token)], token});
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) {
            return left.first > right.first;
        });

        bool found = false;
        for (std::size_t index = 0; index < top_k; ++index) {
            if (ranked[index].second == sampled.new_tokens[step]) {
                found = true;
                break;
            }
        }
        require_true(found, "sampled token was outside top_k");
        prefix.push_back(sampled.new_tokens[step]);
    }
}

void test_generation_clamps_to_context_window(const NativeRuntime& runtime) {
    const std::vector<std::uint64_t> prompt = encode_ascii_prompt("<user>\nTinyCore\n\n<assistant>\n");
    const NativeGenerationResult generated = generate_greedy(runtime, prompt, 8);

    require_true(generated.tokens.size() == 32, "generated tokens should stop at max context length");
    require_true(generated.new_tokens.size() == 3, "generated token count should be clamped to remaining context");
}

}  // namespace

int main() {
    try {
        const NativeRuntime bundle =
            load_native_runtime(artifact_path("reports/runs/ablation_toy/tinycore_recurrent_v0.tcmdl"));
        test_tensor_index_and_bytes(bundle);
        test_composed_q_projection(bundle);
        test_greedy_generation(bundle);
        test_top_k_one_matches_greedy(bundle);
        test_seeded_top_k_sampling_stays_in_top_k(bundle);
        test_generation_clamps_to_context_window(bundle);

        const NativeRuntime extracted =
            load_native_runtime(artifact_path("reports/runs/ablation_toy/tinycore_recurrent_v0_extracted"));
        test_greedy_generation(extracted);
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }

    return 0;
}
