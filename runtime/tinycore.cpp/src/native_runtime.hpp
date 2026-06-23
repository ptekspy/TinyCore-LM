#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tinycore::native {

struct BundleFileEntry {
    std::string name;
    std::string path;
    std::uint64_t offset;
    std::uint64_t length;
    std::string sha256;
};

struct TensorIndexEntry {
    std::string name;
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t offset;
    std::uint64_t length;
    std::string sha256;
};

struct FloatTensorSummary {
    std::uint64_t values;
    double sum_first_values;
    float first_value;
};

struct NativeRuntime {
    std::string manifest_json;
    std::string tensor_data;
    std::vector<TensorIndexEntry> tensors;
};

struct NativeGenerationResult {
    std::string text;
    std::vector<std::uint64_t> tokens;
    std::vector<std::uint64_t> new_tokens;
    std::vector<double> new_token_logits;
};

struct NativeGenerationOptions {
    std::uint64_t max_new_tokens;
    double temperature = 0.0;
    std::uint64_t top_k = 0;
    std::uint64_t seed = 1337;
};

std::string read_file(const std::string& path);
std::string read_tcmdl_header(const std::string& path);
std::string read_binary_slice(const std::string& path, std::uint64_t offset, std::uint64_t length);
NativeRuntime load_native_runtime(const std::string& input_path);
std::vector<std::uint64_t> encode_ascii_prompt(const std::string& prompt);
std::string decode_ascii_tokens(const std::vector<std::uint64_t>& tokens);
std::string extract_string(const std::string& json, const std::string& key);
std::uint64_t sum_section_lengths(const std::string& json);
std::size_t count_sections(const std::string& json);
std::string extract_array(const std::string& json, const std::string& key);
std::string extract_number(const std::string& json, const std::string& key);
std::vector<std::uint64_t> parse_unsigned_list(const std::string& json);
std::size_t count_file_entries(const std::string& json);
std::vector<BundleFileEntry> parse_file_entries(const std::string& files_json);
std::vector<TensorIndexEntry> parse_tensor_entries(const std::string& tensors_json);
const BundleFileEntry* find_file_entry(const std::vector<BundleFileEntry>& entries, const std::string& name);
const TensorIndexEntry* find_tensor_entry(const std::vector<TensorIndexEntry>& entries, const std::string& name);
float read_float32_at(const std::string& data, std::uint64_t byte_offset);
FloatTensorSummary summarize_float32_tensor(
    const std::string& data,
    std::uint64_t offset,
    std::uint64_t length,
    std::size_t sample_values
);
double dot_float32_rows(
    const std::string& data,
    const TensorIndexEntry& left,
    std::uint64_t left_row,
    const TensorIndexEntry& right,
    std::uint64_t right_row
);
float tensor_float32_at_flat(const std::string& data, const TensorIndexEntry& tensor, std::uint64_t flat_index);
std::vector<double> softmax_route(const std::string& data, const TensorIndexEntry& coeff, std::uint64_t virtual_layer);
double composed_linear_value(
    const std::string& data,
    const TensorIndexEntry& basis,
    const TensorIndexEntry& coeff,
    const TensorIndexEntry* low_rank_u,
    const TensorIndexEntry* low_rank_v,
    std::uint64_t virtual_layer,
    std::uint64_t in_index,
    std::uint64_t out_index
);
std::vector<double> read_float32_row(const std::string& data, const TensorIndexEntry& tensor, std::uint64_t row);
std::vector<double> read_float32_vector(const std::string& data, const TensorIndexEntry& tensor);
double sum_first(const std::vector<double>& values, std::size_t count);
std::vector<double> rms_norm(const std::vector<double>& input, const std::vector<double>& weight, double eps);
std::vector<double> composed_linear_matvec(
    const std::string& data,
    const std::vector<double>& input,
    const TensorIndexEntry& basis,
    const TensorIndexEntry& coeff,
    const TensorIndexEntry* low_rank_u,
    const TensorIndexEntry* low_rank_v,
    std::uint64_t virtual_layer
);
std::vector<std::vector<double>> composed_linear_rows(
    const std::string& data,
    const std::vector<std::vector<double>>& rows,
    const TensorIndexEntry& basis,
    const TensorIndexEntry& coeff,
    const TensorIndexEntry* low_rank_u,
    const TensorIndexEntry* low_rank_v,
    std::uint64_t virtual_layer
);
std::vector<std::vector<double>> rms_norm_rows(
    const std::vector<std::vector<double>>& rows,
    const std::vector<double>& weight,
    double eps
);
std::vector<double> mean_rows(const std::vector<std::vector<double>>& rows);
std::vector<double> dense_linear_matvec(const std::string& data, const std::vector<double>& input, const TensorIndexEntry& weight);
std::vector<double> gru_cell_update(
    const std::string& data,
    const std::vector<double>& input,
    const std::vector<double>& previous_state,
    const TensorIndexEntry& weight_ih,
    const TensorIndexEntry& weight_hh,
    const TensorIndexEntry& bias_ih,
    const TensorIndexEntry& bias_hh
);
std::vector<std::vector<double>> causal_attention_rows(
    const std::vector<std::vector<double>>& q,
    const std::vector<std::vector<double>>& k,
    const std::vector<std::vector<double>>& v,
    std::uint64_t n_heads
);
std::vector<double> embedding_head_logits(const std::string& data, const std::vector<double>& input, const TensorIndexEntry& weight);
std::vector<double> native_logits_for(const NativeRuntime& runtime, const std::vector<std::uint64_t>& input_tokens);
NativeGenerationResult generate_greedy(
    const NativeRuntime& runtime,
    const std::vector<std::uint64_t>& prompt_tokens,
    std::uint64_t max_new_tokens
);
NativeGenerationResult generate_with_options(
    const NativeRuntime& runtime,
    const std::vector<std::uint64_t>& prompt_tokens,
    const NativeGenerationOptions& options
);
double sigmoid(double value);
double silu(double value);
std::pair<std::uint64_t, double> argmax_value(const std::vector<double>& values);

}  // namespace tinycore::native
