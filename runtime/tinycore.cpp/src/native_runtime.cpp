#include "native_runtime.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tinycore::native {

std::string read_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open manifest: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::string read_tcmdl_header(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open bundle: " + path);
    }
    char magic[6];
    input.read(magic, sizeof(magic));
    if (input.gcount() != static_cast<std::streamsize>(sizeof(magic)) || std::string(magic, sizeof(magic)) != std::string("TCMDL\0", 6)) {
        throw std::runtime_error("bundle does not start with TCMDL magic");
    }
    std::uint64_t header_length = 0;
    input.read(reinterpret_cast<char*>(&header_length), sizeof(header_length));
    if (input.gcount() != static_cast<std::streamsize>(sizeof(header_length))) {
        throw std::runtime_error("bundle is missing header length");
    }
    std::string header(header_length, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header_length));
    if (input.gcount() != static_cast<std::streamsize>(header_length)) {
        throw std::runtime_error("bundle is missing header bytes");
    }
    return header;
}

std::string read_binary_slice(const std::string& path, std::uint64_t offset, std::uint64_t length) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open bundle: " + path);
    }
    input.seekg(static_cast<std::streamoff>(offset));
    std::string data(length, '\0');
    input.read(data.data(), static_cast<std::streamsize>(length));
    if (input.gcount() != static_cast<std::streamsize>(length)) {
        throw std::runtime_error("failed to read bundle payload slice");
    }
    return data;
}

NativeRuntime load_native_runtime(const std::string& input_path) {
    if (input_path.ends_with(".tcmdl")) {
        const std::string header_json = read_tcmdl_header(input_path);
        const std::vector<BundleFileEntry> files = parse_file_entries(extract_array(header_json, "files"));
        const std::uint64_t payload_start = 6 + sizeof(std::uint64_t) + header_json.size();

        const BundleFileEntry* manifest = find_file_entry(files, "manifest.json");
        const BundleFileEntry* tensor_index = find_file_entry(files, "tensor_index.json");
        const BundleFileEntry* tensor_data = find_file_entry(files, "tensors.bin");
        if (manifest == nullptr || tensor_index == nullptr || tensor_data == nullptr) {
            throw std::runtime_error("bundle must contain manifest.json, tensor_index.json, and tensors.bin");
        }

        const std::string manifest_json =
            read_binary_slice(input_path, payload_start + manifest->offset, manifest->length);
        const std::string tensor_index_json =
            read_binary_slice(input_path, payload_start + tensor_index->offset, tensor_index->length);
        return NativeRuntime{
            .manifest_json = manifest_json,
            .tensor_data = read_binary_slice(input_path, payload_start + tensor_data->offset, tensor_data->length),
            .tensors = parse_tensor_entries(extract_array(tensor_index_json, "tensors")),
        };
    }

    std::filesystem::path artifact_path(input_path);
    if (artifact_path.filename() == "manifest.json") {
        artifact_path = artifact_path.parent_path();
    }
    return NativeRuntime{
        .manifest_json = read_file((artifact_path / "manifest.json").string()),
        .tensor_data = read_file((artifact_path / "tensors.bin").string()),
        .tensors = parse_tensor_entries(extract_array(read_file((artifact_path / "tensor_index.json").string()), "tensors")),
    };
}

std::vector<std::uint64_t> encode_ascii_prompt(const std::string& prompt) {
    std::vector<std::uint64_t> tokens;
    tokens.reserve(prompt.size());
    for (unsigned char character : prompt) {
        tokens.push_back(static_cast<std::uint64_t>(character));
    }
    return tokens;
}

std::string decode_ascii_tokens(const std::vector<std::uint64_t>& tokens) {
    std::string text;
    text.reserve(tokens.size());
    for (std::uint64_t token : tokens) {
        text.push_back(static_cast<char>(token));
    }
    return text;
}

std::string extract_string(const std::string& json, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        return "";
    }
    return match[1].str();
}

std::uint64_t sum_section_lengths(const std::string& json) {
    const std::regex pattern("\"length\"\\s*:\\s*(\\d+)");
    std::uint64_t total = 0;
    for (auto it = std::sregex_iterator(json.begin(), json.end(), pattern); it != std::sregex_iterator(); ++it) {
        total += static_cast<std::uint64_t>(std::stoull((*it)[1].str()));
    }
    return total;
}

std::size_t count_sections(const std::string& json) {
    const std::regex pattern("\"offset\"\\s*:");
    return static_cast<std::size_t>(std::distance(std::sregex_iterator(json.begin(), json.end(), pattern), std::sregex_iterator()));
}

std::string extract_array(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) {
        return "";
    }
    const std::size_t start = json.find('[', key_pos + needle.size());
    if (start == std::string::npos) {
        return "";
    }
    int depth = 0;
    for (std::size_t index = start; index < json.size(); ++index) {
        if (json[index] == '[') {
            ++depth;
        } else if (json[index] == ']') {
            --depth;
            if (depth == 0) {
                return json.substr(start, index - start + 1);
            }
        }
    }
    return "";
}

std::string extract_number(const std::string& json, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(\\d+)");
    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        return "0";
    }
    return match[1].str();
}

std::vector<std::uint64_t> parse_unsigned_list(const std::string& json) {
    std::vector<std::uint64_t> values;
    const std::regex pattern("(\\d+)");
    for (auto it = std::sregex_iterator(json.begin(), json.end(), pattern); it != std::sregex_iterator(); ++it) {
        values.push_back(static_cast<std::uint64_t>(std::stoull((*it)[1].str())));
    }
    return values;
}

std::size_t count_file_entries(const std::string& json) {
    const std::regex pattern("\"sha256\"\\s*:");
    return static_cast<std::size_t>(std::distance(std::sregex_iterator(json.begin(), json.end(), pattern), std::sregex_iterator()));
}

std::vector<BundleFileEntry> parse_file_entries(const std::string& files_json) {
    std::vector<BundleFileEntry> entries;
    const std::regex object_pattern("\\{[^\\{\\}]*\"sha256\"\\s*:\\s*\"[^\"]*\"[^\\{\\}]*\\}");
    for (auto it = std::sregex_iterator(files_json.begin(), files_json.end(), object_pattern); it != std::sregex_iterator(); ++it) {
        const std::string object = (*it).str();
        entries.push_back(BundleFileEntry{
            .name = extract_string(object, "name"),
            .path = extract_string(object, "path"),
            .offset = static_cast<std::uint64_t>(std::stoull(extract_number(object, "offset"))),
            .length = static_cast<std::uint64_t>(std::stoull(extract_number(object, "length"))),
            .sha256 = extract_string(object, "sha256"),
        });
    }
    return entries;
}

std::vector<TensorIndexEntry> parse_tensor_entries(const std::string& tensors_json) {
    std::vector<TensorIndexEntry> entries;
    const std::regex object_pattern("\\{[^\\{\\}]*\"sha256\"\\s*:\\s*\"[^\"]*\"[^\\{\\}]*\\}");
    for (auto it = std::sregex_iterator(tensors_json.begin(), tensors_json.end(), object_pattern); it != std::sregex_iterator(); ++it) {
        const std::string object = (*it).str();
        entries.push_back(TensorIndexEntry{
            .name = extract_string(object, "name"),
            .dtype = extract_string(object, "dtype"),
            .shape = parse_unsigned_list(extract_array(object, "shape")),
            .offset = static_cast<std::uint64_t>(std::stoull(extract_number(object, "offset"))),
            .length = static_cast<std::uint64_t>(std::stoull(extract_number(object, "length"))),
            .sha256 = extract_string(object, "sha256"),
        });
    }
    return entries;
}

const BundleFileEntry* find_file_entry(const std::vector<BundleFileEntry>& entries, const std::string& name) {
    for (const BundleFileEntry& entry : entries) {
        if (entry.name == name || entry.path == name) {
            return &entry;
        }
    }
    return nullptr;
}

const TensorIndexEntry* find_tensor_entry(const std::vector<TensorIndexEntry>& entries, const std::string& name) {
    for (const TensorIndexEntry& entry : entries) {
        if (entry.name == name) {
            return &entry;
        }
    }
    return nullptr;
}

const TensorIndexEntry& require_tensor(const std::vector<TensorIndexEntry>& tensors, const std::string& name) {
    const TensorIndexEntry* tensor = find_tensor_entry(tensors, name);
    if (tensor == nullptr) {
        throw std::runtime_error("missing tensor: " + name);
    }
    return *tensor;
}

std::pair<std::uint64_t, double> sample_value(
    const std::vector<double>& logits,
    double temperature,
    std::uint64_t top_k,
    std::mt19937_64& rng
) {
    if (temperature <= 0.0 || top_k == 1) {
        return argmax_value(logits);
    }
    if (logits.empty()) {
        throw std::runtime_error("sampling requires non-empty logits");
    }

    std::vector<std::uint64_t> candidates;
    candidates.reserve(logits.size());
    for (std::uint64_t index = 0; index < logits.size(); ++index) {
        candidates.push_back(index);
    }
    std::sort(candidates.begin(), candidates.end(), [&](std::uint64_t left, std::uint64_t right) {
        return logits[static_cast<std::size_t>(left)] > logits[static_cast<std::size_t>(right)];
    });
    if (top_k > 0 && top_k < candidates.size()) {
        candidates.resize(static_cast<std::size_t>(top_k));
    }

    double max_logit = -std::numeric_limits<double>::infinity();
    for (std::uint64_t token : candidates) {
        max_logit = std::max(max_logit, logits[static_cast<std::size_t>(token)]);
    }

    std::vector<double> weights;
    weights.reserve(candidates.size());
    double total = 0.0;
    for (std::uint64_t token : candidates) {
        const double weight = std::exp((logits[static_cast<std::size_t>(token)] - max_logit) / temperature);
        weights.push_back(weight);
        total += weight;
    }
    if (total <= 0.0) {
        return argmax_value(logits);
    }

    std::uniform_real_distribution<double> distribution(0.0, total);
    double draw = distribution(rng);
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        if (draw <= weights[index]) {
            const std::uint64_t token = candidates[index];
            return {token, logits[static_cast<std::size_t>(token)]};
        }
        draw -= weights[index];
    }
    const std::uint64_t token = candidates.back();
    return {token, logits[static_cast<std::size_t>(token)]};
}

float read_float32_at(const std::string& data, std::uint64_t byte_offset) {
    if (byte_offset + sizeof(float) > data.size()) {
        throw std::runtime_error("float32 read points outside tensors.bin");
    }
    float value = 0.0F;
    std::memcpy(&value, data.data() + byte_offset, sizeof(float));
    return value;
}

FloatTensorSummary summarize_float32_tensor(const std::string& data, std::uint64_t offset, std::uint64_t length, std::size_t sample_values) {
    if (offset + length > data.size()) {
        throw std::runtime_error("tensor entry points outside tensors.bin");
    }
    if (length % sizeof(float) != 0) {
        throw std::runtime_error("float32 tensor byte length is not divisible by 4");
    }
    const std::uint64_t values = length / sizeof(float);
    const std::size_t samples = static_cast<std::size_t>(std::min<std::uint64_t>(values, sample_values));
    double sum = 0.0;
    float first = 0.0F;
    for (std::size_t index = 0; index < samples; ++index) {
        float value = 0.0F;
        std::memcpy(&value, data.data() + offset + index * sizeof(float), sizeof(float));
        if (index == 0) {
            first = value;
        }
        sum += value;
    }
    return FloatTensorSummary{.values = values, .sum_first_values = sum, .first_value = first};
}

double dot_float32_rows(
    const std::string& data,
    const TensorIndexEntry& left,
    std::uint64_t left_row,
    const TensorIndexEntry& right,
    std::uint64_t right_row
) {
    if (left.dtype != "float32" || right.dtype != "float32") {
        throw std::runtime_error("embedding probe only supports float32 tensors");
    }
    if (left.shape.size() != 2 || right.shape.size() != 2 || left.shape[1] != right.shape[1]) {
        throw std::runtime_error("embedding probe requires two rank-2 tensors with matching width");
    }
    if (left_row >= left.shape[0] || right_row >= right.shape[0]) {
        throw std::runtime_error("embedding probe row is out of range");
    }
    const std::uint64_t width = left.shape[1];
    double result = 0.0;
    for (std::uint64_t index = 0; index < width; ++index) {
        const std::uint64_t left_offset = left.offset + (left_row * width + index) * sizeof(float);
        const std::uint64_t right_offset = right.offset + (right_row * width + index) * sizeof(float);
        result += static_cast<double>(read_float32_at(data, left_offset)) * static_cast<double>(read_float32_at(data, right_offset));
    }
    return result;
}

float tensor_float32_at_flat(const std::string& data, const TensorIndexEntry& tensor, std::uint64_t flat_index) {
    if (tensor.dtype != "float32") {
        throw std::runtime_error("tensor probe only supports float32 tensors");
    }
    const std::uint64_t byte_offset = tensor.offset + flat_index * sizeof(float);
    if (byte_offset + sizeof(float) > tensor.offset + tensor.length) {
        throw std::runtime_error("tensor flat index is out of range");
    }
    return read_float32_at(data, byte_offset);
}

std::vector<double> softmax_route(const std::string& data, const TensorIndexEntry& coeff, std::uint64_t virtual_layer) {
    if (coeff.shape.size() != 2) {
        throw std::runtime_error("route coefficient tensor must be rank-2");
    }
    const std::uint64_t rank = coeff.shape[1];
    double max_value = -std::numeric_limits<double>::infinity();
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(rank));
    for (std::uint64_t index = 0; index < rank; ++index) {
        const double value = tensor_float32_at_flat(data, coeff, virtual_layer * rank + index);
        values.push_back(value);
        if (value > max_value) {
            max_value = value;
        }
    }
    double total = 0.0;
    for (double& value : values) {
        value = std::exp(value - max_value);
        total += value;
    }
    for (double& value : values) {
        value /= total;
    }
    return values;
}

double composed_linear_value(
    const std::string& data,
    const TensorIndexEntry& basis,
    const TensorIndexEntry& coeff,
    const TensorIndexEntry* low_rank_u,
    const TensorIndexEntry* low_rank_v,
    std::uint64_t virtual_layer,
    std::uint64_t in_index,
    std::uint64_t out_index
) {
    if (basis.shape.size() != 3) {
        throw std::runtime_error("basis tensor must be rank-3");
    }
    const std::uint64_t rank = basis.shape[0];
    const std::uint64_t in_features = basis.shape[1];
    const std::uint64_t out_features = basis.shape[2];
    const std::vector<double> alpha = softmax_route(data, coeff, virtual_layer);
    double value = 0.0;
    for (std::uint64_t basis_index = 0; basis_index < rank; ++basis_index) {
        const std::uint64_t flat_index = basis_index * in_features * out_features + in_index * out_features + out_index;
        value += alpha[basis_index] * tensor_float32_at_flat(data, basis, flat_index);
    }
    if (low_rank_u != nullptr && low_rank_v != nullptr) {
        if (low_rank_u->shape.size() != 3 || low_rank_v->shape.size() != 3 || low_rank_u->shape[2] != low_rank_v->shape[1]) {
            throw std::runtime_error("low-rank tensors must be rank-3 with matching rank dimension");
        }
        const std::uint64_t low_rank = low_rank_u->shape[2];
        for (std::uint64_t rank_index = 0; rank_index < low_rank; ++rank_index) {
            const std::uint64_t u_index = virtual_layer * in_features * low_rank + in_index * low_rank + rank_index;
            const std::uint64_t v_index = virtual_layer * low_rank * out_features + rank_index * out_features + out_index;
            value += static_cast<double>(tensor_float32_at_flat(data, *low_rank_u, u_index)) *
                static_cast<double>(tensor_float32_at_flat(data, *low_rank_v, v_index));
        }
    }
    return value;
}

std::vector<double> read_float32_row(const std::string& data, const TensorIndexEntry& tensor, std::uint64_t row) {
    if (tensor.dtype != "float32" || tensor.shape.size() != 2) {
        throw std::runtime_error("row read requires a rank-2 float32 tensor");
    }
    if (row >= tensor.shape[0]) {
        throw std::runtime_error("row read is out of range");
    }
    const std::uint64_t width = tensor.shape[1];
    std::vector<double> result;
    result.reserve(static_cast<std::size_t>(width));
    for (std::uint64_t index = 0; index < width; ++index) {
        result.push_back(read_float32_at(data, tensor.offset + (row * width + index) * sizeof(float)));
    }
    return result;
}

std::vector<double> read_float32_vector(const std::string& data, const TensorIndexEntry& tensor) {
    if (tensor.dtype != "float32" || tensor.shape.size() != 1) {
        throw std::runtime_error("vector read requires a rank-1 float32 tensor");
    }
    std::vector<double> result;
    result.reserve(static_cast<std::size_t>(tensor.shape[0]));
    for (std::uint64_t index = 0; index < tensor.shape[0]; ++index) {
        result.push_back(read_float32_at(data, tensor.offset + index * sizeof(float)));
    }
    return result;
}

double sum_first(const std::vector<double>& values, std::size_t count) {
    double total = 0.0;
    const std::size_t limit = std::min(values.size(), count);
    for (std::size_t index = 0; index < limit; ++index) {
        total += values[index];
    }
    return total;
}

std::vector<double> rms_norm(const std::vector<double>& input, const std::vector<double>& weight, double eps) {
    if (input.size() != weight.size()) {
        throw std::runtime_error("RMSNorm input and weight sizes differ");
    }
    double mean_square = 0.0;
    for (double value : input) {
        mean_square += value * value;
    }
    mean_square /= static_cast<double>(input.size());
    const double scale = 1.0 / std::sqrt(mean_square + eps);
    std::vector<double> output;
    output.reserve(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        output.push_back(input[index] * scale * weight[index]);
    }
    return output;
}

std::vector<double> composed_linear_matvec(
    const std::string& data,
    const std::vector<double>& input,
    const TensorIndexEntry& basis,
    const TensorIndexEntry& coeff,
    const TensorIndexEntry* low_rank_u,
    const TensorIndexEntry* low_rank_v,
    std::uint64_t virtual_layer
) {
    if (basis.shape.size() != 3 || input.size() != basis.shape[1]) {
        throw std::runtime_error("composed matvec shape mismatch");
    }
    std::vector<double> output;
    output.reserve(static_cast<std::size_t>(basis.shape[2]));
    for (std::uint64_t out_index = 0; out_index < basis.shape[2]; ++out_index) {
        double value = 0.0;
        for (std::uint64_t in_index = 0; in_index < basis.shape[1]; ++in_index) {
            value += input[static_cast<std::size_t>(in_index)] *
                composed_linear_value(data, basis, coeff, low_rank_u, low_rank_v, virtual_layer, in_index, out_index);
        }
        output.push_back(value);
    }
    return output;
}

std::vector<std::vector<double>> composed_linear_rows(
    const std::string& data,
    const std::vector<std::vector<double>>& rows,
    const TensorIndexEntry& basis,
    const TensorIndexEntry& coeff,
    const TensorIndexEntry* low_rank_u,
    const TensorIndexEntry* low_rank_v,
    std::uint64_t virtual_layer
) {
    std::vector<std::vector<double>> output;
    output.reserve(rows.size());
    for (const std::vector<double>& row : rows) {
        output.push_back(composed_linear_matvec(data, row, basis, coeff, low_rank_u, low_rank_v, virtual_layer));
    }
    return output;
}

std::vector<std::vector<double>> rms_norm_rows(
    const std::vector<std::vector<double>>& rows,
    const std::vector<double>& weight,
    double eps
) {
    std::vector<std::vector<double>> output;
    output.reserve(rows.size());
    for (const std::vector<double>& row : rows) {
        output.push_back(rms_norm(row, weight, eps));
    }
    return output;
}

std::vector<double> mean_rows(const std::vector<std::vector<double>>& rows) {
    if (rows.empty()) {
        throw std::runtime_error("mean_rows requires at least one row");
    }
    std::vector<double> mean(rows.front().size(), 0.0);
    for (const std::vector<double>& row : rows) {
        for (std::size_t index = 0; index < row.size(); ++index) {
            mean[index] += row[index];
        }
    }
    for (double& value : mean) {
        value /= static_cast<double>(rows.size());
    }
    return mean;
}

std::vector<double> dense_linear_matvec(const std::string& data, const std::vector<double>& input, const TensorIndexEntry& weight);
double sigmoid(double value);

std::vector<double> gru_cell_update(
    const std::string& data,
    const std::vector<double>& input,
    const std::vector<double>& previous_state,
    const TensorIndexEntry& weight_ih,
    const TensorIndexEntry& weight_hh,
    const TensorIndexEntry& bias_ih,
    const TensorIndexEntry& bias_hh
) {
    const std::uint64_t state_dim = previous_state.size();
    std::vector<double> gi = dense_linear_matvec(data, input, weight_ih);
    std::vector<double> gh = dense_linear_matvec(data, previous_state, weight_hh);
    const std::vector<double> bias_i = read_float32_vector(data, bias_ih);
    const std::vector<double> bias_h = read_float32_vector(data, bias_hh);
    for (std::size_t index = 0; index < gi.size(); ++index) {
        gi[index] += bias_i[index];
        gh[index] += bias_h[index];
    }
    std::vector<double> next_state(state_dim, 0.0);
    for (std::uint64_t index = 0; index < state_dim; ++index) {
        const double reset = sigmoid(gi[index] + gh[index]);
        const double update = sigmoid(gi[state_dim + index] + gh[state_dim + index]);
        const double candidate = std::tanh(gi[2 * state_dim + index] + reset * gh[2 * state_dim + index]);
        next_state[static_cast<std::size_t>(index)] =
            (1.0 - update) * candidate + update * previous_state[static_cast<std::size_t>(index)];
    }
    return next_state;
}

std::vector<std::vector<double>> causal_attention_rows(
    const std::vector<std::vector<double>>& q,
    const std::vector<std::vector<double>>& k,
    const std::vector<std::vector<double>>& v,
    std::uint64_t n_heads
) {
    if (q.empty() || q.size() != k.size() || q.size() != v.size() || q.front().size() % n_heads != 0) {
        throw std::runtime_error("causal attention shape mismatch");
    }
    const std::uint64_t seq = q.size();
    const std::uint64_t dim = q.front().size();
    const std::uint64_t head_dim = dim / n_heads;
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    std::vector<std::vector<double>> output(static_cast<std::size_t>(seq), std::vector<double>(static_cast<std::size_t>(dim), 0.0));
    for (std::uint64_t token = 0; token < seq; ++token) {
        for (std::uint64_t head = 0; head < n_heads; ++head) {
            std::vector<double> scores;
            scores.reserve(static_cast<std::size_t>(token + 1));
            double max_score = -std::numeric_limits<double>::infinity();
            for (std::uint64_t source = 0; source <= token; ++source) {
                double score = 0.0;
                for (std::uint64_t channel = 0; channel < head_dim; ++channel) {
                    const std::uint64_t offset = head * head_dim + channel;
                    score += q[static_cast<std::size_t>(token)][static_cast<std::size_t>(offset)] *
                        k[static_cast<std::size_t>(source)][static_cast<std::size_t>(offset)];
                }
                score *= scale;
                scores.push_back(score);
                if (score > max_score) {
                    max_score = score;
                }
            }
            double denom = 0.0;
            for (double& score : scores) {
                score = std::exp(score - max_score);
                denom += score;
            }
            for (std::uint64_t source = 0; source <= token; ++source) {
                const double weight = scores[static_cast<std::size_t>(source)] / denom;
                for (std::uint64_t channel = 0; channel < head_dim; ++channel) {
                    const std::uint64_t offset = head * head_dim + channel;
                    output[static_cast<std::size_t>(token)][static_cast<std::size_t>(offset)] +=
                        weight * v[static_cast<std::size_t>(source)][static_cast<std::size_t>(offset)];
                }
            }
        }
    }
    return output;
}

std::vector<double> dense_linear_matvec(const std::string& data, const std::vector<double>& input, const TensorIndexEntry& weight) {
    if (weight.dtype != "float32" || weight.shape.size() != 2 || weight.shape[1] != input.size()) {
        throw std::runtime_error("dense linear matvec shape mismatch");
    }
    std::vector<double> output;
    output.reserve(static_cast<std::size_t>(weight.shape[0]));
    for (std::uint64_t out_index = 0; out_index < weight.shape[0]; ++out_index) {
        double value = 0.0;
        for (std::uint64_t in_index = 0; in_index < weight.shape[1]; ++in_index) {
            const std::uint64_t flat = out_index * weight.shape[1] + in_index;
            value += static_cast<double>(tensor_float32_at_flat(data, weight, flat)) * input[static_cast<std::size_t>(in_index)];
        }
        output.push_back(value);
    }
    return output;
}

std::vector<double> embedding_head_logits(const std::string& data, const std::vector<double>& input, const TensorIndexEntry& weight) {
    if (weight.dtype != "float32" || weight.shape.size() != 2 || weight.shape[1] != input.size()) {
        throw std::runtime_error("lm head shape mismatch");
    }
    std::vector<double> logits;
    logits.reserve(static_cast<std::size_t>(weight.shape[0]));
    for (std::uint64_t token = 0; token < weight.shape[0]; ++token) {
        double value = 0.0;
        for (std::uint64_t index = 0; index < weight.shape[1]; ++index) {
            value += input[static_cast<std::size_t>(index)] *
                static_cast<double>(read_float32_at(data, weight.offset + (token * weight.shape[1] + index) * sizeof(float)));
        }
        logits.push_back(value);
    }
    return logits;
}

double sigmoid(double value) {
    return 1.0 / (1.0 + std::exp(-value));
}

double silu(double value) {
    return value * sigmoid(value);
}

std::pair<std::uint64_t, double> argmax_value(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::runtime_error("argmax requires non-empty vector");
    }
    std::uint64_t best_index = 0;
    double best_value = values[0];
    for (std::size_t index = 1; index < values.size(); ++index) {
        if (values[index] > best_value) {
            best_index = static_cast<std::uint64_t>(index);
            best_value = values[index];
        }
    }
    return {best_index, best_value};
}

std::vector<double> native_logits_for(
    const NativeRuntime& runtime,
    const std::vector<std::uint64_t>& input_tokens
) {
    const std::string& tensor_data = runtime.tensor_data;
    const std::vector<TensorIndexEntry>& tensors = runtime.tensors;
    const TensorIndexEntry& token_embedding = require_tensor(tensors, "token_emb.weight");
    const TensorIndexEntry& lm_head = require_tensor(tensors, "lm_head.weight");
    const TensorIndexEntry& q_basis = require_tensor(tensors, "block.q.basis");
    const TensorIndexEntry& q_coeff = require_tensor(tensors, "block.q.coeff");
    const TensorIndexEntry& q_u = require_tensor(tensors, "block.q.u");
    const TensorIndexEntry& q_v = require_tensor(tensors, "block.q.v");
    const TensorIndexEntry& k_basis = require_tensor(tensors, "block.k.basis");
    const TensorIndexEntry& k_coeff = require_tensor(tensors, "block.k.coeff");
    const TensorIndexEntry& k_u = require_tensor(tensors, "block.k.u");
    const TensorIndexEntry& k_v = require_tensor(tensors, "block.k.v");
    const TensorIndexEntry& v_basis = require_tensor(tensors, "block.v.basis");
    const TensorIndexEntry& v_coeff = require_tensor(tensors, "block.v.coeff");
    const TensorIndexEntry& v_u = require_tensor(tensors, "block.v.u");
    const TensorIndexEntry& v_v = require_tensor(tensors, "block.v.v");
    const TensorIndexEntry& o_basis = require_tensor(tensors, "block.o.basis");
    const TensorIndexEntry& o_coeff = require_tensor(tensors, "block.o.coeff");
    const TensorIndexEntry& o_u = require_tensor(tensors, "block.o.u");
    const TensorIndexEntry& o_v = require_tensor(tensors, "block.o.v");
    const TensorIndexEntry& up_basis = require_tensor(tensors, "block.up.basis");
    const TensorIndexEntry& up_coeff = require_tensor(tensors, "block.up.coeff");
    const TensorIndexEntry& up_u = require_tensor(tensors, "block.up.u");
    const TensorIndexEntry& up_v = require_tensor(tensors, "block.up.v");
    const TensorIndexEntry& gate_basis = require_tensor(tensors, "block.gate.basis");
    const TensorIndexEntry& gate_coeff = require_tensor(tensors, "block.gate.coeff");
    const TensorIndexEntry& gate_u = require_tensor(tensors, "block.gate.u");
    const TensorIndexEntry& gate_v = require_tensor(tensors, "block.gate.v");
    const TensorIndexEntry& down_basis = require_tensor(tensors, "block.down.basis");
    const TensorIndexEntry& down_coeff = require_tensor(tensors, "block.down.coeff");
    const TensorIndexEntry& down_u = require_tensor(tensors, "block.down.u");
    const TensorIndexEntry& down_v = require_tensor(tensors, "block.down.v");
    const TensorIndexEntry& state_gate_tensor = require_tensor(tensors, "block.state_gate");
    const TensorIndexEntry& state_weight_ih = require_tensor(tensors, "block.state_cell.weight_ih");
    const TensorIndexEntry& state_weight_hh = require_tensor(tensors, "block.state_cell.weight_hh");
    const TensorIndexEntry& state_bias_ih = require_tensor(tensors, "block.state_cell.bias_ih");
    const TensorIndexEntry& state_bias_hh = require_tensor(tensors, "block.state_cell.bias_hh");
    const TensorIndexEntry& state_proj_weight = require_tensor(tensors, "block.state_proj.weight");
    const TensorIndexEntry& position_embedding = require_tensor(tensors, "pos_emb.weight");
    const TensorIndexEntry& norm1_weight = require_tensor(tensors, "block.norm1.weight");
    const TensorIndexEntry& norm2_weight = require_tensor(tensors, "block.norm2.weight");
    const TensorIndexEntry& final_norm_weight = require_tensor(tensors, "norm.weight");

    const std::uint64_t n_heads =
        static_cast<std::uint64_t>(std::stoull(extract_number(runtime.manifest_json, "n_heads")));
    std::vector<std::vector<double>> rows;
    rows.reserve(input_tokens.size());
    for (std::size_t row = 0; row < input_tokens.size(); ++row) {
        rows.push_back(read_float32_row(tensor_data, token_embedding, input_tokens[row]));
        const std::vector<double> pos =
            read_float32_row(tensor_data, position_embedding, static_cast<std::uint64_t>(row));
        for (std::size_t index = 0; index < rows.back().size(); ++index) {
            rows.back()[index] += pos[index];
        }
    }

    std::vector<double> state(static_cast<std::size_t>(state_weight_hh.shape[1]), 0.0);
    for (std::uint64_t virtual_layer = 0; virtual_layer < q_coeff.shape[0]; ++virtual_layer) {
        const std::vector<std::vector<double>> normalized =
            rms_norm_rows(rows, read_float32_vector(tensor_data, norm1_weight), 1e-6);
        const std::vector<std::vector<double>> q_rows =
            composed_linear_rows(tensor_data, normalized, q_basis, q_coeff, &q_u, &q_v, virtual_layer);
        const std::vector<std::vector<double>> k_rows =
            composed_linear_rows(tensor_data, normalized, k_basis, k_coeff, &k_u, &k_v, virtual_layer);
        const std::vector<std::vector<double>> v_rows =
            composed_linear_rows(tensor_data, normalized, v_basis, v_coeff, &v_u, &v_v, virtual_layer);
        const std::vector<std::vector<double>> attention = causal_attention_rows(q_rows, k_rows, v_rows, n_heads);
        const std::vector<std::vector<double>> o_rows =
            composed_linear_rows(tensor_data, attention, o_basis, o_coeff, &o_u, &o_v, virtual_layer);
        for (std::size_t row = 0; row < rows.size(); ++row) {
            for (std::size_t index = 0; index < rows[row].size(); ++index) {
                rows[row][index] += o_rows[row][index];
            }
        }

        const std::vector<double> pooled = mean_rows(rows);
        state = gru_cell_update(tensor_data, pooled, state, state_weight_ih, state_weight_hh, state_bias_ih, state_bias_hh);
        const double gate = sigmoid(tensor_float32_at_flat(tensor_data, state_gate_tensor, virtual_layer));
        const std::vector<double> state_projection = dense_linear_matvec(tensor_data, state, state_proj_weight);
        for (std::vector<double>& row : rows) {
            for (std::size_t index = 0; index < row.size(); ++index) {
                row[index] += gate * state_projection[index];
            }
        }

        const std::vector<std::vector<double>> norm2 =
            rms_norm_rows(rows, read_float32_vector(tensor_data, norm2_weight), 1e-6);
        const std::vector<std::vector<double>> up_rows =
            composed_linear_rows(tensor_data, norm2, up_basis, up_coeff, &up_u, &up_v, virtual_layer);
        const std::vector<std::vector<double>> gate_rows =
            composed_linear_rows(tensor_data, norm2, gate_basis, gate_coeff, &gate_u, &gate_v, virtual_layer);
        std::vector<std::vector<double>> hidden = up_rows;
        for (std::size_t row = 0; row < hidden.size(); ++row) {
            for (std::size_t index = 0; index < hidden[row].size(); ++index) {
                hidden[row][index] *= silu(gate_rows[row][index]);
            }
        }
        const std::vector<std::vector<double>> down_rows =
            composed_linear_rows(tensor_data, hidden, down_basis, down_coeff, &down_u, &down_v, virtual_layer);
        for (std::size_t row = 0; row < rows.size(); ++row) {
            for (std::size_t index = 0; index < rows[row].size(); ++index) {
                rows[row][index] += down_rows[row][index];
            }
        }
    }

    const std::vector<double> final_norm =
        rms_norm(rows.back(), read_float32_vector(tensor_data, final_norm_weight), 1e-6);
    return embedding_head_logits(tensor_data, final_norm, lm_head);
}

NativeGenerationResult generate_greedy(
    const NativeRuntime& runtime,
    const std::vector<std::uint64_t>& prompt_tokens,
    std::uint64_t max_new_tokens
) {
    return generate_with_options(
        runtime,
        prompt_tokens,
        NativeGenerationOptions{
            .max_new_tokens = max_new_tokens,
            .temperature = 0.0,
            .top_k = 0,
            .seed = 1337,
        }
    );
}

NativeGenerationResult generate_with_options(
    const NativeRuntime& runtime,
    const std::vector<std::uint64_t>& prompt_tokens,
    const NativeGenerationOptions& options
) {
    const TensorIndexEntry& position_embedding = require_tensor(runtime.tensors, "pos_emb.weight");
    if (position_embedding.shape.size() != 2) {
        throw std::runtime_error("position embedding must be rank-2");
    }
    if (prompt_tokens.size() >= position_embedding.shape[0]) {
        throw std::runtime_error("prompt length exceeds native runtime context window");
    }
    const std::uint64_t available_new_tokens =
        position_embedding.shape[0] - static_cast<std::uint64_t>(prompt_tokens.size());
    const std::uint64_t max_new_tokens = std::min(options.max_new_tokens, available_new_tokens);
    std::vector<std::uint64_t> tokens = prompt_tokens;
    std::vector<std::uint64_t> new_tokens;
    std::vector<double> new_token_logits;
    new_tokens.reserve(static_cast<std::size_t>(max_new_tokens));
    new_token_logits.reserve(static_cast<std::size_t>(max_new_tokens));
    std::mt19937_64 rng(options.seed);
    for (std::uint64_t step = 0; step < max_new_tokens; ++step) {
        const std::vector<double> logits = native_logits_for(runtime, tokens);
        const auto [next_token, next_logit] =
            sample_value(logits, options.temperature, options.top_k, rng);
        tokens.push_back(next_token);
        new_tokens.push_back(next_token);
        new_token_logits.push_back(next_logit);
    }
    return NativeGenerationResult{
        .text = decode_ascii_tokens(tokens),
        .tokens = tokens,
        .new_tokens = new_tokens,
        .new_token_logits = new_token_logits,
    };
}

}  // namespace tinycore::native
