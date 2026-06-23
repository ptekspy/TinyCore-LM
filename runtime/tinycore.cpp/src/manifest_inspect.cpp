#include "native_runtime.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace tinycore::native;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: tinycore-inspect-manifest <manifest.json|artifact-dir|model.tcmdl>\n";
        return 2;
    }

    std::string path = argv[1];
    const bool is_bundle = path.ends_with(".tcmdl");
    if (!is_bundle && !path.ends_with(".json")) {
        if (!path.empty() && path.back() != '/') {
            path += "/";
        }
        path += "manifest.json";
    }

    try {
        const std::string json = is_bundle ? read_tcmdl_header(path) : read_file(path);
        const std::string format = extract_string(json, "format");
        if (format != "TCMDL") {
            std::cerr << "format is not TCMDL\n";
            return 1;
        }
        std::cout << "format=" << format << "\n";
        std::cout << "format_version=" << extract_string(json, "format_version") << "\n";
        std::cout << "architecture=" << extract_string(json, "architecture") << "\n";
        std::cout << "model_name=" << extract_string(json, "model_name") << "\n";
        const std::string section_json = extract_array(json, is_bundle ? "tensor_sections" : "sections");
        std::cout << "sections=" << count_sections(section_json) << "\n";
        std::cout << "section_bytes=" << sum_section_lengths(section_json) << "\n";
        if (is_bundle) {
            const std::uint64_t header_payload_length = static_cast<std::uint64_t>(std::stoull(extract_number(json, "payload_length")));
            const std::uint64_t payload_start = 6 + sizeof(std::uint64_t) + json.size();
            const std::uint64_t actual_payload_length = static_cast<std::uint64_t>(std::filesystem::file_size(path)) - payload_start;
            const std::string files_json = extract_array(json, "files");
            const std::vector<BundleFileEntry> files = parse_file_entries(files_json);
            std::cout << "files=" << files.size() << "\n";
            std::cout << "payload_length=" << header_payload_length << "\n";
            std::cout << "payload_actual=" << actual_payload_length << "\n";
            std::cout << "payload_ok=" << (header_payload_length == actual_payload_length ? "true" : "false") << "\n";
            for (std::size_t index = 0; index < files.size(); ++index) {
                const BundleFileEntry& file = files[index];
                const std::string sha_prefix = file.sha256.size() > 12 ? file.sha256.substr(0, 12) : file.sha256;
                std::cout << "file." << index << ".name=" << file.name << "\n";
                std::cout << "file." << index << ".path=" << file.path << "\n";
                std::cout << "file." << index << ".offset=" << file.offset << "\n";
                std::cout << "file." << index << ".length=" << file.length << "\n";
                std::cout << "file." << index << ".sha256_prefix=" << sha_prefix << "\n";
            }
            if (header_payload_length != actual_payload_length) {
                return 1;
            }
            const BundleFileEntry* manifest_file = find_file_entry(files, "manifest");
            if (manifest_file != nullptr) {
                const std::string embedded_manifest =
                    read_binary_slice(path, payload_start + manifest_file->offset, manifest_file->length);
                const std::string embedded_section_json = extract_array(embedded_manifest, "sections");
                const std::size_t embedded_sections = count_sections(embedded_section_json);
                const std::uint64_t embedded_section_bytes = sum_section_lengths(embedded_section_json);
                const std::size_t header_sections = count_sections(section_json);
                const std::uint64_t header_section_bytes = sum_section_lengths(section_json);
                const bool manifest_metadata_ok =
                    embedded_sections == header_sections && embedded_section_bytes == header_section_bytes;
                std::cout << "embedded_manifest_sections=" << embedded_sections << "\n";
                std::cout << "embedded_manifest_section_bytes=" << embedded_section_bytes << "\n";
                std::cout << "embedded_manifest_metadata_ok=" << (manifest_metadata_ok ? "true" : "false") << "\n";
                if (!manifest_metadata_ok) {
                    return 1;
                }
            }
            const BundleFileEntry* tensor_index_file = find_file_entry(files, "tensor_index");
            const BundleFileEntry* tensor_data_file = find_file_entry(files, "tensors");
            if (tensor_index_file != nullptr) {
                const std::string tensor_index =
                    read_binary_slice(path, payload_start + tensor_index_file->offset, tensor_index_file->length);
                const std::uint64_t tensor_index_count = static_cast<std::uint64_t>(std::stoull(extract_number(tensor_index, "num_tensors")));
                const std::uint64_t tensor_index_bytes = static_cast<std::uint64_t>(std::stoull(extract_number(tensor_index, "total_bytes")));
                const std::uint64_t header_section_count = static_cast<std::uint64_t>(count_sections(section_json));
                const std::uint64_t header_section_bytes = sum_section_lengths(section_json);
                const bool tensor_index_metadata_ok =
                    tensor_index_count == header_section_count && tensor_index_bytes == header_section_bytes;
                std::cout << "tensor_index_tensors=" << tensor_index_count << "\n";
                std::cout << "tensor_index_bytes=" << tensor_index_bytes << "\n";
                if (tensor_data_file != nullptr) {
                    std::cout << "tensor_data_bytes=" << tensor_data_file->length << "\n";
                }
                std::cout << "tensor_index_metadata_ok=" << (tensor_index_metadata_ok ? "true" : "false") << "\n";
                if (!tensor_index_metadata_ok || (tensor_data_file != nullptr && tensor_data_file->length != tensor_index_bytes)) {
                    return 1;
                }
                if (tensor_data_file != nullptr) {
                    const std::string tensor_data =
                        read_binary_slice(path, payload_start + tensor_data_file->offset, tensor_data_file->length);
                    const std::vector<TensorIndexEntry> tensors = parse_tensor_entries(extract_array(tensor_index, "tensors"));
                    if (!tensors.empty()) {
                        const TensorIndexEntry& first_tensor = tensors.front();
                        const TensorIndexEntry& last_tensor = tensors.back();
                        if (first_tensor.dtype == "float32") {
                            const FloatTensorSummary summary =
                                summarize_float32_tensor(tensor_data, first_tensor.offset, first_tensor.length, 8);
                            std::cout << "tensor_probe.0.name=" << first_tensor.name << "\n";
                            std::cout << "tensor_probe.0.values=" << summary.values << "\n";
                            std::cout << "tensor_probe.0.first_value=" << summary.first_value << "\n";
                            std::cout << "tensor_probe.0.sum_first_8=" << summary.sum_first_values << "\n";
                        }
                        if (last_tensor.dtype == "float32") {
                            const FloatTensorSummary summary =
                                summarize_float32_tensor(tensor_data, last_tensor.offset, last_tensor.length, 8);
                            std::cout << "tensor_probe.1.name=" << last_tensor.name << "\n";
                            std::cout << "tensor_probe.1.values=" << summary.values << "\n";
                            std::cout << "tensor_probe.1.first_value=" << summary.first_value << "\n";
                            std::cout << "tensor_probe.1.sum_first_8=" << summary.sum_first_values << "\n";
                        }
                        const TensorIndexEntry* token_embedding = find_tensor_entry(tensors, "token_emb.weight");
                        const TensorIndexEntry* lm_head = find_tensor_entry(tensors, "lm_head.weight");
                        if (token_embedding != nullptr && lm_head != nullptr) {
                            if (token_embedding->shape.size() != 2 || lm_head->shape.size() != 2) {
                                throw std::runtime_error("embedding probe tensors must be rank-2");
                            }
                            const std::uint64_t token_id = 84;
                            double best_logit = -std::numeric_limits<double>::infinity();
                            std::uint64_t best_token = 0;
                            double self_logit = 0.0;
                            for (std::uint64_t candidate = 0; candidate < lm_head->shape[0]; ++candidate) {
                                const double logit = dot_float32_rows(tensor_data, *token_embedding, token_id, *lm_head, candidate);
                                if (candidate == token_id) {
                                    self_logit = logit;
                                }
                                if (logit > best_logit) {
                                    best_logit = logit;
                                    best_token = candidate;
                                }
                            }
                            std::cout << "embedding_probe.token_id=" << token_id << "\n";
                            std::cout << "embedding_probe.vocab=" << lm_head->shape[0] << "\n";
                            std::cout << "embedding_probe.d_model=" << lm_head->shape[1] << "\n";
                            std::cout << "embedding_probe.top_token=" << best_token << "\n";
                            std::cout << "embedding_probe.top_logit=" << best_logit << "\n";
                            std::cout << "embedding_probe.self_logit=" << self_logit << "\n";
                        }
                        const TensorIndexEntry* q_basis = find_tensor_entry(tensors, "block.q.basis");
                        const TensorIndexEntry* q_coeff = find_tensor_entry(tensors, "block.q.coeff");
                        const TensorIndexEntry* q_u = find_tensor_entry(tensors, "block.q.u");
                        const TensorIndexEntry* q_v = find_tensor_entry(tensors, "block.q.v");
                        const TensorIndexEntry* k_basis = find_tensor_entry(tensors, "block.k.basis");
                        const TensorIndexEntry* k_coeff = find_tensor_entry(tensors, "block.k.coeff");
                        const TensorIndexEntry* k_u = find_tensor_entry(tensors, "block.k.u");
                        const TensorIndexEntry* k_v = find_tensor_entry(tensors, "block.k.v");
                        const TensorIndexEntry* v_basis = find_tensor_entry(tensors, "block.v.basis");
                        const TensorIndexEntry* v_coeff = find_tensor_entry(tensors, "block.v.coeff");
                        const TensorIndexEntry* v_u = find_tensor_entry(tensors, "block.v.u");
                        const TensorIndexEntry* v_v = find_tensor_entry(tensors, "block.v.v");
                        const TensorIndexEntry* o_basis = find_tensor_entry(tensors, "block.o.basis");
                        const TensorIndexEntry* o_coeff = find_tensor_entry(tensors, "block.o.coeff");
                        const TensorIndexEntry* o_u = find_tensor_entry(tensors, "block.o.u");
                        const TensorIndexEntry* o_v = find_tensor_entry(tensors, "block.o.v");
                        const TensorIndexEntry* up_basis = find_tensor_entry(tensors, "block.up.basis");
                        const TensorIndexEntry* up_coeff = find_tensor_entry(tensors, "block.up.coeff");
                        const TensorIndexEntry* up_u = find_tensor_entry(tensors, "block.up.u");
                        const TensorIndexEntry* up_v = find_tensor_entry(tensors, "block.up.v");
                        const TensorIndexEntry* gate_basis = find_tensor_entry(tensors, "block.gate.basis");
                        const TensorIndexEntry* gate_coeff = find_tensor_entry(tensors, "block.gate.coeff");
                        const TensorIndexEntry* gate_u = find_tensor_entry(tensors, "block.gate.u");
                        const TensorIndexEntry* gate_v = find_tensor_entry(tensors, "block.gate.v");
                        const TensorIndexEntry* down_basis = find_tensor_entry(tensors, "block.down.basis");
                        const TensorIndexEntry* down_coeff = find_tensor_entry(tensors, "block.down.coeff");
                        const TensorIndexEntry* down_u = find_tensor_entry(tensors, "block.down.u");
                        const TensorIndexEntry* down_v = find_tensor_entry(tensors, "block.down.v");
                        const TensorIndexEntry* state_gate_tensor = find_tensor_entry(tensors, "block.state_gate");
                        const TensorIndexEntry* state_weight_ih = find_tensor_entry(tensors, "block.state_cell.weight_ih");
                        const TensorIndexEntry* state_weight_hh = find_tensor_entry(tensors, "block.state_cell.weight_hh");
                        const TensorIndexEntry* state_bias_ih = find_tensor_entry(tensors, "block.state_cell.bias_ih");
                        const TensorIndexEntry* state_bias_hh = find_tensor_entry(tensors, "block.state_cell.bias_hh");
                        const TensorIndexEntry* state_proj_weight = find_tensor_entry(tensors, "block.state_proj.weight");
                        const TensorIndexEntry* position_embedding = find_tensor_entry(tensors, "pos_emb.weight");
                        const TensorIndexEntry* norm1_weight = find_tensor_entry(tensors, "block.norm1.weight");
                        const TensorIndexEntry* norm2_weight = find_tensor_entry(tensors, "block.norm2.weight");
                        const TensorIndexEntry* final_norm_weight = find_tensor_entry(tensors, "norm.weight");
                        if (q_basis != nullptr && q_coeff != nullptr) {
                            if (q_basis->shape.size() != 3) {
                                throw std::runtime_error("composition probe basis tensor must be rank-3");
                            }
                            const std::uint64_t virtual_layer = 0;
                            const std::uint64_t sample_values = 8;
                            const std::vector<double> alpha = softmax_route(tensor_data, *q_coeff, virtual_layer);
                            double sum_first_values = 0.0;
                            double first_value = 0.0;
                            for (std::uint64_t flat_index = 0; flat_index < sample_values; ++flat_index) {
                                const std::uint64_t in_index = flat_index / q_basis->shape[2];
                                const std::uint64_t out_index = flat_index % q_basis->shape[2];
                                const double value = composed_linear_value(
                                    tensor_data, *q_basis, *q_coeff, q_u, q_v, virtual_layer, in_index, out_index);
                                if (flat_index == 0) {
                                    first_value = value;
                                }
                                sum_first_values += value;
                            }
                            std::cout << "composition_probe.name=block.q" << "\n";
                            std::cout << "composition_probe.virtual_layer=" << virtual_layer << "\n";
                            std::cout << "composition_probe.route0=" << alpha[0] << "\n";
                            if (alpha.size() > 1) {
                                std::cout << "composition_probe.route1=" << alpha[1] << "\n";
                            }
                            std::cout << "composition_probe.first_value=" << first_value << "\n";
                            std::cout << "composition_probe.sum_first_8=" << sum_first_values << "\n";
                        }
                        if (
                            token_embedding != nullptr && position_embedding != nullptr && norm1_weight != nullptr &&
                            q_basis != nullptr && q_coeff != nullptr
                        ) {
                            const std::uint64_t token_id = 84;
                            const std::uint64_t position = 0;
                            const std::uint64_t virtual_layer = 0;
                            std::vector<double> x = read_float32_row(tensor_data, *token_embedding, token_id);
                            const std::vector<double> pos = read_float32_row(tensor_data, *position_embedding, position);
                            for (std::size_t index = 0; index < x.size(); ++index) {
                                x[index] += pos[index];
                            }
                            const std::vector<double> norm_weight = read_float32_vector(tensor_data, *norm1_weight);
                            const std::vector<double> normalized = rms_norm(x, norm_weight, 1e-6);
                            const std::vector<double> q_output =
                                composed_linear_matvec(tensor_data, normalized, *q_basis, *q_coeff, q_u, q_v, virtual_layer);
                            const auto [q_top_index, q_top_value] = argmax_value(q_output);
                            double mean_square = 0.0;
                            for (double value : x) {
                                mean_square += value * value;
                            }
                            const double rms = std::sqrt(mean_square / static_cast<double>(x.size()) + 1e-6);
                            std::cout << "block_probe.token_id=" << token_id << "\n";
                            std::cout << "block_probe.position=" << position << "\n";
                            std::cout << "block_probe.virtual_layer=" << virtual_layer << "\n";
                            std::cout << "block_probe.input_first_value=" << x.front() << "\n";
                            std::cout << "block_probe.input_sum_first_8=" << sum_first(x, 8) << "\n";
                            std::cout << "block_probe.norm_rms=" << rms << "\n";
                            std::cout << "block_probe.norm_first_value=" << normalized.front() << "\n";
                            std::cout << "block_probe.norm_sum_first_8=" << sum_first(normalized, 8) << "\n";
                            std::cout << "block_probe.q_first_value=" << q_output.front() << "\n";
                            std::cout << "block_probe.q_sum_first_8=" << sum_first(q_output, 8) << "\n";
                            std::cout << "block_probe.q_top_index=" << q_top_index << "\n";
                            std::cout << "block_probe.q_top_value=" << q_top_value << "\n";
                            if (
                                k_basis != nullptr && k_coeff != nullptr &&
                                v_basis != nullptr && v_coeff != nullptr &&
                                o_basis != nullptr && o_coeff != nullptr
                            ) {
                                const std::vector<double> k_output =
                                    composed_linear_matvec(tensor_data, normalized, *k_basis, *k_coeff, k_u, k_v, virtual_layer);
                                const std::vector<double> v_output =
                                    composed_linear_matvec(tensor_data, normalized, *v_basis, *v_coeff, v_u, v_v, virtual_layer);
                                const std::vector<double> o_output =
                                    composed_linear_matvec(tensor_data, v_output, *o_basis, *o_coeff, o_u, o_v, virtual_layer);
                                std::vector<double> x_attention = x;
                                for (std::size_t index = 0; index < x_attention.size(); ++index) {
                                    x_attention[index] += o_output[index];
                                }
                                const auto [v_top_index, v_top_value] = argmax_value(v_output);
                                const auto [o_top_index, o_top_value] = argmax_value(o_output);
                                const auto [x_attention_top_index, x_attention_top_value] = argmax_value(x_attention);
                                std::cout << "attention_probe.k_first_value=" << k_output.front() << "\n";
                                std::cout << "attention_probe.k_sum_first_8=" << sum_first(k_output, 8) << "\n";
                                std::cout << "attention_probe.v_first_value=" << v_output.front() << "\n";
                                std::cout << "attention_probe.v_sum_first_8=" << sum_first(v_output, 8) << "\n";
                                std::cout << "attention_probe.v_top_index=" << v_top_index << "\n";
                                std::cout << "attention_probe.v_top_value=" << v_top_value << "\n";
                                std::cout << "attention_probe.o_first_value=" << o_output.front() << "\n";
                                std::cout << "attention_probe.o_sum_first_8=" << sum_first(o_output, 8) << "\n";
                                std::cout << "attention_probe.o_top_index=" << o_top_index << "\n";
                                std::cout << "attention_probe.o_top_value=" << o_top_value << "\n";
                                std::cout << "attention_probe.residual_first_value=" << x_attention.front() << "\n";
                                std::cout << "attention_probe.residual_sum_first_8=" << sum_first(x_attention, 8) << "\n";
                                std::cout << "attention_probe.residual_top_index=" << x_attention_top_index << "\n";
                                std::cout << "attention_probe.residual_top_value=" << x_attention_top_value << "\n";
                                if (
                                    state_gate_tensor != nullptr && state_weight_ih != nullptr && state_weight_hh != nullptr &&
                                    state_bias_ih != nullptr && state_bias_hh != nullptr && state_proj_weight != nullptr &&
                                    norm2_weight != nullptr && up_basis != nullptr && up_coeff != nullptr &&
                                    gate_basis != nullptr && gate_coeff != nullptr && down_basis != nullptr && down_coeff != nullptr
                                ) {
                                    const std::uint64_t state_dim = state_weight_hh->shape[1];
                                    std::vector<double> gi = dense_linear_matvec(tensor_data, x_attention, *state_weight_ih);
                                    std::vector<double> gh(state_weight_hh->shape[0], 0.0);
                                    const std::vector<double> bias_ih = read_float32_vector(tensor_data, *state_bias_ih);
                                    const std::vector<double> bias_hh = read_float32_vector(tensor_data, *state_bias_hh);
                                    for (std::size_t index = 0; index < gi.size(); ++index) {
                                        gi[index] += bias_ih[index];
                                        gh[index] += bias_hh[index];
                                    }
                                    std::vector<double> state(state_dim, 0.0);
                                    for (std::uint64_t index = 0; index < state_dim; ++index) {
                                        const double reset = sigmoid(gi[index] + gh[index]);
                                        const double update = sigmoid(gi[state_dim + index] + gh[state_dim + index]);
                                        const double candidate =
                                            std::tanh(gi[2 * state_dim + index] + reset * gh[2 * state_dim + index]);
                                        state[static_cast<std::size_t>(index)] = (1.0 - update) * candidate;
                                    }
                                    const double recurrent_gate =
                                        sigmoid(tensor_float32_at_flat(tensor_data, *state_gate_tensor, virtual_layer));
                                    const std::vector<double> state_projection =
                                        dense_linear_matvec(tensor_data, state, *state_proj_weight);
                                    std::vector<double> x_state = x_attention;
                                    for (std::size_t index = 0; index < x_state.size(); ++index) {
                                        x_state[index] += recurrent_gate * state_projection[index];
                                    }
                                    const std::vector<double> norm2 = rms_norm(x_state, read_float32_vector(tensor_data, *norm2_weight), 1e-6);
                                    const std::vector<double> gate_output =
                                        composed_linear_matvec(tensor_data, norm2, *gate_basis, *gate_coeff, gate_u, gate_v, virtual_layer);
                                    const std::vector<double> up_output =
                                        composed_linear_matvec(tensor_data, norm2, *up_basis, *up_coeff, up_u, up_v, virtual_layer);
                                    std::vector<double> mlp;
                                    mlp.reserve(gate_output.size());
                                    for (std::size_t index = 0; index < gate_output.size(); ++index) {
                                        mlp.push_back(silu(gate_output[index]) * up_output[index]);
                                    }
                                    const std::vector<double> down_output =
                                        composed_linear_matvec(tensor_data, mlp, *down_basis, *down_coeff, down_u, down_v, virtual_layer);
                                    std::vector<double> block_output = x_state;
                                    for (std::size_t index = 0; index < block_output.size(); ++index) {
                                        block_output[index] += down_output[index];
                                    }
                                    const auto [state_top_index, state_top_value] = argmax_value(state);
                                    const auto [down_top_index, down_top_value] = argmax_value(down_output);
                                    const auto [block_top_index, block_top_value] = argmax_value(block_output);
                                    std::cout << "recurrent_probe.gate=" << recurrent_gate << "\n";
                                    std::cout << "recurrent_probe.state_first_value=" << state.front() << "\n";
                                    std::cout << "recurrent_probe.state_sum_first_8=" << sum_first(state, 8) << "\n";
                                    std::cout << "recurrent_probe.state_top_index=" << state_top_index << "\n";
                                    std::cout << "recurrent_probe.state_top_value=" << state_top_value << "\n";
                                    std::cout << "mlp_probe.norm2_first_value=" << norm2.front() << "\n";
                                    std::cout << "mlp_probe.norm2_sum_first_8=" << sum_first(norm2, 8) << "\n";
                                    std::cout << "mlp_probe.gate_sum_first_8=" << sum_first(gate_output, 8) << "\n";
                                    std::cout << "mlp_probe.up_sum_first_8=" << sum_first(up_output, 8) << "\n";
                                    std::cout << "mlp_probe.hidden_sum_first_8=" << sum_first(mlp, 8) << "\n";
                                    std::cout << "mlp_probe.down_first_value=" << down_output.front() << "\n";
                                    std::cout << "mlp_probe.down_sum_first_8=" << sum_first(down_output, 8) << "\n";
                                    std::cout << "mlp_probe.down_top_index=" << down_top_index << "\n";
                                    std::cout << "mlp_probe.down_top_value=" << down_top_value << "\n";
                                    std::cout << "block_output_probe.first_value=" << block_output.front() << "\n";
                                    std::cout << "block_output_probe.sum_first_8=" << sum_first(block_output, 8) << "\n";
                                    std::cout << "block_output_probe.top_index=" << block_top_index << "\n";
                                    std::cout << "block_output_probe.top_value=" << block_top_value << "\n";
                                    if (final_norm_weight != nullptr && lm_head != nullptr) {
                                        const std::vector<double> final_norm =
                                            rms_norm(block_output, read_float32_vector(tensor_data, *final_norm_weight), 1e-6);
                                        const std::vector<double> logits = embedding_head_logits(tensor_data, final_norm, *lm_head);
                                        const auto [logit_top_index, logit_top_value] = argmax_value(logits);
                                        std::cout << "one_layer_logits_probe.final_norm_first_value=" << final_norm.front() << "\n";
                                        std::cout << "one_layer_logits_probe.final_norm_sum_first_8=" << sum_first(final_norm, 8) << "\n";
                                        std::cout << "one_layer_logits_probe.logit0=" << logits.front() << "\n";
                                        std::cout << "one_layer_logits_probe.logits_sum_first_8=" << sum_first(logits, 8) << "\n";
                                        std::cout << "one_layer_logits_probe.top_token=" << logit_top_index << "\n";
                                        std::cout << "one_layer_logits_probe.top_logit=" << logit_top_value << "\n";
                                        if (logits.size() > 84) {
                                            std::cout << "one_layer_logits_probe.token84_logit=" << logits[84] << "\n";
                                        }
                                        std::vector<double> all_x = read_float32_row(tensor_data, *token_embedding, token_id);
                                        const std::vector<double> all_pos = read_float32_row(tensor_data, *position_embedding, position);
                                        for (std::size_t index = 0; index < all_x.size(); ++index) {
                                            all_x[index] += all_pos[index];
                                        }
                                        std::vector<double> all_state(state_dim, 0.0);
                                        const std::uint64_t virtual_layers = q_coeff->shape[0];
                                        for (std::uint64_t layer = 0; layer < virtual_layers; ++layer) {
                                            const std::vector<double> all_norm1 =
                                                rms_norm(all_x, read_float32_vector(tensor_data, *norm1_weight), 1e-6);
                                            const std::vector<double> all_v =
                                                composed_linear_matvec(tensor_data, all_norm1, *v_basis, *v_coeff, v_u, v_v, layer);
                                            const std::vector<double> all_o =
                                                composed_linear_matvec(tensor_data, all_v, *o_basis, *o_coeff, o_u, o_v, layer);
                                            for (std::size_t index = 0; index < all_x.size(); ++index) {
                                                all_x[index] += all_o[index];
                                            }

                                            std::vector<double> all_gi = dense_linear_matvec(tensor_data, all_x, *state_weight_ih);
                                            std::vector<double> all_gh = dense_linear_matvec(tensor_data, all_state, *state_weight_hh);
                                            for (std::size_t index = 0; index < all_gi.size(); ++index) {
                                                all_gi[index] += bias_ih[index];
                                                all_gh[index] += bias_hh[index];
                                            }
                                            std::vector<double> next_state(state_dim, 0.0);
                                            for (std::uint64_t index = 0; index < state_dim; ++index) {
                                                const double reset = sigmoid(all_gi[index] + all_gh[index]);
                                                const double update = sigmoid(all_gi[state_dim + index] + all_gh[state_dim + index]);
                                                const double candidate =
                                                    std::tanh(all_gi[2 * state_dim + index] + reset * all_gh[2 * state_dim + index]);
                                                next_state[static_cast<std::size_t>(index)] =
                                                    (1.0 - update) * candidate + update * all_state[static_cast<std::size_t>(index)];
                                            }
                                            all_state = next_state;
                                            const double all_recurrent_gate =
                                                sigmoid(tensor_float32_at_flat(tensor_data, *state_gate_tensor, layer));
                                            const std::vector<double> all_state_projection =
                                                dense_linear_matvec(tensor_data, all_state, *state_proj_weight);
                                            for (std::size_t index = 0; index < all_x.size(); ++index) {
                                                all_x[index] += all_recurrent_gate * all_state_projection[index];
                                            }
                                            const std::vector<double> all_norm2 =
                                                rms_norm(all_x, read_float32_vector(tensor_data, *norm2_weight), 1e-6);
                                            const std::vector<double> all_gate =
                                                composed_linear_matvec(tensor_data, all_norm2, *gate_basis, *gate_coeff, gate_u, gate_v, layer);
                                            const std::vector<double> all_up =
                                                composed_linear_matvec(tensor_data, all_norm2, *up_basis, *up_coeff, up_u, up_v, layer);
                                            std::vector<double> all_mlp;
                                            all_mlp.reserve(all_gate.size());
                                            for (std::size_t index = 0; index < all_gate.size(); ++index) {
                                                all_mlp.push_back(silu(all_gate[index]) * all_up[index]);
                                            }
                                            const std::vector<double> all_down =
                                                composed_linear_matvec(tensor_data, all_mlp, *down_basis, *down_coeff, down_u, down_v, layer);
                                            for (std::size_t index = 0; index < all_x.size(); ++index) {
                                                all_x[index] += all_down[index];
                                            }
                                            const auto [all_x_top_index, all_x_top_value] = argmax_value(all_x);
                                            std::cout << "all_layers_probe.layer." << layer << ".first_value=" << all_x.front() << "\n";
                                            std::cout << "all_layers_probe.layer." << layer << ".sum_first_8=" << sum_first(all_x, 8) << "\n";
                                            std::cout << "all_layers_probe.layer." << layer << ".top_index=" << all_x_top_index << "\n";
                                            std::cout << "all_layers_probe.layer." << layer << ".top_value=" << all_x_top_value << "\n";
                                            std::cout << "all_layers_probe.layer." << layer << ".state_first_value=" << all_state.front() << "\n";
                                            std::cout << "all_layers_probe.layer." << layer << ".state_sum_first_8=" << sum_first(all_state, 8) << "\n";
                                        }
                                        const std::vector<double> all_final_norm =
                                            rms_norm(all_x, read_float32_vector(tensor_data, *final_norm_weight), 1e-6);
                                        const std::vector<double> all_logits =
                                            embedding_head_logits(tensor_data, all_final_norm, *lm_head);
                                        const auto [all_logit_top_index, all_logit_top_value] = argmax_value(all_logits);
                                        std::cout << "all_layers_logits_probe.final_norm_first_value=" << all_final_norm.front() << "\n";
                                        std::cout << "all_layers_logits_probe.final_norm_sum_first_8=" << sum_first(all_final_norm, 8) << "\n";
                                        std::cout << "all_layers_logits_probe.logit0=" << all_logits.front() << "\n";
                                        std::cout << "all_layers_logits_probe.logits_sum_first_8=" << sum_first(all_logits, 8) << "\n";
                                        std::cout << "all_layers_logits_probe.top_token=" << all_logit_top_index << "\n";
                                        std::cout << "all_layers_logits_probe.top_logit=" << all_logit_top_value << "\n";
                                        if (all_logits.size() > 84) {
                                            std::cout << "all_layers_logits_probe.token84_logit=" << all_logits[84] << "\n";
                                        }
                                        const std::uint64_t n_heads = static_cast<std::uint64_t>(std::stoull(extract_number(json, "n_heads")));
                                        std::vector<std::vector<double>> seq_x;
                                        seq_x.push_back(read_float32_row(tensor_data, *token_embedding, 84));
                                        seq_x.push_back(read_float32_row(tensor_data, *token_embedding, 105));
                                        const std::vector<double> seq_pos0 = read_float32_row(tensor_data, *position_embedding, 0);
                                        const std::vector<double> seq_pos1 = read_float32_row(tensor_data, *position_embedding, 1);
                                        for (std::size_t index = 0; index < seq_x[0].size(); ++index) {
                                            seq_x[0][index] += seq_pos0[index];
                                            seq_x[1][index] += seq_pos1[index];
                                        }
                                        std::vector<double> seq_state(state_dim, 0.0);
                                        for (std::uint64_t layer = 0; layer < virtual_layers; ++layer) {
                                            const std::vector<std::vector<double>> seq_norm1 =
                                                rms_norm_rows(seq_x, read_float32_vector(tensor_data, *norm1_weight), 1e-6);
                                            const std::vector<std::vector<double>> seq_q =
                                                composed_linear_rows(tensor_data, seq_norm1, *q_basis, *q_coeff, q_u, q_v, layer);
                                            const std::vector<std::vector<double>> seq_k =
                                                composed_linear_rows(tensor_data, seq_norm1, *k_basis, *k_coeff, k_u, k_v, layer);
                                            const std::vector<std::vector<double>> seq_v =
                                                composed_linear_rows(tensor_data, seq_norm1, *v_basis, *v_coeff, v_u, v_v, layer);
                                            const std::vector<std::vector<double>> seq_attn =
                                                causal_attention_rows(seq_q, seq_k, seq_v, n_heads);
                                            const std::vector<std::vector<double>> seq_o =
                                                composed_linear_rows(tensor_data, seq_attn, *o_basis, *o_coeff, o_u, o_v, layer);
                                            for (std::size_t row = 0; row < seq_x.size(); ++row) {
                                                for (std::size_t index = 0; index < seq_x[row].size(); ++index) {
                                                    seq_x[row][index] += seq_o[row][index];
                                                }
                                            }
                                            seq_state = gru_cell_update(
                                                tensor_data,
                                                mean_rows(seq_x),
                                                seq_state,
                                                *state_weight_ih,
                                                *state_weight_hh,
                                                *state_bias_ih,
                                                *state_bias_hh
                                            );
                                            const double seq_gate =
                                                sigmoid(tensor_float32_at_flat(tensor_data, *state_gate_tensor, layer));
                                            const std::vector<double> seq_state_projection =
                                                dense_linear_matvec(tensor_data, seq_state, *state_proj_weight);
                                            for (std::vector<double>& row : seq_x) {
                                                for (std::size_t index = 0; index < row.size(); ++index) {
                                                    row[index] += seq_gate * seq_state_projection[index];
                                                }
                                            }
                                            const std::vector<std::vector<double>> seq_norm2 =
                                                rms_norm_rows(seq_x, read_float32_vector(tensor_data, *norm2_weight), 1e-6);
                                            const std::vector<std::vector<double>> seq_gate_rows =
                                                composed_linear_rows(tensor_data, seq_norm2, *gate_basis, *gate_coeff, gate_u, gate_v, layer);
                                            const std::vector<std::vector<double>> seq_up_rows =
                                                composed_linear_rows(tensor_data, seq_norm2, *up_basis, *up_coeff, up_u, up_v, layer);
                                            std::vector<std::vector<double>> seq_mlp = seq_gate_rows;
                                            for (std::size_t row = 0; row < seq_mlp.size(); ++row) {
                                                for (std::size_t index = 0; index < seq_mlp[row].size(); ++index) {
                                                    seq_mlp[row][index] = silu(seq_gate_rows[row][index]) * seq_up_rows[row][index];
                                                }
                                            }
                                            const std::vector<std::vector<double>> seq_down =
                                                composed_linear_rows(tensor_data, seq_mlp, *down_basis, *down_coeff, down_u, down_v, layer);
                                            for (std::size_t row = 0; row < seq_x.size(); ++row) {
                                                for (std::size_t index = 0; index < seq_x[row].size(); ++index) {
                                                    seq_x[row][index] += seq_down[row][index];
                                                }
                                            }
                                            const auto [seq_top_index, seq_top_value] = argmax_value(seq_x.back());
                                            std::cout << "sequence_probe.layer." << layer << ".last_first_value=" << seq_x.back().front() << "\n";
                                            std::cout << "sequence_probe.layer." << layer << ".last_sum_first_8=" << sum_first(seq_x.back(), 8) << "\n";
                                            std::cout << "sequence_probe.layer." << layer << ".last_top_index=" << seq_top_index << "\n";
                                            std::cout << "sequence_probe.layer." << layer << ".last_top_value=" << seq_top_value << "\n";
                                            std::cout << "sequence_probe.layer." << layer << ".state_sum_first_8=" << sum_first(seq_state, 8) << "\n";
                                        }
                                        const std::vector<double> seq_final_norm =
                                            rms_norm(seq_x.back(), read_float32_vector(tensor_data, *final_norm_weight), 1e-6);
                                        const std::vector<double> seq_logits =
                                            embedding_head_logits(tensor_data, seq_final_norm, *lm_head);
                                        const auto [seq_logit_top_index, seq_logit_top_value] = argmax_value(seq_logits);
                                        std::cout << "sequence_logits_probe.final_norm_first_value=" << seq_final_norm.front() << "\n";
                                        std::cout << "sequence_logits_probe.final_norm_sum_first_8=" << sum_first(seq_final_norm, 8) << "\n";
                                        std::cout << "sequence_logits_probe.logit0=" << seq_logits.front() << "\n";
                                        std::cout << "sequence_logits_probe.logits_sum_first_8=" << sum_first(seq_logits, 8) << "\n";
                                        std::cout << "sequence_logits_probe.top_token=" << seq_logit_top_index << "\n";
                                        std::cout << "sequence_logits_probe.top_logit=" << seq_logit_top_value << "\n";
                                        if (seq_logits.size() > 105) {
                                            std::cout << "sequence_logits_probe.token105_logit=" << seq_logits[105] << "\n";
                                        }
                                        const std::vector<std::uint64_t> prompt_tokens = {84, 105, 110, 121, 67, 111, 114, 101};
                                        const NativeRuntime prompt_runtime{
                                            .manifest_json = json,
                                            .tensor_data = tensor_data,
                                            .tensors = tensors,
                                        };
                                        const std::vector<double> prompt_logits = native_logits_for(prompt_runtime, prompt_tokens);
                                        const auto [prompt_top_token, prompt_top_logit] = argmax_value(prompt_logits);
                                        std::cout << "prompt_probe.text=TinyCore" << "\n";
                                        std::cout << "prompt_probe.length=" << prompt_tokens.size() << "\n";
                                        std::cout << "prompt_probe.last_token=" << prompt_tokens.back() << "\n";
                                        std::cout << "prompt_probe.logit0=" << prompt_logits.front() << "\n";
                                        std::cout << "prompt_probe.logits_sum_first_8=" << sum_first(prompt_logits, 8) << "\n";
                                        std::cout << "prompt_probe.top_token=" << prompt_top_token << "\n";
                                        std::cout << "prompt_probe.top_logit=" << prompt_top_logit << "\n";
                                        if (prompt_logits.size() > 101) {
                                            std::cout << "prompt_probe.token101_logit=" << prompt_logits[101] << "\n";
                                        }
                                        if (prompt_logits.size() > 32) {
                                            std::cout << "prompt_probe.token32_logit=" << prompt_logits[32] << "\n";
                                        }

                                        const NativeGenerationResult generated = generate_greedy(prompt_runtime, prompt_tokens, 8);
                                        for (std::size_t step = 0; step < generated.new_tokens.size(); ++step) {
                                            std::cout << "native_generate_probe.step." << step << ".token=" << generated.new_tokens[step] << "\n";
                                            std::cout << "native_generate_probe.step." << step << ".logit=" << generated.new_token_logits[step] << "\n";
                                        }
                                        std::cout << "native_generate_probe.prompt=TinyCore" << "\n";
                                        std::cout << "native_generate_probe.new_tokens=";
                                        for (std::size_t index = 0; index < generated.new_tokens.size(); ++index) {
                                            if (index > 0) {
                                                std::cout << ",";
                                            }
                                            std::cout << generated.new_tokens[index];
                                        }
                                        std::cout << "\n";
                                        std::cout << "native_generate_probe.text=" << generated.text << "\n";
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }

    return 0;
}
