from __future__ import annotations

import torch

from tinycore_model import ModelConfig, SharedTransformerLM, TinyCoreLM, TransformerLM, parameter_report


def test_models_forward_loss_and_accounting() -> None:
    cfg = ModelConfig(vocab_size=128, d_model=32, n_heads=4, n_layers=2, n_virtual_layers=4, basis_rank=2, max_seq_len=16)
    batch = torch.randint(0, cfg.vocab_size, (2, 17))
    baseline = TransformerLM(cfg)
    tinycore = TinyCoreLM(cfg)

    assert baseline(batch[:, :-1]).shape == (2, 16, cfg.vocab_size)
    assert tinycore(batch[:, :-1]).shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(baseline.loss(batch))
    assert torch.isfinite(tinycore.loss(batch))

    baseline_report = parameter_report(baseline)
    tinycore_report = parameter_report(tinycore)
    assert tinycore_report["effective_parameter_count_if_materialized"] > tinycore_report["stored_unique_parameter_count"]
    assert baseline_report["stored_unique_parameter_count"] > 0


def test_shared_transformer_reuses_stored_block_weights() -> None:
    cfg = ModelConfig(
        model_type="shared_layer_transformer_v0",
        vocab_size=128,
        d_model=32,
        n_heads=4,
        n_layers=4,
        max_seq_len=16,
    )
    batch = torch.randint(0, cfg.vocab_size, (2, 17))
    model = SharedTransformerLM(cfg)
    report = parameter_report(model)
    assert model(batch[:, :-1]).shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(model.loss(batch))
    assert report["effective_parameter_count_if_materialized"] > report["stored_unique_parameter_count"]


def test_recurrent_tinycore_forward_loss_and_extra_state_params() -> None:
    basis_cfg = ModelConfig(
        model_type="tinycore_basis_v0",
        vocab_size=128,
        d_model=32,
        n_heads=4,
        n_virtual_layers=4,
        basis_rank=2,
        max_seq_len=16,
    )
    recurrent_cfg = ModelConfig(
        model_type="tinycore_recurrent_v0",
        vocab_size=128,
        d_model=32,
        n_heads=4,
        n_virtual_layers=4,
        basis_rank=2,
        max_seq_len=16,
    )
    batch = torch.randint(0, recurrent_cfg.vocab_size, (2, 17))
    basis_report = parameter_report(TinyCoreLM(basis_cfg))
    recurrent = TinyCoreLM(recurrent_cfg)
    recurrent_report = parameter_report(recurrent)
    assert recurrent(batch[:, :-1]).shape == (2, 16, recurrent_cfg.vocab_size)
    assert torch.isfinite(recurrent.loss(batch))
    assert recurrent_report["stored_unique_parameter_count"] > basis_report["stored_unique_parameter_count"]
