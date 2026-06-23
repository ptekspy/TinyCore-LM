from __future__ import annotations

import torch

from tinycore_model import ComposedLinear


def test_composed_linear_matches_effective_weight() -> None:
    torch.manual_seed(7)
    layer = ComposedLinear(3, 5, n_basis=4, n_virtual_layers=2, low_rank=2, bias=True)
    x = torch.randn(2, 6, 3)
    for virtual_layer in range(2):
        expected = x @ layer.effective_weight(virtual_layer) + layer.bias[virtual_layer]
        actual = layer(x, virtual_layer)
        torch.testing.assert_close(actual, expected)


def test_composed_linear_reports_stored_and_effective_params() -> None:
    layer = ComposedLinear(4, 8, n_basis=2, n_virtual_layers=3, low_rank=0)
    report = layer.parameter_report()
    assert report["stored_unique_parameter_count"] == 2 * 4 * 8 + 3 * 2
    assert report["effective_parameter_count_if_materialized"] == 3 * 4 * 8


def test_low_rank_deltas_add_stored_params_without_changing_effective_shape() -> None:
    layer = ComposedLinear(4, 8, n_basis=2, n_virtual_layers=3, low_rank=2)
    report = layer.parameter_report()
    expected_low_rank = 3 * ((4 * 2) + (2 * 8))
    assert report["stored_unique_parameter_count"] == 2 * 4 * 8 + 3 * 2 + expected_low_rank
    assert report["effective_parameter_count_if_materialized"] == 3 * 4 * 8
