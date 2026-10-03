import numpy as np
import pytest

from mlkit.metrics import baseline_up_rate, block_bootstrap_auc, direction_agreement, strategy_result


def test_perfect_predictions_give_auc_one():
    y = np.array([0, 1] * 50)
    auc, lo, hi = block_bootstrap_auc(y, y.astype(float))
    assert auc == 1.0 and lo == 1.0 and hi == 1.0


def test_random_predictions_interval_contains_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    auc, lo, hi = block_bootstrap_auc(y, rng.random(2000))
    assert lo < 0.5 < hi
    assert lo <= auc <= hi


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 2, 500), rng.random(500)
    assert block_bootstrap_auc(y, p) == block_bootstrap_auc(y, p)


def test_bootstrap_needs_at_least_one_block():
    with pytest.raises(ValueError):
        block_bootstrap_auc(np.array([0, 1]), np.array([0.2, 0.8]))


def test_baseline_up_rate():
    assert baseline_up_rate([1, 1, 0, 0]) == 0.5


def test_direction_agreement():
    assert direction_agreement([0.6, 0.4, 0.7, 0.2], [0.9, 0.1, 0.3, 0.45]) == 0.75


def test_strategy_result_long_short_flat_with_costs():
    p = [0.60, 0.50, 0.40]          # long, flat, short
    fwd = [0.01, 0.05, -0.02]       # +1% on the long, +2% on the short
    result = strategy_result(p, fwd, threshold=0.55, cost_bps=10, step=1)
    assert result["trades"] == 2
    assert result["return"] == pytest.approx(0.01 + 0.02 - 2 * 0.001)


def test_strategy_result_uses_every_step_th_row():
    result = strategy_result([0.9, 0.9, 0.9, 0.9], [0.01] * 4, threshold=0.55, cost_bps=0, step=2)
    assert result["trades"] == 2
