"""Evaluation metrics, identical for every project so results can be compared."""
import numpy as np
from sklearn.metrics import roc_auc_score


def block_bootstrap_auc(y, p, block: int = 24, n: int = 500, seed: int = 42) -> tuple[float, float, float]:
    """AUC plus a 95% interval.

    Resamples blocks of `block` consecutive candles, because neighbouring candles
    look alike: drawing single rows would make the interval too narrow.
    """
    y, p = np.asarray(y), np.asarray(p)
    if len(y) < block:
        raise ValueError(f"need at least {block} rows, got {len(y)}")
    rng = np.random.default_rng(seed)
    starts = np.arange(0, len(y) - block + 1)
    k = len(y) // block
    aucs = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, k)])
        if y[idx].min() != y[idx].max():
            aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(roc_auc_score(y, p)), float(lo), float(hi)


def baseline_up_rate(y) -> float:
    """Share of rows whose label is 1: how one-sided the period was."""
    return float(np.mean(np.asarray(y)))


def direction_agreement(p_a, p_b) -> float:
    """Share of rows where both predictions point the same way (above or below 0.5)."""
    return float(np.mean((np.asarray(p_a) > 0.5) == (np.asarray(p_b) > 0.5)))


def strategy_result(p, fwd_ret, threshold: float, cost_bps: float, step: int) -> dict:
    """Simple trading test: long above `threshold`, short below 1 - threshold.

    Uses every `step`-th row so trades do not overlap, and subtracts `cost_bps`
    per round trip. Informational only; not yet a release gate.
    """
    p = np.asarray(p)[::step]
    r = np.asarray(fwd_ret)[::step]
    pos = np.where(p > threshold, 1, np.where(p < 1 - threshold, -1, 0))
    traded = pos != 0
    ret = float((pos[traded] * r[traded]).sum() - traded.sum() * cost_bps / 10_000)
    return {"return": ret, "trades": int(traded.sum())}
