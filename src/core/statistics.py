"""Statistical helpers: Cohen's d, 95% CI, effect size labels."""

from __future__ import annotations

import math
from typing import Sequence

# Two-tailed t-critical values for 95% CI (df = n-1)
_T_CRIT: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    20: 2.086, 25: 2.060, 30: 2.042,
}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def _t_crit(df: int) -> float:
    """Return approximate two-tailed t-critical value for 95% CI."""
    if df <= 0:
        return float("inf")
    if df in _T_CRIT:
        return _T_CRIT[df]
    # Linear interpolation for df between table entries
    keys = sorted(_T_CRIT.keys())
    for i in range(len(keys) - 1):
        if keys[i] <= df <= keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            t = (df - lo) / (hi - lo)
            return _T_CRIT[lo] + t * (_T_CRIT[hi] - _T_CRIT[lo])
    return 1.96  # z for large df


def cohens_d(pre: Sequence[float], post: Sequence[float]) -> float | None:
    """Return Cohen's d (post - pre) / pooled_std.

    Positive d → post > pre (e.g., velocity increased after training).
    Negative d → post < pre (e.g., ROM decreased under fatigue).
    """
    if len(pre) < 2 or len(post) < 2:
        return None
    mean_diff = _mean(post) - _mean(pre)
    var_pre = _std(pre) ** 2
    var_post = _std(post) ** 2
    pooled = math.sqrt((var_pre + var_post) / 2.0)
    return (mean_diff / pooled) if pooled > 1e-10 else None


def effect_label(d: float | None) -> str:
    """Interpret |d| as a text effect-size label (Turkish)."""
    if d is None:
        return "—"
    ad = abs(d)
    if ad < 0.2:
        return "önemsiz"
    if ad < 0.5:
        return "küçük"
    if ad < 0.8:
        return "orta"
    return "büyük"


def ci_95(values: Sequence[float]) -> tuple[float, float] | None:
    """Return (lower, upper) 95% confidence interval for the mean."""
    n = len(values)
    if n < 2:
        return None
    m = _mean(values)
    s = _std(values)
    margin = _t_crit(n - 1) * s / math.sqrt(n)
    return (m - margin, m + margin)


def compare_metric(
    pre_vals: list[float],
    post_vals: list[float],
) -> dict:
    """Full statistical comparison of a metric between pre and post sessions.

    Returns:
        pre_mean, pre_std, pre_ci,
        post_mean, post_std, post_ci,
        delta, pct_change,
        cohens_d, effect_label,
        n_pre, n_post
    """
    pre_mean  = _mean(pre_vals)  if pre_vals  else None
    post_mean = _mean(post_vals) if post_vals else None
    pre_std   = _std(pre_vals)   if len(pre_vals) > 1  else None
    post_std  = _std(post_vals)  if len(post_vals) > 1 else None

    delta = (post_mean - pre_mean) if (pre_mean is not None and post_mean is not None) else None
    pct   = (delta / abs(pre_mean) * 100.0) if (delta is not None and pre_mean and abs(pre_mean) > 1e-8) else None

    d = cohens_d(pre_vals, post_vals)

    return {
        "pre_mean":     pre_mean,
        "pre_std":      pre_std,
        "pre_ci":       ci_95(pre_vals),
        "post_mean":    post_mean,
        "post_std":     post_std,
        "post_ci":      ci_95(post_vals),
        "delta":        delta,
        "pct_change":   pct,
        "cohens_d":     d,
        "effect_label": effect_label(d),
        "n_pre":        len(pre_vals),
        "n_post":       len(post_vals),
    }
