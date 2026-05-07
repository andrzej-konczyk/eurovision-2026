"""
Sprint 12 US-S12-01 — Market-vs-history signal features.

odds_vs_history_delta captures the gap between market-implied probability
and a normalised historical Grand Final rank score.  A positive value means
the market is more bullish than recent form suggests (potential upside
surprise); a negative value means the market is more sceptical than history.

Designed to mitigate KL-08: LGBM prior bias on the 2025 holdout where
countries with strong market odds but weak/absent 3-year GF history
(Greece, Switzerland 2025) were systematically underestimated.

Temporal isolation: both parent features are pre-contest signals.
  - avg_final_rank_3yr   : computed from years < current year only
  - implied_prob_close   : pre-contest bookmaker odds
No outcome leakage.
"""
from __future__ import annotations

import pandas as pd

_N_FINALISTS = 26  # standard Grand Final field size used to normalise rank


def compute_market_signals(matrix: pd.DataFrame) -> pd.DataFrame:
    """Derive market-vs-history delta from an already-merged feature matrix.

    Input *matrix* must contain:
        implied_prob_close   – GF winner-market implied probability (0-1)
        avg_final_rank_3yr   – mean GF rank over the previous 3 years (1=best)

    NaN in either parent column propagates to the output; downstream
    SimpleImputer handles imputation within the training Pipeline.

    Returns a single-column DataFrame aligned to *matrix*.
    """
    if "implied_prob_close" not in matrix.columns or "avg_final_rank_3yr" not in matrix.columns:
        return pd.DataFrame(
            {"odds_vs_history_delta": pd.Series(dtype=float)}, index=matrix.index
        )
    # Convert rank (1=winner, 26=last) to a probability-like score in [0.04, 1.00].
    # avg_final_rank_3yr = 1  → hist_perf = 1.00  (outstanding history)
    # avg_final_rank_3yr = 26 → hist_perf = 0.04  (poor history)
    # avg_final_rank_3yr = NaN → propagates (no GF appearances)
    hist_perf = (_N_FINALISTS + 1 - matrix["avg_final_rank_3yr"]) / _N_FINALISTS
    delta = matrix["implied_prob_close"] - hist_perf
    return pd.DataFrame({"odds_vs_history_delta": delta.values}, index=matrix.index)
