import numpy as np
import pandas as pd


def calculate_tier_metrics(
    values: np.ndarray,
    groups: np.ndarray,
    defaults: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    """
    Fast screening metrics calculating IV and PD spread for a variable across risk groups.

    Args:
        values: float64[n_obs] - candidate variable values
        groups: int64[n_obs] - risk group assignments
        defaults: int64[n_obs] - default flags (0/1)
        n_bins: number of quantile bins per group

    Returns:
        DataFrame with columns: risk_group, iv, pd_min, pd_max, pd_spread, tier_vol
    """
    unique_groups = np.unique(groups)
    results = []

    # We should exclude NaNs from calculations
    valid_mask = ~np.isnan(values) & ~np.isnan(defaults) & ~np.isnan(groups)
    values = values[valid_mask]
    groups = groups[valid_mask].astype(np.int64)
    defaults = defaults[valid_mask].astype(np.int64)

    for g in unique_groups:
        if np.isnan(g):
            continue

        g_mask = groups == g
        g_values = values[g_mask]
        g_defaults = defaults[g_mask]
        tier_vol = len(g_values)

        if tier_vol == 0:
            results.append(
                {
                    "risk_group": g,
                    "iv": 0.0,
                    "pd_min": np.nan,
                    "pd_max": np.nan,
                    "pd_spread": 0.0,
                    "tier_vol": 0,
                }
            )
            continue

        total_bads = g_defaults.sum()
        total_goods = tier_vol - total_bads

        # Sort by value to bin
        sort_idx = np.argsort(g_values)
        sorted_defaults = g_defaults[sort_idx]

        # Rank-based binning (similar to pd.qcut with duplicates handled implicitly by position)
        bin_assignments = (np.arange(tier_vol) * n_bins) // tier_vol

        iv_sum = 0.0
        pd_list = []

        for b in range(n_bins):
            b_mask = bin_assignments == b
            b_vol = b_mask.sum()

            if b_vol == 0:
                continue

            b_bads = sorted_defaults[b_mask].sum()
            b_goods = b_vol - b_bads

            pd_list.append(b_bads / b_vol)

            # Laplace smoothing for IV
            p_b = (b_bads + 0.5) / (total_bads + 1.0)
            p_g = (b_goods + 0.5) / (total_goods + 1.0)

            iv_sum += (p_g - p_b) * np.log(p_g / p_b)

        if len(pd_list) > 0:
            pd_min = min(pd_list)
            pd_max = max(pd_list)
            pd_spread = pd_max - pd_min
        else:
            pd_min = np.nan
            pd_max = np.nan
            pd_spread = 0.0

        results.append(
            {
                "risk_group": g,
                "iv": iv_sum,
                "pd_min": pd_min,
                "pd_max": pd_max,
                "pd_spread": pd_spread,
                "tier_vol": tier_vol,
            }
        )

    if not results:
        return pd.DataFrame(
            columns=["risk_group", "iv", "pd_min", "pd_max", "pd_spread", "tier_vol"]
        )

    return pd.DataFrame(results)
