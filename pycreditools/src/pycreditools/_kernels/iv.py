from __future__ import annotations

import numpy as np


def iv_cluster(
    pd_values: np.ndarray,
    volumes: np.ndarray,
    max_groups: int,
    min_vol_ratio: float,
    lambda_cross: float = 0.5,
    lambda_vol: float = 0.2,
    monthly_vols: np.ndarray | None = None,
    monthly_bads: np.ndarray | None = None,
) -> np.ndarray:
    """
    IV-based agglomerative clustering with constraints.

    Args:
        pd_values: float64[n_bins] - mean PD per bin
        volumes: int64[n_bins] - volume per bin
        max_groups: exact number of output clusters (algorithm will merge down to this)
        min_vol_ratio: min fraction of total volume per cluster
        lambda_cross: penalty weight for vintage crossings
        lambda_vol: penalty weight for PD volatility
        monthly_vols: int64[n_bins, n_months]
        monthly_bads: int64[n_bins, n_months]

    Returns:
        int64[n_bins] - 1-based group assignments
    """
    n_bins = len(pd_values)
    if n_bins == 0:
        return np.array([], dtype=np.int64)
    if n_bins <= max_groups and (volumes == 0).sum() == 0:
        # Check if all other constraints hold? Actually if we just want to force merges
        # when constraints are violated, we should still run the loop.
        pass

    active = np.ones(n_bins, dtype=bool)
    current_vol = volumes.copy().astype(np.float64)
    current_bads = (pd_values * current_vol).astype(np.float64)

    total_vol = current_vol.sum()
    total_bads = current_bads.sum()
    total_goods = total_vol - total_bads

    if monthly_vols is not None and monthly_bads is not None:
        curr_m_vols = monthly_vols.copy().astype(np.float64)
        curr_m_bads = monthly_bads.copy().astype(np.float64)
    else:
        curr_m_vols = None
        curr_m_bads = None

    group_ids = np.arange(n_bins)
    n_active = n_bins

    def calc_iv(bads, vols):
        if total_goods <= 0 or total_bads <= 0:
            return 0.0
        goods = vols - bads
        p_b = bads / total_bads
        p_g = goods / total_goods
        if p_b <= 0 or p_g <= 0:
            return 0.0
        return (p_g - p_b) * np.log(p_g / p_b)

    while True:
        if n_active <= 1:
            break

        active_indices = np.where(active)[0]
        n_curr = len(active_indices)

        min_cost = np.inf
        best_merge_idx = -1

        for i in range(n_curr - 1):
            idx1 = active_indices[i]
            idx2 = active_indices[i + 1]

            v1 = current_vol[idx1]
            v2 = current_vol[idx2]
            b1 = current_bads[idx1]
            b2 = current_bads[idx2]

            p1 = b1 / v1 if v1 > 0 else 0.0
            p2 = b2 / v2 if v2 > 0 else 0.0

            # Hard skip for monotonicity violation unless it's a forced merge
            # Monotonicity violation: p1 >= p2
            violation = (p1 >= p2) and (v1 > 0) and (v2 > 0)

            # Force merges if volume is 0
            if v1 == 0 or v2 == 0:
                cost = -1e9
            else:
                if violation:
                    cost = -1e6  # prioritize fixing monotonicity over normal merges
                else:
                    # Calculate IV loss
                    iv1 = calc_iv(b1, v1)
                    iv2 = calc_iv(b2, v2)
                    iv_merged = calc_iv(b1 + b2, v1 + v2)
                    iv_loss = iv1 + iv2 - iv_merged

                    cross_penalty = 0.0
                    volatility_penalty = 0.0

                    if curr_m_vols is not None and curr_m_bads is not None:
                        mv = curr_m_vols[idx1] + curr_m_vols[idx2]
                        mb = curr_m_bads[idx1] + curr_m_bads[idx2]

                        valid = mv > 0
                        if valid.any():
                            mp = mb[valid] / mv[valid]
                            volatility_penalty = np.std(mp)

                        # crossings between new merged group and neighbors?
                        # To simplify, the C++ IV clustering engine penalizes crossings
                        # *within* the merged group (i.e. did the two groups cross each other?)
                        mv1 = curr_m_vols[idx1]
                        mv2 = curr_m_vols[idx2]
                        mb1 = curr_m_bads[idx1]
                        mb2 = curr_m_bads[idx2]
                        v_valid = (mv1 > 0) & (mv2 > 0)
                        if v_valid.any():
                            mp1 = mb1[v_valid] / mv1[v_valid]
                            mp2 = mb2[v_valid] / mv2[v_valid]
                            crossings = np.sum(mp1 >= mp2)
                            cross_penalty = crossings

                    cost = iv_loss + lambda_cross * cross_penalty + lambda_vol * volatility_penalty

                    # Force merge if volume below threshold
                    if (v1 / total_vol < min_vol_ratio) or (v2 / total_vol < min_vol_ratio):
                        cost -= 1000.0  # arbitrary large priority but less than monotonicity

            if cost < min_cost:
                min_cost = cost
                best_merge_idx = i

        # Stopping condition
        # If no forced merges are required AND we reached max_groups, stop.
        # Forced merges have cost < -100
        if min_cost >= -100 and n_active <= max_groups:
            break

        # Execute merge
        idx1 = active_indices[best_merge_idx]
        idx2 = active_indices[best_merge_idx + 1]

        current_vol[idx1] += current_vol[idx2]
        current_bads[idx1] += current_bads[idx2]

        if curr_m_vols is not None and curr_m_bads is not None:
            curr_m_vols[idx1] += curr_m_vols[idx2]
            curr_m_bads[idx1] += curr_m_bads[idx2]

        active[idx2] = False
        group_ids[group_ids == idx2] = idx1
        n_active -= 1

    # Remap to 1-based sequential integers
    active_indices = np.where(active)[0]
    final_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(active_indices, 1)}

    result = np.array([final_mapping[g] for g in group_ids], dtype=np.int64)
    return result
