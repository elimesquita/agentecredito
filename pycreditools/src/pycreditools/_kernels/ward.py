from __future__ import annotations

import numpy as np


def ward_cluster(
    pd_values: np.ndarray,
    volumes: np.ndarray,
    max_groups: int,
    min_vol_ratio: float,
    max_crossings: int,
    use_volume_weights: bool = True,
    monthly_vols: np.ndarray | None = None,
    monthly_bads: np.ndarray | None = None,
) -> np.ndarray:
    """
    Ward agglomerative clustering with credit-risk constraints.

    Args:
        pd_values: float64[n_bins] - mean PD per bin
        volumes: int64[n_bins] - volume per bin
        max_groups: max number of output clusters
        min_vol_ratio: min fraction of total volume per cluster
        max_crossings: max vintage inversions between adjacent groups
        use_volume_weights: if False, performs pure distance-based linkage
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

    # State vectors
    # We maintain active groups in a linked list structure to allow O(1) merges,
    # or just use masking since n_bins is typically small (e.g. 100-1000).
    # Since this is pure numpy for small N, masking and array recreation is fine.

    active = np.ones(n_bins, dtype=bool)
    current_pd = pd_values.copy().astype(np.float64)
    current_vol = volumes.copy().astype(np.float64)
    total_vol = current_vol.sum()

    if monthly_vols is not None and monthly_bads is not None:
        curr_m_vols = monthly_vols.copy().astype(np.float64)
        curr_m_bads = monthly_bads.copy().astype(np.float64)
    else:
        curr_m_vols = None
        curr_m_bads = None

    # group_ids tracks which original bins belong to which current cluster index.
    # initially bin i belongs to cluster i
    group_ids = np.arange(n_bins)

    n_active = n_bins

    while True:
        if n_active <= 1:
            break

        active_indices = np.where(active)[0]
        n_curr = len(active_indices)

        min_cost = np.inf
        best_merge_idx = -1  # index in active_indices of the left group

        for i in range(n_curr - 1):
            idx1 = active_indices[i]
            idx2 = active_indices[i + 1]

            v1 = current_vol[idx1]
            v2 = current_vol[idx2]
            p1 = current_pd[idx1]
            p2 = current_pd[idx2]

            # Linkage distance
            if use_volume_weights:
                if v1 + v2 == 0:
                    delta = 0.0
                else:
                    delta = (v1 * v2) / (v1 + v2) * (p1 - p2) ** 2
            else:
                delta = (p1 - p2) ** 2

            cost = delta

            # Priority 0: Zero volume
            if v1 == 0 or v2 == 0:
                cost = -2e9 + delta
            # Priority 1: Monotonicity violation (p1 >= p2)
            elif p1 >= p2:
                cost = -1e9 + delta
            # Priority 2: Volume below min_vol_ratio
            elif (v1 / total_vol) < min_vol_ratio or (v2 / total_vol) < min_vol_ratio:
                cost = -1e6 + delta
            else:
                # Priority 3: Crossings
                if curr_m_vols is not None and curr_m_bads is not None:
                    mv1 = curr_m_vols[idx1]
                    mv2 = curr_m_vols[idx2]
                    mb1 = curr_m_bads[idx1]
                    mb2 = curr_m_bads[idx2]

                    # Compute monthly PDs, ignoring months with zero volume in either group
                    valid_months = (mv1 > 0) & (mv2 > 0)
                    if valid_months.any():
                        mp1 = mb1[valid_months] / mv1[valid_months]
                        mp2 = mb2[valid_months] / mv2[valid_months]
                        crossings = np.sum(mp1 >= mp2)

                        if crossings > max_crossings:
                            cost = -1e3 + delta

            if cost < min_cost:
                min_cost = cost
                best_merge_idx = i

        # Stopping condition: if no constraint violated AND n_active <= max_groups
        if min_cost >= 0 and n_active <= max_groups:
            break

        # Execute merge
        idx1 = active_indices[best_merge_idx]
        idx2 = active_indices[best_merge_idx + 1]

        # Merge idx2 into idx1
        v1 = current_vol[idx1]
        v2 = current_vol[idx2]

        if v1 + v2 > 0:
            current_pd[idx1] = (current_pd[idx1] * v1 + current_pd[idx2] * v2) / (v1 + v2)
        else:
            current_pd[idx1] = 0.0

        current_vol[idx1] = v1 + v2

        if curr_m_vols is not None and curr_m_bads is not None:
            curr_m_vols[idx1] += curr_m_vols[idx2]
            curr_m_bads[idx1] += curr_m_bads[idx2]

        active[idx2] = False
        group_ids[group_ids == idx2] = idx1
        n_active -= 1

    # Remap active groups to 1-based sequential integers
    active_indices = np.where(active)[0]
    final_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(active_indices, 1)}

    result = np.array([final_mapping[g] for g in group_ids], dtype=np.int64)
    return result
