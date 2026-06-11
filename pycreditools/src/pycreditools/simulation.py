from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._types import Quadrant, SimulationMethod
from .policy import CreditPolicy
from .stages import RateStage


@dataclass
class CreditSimResults:
    data: pd.DataFrame
    metadata: dict[str, Any]
    policy: CreditPolicy | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the results to a dictionary."""
        return {
            "metadata": self.metadata,
        }

    def to_decision_dataframe(
        self,
        rating_recipe: Any | None = None,
        rating_labels: dict[int, str] | None = None,
    ) -> pd.DataFrame:
        """Convert simulation results to a simplified decision DataFrame.

        Contains:
        - All input columns (original columns).
        - 'decisao': 'Aprovado' if passed all hard filters, 'Reprovado' otherwise.
        - 'motivo': The position and name of the first failed filter/cutoff stage, or 'Aprovado'.
        - 'rating': The mapped risk rating ('A', 'B', etc.) if rating_recipe is provided.
        """
        pol = self.policy
        if pol is None:
            # Fallback to deserialize from metadata
            policy_dict = self.metadata.get("policy")
            if policy_dict:
                from .policy import CreditPolicy

                try:
                    pol = CreditPolicy.from_dict(policy_dict)
                except Exception:
                    pass
            if pol is None:
                raise ValueError(
                    "No policy reference found in simulation results to extract hard filters."
                )

        df = self.data.copy()

        # 1. Separate input columns
        def is_sim_col(c: str) -> bool:
            if c in (
                "new_approval",
                "approved_pre_rate",
                "quadrant",
                "simulated_default",
                "approved",
                "decisao",
                "motivo",
                "decision",
                "reason",
                "contratou",
                "inadimplente",
                "cenario",
                "hired",
                "defaulted",
                "scenario",
                "rating",
                "Rating",
                "risk_rating",
            ):
                return True
            if c.startswith("stage_"):
                parts = c.split("_")
                if len(parts) >= 3 and parts[1].isdigit():
                    return True
            return False

        input_cols = [c for c in df.columns if not is_sim_col(c)]

        # 2. Extract decision based on hard filters
        from .stages import RateStage

        hard_stages = [
            (i, stage) for i, stage in enumerate(pol.stages) if not isinstance(stage, RateStage)
        ]

        if not hard_stages:
            df["decision"] = "Approved"
            df["reason"] = "Approved"
        else:
            stage_cols = [f"stage_{i}_{stage.name}" for i, stage in hard_stages]

            # Check for failure in any hard stage (< 0.5 for analytical / 0 for stochastic)
            failed_df = pd.DataFrame(index=df.index)
            for col_name in stage_cols:
                if col_name in df.columns:
                    failed_df[col_name] = df[col_name] < 0.5
                else:
                    failed_df[col_name] = False

            has_failure = failed_df.any(axis=1)

            # Reason: Position (1, 2...) and name of filter
            col_to_label = {
                f"stage_{i}_{stage.name}": f"{i + 1}: {stage.name}" for i, stage in hard_stages
            }

            # Vectorized first failed stage
            first_failed_col = np.where(has_failure, failed_df.idxmax(axis=1), None)

            df["decision"] = np.where(has_failure, "Rejected", "Approved")
            df["reason"] = (
                pd.Series(first_failed_col, index=df.index).map(col_to_label).fillna("Approved")
            )

        # 3. Apply optional rating recipe
        active_recipe = (
            rating_recipe
            if rating_recipe is not None
            else (self.policy.rating_recipe if self.policy is not None else None)
        )
        if active_recipe is not None:
            if isinstance(active_recipe, dict):
                # Segmented ratings by store/region
                segment_col = "region"
                for c in ["region", "loja", "safra"]:
                    if c in df.columns:
                        segment_col = c
                        break
                df["rating"] = None
                for seg, sub_recipe in active_recipe.items():
                    mask = df[segment_col] == seg
                    if mask.any():
                        pred_seg = sub_recipe.predict(df[mask])
                        if rating_labels is None:
                            rating_labels = {i: chr(64 + i) for i in range(1, 27)}
                        df.loc[mask, "rating"] = pred_seg["risk_rating"].map(rating_labels)
            else:
                pred_df = active_recipe.predict(df)
                if rating_labels is None:
                    rating_labels = {i: chr(64 + i) for i in range(1, 27)}
                df["rating"] = pred_df["risk_rating"].map(rating_labels)
        else:
            df["rating"] = None

        # 4. Add 'hired', 'defaulted', and 'scenario'
        is_analytical = self.metadata.get("method") == "analytical"

        if "new_approval" in df.columns:
            if is_analytical:
                df["hired"] = df["new_approval"]
            else:
                df["hired"] = np.where(df["new_approval"] > 0.5, "Yes", "No")
        else:
            df["hired"] = "No"

        if "simulated_default" in df.columns:
            if is_analytical:
                df["defaulted"] = df["simulated_default"]
            else:
                hired_mask = df["new_approval"] > 0.5
                df["defaulted"] = np.where(hired_mask, df["simulated_default"], np.nan)
        else:
            df["defaulted"] = np.nan

        if "scenario" in df.columns:
            quad_map = {
                "keep_in": "Keep In",
                "swap_in": "Swap In",
                "swap_out": "Swap Out",
                "keep_out": "Keep Out",
            }
            df["scenario"] = df["scenario"].map(quad_map).fillna(df["scenario"])
        else:
            df["scenario"] = "Keep Out"

        # 5. Construct simplified DataFrame (standard & immutable copy)
        simple_cols = input_cols + [
            "decision",
            "reason",
            "hired",
            "defaulted",
            "scenario",
            "rating",
        ]

        return df[simple_cols].copy()


def run_simulation(
    data: pd.DataFrame,
    policy: CreditPolicy,
    method: SimulationMethod | str = SimulationMethod.STOCHASTIC,
) -> CreditSimResults:
    """Run a multi-stage credit policy simulation.

    Args:
        data: Applicant data.
        policy: The credit policy to simulate.
        method: "stochastic" (default) or "analytical".

    Returns:
        CreditSimResults containing the simulation data and metadata.
    """
    if isinstance(method, str):
        method = SimulationMethod(method)

    df = data.copy()

    policy.validate(df)

    stage_approval_cols = []
    df["pass_prob_funnel"] = 1.0

    # Track the cumulative non-RateStage probability for "Approved" (pre-take-up)
    # so we can distinguish Approved vs Hired in the summary.
    df["pass_prob_pre_rate"] = 1.0

    if policy.stages:
        for i, stage in enumerate(policy.stages):
            stage_output_col = f"stage_{i}_{stage.name}"
            stage_approval_cols.append(stage_output_col)

            # Run the stage
            stage_res = stage.apply(df, method=method.value, policy=policy)

            # Cumulative pass probability
            df["pass_prob_funnel"] = df["pass_prob_funnel"] * stage_res
            df[stage_output_col] = df["pass_prob_funnel"]

            # Accumulate only filter/cutoff stages for pre-rate approval
            if not isinstance(stage, RateStage):
                df["pass_prob_pre_rate"] = df["pass_prob_pre_rate"] * stage_res

    if method == SimulationMethod.STOCHASTIC:
        if not stage_approval_cols:
            df["new_approval"] = 1
            df["approved"] = 1
        else:
            df["new_approval"] = df[stage_approval_cols].fillna(0).min(axis=1).astype(int)
            # "approved" = passed all filter/cutoff stages (before rate stages)
            df["approved_pre_rate"] = (df["pass_prob_pre_rate"] > 0).astype(int)
    else:
        df["new_approval"] = df["pass_prob_funnel"]
        # In analytical mode, approved_pre_rate is the probability up to last non-rate stage
        df["approved_pre_rate"] = df["pass_prob_pre_rate"]

    del df["pass_prob_funnel"]
    if "pass_prob_pre_rate" in df.columns:
        del df["pass_prob_pre_rate"]

    df = _classify_scenarios(df, policy, "approved_pre_rate")
    df = _assign_simulated_defaults(df, policy, method)

    metadata = {
        "policy": policy.to_dict(),
        "method": method.value,
    }

    return CreditSimResults(data=df, metadata=metadata, policy=policy)


def _classify_scenarios(
    df: pd.DataFrame, policy: CreditPolicy, new_approval_col: str
) -> pd.DataFrame:
    """Classify applicants into swap_in, swap_out, keep_in, keep_out."""
    old_app = df[policy.current_approval_col].fillna(0).astype(float)
    new_app = df[new_approval_col].fillna(0).astype(float)

    old_flag = (old_app > 0).astype(int)
    new_flag = (new_app > 0).astype(int)

    conditions = [
        (old_flag == 0) & (new_flag == 1),
        (old_flag == 1) & (new_flag == 0),
        (old_flag == 1) & (new_flag == 1),
        (old_flag == 0) & (new_flag == 0),
    ]

    choices = [
        Quadrant.SWAP_IN.value,
        Quadrant.SWAP_OUT.value,
        Quadrant.KEEP_IN.value,
        Quadrant.KEEP_OUT.value,
    ]

    # We use numpy where/select or simply map via a Series mapping to avoid dtype promotion issues.
    # Alternatively, ensure the default is a string (e.g. 'unknown').
    df["scenario"] = np.select(conditions, choices, default="unknown")
    return df


def _assign_simulated_defaults(
    df: pd.DataFrame, policy: CreditPolicy, method: SimulationMethod
) -> pd.DataFrame:
    """Assign default outcomes for the final approved population."""
    swap_ins = df[df["scenario"] == Quadrant.SWAP_IN.value]

    # Initialize simulated_default as float to prevent dtype warnings
    df["simulated_default"] = np.nan

    # Keep_in gets actual default
    keep_in_mask = df["scenario"] == Quadrant.KEEP_IN.value
    df.loc[keep_in_mask, "simulated_default"] = df.loc[
        keep_in_mask, policy.actual_default_col
    ].astype(float)

    if not swap_ins.empty:
        # Per-client baseline PD (score-calibrated)
        baseline_pd = _estimate_swap_in_baseline_pd(df, swap_ins, policy)

        if not policy.stress_scenarios:
            final_probs = baseline_pd
        else:
            if len(policy.stress_scenarios) > 1:
                warnings.warn(
                    f"Multiple stress scenarios active ({len(policy.stress_scenarios)}). "
                    "The simulator will use the maximum (worst-case) stressed PD for each applicant.",
                    UserWarning,
                    stacklevel=2,
                )
            prob_matrix = pd.DataFrame(index=swap_ins.index)
            # Create a copy to prevent warnings when modifying
            swap_ins_temp = swap_ins.copy()
            swap_ins_temp["__baseline_pd"] = baseline_pd.values
            for i, scenario in enumerate(policy.stress_scenarios):
                prob_matrix[f"prob_{i}"] = scenario.apply(swap_ins_temp, "__baseline_pd")
            final_probs = prob_matrix.max(axis=1)

        final_probs = final_probs.clip(0.0, 1.0)

        if method == SimulationMethod.STOCHASTIC:
            random_draws = np.random.random(len(swap_ins))
            outcomes = (random_draws < final_probs).astype(float)
        else:
            outcomes = final_probs

        df.loc[swap_ins.index, "simulated_default"] = outcomes

    return df


def _estimate_swap_in_baseline_pd(
    df: pd.DataFrame,
    swap_ins: pd.DataFrame,
    policy: CreditPolicy,
) -> pd.Series:
    """Estimate per-client baseline PD for Swap Ins using a score→PD
    local calibration built from the Keep In population.
    """
    keep_ins_mask = df["scenario"] == Quadrant.KEEP_IN.value
    actual_default_col = policy.actual_default_col
    global_pd = df.loc[keep_ins_mask, actual_default_col].mean() if keep_ins_mask.any() else 0.0
    if pd.isna(global_pd):
        global_pd = 0.0

    # 1. Rating-based calibration (if rating_recipe is active)
    if policy.rating_recipe is not None:
        try:
            active_recipe = policy.rating_recipe
            if isinstance(active_recipe, dict):
                segment_col = "region"
                for c in ["region", "loja", "safra"]:
                    if c in df.columns:
                        segment_col = c
                        break
                ratings = pd.Series(index=df.index, dtype=object)
                for seg, sub_recipe in active_recipe.items():
                    mask = df[segment_col] == seg
                    if mask.any():
                        pred_seg = sub_recipe.predict(df[mask])
                        ratings.loc[mask] = pred_seg["risk_rating"].map(lambda x: chr(64 + x))
            else:
                pred_df = active_recipe.predict(df)
                ratings = pred_df["risk_rating"].map(lambda x: chr(64 + x))

            keep_in_ratings = ratings.loc[keep_ins_mask]
            keep_in_defaults = df.loc[keep_ins_mask, actual_default_col]
            rating_pd = keep_in_defaults.groupby(keep_in_ratings).mean().fillna(global_pd)
            
            swap_in_ratings = ratings.loc[swap_ins.index]
            baseline = swap_in_ratings.map(rating_pd).fillna(global_pd)
            return pd.Series(baseline.values, index=swap_ins.index).clip(0.0, 1.0)
        except Exception:
            pass

    # 2. Score-based decile calibration (fallback)
    # Identify primary score column (explicitly configured, or fallback to active cutoff score, or last score_cols)
    primary_score = policy.calibration_score_col

    if primary_score is None:
        from .stages import CutoffStage
        cutoff_cols = []
        for stage in policy.stages:
            if isinstance(stage, CutoffStage):
                cutoff_cols.extend(stage.cutoffs.keys())
        for sc in reversed(cutoff_cols):
            if sc in df.columns:
                primary_score = sc
                break

    if primary_score is None and policy.score_cols:
        for sc in reversed(policy.score_cols):
            if sc in df.columns:
                primary_score = sc
                break

    actual_default_col = policy.actual_default_col

    # Adjust min keep ins threshold if custom bins are specified (to allow small bases to run)
    min_keep_ins = 5 if policy.calibration_bins is not None else 50
    if primary_score is None or keep_ins_mask.sum() < min_keep_ins:
        global_pd = df.loc[keep_ins_mask, actual_default_col].mean() if keep_ins_mask.any() else 0.0
        if pd.isna(global_pd):
            global_pd = 0.0
        return pd.Series(global_pd, index=swap_ins.index)

    keep_in_scores = df.loc[keep_ins_mask, primary_score]
    keep_in_defaults = df.loc[keep_ins_mask, actual_default_col]
    global_pd = keep_in_defaults.mean()
    if pd.isna(global_pd):
        global_pd = 0.0

    # Determine reference score distribution for binning (Keep In or Global)
    if policy.calibration_base in ("global", "all", "dataset"):
        reference_scores = df[primary_score]
    else:
        reference_scores = keep_in_scores

    # Determine bins configuration
    cal_bins = policy.calibration_bins
    if cal_bins is None:
        # Legacy/Default dynamic binning
        n_bins = min(20, max(5, len(keep_in_scores) // 200))
    else:
        n_bins = cal_bins

    try:
        if isinstance(n_bins, int):
            _, bin_edges = pd.qcut(reference_scores, q=n_bins, retbins=True, duplicates="drop")
            # Extend edges slightly so that out-of-range values are clipped to nearest bin
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
        else:
            # It's a sequence of bin edges (list or tuple or numpy array)
            # Ensure outer boundaries are infinite to handle out of bounds values
            edges = list(n_bins)
            if edges[0] > -np.inf:
                edges.insert(0, -np.inf)
            if edges[-1] < np.inf:
                edges.append(np.inf)
            bin_edges = np.array(edges)

        keep_in_bins = pd.cut(keep_in_scores, bins=bin_edges, labels=False, include_lowest=True)
        # Group defaults by score bin
        bin_pd = keep_in_defaults.groupby(keep_in_bins).mean()

        # Ensure all bin indices are represented in the mapping (fill missing ones with global_pd)
        all_bin_indices = range(len(bin_edges) - 1)
        bin_pd = bin_pd.reindex(all_bin_indices).fillna(global_pd)

        swap_in_bins = pd.cut(
            swap_ins[primary_score], bins=bin_edges, labels=False, include_lowest=True
        )
        baseline = swap_in_bins.map(bin_pd)
        # Any remaining NaN → global_pd
        baseline = baseline.fillna(global_pd)
    except Exception:
        baseline = pd.Series(global_pd, index=swap_ins.index)

    return pd.Series(baseline.values, index=swap_ins.index).clip(0.0, 1.0)
