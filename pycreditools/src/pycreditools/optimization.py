"""
Optimization engine for credit risk policy cutoffs and Pareto efficiency.
Ported from creditools R package.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import itertools
from dataclasses import dataclass
from typing import Any
import copy

from .policy import CreditPolicy
from .stages import CutoffStage

@dataclass
class OptimizationResult:
    """Result object for credit policy cutoff optimization."""
    best_combination: dict[str, float]
    metrics: dict[str, float]
    all_results: pd.DataFrame
    pareto_frontier: pd.DataFrame
    params: dict[str, Any]
    
    def __repr__(self) -> str:
        return (
            f"OptimizationResult(best_combination={self.best_combination}, "
            f"metrics={self.metrics}, n_combinations={len(self.all_results)})"
        )
        
    def to_dict(self) -> dict[str, Any]:
        """Serialize the best result to a dictionary."""
        return {
            "best_combination": self.best_combination,
            "metrics": self.metrics,
            "params": self.params,
        }
        
    def find_equivalent(
        self,
        target_metric: str = "approval_rate",
        target_value: float = 0.20,
        tolerance: float = 0.01,
    ) -> pd.DataFrame:
        """Find combinations in the grid that match a target metric within tolerance."""
        col_name = "overall_approval_rate" if "approval" in target_metric else "overall_default_rate"
        if col_name not in self.all_results.columns:
            for alternate in ["approval_rate", "default_rate"]:
                if alternate in self.all_results.columns and alternate in target_metric:
                    col_name = alternate
                    break
                    
        if col_name not in self.all_results.columns:
            raise ValueError(f"Column for target metric '{target_metric}' not found in results.")
            
        matches = self.all_results[np.abs(self.all_results[col_name] - target_value) <= tolerance].copy()
        matches["diff"] = np.abs(matches[col_name] - target_value)
        
        if matches.empty:
            closest = self.all_results.copy()
            closest["diff"] = np.abs(closest[col_name] - target_value)
            return closest.sort_values("diff").head(1)
            
        return matches.sort_values("diff").reset_index(drop=True)
        
    def plot(
        self,
        type: str = "tradeoff",
        save_path: str | None = None,
    ) -> Any:
        """Plot the tradeoff combinations space or the efficient Pareto frontier."""
        from .visualization import plot_optimization
        return plot_optimization(self, type=type, save_path=save_path)

def _evaluate_single_stochastic(args) -> dict[str, Any]:
    idx, combo, config, data, target_default_rate, min_approval_rate = args
    import dataclasses
    from .stages import CutoffStage
    
    # Create a temporary policy inserting the CutoffStage first
    temp_policy = copy.deepcopy(config)
    cutoff_stage = CutoffStage(name="opt_cutoff", cutoffs=combo)
    stages_list = [cutoff_stage] + list(temp_policy.stages)
    temp_policy = dataclasses.replace(temp_policy, stages=tuple(stages_list))
    
    sim_results = temp_policy.simulate(data, method="stochastic")
    sim_df = sim_results.data
    
    total = len(sim_df)
    approved = sim_df["new_approval"].sum()
    app_rate = approved / total if total > 0 else 0.0
    
    if approved > 0:
        def_rate = (sim_df["simulated_default"] * sim_df["new_approval"]).sum() / approved
    else:
        def_rate = 0.0
        
    constraints_met = (def_rate <= target_default_rate) and (app_rate >= min_approval_rate)
    tradeoff_score = app_rate - 5.0 * def_rate
    
    row = {
        "combination_id": idx + 1,
        "overall_approval_rate": app_rate,
        "overall_default_rate": def_rate,
        "constraints_met": constraints_met,
        "tradeoff_score": tradeoff_score
    }
    row.update(combo)
    return row

def optimize_cutoffs(
    data: pd.DataFrame,
    config: CreditPolicy,
    cutoff_steps: int = 10,
    target_default_rate: float = 0.05,
    min_approval_rate: float = 0.3,
    method: str = "analytical",
    parallel: bool = False,
    percentiles: tuple[float, float] | None = (0.05, 0.95),
    cutoff_ranges: dict[str, list[float]] | None = None,
) -> OptimizationResult:
    """Systematically evaluates a grid of cutoff combinations for a set of scores to find the optimal set of cutoffs.
    
    Args:
        data: Applicant data.
        config: Base CreditPolicy.
        cutoff_steps: Number of steps to generate for each score cutoff.
        target_default_rate: Maximum acceptable overall default rate.
        min_approval_rate: Minimum acceptable overall approval rate.
        method: "analytical" (fast expected values) or "stochastic" (row sampling).
        parallel: Whether to evaluate combinations in parallel (only used in stochastic).
        percentiles: Optional tuple specifying lower/upper quantiles to bound score ranges (default 5% to 95%).
        cutoff_ranges: Optional dictionary of pre-defined list of cutoffs for each score column.
        
    Returns:
        OptimizationResult object.
    """
    if not (0.0 <= target_default_rate <= 1.0):
        raise ValueError("target_default_rate must be between 0.0 and 1.0.")
    if not (0.0 <= min_approval_rate <= 1.0):
        raise ValueError("min_approval_rate must be between 0.0 and 1.0.")
    if cutoff_steps < 1:
        raise ValueError("cutoff_steps must be at least 1.")
    if data.empty:
        raise ValueError("Input data DataFrame is empty.")

    # 1. Generate cutoff ranges
    score_cols = list(config.score_cols)
    if not score_cols:
        raise ValueError("The policy must define at least one score column in score_cols to optimize cutoffs.")
        
    cutoff_ranges_dict = {}
    if cutoff_ranges is not None:
        cutoff_ranges_dict = cutoff_ranges
    else:
        for col in score_cols:
            if col not in data.columns:
                raise ValueError(f"Score column '{col}' not found in data.")
            vals = data[col].dropna()
            if vals.empty:
                raise ValueError(f"Score column '{col}' has only NaNs or is empty.")
            
            if percentiles is not None:
                min_val = float(np.floor(vals.quantile(percentiles[0])))
                max_val = float(np.ceil(vals.quantile(percentiles[1])))
            else:
                min_val = float(np.floor(vals.min()))
                max_val = float(np.ceil(vals.max()))
                
            if cutoff_steps == 1:
                cutoff_ranges_dict[col] = [float(vals.median())]
            else:
                cutoff_ranges_dict[col] = np.linspace(min_val, max_val, cutoff_steps).tolist()
            
    # Create grid of combinations
    keys = list(cutoff_ranges_dict.keys())
    values = list(cutoff_ranges_dict.values())
    grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # 2. Optimized analytical evaluation path
    if method == "analytical":
        # Create a base policy removing all CutoffStage objects
        p_static = copy.deepcopy(config)
        stages_list = [s for s in p_static.stages if not isinstance(s, CutoffStage)]
        import dataclasses
        p_static = dataclasses.replace(p_static, stages=tuple(stages_list))
        
        # Run simulation once
        sim_static = p_static.simulate(data, method="analytical")
        
        p_base = sim_static.data["new_approval"].values
        # Get simulated_default. Fallback to actual_default_col if NA
        pd_base = sim_static.data["simulated_default"].values
        actual_defaults = data[config.actual_default_col].values
        
        nas = np.isnan(pd_base)
        if nas.any():
            pd_base = np.where(nas, actual_defaults, pd_base)
        pd_base = np.nan_to_num(pd_base, nan=0.0)
        
        N = len(data)
        results_list = []
        
        # Vectorized grid evaluation
        score_arrays = {col: data[col].values for col in keys}
        
        for i, combo in enumerate(grid):
            # Evaluate combination (vectorized)
            is_above = np.ones(N, dtype=bool)
            for col, val in combo.items():
                is_above &= (score_arrays[col] >= val)
                
            p_final = is_above * p_base
            sum_p = float(p_final.sum())
            app_rate = sum_p / N
            
            if sum_p > 0:
                def_rate = float((p_final * pd_base).sum() / sum_p)
            else:
                def_rate = 0.0
                
            constraints_met = (def_rate <= target_default_rate) and (app_rate >= min_approval_rate)
            tradeoff_score = app_rate - 5.0 * def_rate
            
            row = {
                "combination_id": i + 1,
                "overall_approval_rate": app_rate,
                "overall_default_rate": def_rate,
                "constraints_met": constraints_met,
                "tradeoff_score": tradeoff_score
            }
            row.update(combo)
            results_list.append(row)
            
        all_results = pd.DataFrame(results_list)
        
    else:
        # Stochastic/Full simulation path
        if parallel:
            from ._parallel import parallel_map
            tasks = [(i, c, config, data, target_default_rate, min_approval_rate) for i, c in enumerate(grid)]
            results_list = parallel_map(_evaluate_single_stochastic, tasks, parallel=True)
        else:
            results_list = [
                _evaluate_single_stochastic((i, c, config, data, target_default_rate, min_approval_rate))
                for i, c in enumerate(grid)
            ]
            
        all_results = pd.DataFrame(results_list)
        
    # 3. Find optimal result
    valid = all_results[all_results["constraints_met"]]
    if valid.empty:
        best_row = all_results.sort_values("tradeoff_score", ascending=False).iloc[0]
    else:
        best_row = valid.sort_values("overall_approval_rate", ascending=False).iloc[0]
        
    best_combo = {col: float(best_row[col]) for col in keys}
    best_metrics = {
        "overall_approval_rate": float(best_row["overall_approval_rate"]),
        "overall_default_rate": float(best_row["overall_default_rate"]),
        "tradeoff_score": float(best_row["tradeoff_score"]),
        "constraints_met": bool(best_row["constraints_met"])
    }
    
    # 4. Find Pareto frontier
    pareto_frontier = find_pareto_frontier(all_results)
    
    params = {
        "target_default_rate": target_default_rate,
        "min_approval_rate": min_approval_rate,
        "cutoff_steps": cutoff_steps,
        "method": method
    }
    
    return OptimizationResult(
        best_combination=best_combo,
        metrics=best_metrics,
        all_results=all_results,
        pareto_frontier=pareto_frontier,
        params=params
    )

def find_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Pareto efficient combinations from the results.
    
    We want to maximize approval rate and minimize default rate.
    """
    clean_df = df.dropna(subset=["overall_approval_rate", "overall_default_rate"])
    if clean_df.empty:
        return pd.DataFrame(columns=df.columns)
        
    points = clean_df[["overall_approval_rate", "overall_default_rate"]].drop_duplicates().values
    pareto_points = []
    
    for i, p1 in enumerate(points):
        dominated = False
        for j, p2 in enumerate(points):
            if i == j:
                continue
            # p2 dominates p1 if approval(p2) >= approval(p1) and default(p2) <= default(p1)
            # with at least one strict inequality.
            if (p2[0] >= p1[0] and p2[1] <= p1[1]) and (p2[0] > p1[0] or p2[1] < p1[1]):
                dominated = True
                break
        if not dominated:
            pareto_points.append(p1)
            
    if not pareto_points:
        return pd.DataFrame(columns=df.columns)
        
    pareto_df = pd.DataFrame(pareto_points, columns=["overall_approval_rate", "overall_default_rate"])
    res = pd.merge(df, pareto_df, on=["overall_approval_rate", "overall_default_rate"], how="inner")
    return res.sort_values("overall_approval_rate").reset_index(drop=True)


