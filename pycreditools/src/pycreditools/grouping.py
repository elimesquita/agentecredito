from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._kernels import iv_cluster, ward_cluster


@dataclass
class GroupingRecipe:
    """Serializable recipe for applying grouping to new data."""

    score_cols: list[str]
    quantile_breaks: dict[str, list[float]] = None
    cluster_mapping: dict[str, int] = None
    intervals: list[dict[str, Any]] = None
    segmented_intervals: dict[str, list[dict[str, Any]]] = None
    segment_col: str = None

    def to_dict(self) -> dict[str, Any]:
        d = {"score_cols": self.score_cols}
        if self.quantile_breaks is not None:
            d["quantile_breaks"] = self.quantile_breaks
        if self.cluster_mapping is not None:
            d["cluster_mapping"] = self.cluster_mapping
        if self.intervals is not None:
            d["intervals"] = self.intervals
        if self.segmented_intervals is not None:
            d["segmented_intervals"] = self.segmented_intervals
        if self.segment_col is not None:
            d["segment_col"] = self.segment_col
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GroupingRecipe:
        mapping = None
        if "cluster_mapping" in d and d["cluster_mapping"] is not None:
            mapping = {str(k): v for k, v in d["cluster_mapping"].items()}
        return cls(
            score_cols=d["score_cols"],
            quantile_breaks=d.get("quantile_breaks"),
            cluster_mapping=mapping,
            intervals=d.get("intervals"),
            segmented_intervals=d.get("segmented_intervals"),
            segment_col=d.get("segment_col"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> GroupingRecipe:
        return cls.from_dict(json.loads(s))

    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Apply the grouping recipe to map scores to group numbers."""
        df = new_data.copy()
        df["risk_rating"] = np.nan

        if self.intervals is not None:
            score_col = self.score_cols[0]
            rating_to_group = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
            for f in self.intervals:
                rat = f["rating"]
                g_id = rating_to_group.get(rat, 5)
                mask = (df[score_col] >= f["min_score"]) & (df[score_col] <= f["max_score"])
                df.loc[mask, "risk_rating"] = g_id

        elif self.segmented_intervals is not None:
            score_col = self.score_cols[0]
            segment_col = self.segment_col or "region"
            rating_to_group = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
            for seg, faixas in self.segmented_intervals.items():
                mask_seg = df[segment_col] == seg
                for f in faixas:
                    rat = f["rating"]
                    g_id = rating_to_group.get(rat, 5)
                    mask = (
                        mask_seg
                        & (df[score_col] >= f["min_score"])
                        & (df[score_col] <= f["max_score"])
                    )
                    df.loc[mask, "risk_rating"] = g_id

        else:
            # 1. Apply quantile breaks
            bin_cols = []
            for col in self.score_cols:
                breaks = self.quantile_breaks[col]
                # Use np.digitize. np.digitize returns indices 1..len(bins)
                # We want 0-based for string mapping
                bin_idx = np.digitize(df[col], bins=breaks[1:-1])
                bin_col = f"{col}_bin"
                df[bin_col] = bin_idx
                bin_cols.append(bin_col)

            # 2. Combine to micro-bin keys
            if len(bin_cols) == 1:
                keys = df[bin_cols[0]].astype(str)
            else:
                keys = df[bin_cols].astype(str).agg("-".join, axis=1)

            # 3. Map to groups
            df["risk_rating"] = keys.map(self.cluster_mapping)

            # Cleanup
            for bc in bin_cols:
                if bc in df.columns:
                    del df[bc]

        return df


@dataclass
class RiskGroupResult:
    data: pd.DataFrame
    groups: pd.DataFrame
    recipe: GroupingRecipe
    n_groups: int
    params: dict[str, Any]
    report: pd.DataFrame = None

    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        return self.recipe.predict(new_data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe.to_dict(),
            "n_groups": self.n_groups,
            "params": self.params,
        }


import warnings


def fit_risk_groups(
    data: pd.DataFrame,
    score_cols: str | list[str],
    default_col: str,
    bins: int = 20,
    max_groups: int | None = None,
    min_vol_ratio: float = 0.05,
    max_crossings: int = 1,
    time_col: str | None = None,
    method: str = "ward",
    oot_date: Any | None = None,
) -> RiskGroupResult:
    """Find stable risk groups using clustering.

    Args:
        data: Applicant data.
        score_cols: Score column name or list of score column names.
        default_col: Target default column name (0/1).
        bins: Granularity of initial quantile grid.
        max_groups: Maximum number of final risk groups (ratings).
        min_vol_ratio: Minimum ratio of applicants in each group (e.g. 0.05 = 5%).
        max_crossings: Maximum crossings tolerated before forcing cluster merge.
        time_col: Column containing Safra/date.
        method: 'ward' or 'iv'.
        oot_date: Out-of-Time split date. Data time_col < oot_date is train, >= oot_date is OOT.

    Returns:
        RiskGroupResult object.
    """
    if isinstance(score_cols, str):
        score_cols = [score_cols]

    if max_groups is None:
        max_groups = 5

    if max_groups > bins:
        raise ValueError(f"max_groups ({max_groups}) cannot be greater than bins ({bins}).")

    # Validate columns
    required = list(score_cols) + [default_col]
    if time_col:
        required.append(time_col)
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    # Split train & OOT
    if oot_date is not None and time_col is not None:
        train_data = data[data[time_col] < oot_date].copy()
        oot_data = data[data[time_col] >= oot_date].copy()
    else:
        train_data = data.copy()
        oot_data = pd.DataFrame()

    if train_data.empty:
        raise ValueError(
            "Training dataset (before oot_date) is empty. Check time_col and oot_date."
        )

    # 1. Quantile binning on train_data
    breaks_dict = {}
    bin_cols = []

    # We will apply breaks to the full data later
    full_df = data.copy()

    for col in score_cols:
        quantiles = np.linspace(0, 1, bins + 1)
        q_breaks = np.unique(np.quantile(train_data[col].dropna(), quantiles))

        if len(q_breaks) < 2:
            raise ValueError(f"Column {col} does not have enough variance in training data to bin.")

        breaks_dict[col] = q_breaks.tolist()

        # Digitize train and full
        train_data[f"{col}_bin"] = np.digitize(train_data[col], q_breaks[1:-1])
        full_df[f"{col}_bin"] = np.digitize(full_df[col], q_breaks[1:-1])
        bin_cols.append(f"{col}_bin")

    # 2. Compute PD per combo on train_data
    if len(bin_cols) == 1:
        train_data["_combo_key"] = train_data[bin_cols[0]].astype(str)
        full_df["_combo_key"] = full_df[bin_cols[0]].astype(str)
    else:
        train_data["_combo_key"] = train_data[bin_cols].astype(str).agg("-".join, axis=1)
        full_df["_combo_key"] = full_df[bin_cols].astype(str).agg("-".join, axis=1)

    # Aggregate combinations on train_data
    agg = (
        train_data.groupby("_combo_key")
        .agg(volume=(default_col, "count"), bads=(default_col, "sum"))
        .reset_index()
    )

    agg["pd"] = agg["bads"] / agg["volume"].replace(0, 1)

    # Sort from lowest risk to highest risk
    if len(score_cols) == 1:
        agg["_bin_int"] = agg["_combo_key"].astype(int)
        agg = agg.sort_values("_bin_int", ascending=False).reset_index(drop=True)
        del agg["_bin_int"]
    else:
        agg = agg.sort_values("pd").reset_index(drop=True)

    pd_values = agg["pd"].values
    volumes = agg["volume"].values
    keys = agg["_combo_key"].values

    # Time matrices based on train_data
    monthly_vols = None
    monthly_bads = None

    if time_col is not None and time_col in train_data.columns:
        # Pivot on train_data
        pivot_vol = train_data.pivot_table(
            index="_combo_key", columns=time_col, values=default_col, aggfunc="count", fill_value=0
        )
        pivot_bads = train_data.pivot_table(
            index="_combo_key", columns=time_col, values=default_col, aggfunc="sum", fill_value=0
        )

        pivot_vol = pivot_vol.reindex(keys).fillna(0).values
        pivot_bads = pivot_bads.reindex(keys).fillna(0).values

        monthly_vols = pivot_vol.astype(np.float64)
        monthly_bads = pivot_bads.astype(np.float64)

    # 3. Cluster
    if method in ("ward", "distance"):
        assignments = ward_cluster(
            pd_values=pd_values,
            volumes=volumes.astype(np.float64),
            max_groups=max_groups,
            min_vol_ratio=min_vol_ratio,
            max_crossings=max_crossings,
            use_volume_weights=(method == "ward"),
            monthly_vols=monthly_vols,
            monthly_bads=monthly_bads,
        )
    elif method == "iv":
        assignments = iv_cluster(
            pd_values=pd_values,
            volumes=volumes.astype(np.float64),
            max_groups=max_groups,
            min_vol_ratio=min_vol_ratio,
            monthly_vols=monthly_vols,
            monthly_bads=monthly_bads,
        )
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # 4. Map back to full_df and train_data
    cluster_mapping = dict(zip(keys, assignments.tolist()))
    full_df["risk_rating"] = full_df["_combo_key"].map(cluster_mapping)
    train_data["risk_rating"] = train_data["_combo_key"].map(cluster_mapping)

    # 5. Summary on full_df
    groups_summary = (
        full_df.groupby("risk_rating")
        .agg(volume=(default_col, "count"), pd=(default_col, "mean"))
        .reset_index()
    )

    # 6. Detailed Train / OOT stability report
    if not oot_data.empty:
        oot_mapped = oot_data.copy()
        # Bin and map OOT
        for col in score_cols:
            q_breaks = breaks_dict[col]
            oot_mapped[f"{col}_bin"] = np.digitize(oot_mapped[col], q_breaks[1:-1])
        if len(score_cols) == 1:
            oot_mapped["_combo_key"] = oot_mapped[f"{score_cols[0]}_bin"].astype(str)
        else:
            bin_cols_oot = [f"{c}_bin" for c in score_cols]
            oot_mapped["_combo_key"] = oot_mapped[bin_cols_oot].astype(str).agg("-".join, axis=1)
        oot_mapped["risk_rating"] = oot_mapped["_combo_key"].map(cluster_mapping)

        train_rep = (
            train_data.groupby("risk_rating")
            .agg(volume=(default_col, "count"), pd=(default_col, "mean"))
            .reset_index()
        )
        train_rep["period"] = "Train"

        oot_rep = (
            oot_mapped.groupby("risk_rating")
            .agg(volume=(default_col, "count"), pd=(default_col, "mean"))
            .reset_index()
        )
        oot_rep["period"] = "OOT"

        report_df = pd.concat([train_rep, oot_rep], ignore_index=True)
    else:
        train_rep = (
            train_data.groupby("risk_rating")
            .agg(volume=(default_col, "count"), pd=(default_col, "mean"))
            .reset_index()
        )
        train_rep["period"] = "Train"
        report_df = train_rep

    recipe = GroupingRecipe(
        score_cols=score_cols, quantile_breaks=breaks_dict, cluster_mapping=cluster_mapping
    )

    params = {
        "bins": bins,
        "max_groups": max_groups,
        "min_vol_ratio": min_vol_ratio,
        "method": method,
        "oot_date": oot_date,
    }

    # Cleanup keys and bins
    for bc in bin_cols + ["_combo_key"]:
        if bc in full_df.columns:
            del full_df[bc]

    return RiskGroupResult(
        data=full_df,
        groups=groups_summary,
        recipe=recipe,
        n_groups=len(np.unique(assignments)),
        params=params,
        report=report_df,
    )


def fit_pairwise_risk_groups(
    data: pd.DataFrame,
    primary_score: str,
    challenger_scores: list[str],
    default_col: str,
    time_col: str | None = None,
    bins: int = 20,
    max_groups: int | None = None,
    min_vol_ratio: float = 0.05,
    max_crossings: int = 1,
    method: str = "ward",
    oot_date: Any | None = None,
    parallel: bool = False,
) -> dict[str, RiskGroupResult]:
    """Compare a primary score against multiple challenger scores by running risk grouping in pairs.

    Args:
        data: Applicant data.
        primary_score: Legacy or baseline score column name.
        challenger_scores: List of challenger score column names.
        default_col: Target default column name (0/1).
        time_col: Column containing Safra/date.
        bins: Granularity of initial quantile grid.
        max_groups: Maximum number of final risk groups (ratings).
        min_vol_ratio: Minimum ratio of applicants in each group (e.g. 0.05 = 5%).
        max_crossings: Maximum crossings tolerated before forcing cluster merge.
        method: 'ward' or 'iv'.
        oot_date: Out-of-Time split date.
        parallel: Whether to run comparisons in parallel.

    Returns:
        Dict mapping '{primary_score}_vs_{challenger}' to RiskGroupResult.
    """
    results = {}

    def _run_pair(challenger: str) -> tuple[str, RiskGroupResult]:
        score_cols = [primary_score, challenger]
        res = fit_risk_groups(
            data=data,
            score_cols=score_cols,
            default_col=default_col,
            bins=bins,
            max_groups=max_groups,
            min_vol_ratio=min_vol_ratio,
            max_crossings=max_crossings,
            time_col=time_col,
            method=method,
            oot_date=oot_date,
        )
        key = f"{primary_score}_vs_{challenger}"
        return key, res

    if parallel:
        from ._parallel import parallel_map

        pairs = parallel_map(_run_pair, challenger_scores, parallel=True)
        results = dict(pairs)
    else:
        for challenger in challenger_scores:
            key, res = _run_pair(challenger)
            results[key] = res

    return results


def find_risk_groups(*args, **kwargs) -> RiskGroupResult:
    """Deprecated alias for fit_risk_groups."""
    warnings.warn(
        "find_risk_groups is deprecated and will be removed in a future version. "
        "Use fit_risk_groups instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fit_risk_groups(*args, **kwargs)


