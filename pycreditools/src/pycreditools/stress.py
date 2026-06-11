from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import pandas as pd


class StressScenario(ABC):
    """Base class for stress scenarios."""

    @abstractmethod
    def apply(self, df: pd.DataFrame, pd_col: str) -> pd.Series:
        """Apply the stress scenario to the given DataFrame and PD column.

        Args:
            df: DataFrame containing the applicants (usually swap_ins).
            pd_col: The name of the column containing the baseline PD.

        Returns:
            pd.Series containing the stressed PDs (clipped to 0-1).
        """
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize the stress scenario to a dictionary."""
        pass

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StressScenario:
        """Deserialize a dictionary to a StressScenario object."""
        if "type" not in d:
            raise ValueError("Dictionary must contain a 'type' key.")

        t = d["type"]
        if t == "aggravation":
            return AggravationStress(factor=d.get("factor", 1.5), factor_col=d.get("factor_col"))
        elif t == "monotonic_increase":
            return MonotonicStress(
                score_col=d["score_col"], baseline=d["baseline"], factor=d["factor"]
            )
        elif t == "custom":
            raise ValueError("Custom stress scenarios cannot be deserialized automatically.")
        else:
            raise ValueError(f"Unknown stress scenario type: {t}")


class AggravationStress(StressScenario):
    """A stress scenario that multiplies the baseline PD by a factor."""

    def __init__(self, factor: float = 1.5, factor_col: str | None = None):
        """
        Args:
            factor: The aggravation multiplier (e.g., 1.5 = +50%).
            factor_col: Optional column name containing per-applicant dynamic factors.
        """
        self.factor = factor
        self.factor_col = factor_col

    def apply(self, df: pd.DataFrame, pd_col: str) -> pd.Series:
        if self.factor_col and self.factor_col in df.columns:
            stressed_pd = df[pd_col] * self.factor * df[self.factor_col]
        else:
            stressed_pd = df[pd_col] * self.factor

        return stressed_pd.clip(0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "aggravation",
            "factor": self.factor,
            "factor_col": self.factor_col,
        }


class MonotonicStress(StressScenario):
    """A stress scenario that applies a monotonic score-to-PD mapping."""

    def __init__(self, score_col: str, baseline: float, factor: float):
        """
        Args:
            score_col: The score column to map from.
            baseline: The baseline PD at score 0.
            factor: The rate at which PD decreases as score increases (per 1000 points).
        """
        self.score_col = score_col
        self.baseline = baseline
        self.factor = factor

    def apply(self, df: pd.DataFrame, pd_col: str) -> pd.Series:
        scores = df[self.score_col]
        stressed_pd = self.baseline - (scores / 1000.0) * self.factor
        return stressed_pd.clip(0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "monotonic_increase",
            "score_col": self.score_col,
            "baseline": self.baseline,
            "factor": self.factor,
        }


class CustomStress(StressScenario):
    """A stress scenario that uses a custom user-provided function."""

    def __init__(self, fn: Callable[[pd.DataFrame, str], pd.Series]):
        """
        Args:
            fn: A function that takes the DataFrame and the PD column name,
                and returns a pd.Series of stressed PDs.
        """
        self.fn = fn

    def apply(self, df: pd.DataFrame, pd_col: str) -> pd.Series:
        return self.fn(df, pd_col).clip(0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "custom",
            "fn": str(self.fn),
        }
