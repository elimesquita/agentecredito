from __future__ import annotations

from typing import Any

import pandas as pd


class Expression:
    """Base class for building logical column expressions."""

    def eval(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Subclasses must implement eval()")

    def get_columns(self) -> list[str]:
        raise NotImplementedError("Subclasses must implement get_columns()")

    def __gt__(self, other: Any) -> Expression:
        return BinaryExpr(self, ">", other)

    def __ge__(self, other: Any) -> Expression:
        return BinaryExpr(self, ">=", other)

    def __lt__(self, other: Any) -> Expression:
        return BinaryExpr(self, "<", other)

    def __le__(self, other: Any) -> Expression:
        return BinaryExpr(self, "<=", other)

    def __eq__(self, other: Any) -> Expression:  # type: ignore
        return BinaryExpr(self, "==", other)

    def __ne__(self, other: Any) -> Expression:  # type: ignore
        return BinaryExpr(self, "!=", other)

    def __and__(self, other: Expression) -> Expression:
        return BinaryExpr(self, "&", other)

    def __or__(self, other: Expression) -> Expression:
        return BinaryExpr(self, "|", other)

    def __invert__(self) -> Expression:
        return UnaryExpr(self, "~")

    def __add__(self, other: Any) -> Expression:
        return BinaryExpr(self, "+", other)

    def __sub__(self, other: Any) -> Expression:
        return BinaryExpr(self, "-", other)

    def __mul__(self, other: Any) -> Expression:
        return BinaryExpr(self, "*", other)

    def __truediv__(self, other: Any) -> Expression:
        return BinaryExpr(self, "/", other)

    def __radd__(self, other: Any) -> Expression:
        return BinaryExpr(other, "+", self)

    def __rsub__(self, other: Any) -> Expression:
        return BinaryExpr(other, "-", self)

    def __rmul__(self, other: Any) -> Expression:
        return BinaryExpr(other, "*", self)

    def __rtruediv__(self, other: Any) -> Expression:
        return BinaryExpr(other, "/", self)

    def calibrated(self) -> CalibratedExpression:
        """Create a calibrated version of this expression."""
        return CalibratedExpression(self)


class ColumnExpr(Expression):
    """An expression representing a DataFrame column."""

    def __init__(self, name: str):
        self.name = name

    def eval(self, df: pd.DataFrame) -> pd.Series:
        if self.name not in df.columns:
            raise KeyError(f"Column '{self.name}' not found in DataFrame.")
        return df[self.name]

    def get_columns(self) -> list[str]:
        return [self.name]

    def __repr__(self) -> str:
        return f"col('{self.name}')"


class BinaryExpr(Expression):
    """An expression comparing or operating on two values (at least one being an Expression)."""

    def __init__(self, left: Any, op: str, right: Any):
        self.left = left
        self.op = op
        self.right = right

    def eval(self, df: pd.DataFrame) -> pd.Series:
        left_val = self.left.eval(df) if isinstance(self.left, Expression) else self.left
        right_val = self.right.eval(df) if isinstance(self.right, Expression) else self.right

        if self.op == ">":
            return left_val > right_val
        elif self.op == ">=":
            return left_val >= right_val
        elif self.op == "<":
            return left_val < right_val
        elif self.op == "<=":
            return left_val <= right_val
        elif self.op == "==":
            return left_val == right_val
        elif self.op == "!=":
            return left_val != right_val
        elif self.op == "&":
            return left_val & right_val
        elif self.op == "|":
            return left_val | right_val
        elif self.op == "+":
            return left_val + right_val
        elif self.op == "-":
            return left_val - right_val
        elif self.op == "*":
            return left_val * right_val
        elif self.op == "/":
            return left_val / right_val
        else:
            raise ValueError(f"Unsupported operator: {self.op}")

    def get_columns(self) -> list[str]:
        cols = []
        if isinstance(self.left, Expression):
            cols.extend(self.left.get_columns())
        if isinstance(self.right, Expression):
            cols.extend(self.right.get_columns())
        return cols

    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


class UnaryExpr(Expression):
    """An expression with a unary operator (like NOT)."""

    def __init__(self, expr: Expression, op: str):
        self.expr = expr
        self.op = op

    def eval(self, df: pd.DataFrame) -> pd.Series:
        val = self.expr.eval(df)
        if self.op == "~":
            return ~val
        raise ValueError(f"Unsupported unary operator: {self.op}")

    def get_columns(self) -> list[str]:
        return self.expr.get_columns()

    def __repr__(self) -> str:
        return f"{self.op}({self.expr})"


class CalibratedExpression(Expression):
    """An expression that is evaluated and calibrated on the approved population,
    then mapped to all candidates via score bins.
    """

    def __init__(self, expression: Expression):
        self.expression = expression

    def eval(self, df: pd.DataFrame) -> pd.Series:
        # Fallback if evaluated directly outside a calibrated simulation
        return self.expression.eval(df)

    def get_columns(self) -> list[str]:
        return self.expression.get_columns()

    def calibrate_and_eval(self, df: pd.DataFrame, policy: Any) -> pd.Series:
        import numpy as np

        # 1. Identify approved mask
        approved_mask = df[policy.current_approval_col] == 1
        if not approved_mask.any():
            return pd.Series(0.0, index=df.index)

        # 2. Evaluate expression ONLY on the approved population
        keep_in_vals = self.expression.eval(df.loc[approved_mask])
        global_mean = keep_in_vals.mean()
        if pd.isna(global_mean):
            global_mean = 0.0

        # 3. Determine score column to use for calibration
        primary_score = policy.calibration_score_col
        if primary_score is None:
            # Fallback to active cutoff score, or fallback to last score_cols
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

        if primary_score is None or primary_score not in df.columns:
            return pd.Series(global_mean, index=df.index)

        # 4. Group by score bins of Keep In
        keep_in_scores = df.loc[approved_mask, primary_score]

        if policy.calibration_base in ("global", "all", "dataset"):
            reference_scores = df[primary_score]
        else:
            reference_scores = keep_in_scores

        cal_bins = policy.calibration_bins
        if cal_bins is None:
            n_bins = min(20, max(5, len(keep_in_scores) // 200))
        else:
            n_bins = cal_bins

        try:
            if isinstance(n_bins, int):
                _, bin_edges = pd.qcut(reference_scores, q=n_bins, retbins=True, duplicates="drop")
                bin_edges[0] = -np.inf
                bin_edges[-1] = np.inf
            else:
                edges = list(n_bins)
                if edges[0] > -np.inf:
                    edges.insert(0, -np.inf)
                if edges[-1] < np.inf:
                    edges.append(np.inf)
                bin_edges = np.array(edges)

            keep_in_bins = pd.cut(keep_in_scores, bins=bin_edges, labels=False, include_lowest=True)
            bin_means = keep_in_vals.groupby(keep_in_bins).mean()

            all_bin_indices = range(len(bin_edges) - 1)
            bin_means = bin_means.reindex(all_bin_indices).fillna(global_mean)

            df_bins = pd.cut(df[primary_score], bins=bin_edges, labels=False, include_lowest=True)
            probs = df_bins.map(bin_means).fillna(global_mean)
        except Exception:
            probs = pd.Series(global_mean, index=df.index)

        return pd.Series(probs.values, index=df.index)

    def __repr__(self) -> str:
        return f"{self.expression}.calibrated()"


def col(name: str) -> ColumnExpr:
    """Create a Column Expression builder.

    Example:
        `col("age") >= 18`
    """
    return ColumnExpr(name)


def serialize_expression(expr: Expression) -> dict[str, Any]:
    """Serialize an Expression object to a dict."""
    if isinstance(expr, ColumnExpr):
        return {"type": "column", "name": expr.name}
    elif isinstance(expr, BinaryExpr):
        left = serialize_expression(expr.left) if isinstance(expr.left, Expression) else expr.left
        right = (
            serialize_expression(expr.right) if isinstance(expr.right, Expression) else expr.right
        )
        return {"type": "binary", "left": left, "op": expr.op, "right": right}
    elif isinstance(expr, UnaryExpr):
        return {"type": "unary", "op": expr.op, "expr": serialize_expression(expr.expr)}
    elif isinstance(expr, CalibratedExpression):
        return {"type": "calibrated", "expression": serialize_expression(expr.expression)}
    else:
        raise ValueError(f"Unknown expression type: {type(expr)}")


def deserialize_expression(d: dict[str, Any]) -> Expression:
    """Deserialize a dict to an Expression object."""
    if not isinstance(d, dict) or "type" not in d:
        raise ValueError("Invalid serialized expression representation.")

    t = d["type"]
    if t == "column":
        return ColumnExpr(d["name"])
    elif t == "binary":
        left = (
            deserialize_expression(d["left"])
            if (isinstance(d["left"], dict) and "type" in d["left"])
            else d["left"]
        )
        right = (
            deserialize_expression(d["right"])
            if (isinstance(d["right"], dict) and "type" in d["right"])
            else d["right"]
        )
        return BinaryExpr(left, d["op"], right)
    elif t == "unary":
        return UnaryExpr(deserialize_expression(d["expr"]), d["op"])
    elif t == "calibrated":
        return CalibratedExpression(deserialize_expression(d["expression"]))
    else:
        raise ValueError(f"Unknown serialized expression type: {t}")
