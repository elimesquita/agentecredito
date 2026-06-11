from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any

_CALLABLE_REGISTRY: dict[str, callable] = {}

def register_callable(name: str, func: callable) -> None:
    """Register a custom function so it can be resolved during deserialization."""
    _CALLABLE_REGISTRY[name] = func

def _resolve_callable(name: str) -> callable:
    if name in _CALLABLE_REGISTRY:
        return _CALLABLE_REGISTRY[name]
        
    import sys
    try:
        frame = sys._getframe(0)
        while frame:
            if name in frame.f_locals:
                val = frame.f_locals[name]
                if callable(val):
                    return val
            if name in frame.f_globals:
                val = frame.f_globals[name]
                if callable(val):
                    return val
            frame = frame.f_back
    except Exception:
        pass
        
    def placeholder(df):
        raise ValueError(
            f"Custom function '{name}' was not found in the environment. "
            f"Please define it in your script or register it using "
            f"pycreditools.stages.register_callable('{name}', func)."
        )
    placeholder.__name__ = name
    return placeholder

class Stage(ABC):
    """Base class for credit policy stages."""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def apply(self, df: pd.DataFrame, method: str = "analytical", policy: Any | None = None) -> pd.Series:
        """Apply the stage rule to the DataFrame.
        
        Args:
            df: The applicant data.
            method: "analytical" (returns float probabilities) or "stochastic" (returns 0/1 integers).
            policy: Optional CreditPolicy containing metadata.
            
        Returns:
            pd.Series containing the pass status for each row (0.0/1.0 or 0/1).
        """
        pass
        
    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize the stage to a dictionary."""
        pass
        
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Stage:
        """Deserialize a dictionary to a Stage object."""
        if "type" not in d:
            raise ValueError("Dictionary must contain a 'type' key.")
            
        t = d["type"]
        if t == "cutoff":
            return CutoffStage(
                name=d["name"], 
                cutoffs=d["cutoffs"], 
                direction=d.get("direction", "gte")
            )
        elif t == "filter":
            cond = d["condition"]
            if isinstance(cond, dict):
                if cond.get("type") == "callable":
                    cond = _resolve_callable(cond["name"])
                else:
                    from .expressions import deserialize_expression
                    cond = deserialize_expression(cond)
            return FilterStage(name=d["name"], condition=cond)
        elif t == "rate":
            var_data = d.get("variable")
            if isinstance(var_data, dict):
                if var_data.get("type") == "callable":
                    var_data = _resolve_callable(var_data["name"])
                else:
                    from .expressions import deserialize_expression
                    var_data = deserialize_expression(var_data)
            return RateStage(
                name=d["name"], 
                base_rate=d["base_rate"], 
                variable=var_data,
                calibrate=d.get("calibrate", False)
            )
        else:
            raise ValueError(f"Unknown stage type: {t}")

class CutoffStage(Stage):
    """A stage that requires specific columns to meet or exceed a cutoff value."""
    
    def __init__(self, name: str, cutoffs: dict[str, float], direction: str = "gte"):
        """
        Args:
            name: Stage name.
            cutoffs: Dictionary mapping column names to cutoff values.
            direction: "gte" (>=) or "lte" (<=).
        """
        super().__init__(name)
        self.cutoffs = cutoffs
        self.direction = direction
        
    def apply(self, df: pd.DataFrame, method: str = "analytical", policy: Any | None = None) -> pd.Series:
        # Start with all Trues
        result = pd.Series(True, index=df.index)
        
        for col, val in self.cutoffs.items():
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in data for cutoff stage '{self.name}'.")
                
            if self.direction == "gte":
                result &= (df[col] >= val)
            else:
                result &= (df[col] <= val)
                
        # Fill NAs with False
        result = result.fillna(False)
        
        if method == "stochastic":
            return result.astype(int)
        else:
            return result.astype(float)
            
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "cutoff",
            "name": self.name,
            "cutoffs": self.cutoffs,
            "direction": self.direction,
        }

from .expressions import Expression

class FilterStage(Stage):
    """A stage that filters based on an expression, a callable, or a string condition."""
    
    def __init__(self, name: str, condition: str | callable | Expression):
        """
        Args:
            name: Stage name.
            condition: Can be:
                - A string query evaluated by pandas (e.g. 'age >= 18').
                - A pycreditools.Expression (e.g. col('age') >= 18).
                - A callable that takes a DataFrame and returns a boolean Series.
        """
        super().__init__(name)
        self.condition = condition
        
    def apply(self, df: pd.DataFrame, method: str = "analytical", policy: Any | None = None) -> pd.Series:
        try:
            if isinstance(self.condition, Expression):
                result = self.condition.eval(df)
            elif callable(self.condition):
                result = self.condition(df)
            else:
                # Fallback to string evaluation
                result = df.eval(self.condition)
                
            if not isinstance(result, pd.Series):
                result = pd.Series(result, index=df.index)
        except Exception as e:
            cond_repr = repr(self.condition)
            raise ValueError(f"Failed to evaluate filter condition {cond_repr}: {e}")
            
        result = result.fillna(False)
        
        if method == "stochastic":
            return result.astype(int)
        else:
            return result.astype(float)
            
    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.condition, Expression):
            from .expressions import serialize_expression
            cond_data = serialize_expression(self.condition)
        elif callable(self.condition):
            cond_data = {
                "type": "callable",
                "name": getattr(self.condition, "__name__", str(self.condition))
            }
        else:
            cond_data = str(self.condition)
            
        return {
            "type": "filter",
            "name": self.name,
            "condition": cond_data,
        }

class RateStage(Stage):
    """A stage that applies a probability of passing."""
    
    def __init__(
        self,
        name: str,
        base_rate: float,
        variable: str | float | Expression | callable | None = None,
        calibrate: bool = False,
    ):
        """
        Args:
            name: Stage name.
            base_rate: The base probability of passing (0.0 to 1.0).
            variable: Optional column name, expression, callable, or numeric multiplier for the base rate.
            calibrate: If True, wraps expression variable in CalibratedExpression.
        """
        super().__init__(name)
        self.base_rate = base_rate
        self.calibrate = calibrate
        
        from .expressions import CalibratedExpression, Expression
        if calibrate and variable is not None:
            if isinstance(variable, Expression) and not isinstance(variable, CalibratedExpression):
                self.variable = CalibratedExpression(variable)
            else:
                self.variable = variable
        else:
            self.variable = variable
            
    def apply(self, df: pd.DataFrame, method: str = "analytical", policy: Any | None = None) -> pd.Series:
        from .expressions import CalibratedExpression, Expression
        
        # 1. Compute probabilities based on self.variable
        if self.variable is not None:
            if isinstance(self.variable, Expression):
                if isinstance(self.variable, CalibratedExpression):
                    probs = (self.base_rate * self.variable.calibrate_and_eval(df, policy)).clip(0.0, 1.0)
                else:
                    probs = (self.base_rate * self.variable.eval(df)).clip(0.0, 1.0)
            elif callable(self.variable):
                probs = (self.base_rate * self.variable(df)).clip(0.0, 1.0)
            elif isinstance(self.variable, str) and self.variable in df.columns:
                probs = (self.base_rate * df[self.variable]).clip(0.0, 1.0)
            else:
                try:
                    mult = float(self.variable)
                    probs = pd.Series(np.clip(self.base_rate * mult, 0.0, 1.0), index=df.index)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Variable '{self.variable}' must be a column name, expression, callable, or a numeric value."
                    )
        else:
            probs = pd.Series(self.base_rate, index=df.index)
            
        if not isinstance(probs, pd.Series):
            probs = pd.Series(probs, index=df.index)
            
        probs = probs.fillna(0.0)
        
        # 2. Determine if this is the conversion stage
        is_conversion_stage = False
        hired_col = policy.current_hired_col if policy is not None else None
        if hired_col is None and "hired" in df.columns:
            hired_col = "hired"
            
        if hired_col is not None and hired_col in df.columns:
            if self.calibrate:
                is_conversion_stage = True
            elif self.name.lower() in ("conversao", "conversion", "hired", "take_up", "take_up_rate"):
                is_conversion_stage = True
            else:
                rate_stages = [s for s in (policy.stages if policy is not None else []) if isinstance(s, RateStage)]
                if rate_stages and rate_stages[-1] is self:
                    is_conversion_stage = True
                    
        # 3. Apply Keep In bypass / deterministic behavior if policy is provided
        if policy is not None and policy.current_approval_col in df.columns:
            keep_ins_mask = df[policy.current_approval_col] == 1
            if keep_ins_mask.any():
                probs = probs.copy()
                if is_conversion_stage:
                    probs.loc[keep_ins_mask] = df.loc[keep_ins_mask, hired_col].fillna(0.0)
                else:
                    probs.loc[keep_ins_mask] = 1.0
                    
        if method == "stochastic":
            if policy is not None and policy.current_approval_col in df.columns:
                keep_ins_mask = df[policy.current_approval_col] == 1
                outcomes = pd.Series(0, index=df.index, dtype=int)
                
                swap_ins_mask = ~keep_ins_mask
                if swap_ins_mask.any():
                    random_draws = np.random.random(swap_ins_mask.sum())
                    outcomes.loc[swap_ins_mask] = (random_draws < probs.loc[swap_ins_mask]).astype(int)
                    
                if keep_ins_mask.any():
                    if is_conversion_stage:
                        outcomes.loc[keep_ins_mask] = df.loc[keep_ins_mask, hired_col].fillna(0.0).astype(int)
                    else:
                        outcomes.loc[keep_ins_mask] = 1
                return outcomes
            else:
                random_draws = np.random.random(len(df))
                return (random_draws < probs).astype(int)
        else:
            return probs.astype(float)
            
    def to_dict(self) -> dict[str, Any]:
        from .expressions import Expression, serialize_expression
        var_data = self.variable
        if isinstance(self.variable, Expression):
            var_data = serialize_expression(self.variable)
        elif callable(self.variable):
            var_data = {
                "type": "callable",
                "name": getattr(self.variable, "__name__", str(self.variable))
            }
        return {
            "type": "rate",
            "name": self.name,
            "base_rate": self.base_rate,
            "variable": var_data,
            "calibrate": self.calibrate,
        }
