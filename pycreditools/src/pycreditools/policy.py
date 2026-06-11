from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .grouping import GroupingRecipe
from .stages import CutoffStage, FilterStage, RateStage, Stage
from .stress import AggravationStress, StressScenario


@dataclass(frozen=True)
class CreditPolicy:
    """Configuration for a credit policy simulation.

    This class is immutable. Methods like `add_stage` and `add_stress` return
    a new instance of CreditPolicy rather than modifying the current one.
    """

    applicant_id_col: str
    score_cols: tuple[str, ...]
    current_approval_col: str
    actual_default_col: str
    time_col: str | None = None
    stages: tuple[Stage, ...] = field(default_factory=tuple)
    stress_scenarios: tuple[StressScenario, ...] = field(default_factory=tuple)
    rating_recipe: GroupingRecipe | dict[str, GroupingRecipe] | None = None
    calibration_score_col: str | None = None
    calibration_bins: int | tuple[float, ...] | list[float] | None = None
    calibration_base: str = "keep_in"
    current_hired_col: str | None = None


    def __post_init__(self) -> None:
        # Coerce score_cols to tuple of strings
        if isinstance(self.score_cols, str):
            score_tuple = (self.score_cols,)
        else:
            try:
                score_tuple = tuple(self.score_cols)
            except TypeError:
                score_tuple = (str(self.score_cols),)

        object.__setattr__(self, "score_cols", score_tuple)

        # Ensure stages and stress_scenarios are also tuples
        if not isinstance(self.stages, tuple):
            object.__setattr__(self, "stages", tuple(self.stages))
        if not isinstance(self.stress_scenarios, tuple):
            object.__setattr__(self, "stress_scenarios", tuple(self.stress_scenarios))

        # Coerce calibration_bins to tuple if list/iterable
        if self.calibration_bins is not None and not isinstance(self.calibration_bins, int):
            try:
                object.__setattr__(self, "calibration_bins", tuple(self.calibration_bins))
            except TypeError:
                pass

    def add_stage(self, stage: Stage) -> CreditPolicy:
        """Return a new CreditPolicy with the given stage appended."""
        new_stages = self.stages + (stage,)
        return self._replace(stages=new_stages)

    def add_stress(self, scenario: StressScenario) -> CreditPolicy:
        """Return a new CreditPolicy with the given stress scenario appended."""
        new_scenarios = self.stress_scenarios + (scenario,)
        return self._replace(stress_scenarios=new_scenarios)

    def with_rating(
        self, rating_recipe: GroupingRecipe | dict[str, GroupingRecipe] | None
    ) -> CreditPolicy:
        """Return a new CreditPolicy with the given rating recipe."""
        return self._replace(rating_recipe=rating_recipe)

    def with_calibration(
        self,
        score_col: str | None = None,
        bins: int | tuple[float, ...] | list[float] | None = None,
        base: str = "keep_in",
    ) -> CreditPolicy:
        """Return a new CreditPolicy with updated calibration settings."""
        return self._replace(
            calibration_score_col=score_col,
            calibration_bins=bins,
            calibration_base=base,
        )

    def _replace(self, **kwargs) -> CreditPolicy:

        """Helper to create a new instance with replaced fields."""
        import dataclasses

        return dataclasses.replace(self, **kwargs)

    # --- Fluid Builder Methods ---

    def cutoff(self, name: str, cutoffs: dict[str, float], direction: str = "gte") -> CreditPolicy:
        """Add a CutoffStage to the policy."""
        return self.add_stage(CutoffStage(name=name, cutoffs=cutoffs, direction=direction))

    def filter(self, name: str, condition: Any) -> CreditPolicy:
        """Add a FilterStage to the policy. Condition can be an Expression, callable, or string."""
        return self.add_stage(FilterStage(name=name, condition=condition))

    def rate(
        self,
        name: str,
        base_rate: float,
        variable: str | float | None = None,
        calibrate: bool = False,
    ) -> CreditPolicy:
        """Add a RateStage to the policy."""
        return self.add_stage(
            RateStage(name=name, base_rate=base_rate, variable=variable, calibrate=calibrate)
        )

    def stress_aggravation(self, factor: float) -> CreditPolicy:
        """Add an AggravationStress scenario."""
        return self.add_stress(AggravationStress(factor=factor))

    def simulate(self, df: pd.DataFrame, method: str = "analytical") -> Any:
        """Run simulation directly from the policy object.

        Args:
            df: The applicant data.
            method: "analytical" or "stochastic".

        Returns:
            CreditSimResults object.
        """
        from .simulation import SimulationMethod, run_simulation

        sim_method = (
            SimulationMethod.ANALYTICAL if method == "analytical" else SimulationMethod.STOCHASTIC
        )
        return run_simulation(df, self, method=sim_method)

    def validate(self, df: pd.DataFrame) -> None:
        """Validate that the policy can be run on the given DataFrame.

        Raises:
            ValueError: If required columns are missing.
        """
        required_cols = list(self.score_cols) + [
            self.applicant_id_col,
            self.current_approval_col,
            self.actual_default_col,
        ]

        # Add columns from stages
        from .expressions import Expression
        from .stages import CutoffStage, FilterStage

        for stage in self.stages:
            if isinstance(stage, CutoffStage):
                required_cols.extend(stage.cutoffs.keys())
            elif isinstance(stage, FilterStage) and isinstance(stage.condition, Expression):
                required_cols.extend(stage.condition.get_columns())

        if self.time_col is not None:
            required_cols.append(self.time_col)

        required_cols = list(set(required_cols))  # deduplicate
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns in data: {missing}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy to a dictionary."""
        recipe_data = None
        if self.rating_recipe is not None:
            if isinstance(self.rating_recipe, dict):
                recipe_data = {
                    "type": "segmented_recipes",
                    "recipes": {k: v.to_dict() for k, v in self.rating_recipe.items()},
                }
            else:
                recipe_data = self.rating_recipe.to_dict()

        return {
            "applicant_id_col": self.applicant_id_col,
            "score_cols": self.score_cols,
            "current_approval_col": self.current_approval_col,
            "actual_default_col": self.actual_default_col,
            "time_col": self.time_col,
            "stages": [s.to_dict() for s in self.stages],
            "stress_scenarios": [s.to_dict() for s in self.stress_scenarios],
            "rating_recipe": recipe_data,
            "calibration_score_col": self.calibration_score_col,
            "calibration_bins": self.calibration_bins,
            "calibration_base": self.calibration_base,
            "current_hired_col": self.current_hired_col,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CreditPolicy:
        """Deserialize a dictionary to a CreditPolicy object."""
        stages = tuple(Stage.from_dict(s) for s in d.get("stages", []))
        stress_scenarios = tuple(StressScenario.from_dict(s) for s in d.get("stress_scenarios", []))

        # Deserialize rating recipe
        recipe_dict = d.get("rating_recipe")
        recipe = None
        if recipe_dict:
            if isinstance(recipe_dict, dict) and recipe_dict.get("type") == "segmented_recipes":
                recipe = {k: GroupingRecipe.from_dict(v) for k, v in recipe_dict["recipes"].items()}
            else:
                recipe = GroupingRecipe.from_dict(recipe_dict)

        return cls(
            applicant_id_col=d["applicant_id_col"],
            score_cols=d["score_cols"],
            current_approval_col=d["current_approval_col"],
            actual_default_col=d["actual_default_col"],
            time_col=d.get("time_col"),
            stages=stages,
            stress_scenarios=stress_scenarios,
            rating_recipe=recipe,
            calibration_score_col=d.get("calibration_score_col"),
            calibration_bins=d.get("calibration_bins"),
            calibration_base=d.get("calibration_base", "keep_in"),
            current_hired_col=d.get("current_hired_col"),
        )

    def describe(self) -> str:
        """Return a human-readable description of the policy."""
        lines = [
            "CreditPolicy:",
            f"  Applicant ID: {self.applicant_id_col}",
            f"  Score columns: {', '.join(self.score_cols)}",
            f"  Current approval: {self.current_approval_col}",
            f"  Actual default: {self.actual_default_col}",
        ]




        if self.stages:
            lines.append("  Stages:")
            for i, stage in enumerate(self.stages, 1):
                lines.append(f"    {i}. {stage.name} ({stage.__class__.__name__})")
        else:
            lines.append("  Stages: None")

        if self.stress_scenarios:
            lines.append("  Stress Scenarios:")
            for i, scenario in enumerate(self.stress_scenarios, 1):
                lines.append(f"    {i}. {scenario.__class__.__name__}")
        else:
            lines.append("  Stress Scenarios: None")

        if self.rating_recipe:
            lines.append("  Rating Recipe: Yes")
        else:
            lines.append("  Rating Recipe: None")

        return "\n".join(lines)

    def export(
        self,
        rating_recipe: Any | None = None,
        path: str | None = None,
        clean: bool = False,
    ) -> Any:
        """Export this policy and an optional rating recipe for deployment.

        Args:
            rating_recipe: Optional GroupingRecipe.
            path: Optional file path to save the deployment JSON file.
            clean: If True, exports only hard rules and cutoffs (omitting study metadata and rates).

            If rating_recipe is None, defaults to self.rating_recipe.

        Returns:
            DeploymentPolicy instance.
        """
        from .deployment import DeploymentPolicy

        recipe = rating_recipe if rating_recipe is not None else self.rating_recipe
        dep = DeploymentPolicy(policy=self, rating_recipe=recipe)
        if path is not None:
            dep.save(path, clean=clean)
        return dep
