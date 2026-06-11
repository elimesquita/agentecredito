from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .grouping import GroupingRecipe
from .policy import CreditPolicy


@dataclass
class DeploymentPolicy:
    """A unified decision policy packaged for deployment/implementation.

    It bundles a CreditPolicy (filters, cutoffs, rates) and an optional GroupingRecipe (clustering/ratings).
    """

    policy: CreditPolicy
    rating_recipe: GroupingRecipe | None = None

    def to_dict(self, clean: bool = False) -> dict[str, Any]:
        """Serialize the deployment policy to a dictionary."""
        return self.to_production_rules(clean=clean)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DeploymentPolicy:
        """Deserialize a dictionary to a DeploymentPolicy."""
        # Check if loading old representation for backward compatibility
        if "policy" in d:
            policy = CreditPolicy.from_dict(d["policy"])
            recipe_dict = d.get("rating_recipe")
            recipe = None
            if recipe_dict:
                if isinstance(recipe_dict, dict) and recipe_dict.get("type") == "segmented_recipes":
                    recipe = {
                        k: GroupingRecipe.from_dict(v) for k, v in recipe_dict["recipes"].items()
                    }
                else:
                    recipe = GroupingRecipe.from_dict(recipe_dict)
            return cls(policy=policy, rating_recipe=recipe)

        # Parse new representation
        metadata = d.get("metadata", {})
        applicant_id_col = metadata.get("applicant_id_col", "applicant_id")
        score_cols = metadata.get("score_cols", [])
        if isinstance(score_cols, list):
            score_cols = tuple(score_cols)
        current_approval_col = metadata.get("current_approval_col", "approved")
        actual_default_col = metadata.get("actual_default_col", "defaulted")
        time_col = metadata.get("time_col")

        from .stages import CutoffStage, FilterStage, RateStage

        stages = []
        stages_list = d.get("funnel_stages", [])
        for s in stages_list:
            t = s.get("type")
            name = s.get("name")
            if t == "filter":
                expr = s.get("expression")
                if isinstance(expr, str) and expr.startswith("callable:"):
                    from .stages import _resolve_callable

                    func_name = expr.split("callable:", 1)[1]
                    stages.append(
                        FilterStage(name=name, condition=_resolve_callable(func_name))
                    )
                else:
                    stages.append(FilterStage(name=name, condition=expr))
            elif t == "score_cutoff":
                stages.append(
                    CutoffStage(
                        name=name,
                        cutoffs=s.get("values"),
                        direction="gte" if s.get("operator") == ">=" else "lte",
                    )
                )
            elif t == "rate_conversion":
                stages.append(
                    RateStage(
                        name=name,
                        base_rate=s.get("base_rate"),
                        variable=s.get("propensity_column"),
                    )
                )

        policy = CreditPolicy(
            applicant_id_col=applicant_id_col,
            score_cols=score_cols,
            current_approval_col=current_approval_col,
            actual_default_col=actual_default_col,
            time_col=time_col,
            stages=tuple(stages),
        )

        ratings_dict = d.get("rating_classification")
        recipe = None
        if ratings_dict:
            tipo = ratings_dict.get("type")
            if tipo == "matrix":
                score_colunas = ratings_dict.get("score_columns", [])
                limites_quantis = ratings_dict.get("quantile_breaks", {})
                mapeamento_celulas = ratings_dict.get("cell_mapping", {})
                cluster_mapping = {}
                for k, v in mapeamento_celulas.items():
                    if isinstance(v, str) and len(v) == 1:
                        cluster_mapping[k] = ord(v.upper()) - 64
                    else:
                        cluster_mapping[k] = int(v)
                recipe = GroupingRecipe(
                    score_cols=score_colunas,
                    quantile_breaks=limites_quantis,
                    cluster_mapping=cluster_mapping,
                )
            elif tipo == "segmented_matrix":
                coluna_segmentacao = ratings_dict.get("segmentation_column", "region")
                recipe = {}
                for seg, info in ratings_dict.get("segments", {}).items():
                    score_colunas = info.get("score_columns", [])
                    limites_quantis = info.get("quantile_breaks", {})
                    mapeamento_celulas = info.get("cell_mapping", {})
                    cluster_mapping = {}
                    for k, v in mapeamento_celulas.items():
                        if isinstance(v, str) and len(v) == 1:
                            cluster_mapping[k] = ord(v.upper()) - 64
                        else:
                            cluster_mapping[k] = int(v)
                    recipe[seg] = GroupingRecipe(
                        score_cols=score_colunas,
                        quantile_breaks=limites_quantis,
                        cluster_mapping=cluster_mapping,
                        segment_col=coluna_segmentacao,
                    )
            else:
                score_coluna = ratings_dict.get("score_column")
                if "segments" in ratings_dict:
                    recipe = {}
                    coluna_segmentacao = ratings_dict.get("segmentation_column", "region")
                    seg_data = ratings_dict.get("segments", {})
                    for seg, faixas in seg_data.items():
                        normalized_faixas = []
                        for f in faixas:
                            normalized_faixas.append({
                                "rating": f.get("rating"),
                                "min_score": f.get("min_score"),
                                "max_score": f.get("max_score"),
                            })
                        recipe[seg] = GroupingRecipe(
                            score_cols=[score_coluna], intervals=normalized_faixas, segment_col=coluna_segmentacao
                        )
                elif "intervals" in ratings_dict:
                    faixas = ratings_dict.get("intervals", [])
                    normalized_faixas = []
                    for f in faixas:
                        normalized_faixas.append({
                            "rating": f.get("rating"),
                            "min_score": f.get("min_score"),
                            "max_score": f.get("max_score"),
                        })
                    recipe = GroupingRecipe(score_cols=[score_coluna], intervals=normalized_faixas)

        return cls(policy=policy, rating_recipe=recipe)

    def save(self, path: str, clean: bool = False) -> None:
        """Save the deployment policy to a JSON file.

        Args:
            path: Target JSON file path.
            clean: If True, saves a minimal representation containing only hard filters
                   and score cutoffs (omitting RateStages, metadata, and rating recipes).
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(clean=clean), f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> DeploymentPolicy:
        """Load a deployment policy from a JSON file."""
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

    def to_production_rules(self, clean: bool = False) -> dict[str, Any]:
        """Convert the policy to a simplified, production-friendly rules dictionary.

        Args:
            clean: If True, returns a minimal representation containing only hard filters
                   and score cutoffs (omitting RateStages, metadata, and rating recipes).
        """
        import numpy as np

        # 2. Clean stages/filters
        stages_list = []
        for stage in self.policy.stages:
            from .stages import CutoffStage, FilterStage, RateStage

            if isinstance(stage, FilterStage):
                if callable(stage.condition):
                    cond_repr = (
                        f"callable:{getattr(stage.condition, '__name__', str(stage.condition))}"
                    )
                elif isinstance(stage.condition, str):
                    cond_repr = stage.condition
                else:
                    cond_repr = repr(stage.condition)
                    cond_repr = cond_repr.replace("col('", "").replace("')", "")
                stages_list.append(
                    {
                        "position": len(stages_list) + 1,
                        "name": stage.name,
                        "type": "filter",
                        "expression": cond_repr,
                    }
                )
            elif isinstance(stage, CutoffStage):
                stages_list.append(
                    {
                        "position": len(stages_list) + 1,
                        "name": stage.name,
                        "type": "score_cutoff",
                        "columns": list(stage.cutoffs.keys()),
                        "operator": ">=" if stage.direction == "gte" else "<=",
                        "values": stage.cutoffs,
                    }
                )
            elif isinstance(stage, RateStage) and not clean:
                stages_list.append(
                    {
                        "position": len(stages_list) + 1,
                        "name": stage.name,
                        "type": "rate_conversion",
                        "base_rate": stage.base_rate,
                        "propensity_column": stage.variable,
                    }
                )

        if clean:
            return {"funnel_stages": stages_list}

        # 1. Metadata
        metadata = {
            "applicant_id_col": self.policy.applicant_id_col,
            "score_cols": list(self.policy.score_cols),
            "current_approval_col": self.policy.current_approval_col,
            "actual_default_col": self.policy.actual_default_col,
            "time_col": self.policy.time_col,
        }

        # 3. Clean rating ranges from recipe
        ratings = None
        if self.rating_recipe is not None:
            if isinstance(self.rating_recipe, dict):
                is_matrix = any(len(r.score_cols) > 1 for r in self.rating_recipe.values())
                segment_col = "region"
                for r in self.rating_recipe.values():
                    if r.segment_col:
                        segment_col = r.segment_col
                        break

                if is_matrix:
                    segmentos_dict = {}
                    for seg, recipe in self.rating_recipe.items():
                        cluster_to_rating = {i: chr(64 + i) for i in range(1, 27)}
                        mapeamento_celulas = {}
                        if recipe.cluster_mapping:
                            for k, v in recipe.cluster_mapping.items():
                                mapeamento_celulas[k] = cluster_to_rating.get(v, "E")

                        segmentos_dict[seg] = {
                            "score_columns": list(recipe.score_cols),
                            "quantile_breaks": recipe.quantile_breaks or {},
                            "cell_mapping": mapeamento_celulas,
                        }
                    ratings = {
                        "type": "segmented_matrix",
                        "segmentation_column": segment_col,
                        "segments": segmentos_dict,
                    }
                else:
                    segmentos_dict = {}
                    score_col = None
                    for seg, recipe in self.rating_recipe.items():
                        if not score_col and recipe.score_cols:
                            score_col = recipe.score_cols[0]

                        if recipe.intervals is not None:
                            intervals_list = []
                            for f in recipe.intervals:
                                intervals_list.append({
                                    "rating": f["rating"],
                                    "min_score": f.get("min_score"),
                                    "max_score": f.get("max_score"),
                                })
                            segmentos_dict[seg] = intervals_list
                        else:
                            breaks = np.array(recipe.quantile_breaks[score_col])
                            mapping = recipe.cluster_mapping
                            cluster_to_rating = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}

                            score_to_rating = {}
                            for s in range(0, 1001):
                                bin_idx = np.digitize([s], bins=breaks[1:-1])[0]
                                cid = mapping.get(str(bin_idx))
                                score_to_rating[s] = cluster_to_rating.get(cid, "E")

                            faixas = {}
                            for s, r in score_to_rating.items():
                                if r not in faixas:
                                    faixas[r] = []
                                faixas[r].append(s)

                            faixas_list = []
                            for r in sorted(faixas.keys()):
                                faixas_list.append(
                                    {
                                        "rating": r,
                                        "min_score": min(faixas[r]),
                                        "max_score": max(faixas[r]),
                                    }
                                )
                            segmentos_dict[seg] = faixas_list

                    ratings = {
                        "score_column": score_col,
                        "segmentation_column": segment_col,
                        "segments": segmentos_dict,
                    }
            else:
                if len(self.rating_recipe.score_cols) > 1:
                    cluster_to_rating = {i: chr(64 + i) for i in range(1, 27)}
                    mapeamento_celulas = {}
                    if self.rating_recipe.cluster_mapping:
                        for k, v in self.rating_recipe.cluster_mapping.items():
                            mapeamento_celulas[k] = cluster_to_rating.get(v, "E")

                    ratings = {
                        "type": "matrix",
                        "score_columns": list(self.rating_recipe.score_cols),
                        "quantile_breaks": self.rating_recipe.quantile_breaks or {},
                        "cell_mapping": mapeamento_celulas,
                    }
                else:
                    score_col = (
                        self.rating_recipe.score_cols[0] if self.rating_recipe.score_cols else None
                    )
                    if score_col:
                        if self.rating_recipe.intervals is not None:
                            intervals_list = []
                            for f in self.rating_recipe.intervals:
                                intervals_list.append({
                                    "rating": f["rating"],
                                    "min_score": f.get("min_score"),
                                    "max_score": f.get("max_score"),
                                })
                            faixas_list = intervals_list
                        else:
                            breaks = np.array(self.rating_recipe.quantile_breaks[score_col])
                            mapping = self.rating_recipe.cluster_mapping

                            cluster_to_rating = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
                            score_to_rating = {}
                            for s in range(0, 1001):
                                bin_idx = np.digitize([s], bins=breaks[1:-1])[0]
                                cid = mapping.get(str(bin_idx))
                                score_to_rating[s] = cluster_to_rating.get(cid, "E")

                            faixas = {}
                            for s, r in score_to_rating.items():
                                if r not in faixas:
                                    faixas[r] = []
                                faixas[r].append(s)

                            faixas_list = []
                            for r in sorted(faixas.keys()):
                                faixas_list.append(
                                    {
                                        "rating": r,
                                        "min_score": min(faixas[r]),
                                        "max_score": max(faixas[r]),
                                    }
                                )
                        ratings = {"score_column": score_col, "intervals": faixas_list}

        return {"metadata": metadata, "funnel_stages": stages_list, "rating_classification": ratings}

    def save_production_rules(self, path: str, clean: bool = False) -> None:
        """Save the production-friendly rules to a JSON file.

        Args:
            path: Target JSON file path.
            clean: If True, saves a minimal representation containing only hard filters
                   and score cutoffs.
        """
        self.save(path, clean=clean)

    def predict(
        self, df: pd.DataFrame, simple: bool = True, method: str = "analytical"
    ) -> pd.DataFrame:
        """Apply the complete decision flow (policy + rating) to new data."""
        df_temp = df.copy()
        
        # Detect if we need to mock metadata columns that are missing from df
        mocked_cols = []
        id_col = self.policy.applicant_id_col
        if id_col not in df_temp.columns:
            df_temp[id_col] = range(len(df_temp))
            mocked_cols.append(id_col)
            
        app_col = self.policy.current_approval_col
        if app_col not in df_temp.columns:
            df_temp[app_col] = 1
            mocked_cols.append(app_col)
            
        def_col = self.policy.actual_default_col
        if def_col not in df_temp.columns:
            df_temp[def_col] = 0
            mocked_cols.append(def_col)
            
        sim_res = self.policy.simulate(df_temp, method=method)
        if simple:
            res_df = sim_res.to_decision_dataframe(rating_recipe=self.rating_recipe)
            # Remove mocked cols if they were added and are in simple_cols
            cols_to_drop = [c for c in mocked_cols if c in res_df.columns]
            if cols_to_drop:
                res_df = res_df.drop(columns=cols_to_drop)
            return res_df

        res_df = sim_res.data.copy()
        if self.rating_recipe is not None:
            # Apply ratings to the full DataFrame
            if isinstance(self.rating_recipe, dict):
                segment_col = "region"
                for c in ["region", "loja", "safra"]:
                    if c in res_df.columns:
                        segment_col = c
                        break
                res_df["Rating"] = None
                for seg, recipe in self.rating_recipe.items():
                    mask = res_df[segment_col] == seg
                    if mask.any():
                        pred_seg = recipe.predict(res_df[mask])
                        rating_labels = {i: chr(64 + i) for i in range(1, 27)}
                        res_df.loc[mask, "Rating"] = pred_seg["risk_rating"].map(rating_labels)
            else:
                pred_df = self.rating_recipe.predict(res_df)
                rating_labels = {i: chr(64 + i) for i in range(1, 27)}
                res_df["Rating"] = pred_df["risk_rating"].map(rating_labels)
                
        # Drop mocked columns from non-simple DataFrame as well if they were not in the original input
        cols_to_drop = [c for c in mocked_cols if c in res_df.columns]
        if cols_to_drop:
            res_df = res_df.drop(columns=cols_to_drop)
            
        return res_df
