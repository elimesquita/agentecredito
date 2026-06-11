from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

from pycreditools import CreditPolicy, DeploymentPolicy, fit_risk_groups, optimize_cutoffs


mcp = FastMCP("pycreditools-credit-agent")


SUPPORTED_DATA_EXTENSIONS = {".csv", ".xlsx", ".xls", ".xlsm", ".parquet"}
DEFAULT_ARTIFACT_DIR = "mcp_outputs"


def _ensure_dict(value: dict[str, Any] | str, label: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} must be a dict or valid JSON string.") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{label} JSON must decode to an object.")
        return parsed
    raise ValueError(f"{label} must be a dict or JSON string.")


def _resolve_path(path: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Data file not found: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_DATA_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_DATA_EXTENSIONS))
        raise ValueError(f"Unsupported file extension '{resolved.suffix}'. Use one of: {allowed}.")
    return resolved


def _read_data(path: str, sheet_name: str | int | None = None) -> pd.DataFrame:
    data_path = _resolve_path(path)
    suffix = data_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(data_path, sheet_name=sheet_name or 0)
    if suffix == ".parquet":
        return pd.read_parquet(data_path)

    raise ValueError(f"Unsupported file extension: {suffix}")


def _excel_sheets(path: Path) -> list[str] | None:
    if path.suffix.lower() not in {".xlsx", ".xls", ".xlsm"}:
        return None
    return list(pd.ExcelFile(path).sheet_names)


def _artifact_dir(data_path: str | None = None, output_dir: str | None = None) -> Path:
    if output_dir:
        target = Path(output_dir).expanduser().resolve()
    elif data_path:
        target = Path(data_path).expanduser().resolve().parent / DEFAULT_ARTIFACT_DIR
    else:
        target = Path.cwd() / DEFAULT_ARTIFACT_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_records_csv(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False)
    return str(path)


def _write_json(data: dict[str, Any], path: Path) -> str:
    path.write_text(json.dumps(_json_safe(data), indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and pd.isna(value):
        return None
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    return value


def _summarize_decisions(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": int(len(df))}
    if "decision" in df.columns:
        summary["decisions"] = {
            str(k): int(v) for k, v in df["decision"].value_counts(dropna=False).items()
        }
    if "scenario" in df.columns:
        summary["scenarios"] = {
            str(k): int(v) for k, v in df["scenario"].value_counts(dropna=False).items()
        }
    if "rating" in df.columns:
        summary["ratings"] = {
            str(k): int(v) for k, v in df["rating"].value_counts(dropna=False).items()
        }
    if "hired" in df.columns and pd.api.types.is_numeric_dtype(df["hired"]):
        summary["expected_hired"] = float(df["hired"].sum())
    return _json_safe(summary)


@mcp.tool()
def inspect_credit_data(data_path: str, sheet_name: str | int | None = None) -> dict[str, Any]:
    """Inspect a real client CSV, Excel, or Parquet file before running credit tools."""
    path = _resolve_path(data_path)
    df = _read_data(str(path), sheet_name=sheet_name)
    is_excel = path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}

    return _json_safe({
        "path": str(path),
        "file_type": path.suffix.lower().lstrip("."),
        "sheets": _excel_sheets(path),
        "active_sheet": (sheet_name or 0) if is_excel else None,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "missing_values": {str(k): int(v) for k, v in df.isna().sum().items()},
        "preview": df.head(10).to_dict(orient="records"),
    })


@mcp.tool()
def simulate_credit_policy(
    data_path: str,
    policy: dict[str, Any] | str,
    sheet_name: str | int | None = None,
    method: str = "analytical",
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Run a serialized CreditPolicy against a real client file and save decisions."""
    df = _read_data(data_path, sheet_name=sheet_name)
    credit_policy = CreditPolicy.from_dict(_ensure_dict(policy, "policy"))
    credit_policy.validate(df)

    result = credit_policy.simulate(df, method=method)
    decisions = result.to_decision_dataframe()

    out_dir = _artifact_dir(data_path, output_dir)
    output_path = out_dir / f"{Path(data_path).stem}_credit_decisions.csv"

    return _json_safe({
        "summary": _summarize_decisions(decisions),
        "output_path": _write_records_csv(decisions, output_path),
        "metadata": result.metadata,
    })


@mcp.tool()
def predict_credit_decision(
    data_path: str,
    production_rules: dict[str, Any] | str,
    sheet_name: str | int | None = None,
    method: str = "analytical",
    simple: bool = True,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Apply exported production rules to a real client file and save the output."""
    df = _read_data(data_path, sheet_name=sheet_name)
    deployment = DeploymentPolicy.from_dict(_ensure_dict(production_rules, "production_rules"))

    predictions = deployment.predict(df, simple=simple, method=method)
    out_dir = _artifact_dir(data_path, output_dir)
    output_path = out_dir / f"{Path(data_path).stem}_predictions.csv"

    return _json_safe({
        "summary": _summarize_decisions(predictions),
        "output_path": _write_records_csv(predictions, output_path),
    })


@mcp.tool()
def optimize_score_cutoffs(
    data_path: str,
    policy: dict[str, Any] | str,
    sheet_name: str | int | None = None,
    cutoff_steps: int = 10,
    target_default_rate: float = 0.05,
    min_approval_rate: float = 0.30,
    method: str = "analytical",
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Find score cutoffs that balance approval rate and default rate on client data."""
    df = _read_data(data_path, sheet_name=sheet_name)
    credit_policy = CreditPolicy.from_dict(_ensure_dict(policy, "policy"))
    credit_policy.validate(df)

    result = optimize_cutoffs(
        data=df,
        config=credit_policy,
        cutoff_steps=cutoff_steps,
        target_default_rate=target_default_rate,
        min_approval_rate=min_approval_rate,
        method=method,
    )

    out_dir = _artifact_dir(data_path, output_dir)
    all_results_path = out_dir / f"{Path(data_path).stem}_cutoff_grid.csv"
    pareto_path = out_dir / f"{Path(data_path).stem}_pareto_frontier.csv"

    return _json_safe({
        "best": result.to_dict(),
        "all_results_path": _write_records_csv(result.all_results, all_results_path),
        "pareto_frontier_path": _write_records_csv(result.pareto_frontier, pareto_path),
    })


@mcp.tool()
def fit_risk_ratings(
    data_path: str,
    score_cols: str | list[str],
    default_col: str,
    sheet_name: str | int | None = None,
    bins: int = 20,
    max_groups: int = 5,
    min_vol_ratio: float = 0.05,
    max_crossings: int = 1,
    time_col: str | None = None,
    method: str = "ward",
    oot_date: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Create risk ratings from real historical client data and save the recipe."""
    df = _read_data(data_path, sheet_name=sheet_name)

    result = fit_risk_groups(
        data=df,
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

    out_dir = _artifact_dir(data_path, output_dir)
    groups_path = out_dir / f"{Path(data_path).stem}_risk_groups.csv"
    recipe_path = out_dir / f"{Path(data_path).stem}_rating_recipe.json"
    report_path = out_dir / f"{Path(data_path).stem}_rating_report.csv"

    response = {
        "n_groups": result.n_groups,
        "recipe": result.recipe.to_dict(),
        "groups_path": _write_records_csv(result.groups, groups_path),
        "recipe_path": _write_json(result.recipe.to_dict(), recipe_path),
    }
    if result.report is not None:
        response["report_path"] = _write_records_csv(result.report, report_path)
    return _json_safe(response)


@mcp.tool()
def export_production_rules(
    policy: dict[str, Any] | str,
    rating_recipe: dict[str, Any] | str | None = None,
    clean: bool = False,
    output_dir: str | None = None,
    output_name: str = "production_rules.json",
) -> dict[str, Any]:
    """Export a policy and optional rating recipe as production-friendly JSON rules."""
    credit_policy = CreditPolicy.from_dict(_ensure_dict(policy, "policy"))

    recipe_obj = None
    if rating_recipe is not None:
        from pycreditools import GroupingRecipe

        recipe_obj = GroupingRecipe.from_dict(_ensure_dict(rating_recipe, "rating_recipe"))

    deployment = credit_policy.export(rating_recipe=recipe_obj)
    rules = deployment.to_production_rules(clean=clean)
    output_path = _artifact_dir(output_dir=output_dir) / output_name

    return _json_safe({
        "rules": rules,
        "output_path": _write_json(rules, output_path),
    })


@mcp.tool()
def explain_policy(policy: dict[str, Any] | str) -> dict[str, Any]:
    """Return a human-readable description of a serialized CreditPolicy."""
    credit_policy = CreditPolicy.from_dict(_ensure_dict(policy, "policy"))
    return _json_safe({
        "description": credit_policy.describe(),
        "policy": credit_policy.to_dict(),
    })


if __name__ == "__main__":
    mcp.run()
