from __future__ import annotations

# Import lightweight database and path helpers for local evidence discovery.
import sqlite3
from dataclasses import dataclass
from pathlib import Path

# Import pandas for consistent CSV and tabular data handling.
import pandas as pd


# Store every validated dashboard source in one immutable data container.
@dataclass(frozen=True)
class DashboardData:
    project_root: Path
    overall_metrics: pd.DataFrame
    overall_improvements: pd.DataFrame
    zone_metrics: pd.DataFrame
    zone_summary: pd.DataFrame
    zone_improvements: pd.DataFrame
    low_demand_definition: pd.DataFrame
    low_demand_pooled: pd.DataFrame
    low_demand_summary: pd.DataFrame
    low_demand_improvements: pd.DataFrame
    low_demand_zone_improvements: pd.DataFrame
    baseline_predictions: pd.DataFrame
    enriched_predictions: pd.DataFrame
    baseline_importance: pd.DataFrame
    enriched_importance: pd.DataFrame
    subway_context: pd.DataFrame
    vehicle_context: pd.DataFrame
    geometry_index: pd.DataFrame
    map_only_zones: pd.DataFrame

    # Count distinct zones represented in the evaluation evidence.
    @property
    def evaluated_zone_count(self) -> int:
        return int(self.zone_metrics["LocationID"].nunique())

    # Count distinct hourly timestamps in the untouched test period.
    @property
    def test_hour_count(self) -> int:
        return int(self.enriched_predictions["pickup_hour"].nunique())

    # Count test prediction records across every evaluated zone and hour.
    @property
    def prediction_row_count(self) -> int:
        return int(len(self.enriched_predictions))

    # Count zones in the training-defined low-demand outer-borough group.
    @property
    def low_demand_zone_count(self) -> int:
        return int(self.low_demand_definition["LocationID"].nunique())


# Derive the FYP project root relative to this module instead of hard-coding a machine path.
def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Convert a model choice into the matching prediction-column name.
def prediction_column(algorithm: str, configuration: str) -> str:
    return f"{algorithm.lower().replace(' ', '_')}_{configuration.lower()}"


# Read a required CSV and fail with a clear source-path message when it is missing.
def _read_csv(path: Path, *, dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required dashboard source was not found: {path}")
    return pd.read_csv(path, parse_dates=dates)


# Read taxi-zone identifiers and labels directly from the GeoPackage index table.
def _geometry_index(gpkg_path: Path) -> pd.DataFrame:
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Required dashboard map was not found: {gpkg_path}")
    with sqlite3.connect(gpkg_path) as connection:
        return pd.read_sql_query(
            "SELECT LocationID, borough, zone FROM nyc_taxi_zones",
            connection,
        )


# Normalise shared identifier and Boolean columns before any joins or filters.
def _normalise_zone_types(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "LocationID" in result:
        result["LocationID"] = pd.to_numeric(result["LocationID"], errors="raise").astype(int)
    if "Unusually high error" in result:
        result["Unusually high error"] = (
            result["Unusually high error"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
            .astype(bool)
        )
    return result


# Load every authoritative CSV and map index required by the dashboard.
def load_dashboard_data(project_root: Path | str | None = None) -> DashboardData:
    root = Path(project_root) if project_root is not None else default_project_root()
    root = root.resolve()
    outputs = root / "processed_outputs"
    evaluation = outputs / "model_results" / "evaluation"
    baseline = outputs / "model_results" / "baseline"
    enriched = outputs / "model_results" / "enriched"

    # Load and normalise the large zone, prediction, context, and map-index sources.
    zone_metrics = _normalise_zone_types(
        _read_csv(evaluation / "zone_level_test_metrics.csv")
    )
    zone_improvements = _normalise_zone_types(
        _read_csv(evaluation / "zone_level_baseline_vs_enriched_improvements.csv")
    )
    low_definition = _normalise_zone_types(
        _read_csv(evaluation / "low_demand_outer_borough_zone_definition.csv")
    )
    low_zone_improvements = _normalise_zone_types(
        _read_csv(evaluation / "low_demand_zone_level_improvements.csv")
    )
    baseline_predictions = _normalise_zone_types(
        _read_csv(baseline / "baseline_test_predictions.csv", dates=["pickup_hour"])
    )
    enriched_predictions = _normalise_zone_types(
        _read_csv(enriched / "enriched_test_predictions.csv", dates=["pickup_hour"])
    )
    subway_context = _normalise_zone_types(
        _read_csv(outputs / "mta_subway" / "nyc_taxi_zone_subway_proximity.csv")
    )
    vehicle_context = _normalise_zone_types(
        _read_csv(
            outputs
            / "spatial_joining"
            / "nyc_taxi_zones_with_zero_vehicle_context.csv"
        )
    )
    geometry_index = _normalise_zone_types(
        _geometry_index(outputs / "taxi_zone" / "nyc_taxi_zones_cleaned.gpkg")
    )

    # Identify map geometries that have no corresponding model-evaluation records.
    evaluated_ids = set(zone_metrics["LocationID"].unique())
    map_only = geometry_index.loc[~geometry_index["LocationID"].isin(evaluated_ids)].copy()

    # Assemble all sources before enforcing the approved dataset cardinalities.
    data = DashboardData(
        project_root=root,
        overall_metrics=_read_csv(evaluation / "overall_test_metrics.csv"),
        overall_improvements=_read_csv(
            evaluation / "overall_baseline_vs_enriched_improvements.csv"
        ),
        zone_metrics=zone_metrics,
        zone_summary=_read_csv(evaluation / "zone_performance_summary.csv"),
        zone_improvements=zone_improvements,
        low_demand_definition=low_definition,
        low_demand_pooled=_read_csv(evaluation / "low_demand_pooled_test_metrics.csv"),
        low_demand_summary=_read_csv(
            evaluation / "low_demand_zone_performance_summary.csv"
        ),
        low_demand_improvements=_read_csv(
            evaluation / "low_demand_improvement_summary.csv"
        ),
        low_demand_zone_improvements=low_zone_improvements,
        baseline_predictions=baseline_predictions,
        enriched_predictions=enriched_predictions,
        baseline_importance=_read_csv(
            baseline / "baseline_permutation_feature_importance.csv"
        ),
        enriched_importance=_read_csv(
            enriched / "enriched_permutation_feature_importance.csv"
        ),
        subway_context=subway_context,
        vehicle_context=vehicle_context,
        geometry_index=geometry_index,
        map_only_zones=map_only,
    )
    _validate_dashboard_data(data)
    return data


# Reject incomplete or mismatched evidence before Streamlit renders misleading results.
def _validate_dashboard_data(data: DashboardData) -> None:
    checks = {
        "analysis-ready zones": (data.evaluated_zone_count, 259),
        "test hours": (data.test_hour_count, 432),
        "test prediction rows": (data.prediction_row_count, 111_888),
        "low-demand zones": (data.low_demand_zone_count, 49),
        "map geometries": (len(data.geometry_index), 262),
    }
    failures = [
        f"{name}: expected {expected:,}, found {actual:,}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if failures:
        raise ValueError("Dashboard source validation failed: " + "; ".join(failures))
