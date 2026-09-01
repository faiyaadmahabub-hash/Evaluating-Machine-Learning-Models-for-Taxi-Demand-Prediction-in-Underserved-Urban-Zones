from __future__ import annotations

# Import immutable result containers and numerical helpers.
from dataclasses import dataclass
from math import sqrt
from typing import Iterable

# Import pandas and the validated dashboard-data contract.
import pandas as pd

from .data import DashboardData


# Store the accessibility thresholds used for contextual concern flags.
@dataclass(frozen=True)
class PlanningThresholds:
    subway_q75_km: float
    zero_vehicle_q75: float


# Store a consistently calculated set of regression metrics.
@dataclass(frozen=True)
class RegressionMetrics:
    observations: int
    rmse: float
    mae: float
    r2: float


# Select one feature-set, algorithm, and configuration from a published metric table.
def model_slice(
    frame: pd.DataFrame,
    *,
    dataset: str,
    algorithm: str,
    configuration: str,
) -> pd.DataFrame:
    mask = (
        frame["Dataset"].eq(dataset)
        & frame["Algorithm"].eq(algorithm)
        & frame["Configuration"].eq(configuration)
    )
    return frame.loc[mask].copy()


# Apply optional borough and zone filters without modifying the source frame.
def apply_zone_filters(
    frame: pd.DataFrame,
    *,
    boroughs: Iterable[str] | None,
    zones: Iterable[str] | None,
) -> pd.DataFrame:
    result = frame.copy()
    if boroughs:
        result = result.loc[result["borough"].isin(list(boroughs))]
    if zones:
        result = result.loc[result["zone"].isin(list(zones))]
    return result


# Recalculate RMSE, MAE, and R² from the filtered prediction records.
def calculate_prediction_metrics(
    frame: pd.DataFrame,
    *,
    prediction: str,
) -> RegressionMetrics:
    usable = frame[["actual_pickups", prediction]].dropna()
    if usable.empty:
        return RegressionMetrics(0, float("nan"), float("nan"), float("nan"))
    actual = usable["actual_pickups"].astype(float)
    predicted = usable[prediction].astype(float)
    error = actual - predicted
    squared_error = float((error**2).sum())
    denominator = float(((actual - actual.mean()) ** 2).sum())
    return RegressionMetrics(
        observations=int(len(usable)),
        rmse=sqrt(squared_error / len(usable)),
        mae=float(error.abs().mean()),
        r2=(1.0 - squared_error / denominator) if denominator > 0 else float("nan"),
    )


# Aggregate actual and predicted histories for an overall, borough, or zone scope.
def aggregate_scope_history(
    predictions: pd.DataFrame,
    *,
    boroughs: Iterable[str] | None,
    zone: str | None,
    prediction: str,
) -> pd.DataFrame:
    selected = predictions.copy()
    if boroughs:
        selected = selected.loc[selected["borough"].isin(list(boroughs))]
    else:
        selected = selected.iloc[0:0]
    if zone is not None:
        selected = selected.loc[selected["zone"].eq(zone)]
        return selected[["pickup_hour", "actual_pickups", prediction]].sort_values(
            "pickup_hour"
        )
    return (
        selected.groupby("pickup_hour", as_index=False)[["actual_pickups", prediction]]
        .sum()
        .sort_values("pickup_hour")
    )


# Combine final-model performance with subway and household-vehicle context.
def build_planning_evidence(
    data: DashboardData,
) -> tuple[pd.DataFrame, PlanningThresholds]:
    final_metrics = model_slice(
        data.zone_metrics,
        dataset="Enriched",
        algorithm="Random Forest",
        configuration="Tuned",
    )
    subway = data.subway_context[["LocationID", "subway_proximity_km"]].copy()
    vehicle = data.vehicle_context[
        [
            "LocationID",
            "zero_vehicle_household_rate",
            "census_reliability",
            "census_coverage_ratio",
        ]
    ].copy()
    low = data.low_demand_definition[["LocationID"]].assign(Low_demand=True)

    # Join each contextual source one-to-one by the reporting-only LocationID key.
    evidence = (
        final_metrics.merge(subway, on="LocationID", how="inner", validate="one_to_one")
        .merge(vehicle, on="LocationID", how="inner", validate="one_to_one")
        .merge(low, on="LocationID", how="left", validate="one_to_one")
    )
    evidence["Low_demand"] = evidence["Low_demand"].astype("boolean").fillna(False).astype(bool)

    # Define limited-access thresholds from the 75th percentile of evaluated zones.
    thresholds = PlanningThresholds(
        subway_q75_km=float(evidence["subway_proximity_km"].quantile(0.75)),
        zero_vehicle_q75=float(evidence["zero_vehicle_household_rate"].quantile(0.75)),
    )
    evidence["Negative R2"] = evidence["Zone R2"].lt(0)
    evidence["Limited subway access"] = evidence["subway_proximity_km"].ge(
        thresholds.subway_q75_km
    )
    evidence["High zero-vehicle rate"] = evidence[
        "zero_vehicle_household_rate"
    ].ge(thresholds.zero_vehicle_q75)
    evidence["Performance concern"] = evidence["Negative R2"] | evidence[
        "Unusually high error"
    ]
    evidence["Context concern"] = evidence["Limited subway access"] | evidence[
        "High zero-vehicle rate"
    ]
    evidence["Planning review"] = evidence["Performance concern"] & evidence[
        "Context concern"
    ]

    # Translate Boolean review conditions into readable evidence labels.
    def reasons(row: pd.Series) -> str:
        labels: list[str] = []
        if bool(row["Unusually high error"]):
            labels.append("unusually high RMSE")
        if bool(row["Negative R2"]):
            labels.append("negative R²")
        if bool(row["Limited subway access"]):
            labels.append("limited subway access")
        if bool(row["High zero-vehicle rate"]):
            labels.append("high zero-vehicle household rate")
        return "; ".join(labels) if labels else "no review threshold triggered"

    evidence["Planning reasons"] = evidence.apply(reasons, axis=1)
    return evidence, thresholds


# Build source-backed demand and deployment evidence for one historical test hour.
def build_demand_evidence(
    data: DashboardData,
    pickup_hour: pd.Timestamp,
) -> tuple[pd.DataFrame, float]:
    """Combine one historical test hour's final-model demand with planning context."""
    # Select the actual and tuned enriched Random Forest values for all 259 zones.
    selected_hour = pd.Timestamp(pickup_hour)
    hourly = data.enriched_predictions.loc[
        data.enriched_predictions["pickup_hour"].eq(selected_hour),
        ["LocationID", "actual_pickups", "random_forest_tuned"],
    ].copy()
    if hourly.empty:
        raise ValueError(
            f"No enriched-model predictions are available for {selected_hour}."
        )
    if len(hourly) != data.evaluated_zone_count or hourly["LocationID"].duplicated().any():
        raise ValueError(
            "Hourly enriched-model predictions do not contain one row for each evaluated zone."
        )

    hourly = hourly.rename(
        columns={
            "actual_pickups": "Actual pickups",
            "random_forest_tuned": "Predicted pickups",
        }
    )
    # Merge demand with context, then flag the highest-demand quartile for that hour.
    planning, _ = build_planning_evidence(data)
    evidence = planning.merge(
        hourly,
        on="LocationID",
        how="inner",
        validate="one_to_one",
    )
    threshold = float(evidence["Predicted pickups"].quantile(0.75))
    evidence["High predicted demand"] = evidence["Predicted pickups"].ge(threshold)
    evidence["Deployment consideration"] = evidence["High predicted demand"] & evidence[
        "Context concern"
    ]
    evidence["Reliability warning"] = evidence["Performance concern"]
    evidence["Prediction difference"] = (
        evidence["Predicted pickups"] - evidence["Actual pickups"]
    )

    # Explain which demand and accessibility conditions each zone triggered.
    def deployment_reasons(row: pd.Series) -> str:
        labels: list[str] = []
        if bool(row["High predicted demand"]):
            labels.append("high predicted demand")
        if bool(row["Limited subway access"]):
            labels.append("limited subway access")
        if bool(row["High zero-vehicle rate"]):
            labels.append("high zero-vehicle household rate")
        return "; ".join(labels) if labels else "deployment threshold not met"

    evidence["Deployment reasons"] = evidence.apply(deployment_reasons, axis=1)
    return evidence, threshold
