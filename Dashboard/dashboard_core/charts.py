from __future__ import annotations

# Import paths, spatial tables, tabular data, and Plotly chart builders.
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .data import DashboardData


# Define the shared dark theme and algorithm colours used across all charts.
PAPER = "#07111f"
PANEL = "#0d1b2a"
GRID = "rgba(148, 163, 184, 0.14)"
TEXT = "#e5edf7"
MUTED = "#9fb0c5"
CYAN = "#22d3ee"
ORANGE = "#fb923c"
ALGORITHM_COLORS = {"Random Forest": CYAN, "XGBoost": ORANGE}


# Apply consistent layout, typography, axes, and hover styling to Plotly figures.
def _style_figure(figure: go.Figure, *, height: int = 410) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=62, b=24),
        paper_bgcolor=PAPER,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter, Segoe UI, sans-serif"),
        title=dict(font=dict(size=18, color=TEXT), x=0.02),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(bgcolor="#10243a", font_color=TEXT),
    )
    figure.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, title_font_color=MUTED)
    figure.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, title_font_color=MUTED)
    return figure


# Inner-join the 262-zone map to the 259 evaluated LocationIDs and convert to WGS84.
def load_analysis_map(
    data: DashboardData,
    evidence: pd.DataFrame,
) -> gpd.GeoDataFrame:
    gpkg = (
        Path(data.project_root)
        / "processed_outputs"
        / "taxi_zone"
        / "nyc_taxi_zones_cleaned.gpkg"
    )
    geometry = gpd.read_file(gpkg, layer="nyc_taxi_zones")
    geometry["LocationID"] = pd.to_numeric(
        geometry["LocationID"], errors="raise"
    ).astype(int)
    mapped = geometry[["LocationID", "geometry"]].merge(
        evidence,
        on="LocationID",
        how="inner",
        validate="one_to_one",
    )
    mapped = gpd.GeoDataFrame(mapped, geometry="geometry", crs=geometry.crs)
    if mapped.crs is None:
        raise ValueError("Taxi-zone geometry has no coordinate reference system.")
    if mapped.crs.to_epsg() != 4326:
        mapped = mapped.to_crs(4326)
    return mapped.sort_values("LocationID").reset_index(drop=True)


# Draw a zone choropleth and optionally outline, centre, and zoom to one selected zone.
def zone_choropleth(
    mapped: gpd.GeoDataFrame,
    *,
    value_column: str,
    title: str,
    color_scale: str = "Viridis",
    midpoint: float | None = None,
    selected_location_id: int | None = None,
) -> go.Figure:
    geojson = mapped[["LocationID", "geometry"]].__geo_interface__
    hover_data = {
        "LocationID": True,
        "borough": True,
        "zone": True,
        value_column: ":.4f",
    }
    figure = px.choropleth_map(
        mapped,
        geojson=geojson,
        locations="LocationID",
        featureidkey="properties.LocationID",
        color=value_column,
        color_continuous_scale=color_scale,
        color_continuous_midpoint=midpoint,
        hover_name="zone",
        hover_data=hover_data,
        map_style="carto-darkmatter",
        center={"lat": 40.7128, "lon": -74.0060},
        zoom=8.7,
        opacity=0.84,
        title=title,
    )
    figure.update_traces(marker_line_width=0.45, marker_line_color="#cbd5e1")

    # Isolate the requested zone while keeping the full choropleth visible underneath.
    selected = mapped.iloc[0:0]
    if selected_location_id is not None:
        selected = mapped.loc[mapped["LocationID"].eq(int(selected_location_id))]
    if not selected.empty:
        selected_geojson = selected[["LocationID", "geometry"]].__geo_interface__
        figure.add_trace(
            go.Choroplethmap(
                geojson=selected_geojson,
                locations=selected["LocationID"],
                featureidkey="properties.LocationID",
                z=[1] * len(selected),
                colorscale=[[0, "#22d3ee"], [1, "#22d3ee"]],
                showscale=False,
                marker=dict(line=dict(color="#f8fafc", width=4), opacity=0.38),
                name="Selected zone",
                hoverinfo="skip",
            )
        )
        # Use a projected representative point for reliable map centring within the polygon.
        selected_projected = selected.to_crs(2263)
        point = selected_projected.geometry.representative_point().to_crs(4326).iloc[0]
        figure.update_layout(
            map=dict(center={"lat": point.y, "lon": point.x}, zoom=10.6)
        )
    figure.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=55, b=0),
        paper_bgcolor=PAPER,
        font=dict(color=TEXT, family="Inter, Segoe UI, sans-serif"),
        title=dict(font=dict(size=18), x=0.02),
        coloraxis_colorbar=dict(title=value_column, thickness=12),
    )
    return figure


# Compare one published metric across models without mixing incompatible scales.
def metric_bar_chart(
    frame: pd.DataFrame,
    *,
    metric: str,
    title: str,
    lower_is_better: bool,
) -> go.Figure:
    plotted = frame.copy()
    plotted["Model setting"] = (
        plotted["Dataset"] + " • " + plotted["Configuration"]
    )
    labels = {"RMSE": "RMSE", "MAE": "MAE", "R2": "R²"}
    figure = px.bar(
        plotted,
        x="Model setting",
        y=metric,
        color="Algorithm",
        barmode="group",
        color_discrete_map=ALGORITHM_COLORS,
        category_orders={
            "Model setting": [
                "Baseline • Default",
                "Baseline • Tuned",
                "Enriched • Default",
                "Enriched • Tuned",
            ]
        },
        labels={metric: labels[metric]},
        title=title,
    )
    figure.update_traces(
        texttemplate="%{y:.4f}" if metric == "R2" else "%{y:.2f}",
        textposition="outside",
        cliponaxis=False,
    )
    figure.update_yaxes(title=labels[metric])
    figure.update_xaxes(title=None, tickangle=-15)
    if metric == "R2":
        low = min(-0.05, float(plotted[metric].min()) - 0.03)
        high = max(1.0, float(plotted[metric].max()) + 0.08)
        figure.update_yaxes(range=[low, high])
    _style_figure(figure, height=430)
    figure.add_annotation(
        text="Lower is better" if lower_is_better else "Higher is better",
        xref="paper",
        yref="paper",
        x=0,
        y=1.09,
        showarrow=False,
        font=dict(color=MUTED, size=11),
    )
    return figure


# Plot actual and predicted hourly pickups for the selected taxi zone.
def prediction_trend_chart(
    frame: pd.DataFrame,
    *,
    prediction: str,
    title: str,
    selected_hour: pd.Timestamp | None = None,
) -> go.Figure:
    ordered = frame.sort_values("pickup_hour")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=ordered["pickup_hour"],
            y=ordered["actual_pickups"],
            name="Actual",
            mode="lines",
            line=dict(color="#f8fafc", width=2.2),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=ordered["pickup_hour"],
            y=ordered[prediction],
            name="Predicted",
            mode="lines",
            line=dict(color=CYAN, width=2),
        )
    )
    # Mark the timestamp represented by the map and KPI cards when one is supplied.
    if selected_hour is not None:
        marker_hour = pd.Timestamp(selected_hour)
        figure.add_shape(
            type="line",
            x0=marker_hour,
            x1=marker_hour,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color=ORANGE, width=2, dash="dash"),
        )
        figure.add_annotation(
            x=marker_hour,
            y=1,
            yref="paper",
            text="Selected hour",
            showarrow=False,
            yshift=12,
            font=dict(color=ORANGE, size=11),
        )
    figure.update_layout(title=title)
    figure.update_xaxes(title=None)
    figure.update_yaxes(title="Hourly pickups", rangemode="tozero")
    return _style_figure(figure, height=390)


# Plot permutation importance as test-RMSE dependence with uncertainty bars.
def feature_importance_chart(
    frame: pd.DataFrame,
    *,
    model: str,
    title: str,
) -> go.Figure:
    selected = frame.loc[frame["Model"].eq(model)].sort_values("RMSE increase")
    figure = px.bar(
        selected,
        x="RMSE increase",
        y="Feature",
        orientation="h",
        error_x="Standard deviation",
        color="RMSE increase",
        color_continuous_scale="Tealgrn",
        title=title,
    )
    figure.update_layout(coloraxis_showscale=False, showlegend=False)
    figure.update_xaxes(title="Test RMSE increase after permutation")
    figure.update_yaxes(title=None)
    return _style_figure(figure, height=430)


# Compare baseline and enriched zone RMSE against the equal-performance diagonal.
def baseline_enriched_zone_scatter(
    frame: pd.DataFrame,
    *,
    title: str,
) -> go.Figure:
    figure = px.scatter(
        frame,
        x="Baseline zone RMSE",
        y="Enriched zone RMSE",
        color="borough",
        hover_name="zone",
        hover_data=["LocationID", "Zone RMSE reduction (%)"],
        title=title,
    )
    maximum = float(
        max(frame["Baseline zone RMSE"].max(), frame["Enriched zone RMSE"].max())
    )
    figure.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=maximum,
        y1=maximum,
        line=dict(color=MUTED, dash="dash"),
    )
    figure.update_xaxes(title="Baseline zone RMSE", rangemode="tozero")
    figure.update_yaxes(title="Enriched zone RMSE", rangemode="tozero")
    _style_figure(figure, height=450)
    figure.update_layout(
        legend=dict(y=-0.18, x=0.5, xanchor="center", yanchor="top"),
        margin=dict(l=20, r=20, t=62, b=82),
    )
    return figure
