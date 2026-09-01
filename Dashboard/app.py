from __future__ import annotations

# Import standard-library helpers for formatting values and resolving local paths.
import math
import os
from pathlib import Path

# Import the data, mapping, and Streamlit libraries used by the interface.
import geopandas as gpd
import pandas as pd
import streamlit as st

# Import reusable chart builders so visual logic stays outside the page layout.
from dashboard_core.charts import (
    baseline_enriched_zone_scatter,
    feature_importance_chart,
    load_analysis_map,
    metric_bar_chart,
    prediction_trend_chart,
    zone_choropleth,
)
# Import validated data-loading helpers and the model-column name resolver.
from dashboard_core.data import (
    default_project_root,
    load_dashboard_data,
    prediction_column,
)
# Import shared filtering, metric, and planning-evidence calculations.
from dashboard_core.metrics import (
    aggregate_scope_history,
    apply_zone_filters,
    build_demand_evidence,
    build_planning_evidence,
    calculate_prediction_metrics,
    model_slice,
)


# Configure the browser page before any Streamlit content is rendered.
st.set_page_config(
    page_title="NYC Taxi Demand Evidence Dashboard",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Apply the dashboard's custom dark visual styling from the assets folder.
def _load_css() -> None:
    css_path = Path(__file__).resolve().parent / "assets" / "styles.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# Cache validated source data so filter interactions do not reload every CSV.
@st.cache_data(show_spinner="Loading validated model evidence…")
def _cached_data(project_root: str):
    return load_dashboard_data(Path(project_root))


# Cache the 259-zone geometry join used by all three objective tabs.
@st.cache_data(show_spinner="Preparing the 259-zone map…")
def _cached_map(project_root: str):
    loaded = load_dashboard_data(Path(project_root))
    evidence, _ = build_planning_evidence(loaded)
    return load_analysis_map(loaded, evidence)


# Format numeric KPI values while displaying undefined results safely.
def _metric_value(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "Undefined"
    return f"{value:,.{digits}f}"


# Display unavailable regression scores as a clear explanation instead of a blank cell.
def _display_number_or_not_defined(value: float, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "Not defined"
    return f"{float(value):,.{digits}f}"


# Render a consistent title and explanation at the start of each objective.
def _section_intro(kicker: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="section-intro">
          <div class="eyebrow">{kicker}</div>
          <h2>{title}</h2>
          <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Show a friendly empty-state message when the current filters return no zones.
def _render_empty(message: str) -> None:
    st.info(message, icon="ℹ️")


# Resolve the FYP root, load authoritative evidence, and stop cleanly on source errors.
_load_css()
project_root = Path(os.environ.get("FYP_PROJECT_ROOT", default_project_root())).resolve()

try:
    data = _cached_data(str(project_root))
    planning_evidence, planning_thresholds = build_planning_evidence(data)
    analysis_map = _cached_map(str(project_root))
except (FileNotFoundError, ValueError, OSError) as error:
    st.error("The dashboard could not load its validated source files.")
    st.code(str(error))
    st.stop()


# Display the dashboard identity and historical evidence scope.
st.markdown(
    """
    <div class="hero">
      <div>
        <div class="eyebrow">FYP · Historical decision support · January–March 2025</div>
        <h1>NYC Taxi Demand Evidence Dashboard</h1>
        <p>Zone-level model performance, contextual enrichment, and transparent planning evidence across New York City.</p>
      </div>
      <div class="hero-badge"><span></span>Validated test evidence</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Explain why analytical maps contain 259 evaluated zones instead of 262 geometries.
st.info(
    "The source map contains 262 taxi-zone geometries. Model evidence is available for 259 analysis-ready zones; "
    "LocationIDs 103, 104 and 105 (Governor's Island/Ellis Island/Liberty Island) have no evaluated model rows and "
    "are therefore excluded from analytical maps.",
    icon="🗺️",
)

# Organise the interface around the three approved research objectives.
tab1, tab2, tab3 = st.tabs(
    [
        "Objective 1 · Zone performance",
        "Objective 2 · Contextual enrichment",
        "Objective 3 · Planning evidence",
    ]
)


# Objective 1 compares models and exposes zone-level prediction performance.
with tab1:
    _section_intro(
        "OBJECTIVE 1",
        "Model performance by taxi zone",
        "Compare Random Forest and XGBoost using compatible metrics, then inspect where the selected model succeeds or struggles during the untouched 432-hour test period.",
    )

    # Collect the model, feature-set, borough, and zone selections in one control row.
    controls = st.columns([1.05, 1.05, 1.05, 1.3, 1.6])
    with controls[0]:
        dataset_choice = st.selectbox(
            "Feature set",
            ["Enriched", "Baseline"],
            key="o1_dataset",
        )
    with controls[1]:
        algorithm_choice = st.selectbox(
            "Algorithm",
            ["Random Forest", "XGBoost"],
            key="o1_algorithm",
        )
    with controls[2]:
        configuration_choice = st.selectbox(
            "Configuration",
            ["Tuned", "Default"],
            key="o1_configuration",
        )

    # Select the published zone metrics matching the chosen model configuration.
    selected_zone_metrics = model_slice(
        data.zone_metrics,
        dataset=dataset_choice,
        algorithm=algorithm_choice,
        configuration=configuration_choice,
    )
    borough_options = sorted(selected_zone_metrics["borough"].dropna().unique())
    with controls[3]:
        borough_choice = st.multiselect(
            "Borough",
            borough_options,
            default=borough_options,
            key="o1_boroughs",
        )
    available_zone_names = sorted(
        selected_zone_metrics.loc[
            selected_zone_metrics["borough"].isin(borough_choice), "zone"
        ].dropna().unique()
    )
    all_zones_option = "All selected zones"
    zone_options = [all_zones_option, *available_zone_names]
    # Return to the overall scope if a borough change removes the selected zone.
    if st.session_state.get("o1_zone", all_zones_option) not in zone_options:
        st.session_state["o1_zone"] = all_zones_option
    with controls[4]:
        zone_choice = st.selectbox(
            "Focus zone",
            zone_options,
            key="o1_zone",
        )

    # Apply the chosen borough and optional zone filters to metrics and predictions.
    zones_filter = None if zone_choice == all_zones_option else [zone_choice]
    filtered_zone_metrics = apply_zone_filters(
        selected_zone_metrics,
        boroughs=borough_choice,
        zones=zones_filter,
    )
    source_predictions = (
        data.enriched_predictions
        if dataset_choice == "Enriched"
        else data.baseline_predictions
    )
    filtered_predictions = source_predictions.loc[
        source_predictions["borough"].isin(borough_choice)
    ]
    if zones_filter:
        filtered_predictions = filtered_predictions.loc[
            filtered_predictions["zone"].isin(zones_filter)
        ]
    prediction_name = prediction_column(algorithm_choice, configuration_choice)
    filtered_metrics = calculate_prediction_metrics(
        filtered_predictions,
        prediction=prediction_name,
    )

    # Render either an empty-state message or the complete Objective 1 evidence view.
    if filtered_zone_metrics.empty:
        _render_empty("No zones match the selected filters. Broaden the borough or zone selection.")
    else:
        # Summarise filtered accuracy, typical zone error, and diagnostic counts.
        kpis = st.columns(5)
        kpis[0].metric("Filtered RMSE", _metric_value(filtered_metrics.rmse))
        kpis[1].metric("Filtered R²", _metric_value(filtered_metrics.r2))
        kpis[2].metric(
            "Median zone RMSE",
            _metric_value(float(filtered_zone_metrics["Zone RMSE"].median())),
        )
        kpis[3].metric(
            "Negative-R² zones",
            f"{int(filtered_zone_metrics['Zone R2'].lt(0).sum()):,}",
        )
        kpis[4].metric(
            "High-error zones",
            f"{int(filtered_zone_metrics['Unusually high error'].sum()):,}",
        )

        # Draw RMSE and R² on separate map scales to avoid incompatible comparisons.
        map_layer = st.radio(
            "Map measure",
            ["Zone RMSE", "Zone R2"],
            horizontal=True,
            key="o1_map_layer",
        )
        mapped_selected = analysis_map[["LocationID", "geometry"]].merge(
            filtered_zone_metrics,
            on="LocationID",
            how="inner",
            validate="one_to_one",
        )
        mapped_selected = gpd.GeoDataFrame(
            mapped_selected,
            geometry="geometry",
            crs=analysis_map.crs,
        )
        st.plotly_chart(
            zone_choropleth(
                mapped_selected,
                value_column=map_layer,
                title=f"{dataset_choice} · {algorithm_choice} · {configuration_choice} · {map_layer}",
                color_scale="YlOrRd" if map_layer == "Zone RMSE" else "RdYlGn",
                midpoint=0 if map_layer == "Zone R2" else None,
            ),
            width="stretch",
            key="o1_map",
        )
        st.caption(
            "This map shows the selected model's absolute performance. Similar colours across model choices mean "
            "the models have a similar geographical pattern; use the improvement view below to see smaller differences."
        )

        # Match the historical trend to the complete borough scope or chosen zone.
        all_zone_scope = zone_choice == all_zones_option
        objective1_history = aggregate_scope_history(
            source_predictions,
            boroughs=borough_choice,
            zone=None if all_zone_scope else zone_choice,
            prediction=prediction_name,
        )
        if all_zone_scope:
            all_boroughs_selected = set(borough_choice) == set(borough_options)
            history_scope = (
                "all NYC zones"
                if all_boroughs_selected
                else ", ".join(borough_choice)
            )
        else:
            history_scope = zone_choice
        st.plotly_chart(
            prediction_trend_chart(
                objective1_history,
                prediction=prediction_name,
                title=f"Actual and predicted pickups · {history_scope}",
            ),
            width="stretch",
            key="o1_trend",
        )

        # Compare baseline and enriched zone error for the same model and current location scope.
        selected_improvement = data.zone_improvements.loc[
            data.zone_improvements["Algorithm"].eq(algorithm_choice)
            & data.zone_improvements["Configuration"].eq(configuration_choice)
        ].copy()
        selected_improvement = apply_zone_filters(
            selected_improvement,
            boroughs=borough_choice,
            zones=zones_filter,
        )
        st.markdown("#### What changed after adding the contextual features?")
        st.caption(
            f"This matched comparison keeps {algorithm_choice} and {configuration_choice.lower()} settings fixed. "
            "With every borough selected and no individual focus zone, it covers all 259 evaluated zones."
        )
        improvement_count = int(selected_improvement["RMSE improved"].sum())
        compared_count = int(len(selected_improvement))
        improvement_kpis = st.columns(4)
        improvement_kpis[0].metric("Zones compared", f"{compared_count:,}")
        improvement_kpis[1].metric("Zones improved", f"{improvement_count:,}")
        improvement_kpis[2].metric(
            "Zones not improved",
            f"{compared_count - improvement_count:,}",
        )
        improvement_kpis[3].metric(
            "Median RMSE reduction",
            f"{float(selected_improvement['Zone RMSE reduction (%)'].median()):.2f}%",
        )

        # Join the matched improvement records to geometry for a difference-focused map.
        improvement_map = analysis_map[["LocationID", "geometry"]].merge(
            selected_improvement,
            on="LocationID",
            how="inner",
            validate="one_to_one",
        )
        improvement_map = gpd.GeoDataFrame(
            improvement_map,
            geometry="geometry",
            crs=analysis_map.crs,
        )
        improvement_left, improvement_right = st.columns([1.15, 0.85])
        with improvement_left:
            st.plotly_chart(
                zone_choropleth(
                    improvement_map,
                    value_column="Zone RMSE reduction (%)",
                    title="Zone RMSE change after contextual enrichment",
                    color_scale="RdYlGn",
                    midpoint=0,
                ),
                width="stretch",
                key="o1_improvement_map",
            )
        with improvement_right:
            st.plotly_chart(
                baseline_enriched_zone_scatter(
                    selected_improvement,
                    title="Baseline versus enriched zone RMSE",
                ),
                width="stretch",
                key="o1_improvement_scatter",
            )
        st.caption(
            "Positive percentages mean that enrichment reduced RMSE; negative percentages mean that RMSE increased."
        )

        # List each matched zone result so small changes remain visible when map colours look similar.
        improvement_table = selected_improvement[
            [
                "LocationID",
                "borough",
                "zone",
                "Baseline zone RMSE",
                "Enriched zone RMSE",
                "Zone RMSE reduction (%)",
                "Baseline zone R2",
                "Enriched zone R2",
                "RMSE improved",
            ]
        ].sort_values("Zone RMSE reduction (%)", ascending=False)
        improvement_table["Baseline zone R²"] = improvement_table[
            "Baseline zone R2"
        ].map(_display_number_or_not_defined)
        improvement_table["Enriched zone R²"] = improvement_table[
            "Enriched zone R2"
        ].map(_display_number_or_not_defined)
        improvement_table = improvement_table.drop(
            columns=["Baseline zone R2", "Enriched zone R2"]
        ).rename(columns={"RMSE improved": "RMSE improved?"})
        st.dataframe(
            improvement_table,
            hide_index=True,
            width="stretch",
            height=390,
            column_config={
                "Baseline zone RMSE": st.column_config.NumberColumn(format="%.3f"),
                "Enriched zone RMSE": st.column_config.NumberColumn(format="%.3f"),
                "Zone RMSE reduction (%)": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

        # Compare each published metric in its own chart and measurement scale.
        st.markdown("#### Published overall model comparison")
        st.caption(
            "These fixed charts show all eight models together, so they intentionally remain visible when a selector changes. "
            "Each metric has its own scale and interpretation."
        )
        selected_overall = model_slice(
            data.overall_metrics,
            dataset=dataset_choice,
            algorithm=algorithm_choice,
            configuration=configuration_choice,
        ).iloc[0]
        st.info(
            f"Current selection: {dataset_choice} · {algorithm_choice} · {configuration_choice} — "
            f"RMSE {_metric_value(float(selected_overall['RMSE']))}, "
            f"MAE {_metric_value(float(selected_overall['MAE']))}, "
            f"R² {_metric_value(float(selected_overall['R2']))}."
        )
        compare_left, compare_right = st.columns(2)
        compare_left.plotly_chart(
            metric_bar_chart(
                data.overall_metrics,
                metric="RMSE",
                title="Overall test RMSE",
                lower_is_better=True,
            ),
            width="stretch",
            key="o1_rmse_comparison",
        )
        compare_right.plotly_chart(
            metric_bar_chart(
                data.overall_metrics,
                metric="MAE",
                title="Overall test MAE",
                lower_is_better=True,
            ),
            width="stretch",
            key="o1_mae_comparison",
        )
        st.plotly_chart(
            metric_bar_chart(
                data.overall_metrics,
                metric="R2",
                title="Overall test R²",
                lower_is_better=False,
            ),
            width="stretch",
            key="o1_r2_comparison",
        )

        # Provide a sortable zone-level table for detailed diagnostic lookup.
        table = filtered_zone_metrics[
            [
                "LocationID",
                "borough",
                "zone",
                "Mean actual demand",
                "Zone RMSE",
                "Zone MAE",
                "Zone R2",
                "Mean bias",
                "Unusually high error",
            ]
        ].sort_values("Zone RMSE", ascending=False)
        table["Zone R²"] = table["Zone R2"].map(_display_number_or_not_defined)
        table = table.drop(columns=["Zone R2"])
        st.markdown("#### Zone-level diagnostic table")
        st.caption(
            "Not defined means actual pickups did not change during the test period, so R² cannot be calculated. "
            "This is not missing data; RMSE and MAE still show the prediction error."
        )
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            height=390,
            column_config={
                "Zone RMSE": st.column_config.NumberColumn(format="%.3f"),
                "Zone MAE": st.column_config.NumberColumn(format="%.3f"),
                "Mean bias": st.column_config.NumberColumn(format="%.3f"),
                "Mean actual demand": st.column_config.NumberColumn(format="%.3f"),
            },
        )


# Objective 2 evaluates contextual enrichment in the fixed 49-zone subgroup.
with tab2:
    _section_intro(
        "OBJECTIVE 2",
        "Contextual enrichment in low-demand outer-borough zones",
        "Test whether subway proximity and zero-vehicle household rate improve prediction accuracy in the 49 zones defined from training-period demand only.",
    )

    # Let the viewer compare matching algorithms and tuning configurations.
    o2_controls = st.columns([1, 1, 2])
    with o2_controls[0]:
        o2_algorithm = st.selectbox(
            "Algorithm",
            ["Random Forest", "XGBoost"],
            key="o2_algorithm",
        )
    with o2_controls[1]:
        o2_configuration = st.selectbox(
            "Configuration",
            ["Tuned", "Default"],
            key="o2_configuration",
        )
    with o2_controls[2]:
        st.markdown(
            "<div class='method-chip'>Low-demand rule · outer-borough training mean ≤ 0.2093 pickups/hour</div>",
            unsafe_allow_html=True,
        )

    # Retrieve the pooled and zone-level records for the selected comparison.
    pooled_baseline = model_slice(
        data.low_demand_pooled,
        dataset="Baseline",
        algorithm=o2_algorithm,
        configuration=o2_configuration,
    ).iloc[0]
    pooled_enriched = model_slice(
        data.low_demand_pooled,
        dataset="Enriched",
        algorithm=o2_algorithm,
        configuration=o2_configuration,
    ).iloc[0]
    low_improvement = data.low_demand_improvements.loc[
        data.low_demand_improvements["Algorithm"].eq(o2_algorithm)
        & data.low_demand_improvements["Configuration"].eq(o2_configuration)
    ].iloc[0]
    enriched_low_summary = model_slice(
        data.low_demand_summary,
        dataset="Enriched",
        algorithm=o2_algorithm,
        configuration=o2_configuration,
    ).iloc[0]

    # Summarise pooled error reduction and the number of improved low-demand zones.
    o2_kpis = st.columns(5)
    o2_kpis[0].metric("Low-demand zones", "49")
    o2_kpis[1].metric("Baseline pooled RMSE", _metric_value(float(pooled_baseline["RMSE"])))
    o2_kpis[2].metric("Enriched pooled RMSE", _metric_value(float(pooled_enriched["RMSE"])))
    pooled_reduction = 100 * (
        float(pooled_baseline["RMSE"]) - float(pooled_enriched["RMSE"])
    ) / float(pooled_baseline["RMSE"])
    o2_kpis[3].metric("Pooled RMSE reduction", f"{pooled_reduction:.2f}%")
    o2_kpis[4].metric(
        "Zones improved",
        f"{int(low_improvement['Improved_zones'])} / {int(low_improvement['Low_demand_zones'])}",
        help=f"Negative-R² zones after enrichment: {int(enriched_low_summary['Negative_zone_R2_count'])}",
    )

    # Join the 49 zone improvements to geometry for the enrichment map.
    selected_low_improvement = data.low_demand_zone_improvements.loc[
        data.low_demand_zone_improvements["Algorithm"].eq(o2_algorithm)
        & data.low_demand_zone_improvements["Configuration"].eq(o2_configuration)
    ].copy()
    low_map = analysis_map[["LocationID", "geometry"]].merge(
        selected_low_improvement,
        on="LocationID",
        how="inner",
        validate="one_to_one",
    )
    low_map = gpd.GeoDataFrame(low_map, geometry="geometry", crs=analysis_map.crs)

    # Pair the improvement map with a baseline-versus-enriched comparison chart.
    left, right = st.columns([1.15, 0.85])
    with left:
        st.plotly_chart(
            zone_choropleth(
                low_map,
                value_column="Zone RMSE reduction (%)",
                title="RMSE reduction after contextual enrichment",
                color_scale="Tealgrn",
            ),
            width="stretch",
            key="o2_map",
        )
    with right:
        st.plotly_chart(
            baseline_enriched_zone_scatter(
                selected_low_improvement,
                title="Baseline versus enriched zone RMSE",
            ),
            width="stretch",
            key="o2_scatter",
        )
        st.success(
            "All 49 low-demand outer-borough zones recorded lower RMSE after enrichment for this matched model comparison.",
            icon="✅",
        )

    # Show only the tuned-model importance evidence that was actually saved by the notebooks.
    st.markdown("#### Which inputs did the tuned enriched model rely on?")
    if o2_configuration == "Tuned":
        importance_model = o2_algorithm
        st.plotly_chart(
            feature_importance_chart(
                data.enriched_importance,
                model=importance_model,
                title=f"Permutation importance · Tuned enriched {importance_model}",
            ),
            width="stretch",
            key="o2_importance",
        )
        st.warning(
            "Permutation importance shows how strongly the fitted model relied on each input. It does not prove that an input caused taxi demand or caused the improvement.",
            icon="⚠️",
        )
    else:
        st.info(
            "Permutation importance was calculated only for the tuned models in the completed modelling notebooks. "
            "No Default-model importance values were saved, so the dashboard does not invent them.",
            icon="ℹ️",
        )

    # Quantify target sparsity to explain unstable R² in very low-demand zones.
    low_ids = set(data.low_demand_definition["LocationID"])
    low_actual = data.enriched_predictions.loc[
        data.enriched_predictions["LocationID"].isin(low_ids)
    ]
    zero_share = float(low_actual["actual_pickups"].eq(0).mean() * 100)
    st.markdown("#### Why R² remains difficult in low-demand zones")
    st.caption(
        f"Across the 21,168 low-demand test observations, {zero_share:.2f}% record zero pickups. "
        "When actual demand has very little variation, a small absolute miss can still produce a negative R². "
        "R² is not defined when actual pickups did not change at all. This is not missing data."
    )
    # Present the detailed 49-zone improvement results for audit and comparison.
    o2_table = selected_low_improvement[
        [
            "LocationID",
            "borough",
            "zone",
            "Mean actual demand",
            "Baseline zone RMSE",
            "Enriched zone RMSE",
            "Zone RMSE reduction (%)",
            "Enriched zone R2",
            "RMSE improved",
        ]
    ].sort_values("Zone RMSE reduction (%)", ascending=False)
    o2_table["Enriched zone R²"] = o2_table["Enriched zone R2"].map(
        _display_number_or_not_defined
    )
    o2_table = o2_table.drop(columns=["Enriched zone R2"])
    st.dataframe(
        o2_table,
        hide_index=True,
        width="stretch",
        height=410,
        column_config={
            "Baseline zone RMSE": st.column_config.NumberColumn(format="%.3f"),
            "Enriched zone RMSE": st.column_config.NumberColumn(format="%.3f"),
            "Zone RMSE reduction (%)": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )


# Objective 3 maps historical predicted demand and accessibility-based deployment evidence.
with tab3:
    _section_intro(
        "OBJECTIVE 3",
        "Predicted demand and areas to check for more taxi service",
        "See where the final model expected higher taxi demand, then use subway distance and household vehicle access to decide which areas should be checked more closely.",
    )

    # Derive available dates directly from the 432 stored test-period hours.
    test_hours = pd.DatetimeIndex(
        sorted(pd.to_datetime(data.enriched_predictions["pickup_hour"]).unique())
    )
    test_dates = list(test_hours.normalize().date)
    test_dates = list(dict.fromkeys(test_dates))

    # Collect location, time, map-layer, and zone-profile selections in one row.
    o3_controls = st.columns([1.35, 1.0, 0.8, 1.35, 1.8])
    o3_borough_options = sorted(planning_evidence["borough"].dropna().unique())
    with o3_controls[0]:
        o3_boroughs = st.multiselect(
            "Borough",
            o3_borough_options,
            default=o3_borough_options,
            key="o3_boroughs",
        )
    with o3_controls[1]:
        o3_date = st.selectbox(
            "Historical test date",
            test_dates,
            format_func=lambda value: value.strftime("%d %b %Y"),
            key="o3_date",
        )
    with o3_controls[2]:
        o3_hour = st.slider("Hour", min_value=0, max_value=23, value=0, key="o3_hour")

    # Map readable control labels to the source-backed evidence columns.
    layer_labels = {
        "Predicted demand": "Predicted pickups",
        "Actual demand": "Actual pickups",
        "Subway distance (km)": "subway_proximity_km",
        "Zero-vehicle household rate": "zero_vehicle_household_rate",
    }
    with o3_controls[3]:
        layer_label = st.selectbox(
            "Map layer",
            list(layer_labels),
            key="o3_layer",
        )

    # Build one evidence record per evaluated zone for the selected historical hour.
    selected_hour = pd.Timestamp(
        f"{o3_date:%Y-%m-%d} {int(o3_hour):02d}:00:00"
    )
    demand_evidence, demand_threshold = build_demand_evidence(data, selected_hour)
    filtered_evidence = demand_evidence.loc[
        demand_evidence["borough"].isin(o3_boroughs)
    ].copy()
    focus_options = sorted(filtered_evidence["zone"].dropna().unique())
    all_zones_option = "All selected zones"
    zone_options = [all_zones_option, *focus_options]
    # Reset a stale zone automatically when its borough is removed from the scope.
    if st.session_state.get("o3_focus_zone", all_zones_option) not in zone_options:
        st.session_state["o3_focus_zone"] = all_zones_option
    with o3_controls[4]:
        if focus_options:
            o3_focus_zone = st.selectbox(
                "Zone evidence profile",
                zone_options,
                index=0,
                key="o3_focus_zone",
            )
        else:
            o3_focus_zone = all_zones_option
            st.selectbox(
                "Zone evidence profile",
                [all_zones_option],
                disabled=True,
                key="o3_focus_zone",
            )

    # Render either an empty state or the full demand-and-deployment evidence view.
    if filtered_evidence.empty:
        _render_empty("No zones match the selected borough filter.")
    else:
        all_scope = o3_focus_zone == all_zones_option
        scope_evidence = (
            filtered_evidence
            if all_scope
            else filtered_evidence.loc[filtered_evidence["zone"].eq(o3_focus_zone)]
        ).copy()
        # Show aggregate demand and planning KPIs only for the overall or borough scope.
        if all_scope:
            peak = filtered_evidence.sort_values("Predicted pickups", ascending=False).iloc[0]
            predicted_total = float(filtered_evidence["Predicted pickups"].sum())
            actual_total = float(filtered_evidence["Actual pickups"].sum())
            o3_kpis = st.columns(5)
            o3_kpis[0].metric("Total predicted pickups", f"{predicted_total:,.1f}")
            o3_kpis[1].metric("Total actual pickups", f"{actual_total:,.0f}")
            o3_kpis[2].metric(
                "Prediction difference",
                f"{predicted_total - actual_total:+,.1f}",
            )
            o3_kpis[3].metric("Peak demand zone", str(peak["zone"]))
            o3_kpis[4].metric(
                "Areas to check",
                f"{int(filtered_evidence['Deployment consideration'].sum()):,}",
                help=(
                    "Number of areas with high expected taxi demand and at least one "
                    "travel-choice concern at the selected hour."
                ),
            )
            st.caption(
                "These areas have high expected taxi demand and may have fewer easy travel options. "
                "Check current taxi availability and passenger waiting times before deciding whether more taxis are needed."
            )

        # Join the selected hourly evidence to map geometry without changing other tabs.
        mapped_evidence = analysis_map[["LocationID", "geometry"]].merge(
            filtered_evidence,
            on="LocationID",
            how="inner",
            validate="one_to_one",
        )
        mapped_evidence = gpd.GeoDataFrame(
            mapped_evidence,
            geometry="geometry",
            crs=analysis_map.crs,
        )
        selected_layer = layer_labels[layer_label]
        # Pass the selected LocationID so the map highlights and centres on that zone.
        focus_location_id = None
        # Resolve an individual zone only when the overall-scope option is inactive.
        if not all_scope:
            focus_location_id = int(
                filtered_evidence.loc[
                    filtered_evidence["zone"].eq(o3_focus_zone), "LocationID"
                ].iloc[0]
            )
        st.plotly_chart(
            zone_choropleth(
                mapped_evidence,
                value_column=selected_layer,
                title=f"{layer_label} · {selected_hour:%d %b %Y, %H:00}",
                color_scale="YlOrRd" if selected_layer in {"Predicted pickups", "Actual pickups"} else "PuBuGn",
                selected_location_id=focus_location_id,
            ),
            width="stretch",
            key="o3_map",
        )
        st.caption(
            "All selected zones keeps the full borough scope visible; choosing one zone outlines and centres it. "
            "Change the date or hour to redraw the demand heatmap from the stored test-period predictions."
        )

        # Replace the individual profile with a concise summary for aggregate scopes.
        if all_scope:
            all_boroughs_selected = set(o3_boroughs) == set(o3_borough_options)
            scope_title = (
                "Overall NYC evidence"
                if all_boroughs_selected
                else "Selected borough evidence · " + ", ".join(o3_boroughs)
            )
            st.markdown(
                f"""
                <div class="zone-profile">
                  <div>
                    <div class="eyebrow">CURRENT SCOPE</div>
                    <h3>{scope_title}</h3>
                    <p>{len(filtered_evidence)} evaluated zones are included in the map, KPIs, table and historical trend.</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Keep the existing context and reliability evidence for an individual zone.
            focus = filtered_evidence.loc[
                filtered_evidence["zone"].eq(o3_focus_zone)
            ].iloc[0]
            status_class = "status-review" if bool(focus["Deployment consideration"]) else "status-clear"
            status_text = (
                "Check this area"
                if bool(focus["Deployment consideration"])
                else "Not marked for checking at this hour"
            )
            plain_reasons = (
                str(focus["Deployment reasons"])
                .replace("high predicted demand", "high expected taxi demand")
                .replace("limited subway access", "farther from a subway")
                .replace(
                    "high zero-vehicle household rate",
                    "many households without a vehicle",
                )
                .replace(
                    "deployment threshold not met",
                    "no area-checking condition was met",
                )
            )
            st.markdown(
                f"""
                <div class="zone-profile">
                  <div>
                    <div class="eyebrow">SELECTED ZONE · LOCATIONID {int(focus['LocationID'])}</div>
                    <h3>{focus['zone']} · {focus['borough']}</h3>
                    <p>{plain_reasons}</p>
                  </div>
                  <div class="{status_class}">{status_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            profile_kpis = st.columns(6)
            profile_kpis[0].metric(
                "Predicted pickups",
                f"{float(focus['Predicted pickups']):,.1f}",
            )
            profile_kpis[1].metric(
                "Actual pickups",
                f"{float(focus['Actual pickups']):,.0f}",
            )
            profile_kpis[2].metric(
                "Subway distance",
                f"{float(focus['subway_proximity_km']):.2f} km",
            )
            profile_kpis[3].metric(
                "No-vehicle homes",
                f"{float(focus['zero_vehicle_household_rate']) * 100:.1f}%",
            )
            profile_kpis[4].metric(
                "Prediction error",
                _metric_value(float(focus["Zone RMSE"])),
                help="The usual size of this area's prediction errors across all 432 test hours. Lower is better.",
            )
            profile_kpis[5].metric(
                "Demand pattern",
                _metric_value(float(focus["Zone R2"])),
                help="How well the model followed changes in this area's demand. Higher is generally better.",
            )

        # Aggregate the full 432-hour trend to the current borough or zone scope.
        focus_history = aggregate_scope_history(
            data.enriched_predictions,
            boroughs=o3_boroughs,
            zone=None if all_scope else o3_focus_zone,
            prediction="random_forest_tuned",
        )
        if all_scope:
            trend_scope = (
                "All NYC zones"
                if set(o3_boroughs) == set(o3_borough_options)
                else ", ".join(o3_boroughs)
            )
            trend_title = f"Aggregated historical test evidence · {trend_scope}"
        else:
            trend_title = f"Historical test evidence · {o3_focus_zone}"
        st.plotly_chart(
            prediction_trend_chart(
                focus_history,
                prediction="random_forest_tuned",
                title=trend_title,
                selected_hour=selected_hour,
            ),
            width="stretch",
            key="o3_trend",
        )

        # Explain the screening rule in plain English before listing the evidence.
        st.markdown("#### How areas are selected for checking")
        st.markdown(
            "High expected taxi demand means the area is among the busiest quarter of the 259 evaluated zones "
            "at the selected hour. An area may have fewer easy travel choices when it is among the quarter "
            "farthest from a subway station or among the quarter with the most households without a vehicle. "
            "An area is marked only when high expected demand and at least one travel-choice condition are present."
        )
        st.caption(
            f"For this selected hour, the demand cut-off is {demand_threshold:.3f} predicted pickups. "
            f"The fixed comparison points are {planning_thresholds.subway_q75_km:.3f} km from a subway and "
            f"{planning_thresholds.zero_vehicle_q75:.1%} of households without a vehicle. "
            "These are dashboard comparison limits, not official NYC service rules."
        )
        st.markdown("#### Areas to check and supporting evidence")
        planning_table = scope_evidence[
            [
                "LocationID",
                "borough",
                "zone",
                "Predicted pickups",
                "Actual pickups",
                "Prediction difference",
                "subway_proximity_km",
                "zero_vehicle_household_rate",
                "Deployment consideration",
                "Reliability warning",
                "Deployment reasons",
            ]
        ].sort_values(
            ["Deployment consideration", "Predicted pickups"],
            ascending=[False, False],
        )
        planning_table["Deployment reasons"] = (
            planning_table["Deployment reasons"]
            .str.replace("high predicted demand", "high expected taxi demand", regex=False)
            .str.replace("limited subway access", "farther from a subway", regex=False)
            .str.replace(
                "high zero-vehicle household rate",
                "many households without a vehicle",
                regex=False,
            )
            .str.replace(
                "deployment threshold not met",
                "no area-checking condition was met",
                regex=False,
            )
        )
        planning_table = planning_table.rename(
            columns={
                "Deployment consideration": "Area marked for checking",
                "Reliability warning": "Use prediction carefully",
                "Deployment reasons": "Why the area was marked",
            }
        )
        st.dataframe(
            planning_table,
            hide_index=True,
            width="stretch",
            height=430,
            column_config={
                "Predicted pickups": st.column_config.NumberColumn(format="%.2f"),
                "Actual pickups": st.column_config.NumberColumn(format="%.0f"),
                "Prediction difference": st.column_config.NumberColumn(format="%+.2f"),
                "subway_proximity_km": st.column_config.NumberColumn(
                    "Subway distance (km)", format="%.3f"
                ),
                "zero_vehicle_household_rate": st.column_config.NumberColumn(
                    "Zero-vehicle rate", format="%.3f"
                ),
            },
        )
        st.warning(
            "This is historical test-period planning evidence, not a live forecast or an automated deployment instruction. The dashboard does not measure current taxi supply, passenger waiting time, unmet demand, or service adequacy; planners must review those operational factors before deployment decisions.",
            icon="⚠️",
        )


# Close the page with the evaluation period and modelling-scope reminder.
st.markdown(
    """
    <div class="footer-note">
      <strong>Evidence coverage:</strong> NYC yellow taxi pickup counts from January–March 2025. Final evaluation uses 432 untouched hours from 14–31 March 2025. LocationID is retained for reporting and was not used as a model predictor.
    </div>
    """,
    unsafe_allow_html=True,
)
