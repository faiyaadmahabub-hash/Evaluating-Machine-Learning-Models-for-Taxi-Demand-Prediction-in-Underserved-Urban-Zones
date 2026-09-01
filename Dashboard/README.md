# NYC Taxi Demand Evidence Dashboard

This local Streamlit application presents the completed FYP model evidence across three research objectives:

1. zone-level Random Forest and XGBoost performance;
2. contextual enrichment in 49 low-demand outer-borough zones; and
3. historical predicted-demand heatmaps and transparent deployment-consideration evidence combining final-model demand with accessibility context.

The application is historical decision support. It does not provide live prediction, calibrated confidence, congestion estimates, or automated fleet-deployment recommendations.

## Folder placement

Keep this `Dashboard` folder directly inside the FYP project folder. The application derives all source paths relative to that location and does not contain machine-specific data paths.

Expected structure:

```text
fyp/
├── Dashboard/
│   ├── app.py
│   ├── dashboard_core/
│   ├── assets/
│   └── requirements.txt
└── processed_outputs/
```

## Local setup

Use Python 3.11 or 3.12. From the `Dashboard` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If `python` is not recognised, use the Python environment that ran the project notebooks or repair/reinstall the Python launcher first.

If an older `.venv` reports `Unable to create process`, recreate it with the currently installed Python:

```powershell
Rename-Item .venv .venv_old
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deployment-consideration definition

A zone receives an exploratory deployment-consideration flag for the selected historical hour only when both conditions are true:

- **Demand pressure:** tuned enriched Random Forest predicted pickups are at or above the 75th percentile across the 259 evaluated zones for that hour; and
- **Context concern:** subway distance or zero-vehicle household rate is at or above the 75th percentile among the 259 evaluated zones.

The underlying reasons remain visible for every zone. Zone RMSE and R² remain visible only as reliability context. The flag prompts human investigation and is not proof that a community lacks taxi service or requires immediate deployment.

## Objective 3 selection behaviour

`All selected zones` is the default evidence profile. With all boroughs selected it aggregates the 259 evaluated zones; with a borough subset it aggregates only zones in that scope. Choosing one zone highlights and centres its polygon while retaining the existing zone-level subway, zero-vehicle, RMSE and R² evidence. The actual-versus-predicted trend follows the same NYC, borough, or individual-zone scope and marks the hour represented by the map.
