"""Check committed data and extract a historical notebook figure; no model training."""
from pathlib import Path
import base64
import json
import sqlite3
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
outputs = ROOT / "processed_outputs"
demand = pd.read_csv(outputs / "yellow_taxi/nyc_yellow_taxi_zone_hour_demand_2025_q1.csv")
assert len(demand) == 565658
assert demand.LocationID.nunique() == 262
assert demand.pickup_hour.nunique() == 2159
assert not demand.duplicated(["LocationID", "pickup_hour"]).any()
assert demand.pickup_count.ge(0).all()
assert demand.pickup_count.sum() == 10218621
with sqlite3.connect(outputs / "taxi_zone/nyc_taxi_zones_cleaned.gpkg") as connection:
    count = connection.execute("SELECT COUNT(*) FROM nyc_taxi_zones").fetchone()[0]
assert count == 262
notebooks = [p for folder in ["data processing pipeline", "data understanding", "model_building"]
             for p in (ROOT / folder).glob("*.ipynb")]
for path in notebooks:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
print(f"PASS: {len(notebooks)} valid notebook documents; 565,658 unique zone-hours; "
      "262 zones; 2,159 hours; 10,218,621 pickups; 262 map polygons.")
evaluation = json.loads((ROOT / "model_building/04_Overall_and_Zone_Level_Model_Evaluation.ipynb").read_text(encoding="utf-8"))
figure_cell = next(c for c in evaluation["cells"] if 'title="Overall Test RMSE: Baseline and Enriched Models"' in ''.join(c.get("source", [])))
encoded = next(o["data"]["image/png"] for o in figure_cell["outputs"] if "image/png" in o.get("data", {}))
if isinstance(encoded, list):
    encoded = "".join(encoded)
asset = ROOT / "docs/assets/historical-test-rmse.png"
asset.parent.mkdir(parents=True, exist_ok=True)
asset.write_bytes(base64.b64decode(encoded))
print("EXTRACTED: original saved notebook RMSE figure (historical; no evaluation rerun).")
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "Dashboard"))
from dashboard_core.data import load_dashboard_data
try:
    load_dashboard_data(ROOT)
except FileNotFoundError as error:
    print(f"BLOCKED: dashboard source validation: {error}")
else:
    print("PASS: dashboard source validation")
