# Verification record

Date: 17 October 2026. Environment: Windows PowerShell, Python 3.10.9; existing system pandas. No package install or model training performed. Copied virtual environments were not used.

## Executed

```powershell
python docs/verify_evidence.py
```

Exit 0. Confirmed 12 notebook JSON documents; 565,658 unique committed zone-hour rows; 262 zones; 2,159 hourly timestamps; nonnegative counts; 10,218,621 total pickups; 262 taxi-zone GeoPackage records.

The script reports an expected **dashboard blocker**: `processed_outputs/model_results/evaluation/zone_level_test_metrics.csv` is missing. The model-results directory is ignored and unavailable in this clone. Full dashboard behavior, model inference and metric recomputation have not been verified. System GeoPandas is also absent; install dashboard requirements in a fresh environment before launching.

## Figure provenance

`docs/assets/historical-test-rmse.png` is decoded directly from the saved `image/png` output in `model_building/04_Overall_and_Zone_Level_Model_Evaluation.ipynb`, the cell calling `plot_overall_metric` for overall test RMSE. It was visually inspected for legibility. README metric values were checked against that notebook's saved overall table and low-demand table. They are historical recorded results, not a rerun. No dashboard screenshot is supplied because its analytical inputs are missing.

## Reproduction constraints

The original long README is preserved in `docs/PROJECT_GUIDE.md`. It documents machine-specific paths, the Census input-folder mismatch and the timestamp-format mismatch between trip export and integration. These are prerequisites for full pipeline reproduction. The check validates committed structure and data invariants; it does not validate all raw-data transformations, test-set isolation by execution, predictive correctness or dashboard interactions.
