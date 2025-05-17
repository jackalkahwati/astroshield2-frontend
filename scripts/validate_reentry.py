import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths (allow fallback if expected locations are missing)
# -----------------------------------------------------------------------------
tip_path = Path('data/tip_window1.json')
if not tip_path.exists():
    tip_path = Path('tip_window1.json')

pred_path = Path('data/my_reentry_preds.csv')
if not pred_path.exists():
    pred_path = Path('my_reentry_preds.csv')

if not tip_path.exists():
    raise FileNotFoundError(f"Ground-truth file not found: {tip_path}")
if not pred_path.exists():
    raise FileNotFoundError(f"Prediction file not found: {pred_path}")

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
# TIP window ground truth JSON (list of dicts)
truth_df = pd.read_json(tip_path)
# Ensure datetime parsing
truth_df['DECAY_EPOCH'] = pd.to_datetime(truth_df['DECAY_EPOCH'], utc=True)
truth_df['NORAD_CAT_ID'] = truth_df['NORAD_CAT_ID'].astype(str)
truth_df['LAT'] = truth_df['LAT'].astype(float)
truth_df['LON'] = truth_df['LON'].astype(float)

# Predictions CSV
dtype = {
    'NORAD_CAT_ID': str,
    't_pred': str,
    'lat_pred': float,
    'lon_pred': float,
}
preds_df = pd.read_csv(pred_path, dtype=dtype)
# Parse prediction timestamp
preds_df['t_pred'] = pd.to_datetime(preds_df['t_pred'], utc=True)

# -----------------------------------------------------------------------------
# Merge on NORAD_CAT_ID
# -----------------------------------------------------------------------------
merged = pd.merge(
    preds_df,
    truth_df[['NORAD_CAT_ID', 'DECAY_EPOCH', 'LAT', 'LON']],
    on='NORAD_CAT_ID',
    how='inner'
)

if merged.empty:
    raise ValueError("No matching NORAD_CAT_IDs between prediction and ground truth files.")

# -----------------------------------------------------------------------------
# Helper: haversine distance (vectorized)
# -----------------------------------------------------------------------------
R_EARTH_KM = 6371.0

lat1 = np.radians(merged['LAT'])
lon1 = np.radians(merged['LON'])
lat2 = np.radians(merged['lat_pred'])
lon2 = np.radians(merged['lon_pred'])

dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
c = 2 * np.arcsin(np.sqrt(a))
merged['dR_km'] = R_EARTH_KM * c

# -----------------------------------------------------------------------------
# ΔT (minutes)
# -----------------------------------------------------------------------------
merged['dT_min'] = (merged['t_pred'] - merged['DECAY_EPOCH']).dt.total_seconds() / 60.0

# Reorder columns
cols = ['NORAD_CAT_ID', 'DECAY_EPOCH', 't_pred', 'dT_min', 'LAT', 'LON', 'lat_pred', 'lon_pred', 'dR_km']
merged = merged[cols]

# -----------------------------------------------------------------------------
# Save artifacts
# -----------------------------------------------------------------------------
artifacts_dir = Path('artifacts')
artifacts_dir.mkdir(exist_ok=True)

delta_report_path = artifacts_dir / 'delta_report.csv'
metrics_summary_path = artifacts_dir / 'metrics_summary.csv'

merged.to_csv(delta_report_path, index=False)

# Summary metrics
summary = {
    'N_events': len(merged),
    'median_abs_dT_min': merged['dT_min'].abs().median(),
    'median_dR_km': merged['dR_km'].median(),
    'p95_dR_km': merged['dR_km'].quantile(0.95),
}

pd.DataFrame([summary]).to_csv(metrics_summary_path, index=False)

# -----------------------------------------------------------------------------
# Console output (concise)
# -----------------------------------------------------------------------------
print("✅ Validation complete")
print()
print(f"• N events = {summary['N_events']}")
print(f"• Median |ΔT| = {summary['median_abs_dT_min']:.2f} min")
print(f"• Median ΔR = {summary['median_dR_km']:.2f} km")
print(f"• 95-percentile ΔR = {summary['p95_dR_km']:.2f} km")
print()
print("Files saved:")
print(f"- {delta_report_path}")
print(f"- {metrics_summary_path}") 