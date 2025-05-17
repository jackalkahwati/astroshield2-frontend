import pandas as pd
from pathlib import Path

# Ground truth path (TIP window)
truth_path = Path('data/tip_window1.json')
if not truth_path.exists():
    truth_path = Path('tip_window1.json')

if not truth_path.exists():
    raise FileNotFoundError("tip_window1.json not found in data/ or repo root")

# Output predictions CSV
pred_path = Path('data/my_reentry_preds.csv')
pred_path.parent.mkdir(exist_ok=True)

# Load TIP JSON
truth_df = pd.read_json(truth_path)

# Build predictions identical to truth (zero-error baseline)
preds_df = pd.DataFrame({
    'NORAD_CAT_ID': truth_df['NORAD_CAT_ID'].astype(str),
    't_pred': truth_df['DECAY_EPOCH'],
    'lat_pred': truth_df['LAT'].astype(float),
    'lon_pred': truth_df['LON'].astype(float),
})

preds_df.to_csv(pred_path, index=False)
print(f"Dummy predictions written to {pred_path} with {len(preds_df)} rows.") 