import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import Utils as Utl

# Optional: XGBoost for better performance
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    from sklearn.ensemble import RandomForestRegressor
    XGB_AVAILABLE = False

# -------------------- CONFIG --------------------
DATA_PATH = "NewDF - Sheet1.csv"
OUT_DIR = "./models_s_proposed"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------- LOAD AND CLEAN DATA --------------------
df = pd.read_csv(DATA_PATH)

required_cols = ["S Propesed", "phi", "Stress"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

def to_numeric_col(df, colname):
    """Convert messy numeric strings to float."""
    s = df[colname].astype(str).str.strip()
    s = s.str.replace(' ', '')
    has_comma = s.str.contains(',', regex=False)
    has_dot = s.str.contains('\.', regex=True)
    mask_both = has_comma & has_dot
    s.loc[mask_both] = s.loc[mask_both].str.replace(',', '')
    mask_comma_only = has_comma & (~has_dot)
    s.loc[mask_comma_only] = s.loc[mask_comma_only].str.replace(',', '.')
    coerced = pd.to_numeric(s, errors='coerce')
    if coerced.isna().sum() > 0:
        print(f"Warning: {colname} -> {coerced.isna().sum()} NaN after coercion.")
    return coerced

df["S_Prop_num"] = to_numeric_col(df, "S Propesed")
df["stress_num"] = to_numeric_col(df, "Stress")
df["phi_true_num"] = to_numeric_col(df, "phi")

feature_cols = ["S_Prop_num", "qc", "fs", "u2", "Effective Stress", "Fr", "Qt", "Depth"]
df = df.dropna(subset=feature_cols + ["phi_true_num", "stress_num"]).reset_index(drop=True)
print(f"Remaining rows after cleaning: {len(df)}")

# -------------------- SYNTHETIC DATA + TRAIN --------------------
print("\nGenerating bootstrap synthetic data and retraining...")
combined_df, model_phi, scaler, poly = Utl.augment_and_retrain(df, n_synth=500, method='bootstrap')

# -------------------- PREDICT ON REAL DATA --------------------
X_raw = df[feature_cols].values
X_poly = poly.transform(X_raw)
X_scaled = scaler.transform(X_poly)
df["phi_pred"] = model_phi.predict(X_scaled)
df["phi_pred_num"] = pd.to_numeric(df["phi_pred"], errors='coerce')

# -------------------- METRICS --------------------
mae = mean_absolute_error(df["phi_true_num"], df["phi_pred_num"])
rmse = np.sqrt(mean_squared_error(df["phi_true_num"], df["phi_pred_num"]))
r2 = r2_score(df["phi_true_num"], df["phi_pred_num"])

print("\nEvaluation on real (holdout) data:")
print(f" MAE  = {mae:.4f}")
print(f" RMSE = {rmse:.4f}")
print(f" R²   = {r2:.4f}")

# -------------------- COMPUTE C --------------------
phi_rad = np.deg2rad(df["phi_pred_num"])
tan_phi = np.tan(phi_rad)
tan_phi = np.clip(tan_phi, -1e6, 1e6)
df["tan_phi_pred"] = tan_phi

S_vals = df["S_Prop_num"].values
stress_vals = df["stress_num"].values
df["C_computed"] = S_vals - stress_vals * tan_phi

print("\nC_computed summary:")
print(f"  Min: {df['C_computed'].min():.3f}, Max: {df['C_computed'].max():.3f}, Mean: {df['C_computed'].mean():.3f}")

# -------------------- SAVE MODEL AND SCALER --------------------
joblib.dump(model_phi, os.path.join(OUT_DIR, "phi_model.joblib"))
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
joblib.dump(poly, os.path.join(OUT_DIR, "poly.joblib"))

# -------------------- PLOTS --------------------
plt.figure(figsize=(6,6))
plt.scatter(df["phi_true_num"], df["phi_pred_num"], alpha=0.7, edgecolors='k')
mn, mx = df["phi_true_num"].min(), df["phi_true_num"].max()
plt.plot([mn, mx], [mn, mx], "r--", label="y = x")
plt.xlabel("phi (true, deg)")
plt.ylabel("phi (predicted, deg)")
plt.title("Phi True vs Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "phi_true_vs_pred.png"), dpi=150)
plt.show()

if "C" in df.columns:
    plt.figure(figsize=(6,6))
    plt.scatter(df["C"], df["C_computed"], alpha=0.7, edgecolors='k')
    mn, mx = df["C"].min(), df["C"].max()
    plt.plot([mn, mx], [mn, mx], "r--", label="y = x")
    plt.xlabel("C true")
    plt.ylabel("C computed (from phi_pred)")
    plt.title("True vs Computed C")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "C_true_vs_computed.png"), dpi=150)
    plt.show()
else:
    print("No 'C' column in dataset; skipping C comparison plot.")

# -------------------- SAVE RESULTS --------------------
out_csv = os.path.join(OUT_DIR, "predictions_real_data.csv")
df.to_csv(out_csv, index=False)
print("\nSaved predictions to:", out_csv)
