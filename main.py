import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_predict, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import Utils as Utl

# try xgboost
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# ---------- config ----------
DATA_PATH = "NewDF - Sheet1.csv"
OUT_DIR = "./models_s_proposed"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- load ----------
df = pd.read_csv(DATA_PATH)

# ---------- check required columns ----------
required_cols = ["S Propesed", "phi", "Stress"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

# ---------- coerce numeric and report parsing issues ----------
def to_numeric_col(df, colname):
    # remove common thousands separators and convert comma decimals -> dot if needed
    s = df[colname].astype(str).str.strip()
    # try handle comma decimal like "1,23" -> "1.23" but be careful with thousand separators "1,234"
    # heuristic: if there is a comma and also a dot -> remove thousands commas
    s2 = s.copy()
    has_comma = s2.str.contains(',', regex=False)
    has_dot = s2.str.contains('\.', regex=True)
    # if both comma and dot exist, remove commas
    mask_both = has_comma & has_dot
    s2.loc[mask_both] = s2.loc[mask_both].str.replace(',', '')
    # if comma exists but no dot, we assume comma is decimal sep and replace with dot
    mask_comma_only = has_comma & (~has_dot)
    s2.loc[mask_comma_only] = s2.loc[mask_comma_only].str.replace(',', '.')
    # remove any spaces
    s2 = s2.str.replace(' ', '')
    # coerce to numeric
    coerced = pd.to_numeric(s2, errors='coerce')
    n_total = len(coerced)
    n_bad = coerced.isna().sum()
    if n_bad > 0:
        print(f"Warning: column '{colname}' has {n_bad}/{n_total} non-numeric values after coercion. They become NaN.")
    return coerced

df["S_Prop_num"] = to_numeric_col(df, "S Propesed")
df["stress_num"] = to_numeric_col(df, "Stress")
df["phi_true_num"] = to_numeric_col(df, "phi")

# drop rows missing required numeric values
before = len(df)
df = df.dropna(subset=["S_Prop_num", "stress_num", "phi_true_num", "qc", "fs", "u2", "Effective Stress", "Fr", "Qt", "Depth"]).reset_index(drop=True)
after = len(df)
print(f"Dropped {before - after} rows with missing numeric S/stress/phi. Remaining rows: {after}")

if len(df) == 0:
    raise RuntimeError("No rows left after numeric coercion. Fix CSV numeric formatting.")

# ---------- build features for phi prediction ----------
feature_cols = [
    "S_Prop_num", "qc", "fs", "u2", "Effective Stress", "Fr", "Qt", "Depth"
]
# Make sure all exist and convert to numeric:
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

X_raw = df[feature_cols].values
y_phi = df["phi_true_num"].values
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_raw)
scaler = RobustScaler().fit(X_poly)
X = scaler.transform(X_poly)

# ---------- train a robust phi model (RandomizedSearchCV) ----------
if XGB_AVAILABLE:
    base_model = XGBRegressor(objective="reg:squarederror", random_state=42, verbosity=0)
    param_dist = {
        "n_estimators": [300, 500],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5],
        "min_child_weight": [1, 5, 10],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "reg_lambda": [0.1, 1, 5],
    }
else:
    base_model = RandomForestRegressor(random_state=42)
    param_dist = {
        "n_estimators": [200, 400],
        "max_features": ["sqrt", 1.0],
        "max_depth": [4, 6, None],
        "min_samples_leaf": [1, 2],
    }


rs = RandomizedSearchCV(base_model, param_distributions=param_dist,
                        n_iter=10, scoring="neg_mean_absolute_error",
                        cv=4, random_state=42, n_jobs=1, verbose=0)
print("Running hyperparameter search for phi model (this may take a short while)...")
rs.fit(X, y_phi)
best_model_phi = rs.best_estimator_
print("Best params (phi):", rs.best_params_)

# ---------- cross-validated prediction diagnostics ----------
cv = KFold(n_splits=5, shuffle=True, random_state=42)
y_phi_pred_cv = cross_val_predict(best_model_phi, X, y_phi, cv=cv, n_jobs=1)

print("Phi CV metrics:")
print(" MAE:", mean_absolute_error(y_phi, y_phi_pred_cv))
print(" RMSE:", np.sqrt(mean_squared_error(y_phi, y_phi_pred_cv)))
print(" R2:", r2_score(y_phi, y_phi_pred_cv))

# train final phi model on full data
best_model_phi.fit(X, y_phi)
DF_C, COMB_Model, c, d = Utl.augment_and_retrain(df, 500, 'bootstrap')
# ---------- predict phi on full df ----------
DF_C["phi_pred"] = COMB_Model.predict(X)
# make sure numeric
DF_C["phi_pred_num"] = pd.to_numeric(DF_C["phi_pred"], errors='coerce')

# ---------- diagnostics: show sample and stats ----------
print("\nSample rows (S_proposed, stress, phi_true, phi_pred):")
print(DF_C[["S_Prop_num", "stress_num", "phi_true_num", "phi_pred_num"]].head(10).to_string(index=False))

print("\nSummary stats:")
for col in ["S_Prop_num", "stress_num", "phi_true_num", "phi_pred_num"]:
    arr = DF_C[col].values
    print(f" {col} -> min:{np.nanmin(arr):.6g}, median:{np.nanmedian(arr):.6g}, mean:{np.nanmean(arr):.6g}, max:{np.nanmax(arr):.6g}")

# ---------- compute tan(phi_pred) safely ----------
# Convert deg->rad; if phi predictions are suspiciously large, print a warning.
phi_pred_rad = np.deg2rad(DF_C["phi_pred_num"].values.astype(float))
# compute tan but clip to avoid overflow; limit tan to a max absolute value (e.g. 1e6)
tan_phi_pred = np.tan(phi_pred_rad)
# detect any infinities / extreme values
inf_mask = ~np.isfinite(tan_phi_pred)
if inf_mask.any():
    print(f"Warning: tan(phi_pred) produced non-finite values for {inf_mask.sum()} rows. They will be set to large finite value.")
    tan_phi_pred[inf_mask] = np.sign(tan_phi_pred[~inf_mask].mean() if (~inf_mask).any() else 0.0) * 1e6

# clip extreme magnitude
tan_clip_limit = 1e6
tan_phi_pred = np.clip(tan_phi_pred, -tan_clip_limit, tan_clip_limit)
DF_C["tan_phi_pred"] = tan_phi_pred

# ---------- compute C = S - stress * tan(phi_pred) ----------
S_vals = DF_C["S_Prop_num"].values.astype(float)
stress_vals = DF_C["stress_num"].values.astype(float)
DF_C["C_computed"] = S_vals - stress_vals * df["tan_phi_pred"].values

# ---------- diagnostics after compute ----------
print("\nAfter compute - summary for computed C:")
arrC = DF_C["C_computed"].values
print(f" C_computed -> min:{np.nanmin(arrC):.6g}, median:{np.nanmedian(arrC):.6g}, mean:{np.nanmean(arrC):.6g}, max:{np.nanmax(arrC):.6g}")
n_zero = np.sum(np.isclose(arrC, 0.0, atol=1e-12))
print(f" Count exactly zero (within 1e-12): {n_zero} / {len(arrC)}")
print(DF_C[["S_Prop_num", "stress_num", "phi_true_num", "phi_pred_num", "C_computed"]].head(10).to_string(index=False))
# show rows where C_computed is very close to zero for inspection
if n_zero > 0:
    print("\nRows where C_computed approximately zero (first 10):")
    print(df.loc[np.isclose(df["C_computed"].values, 0.0, atol=1e-12),
                 ["S_Prop_num","stress_num","phi_true_num","phi_pred_num","tan_phi_pred","C_computed"]].head(10).to_string(index=False))

# ---------- quick sanity checks that often reveal root cause ----------
print("\nSanity checks:")
print("  Are all stress values the same? ->", np.allclose(stress_vals, stress_vals[0]))
print("  Are all S values the same? ->", np.allclose(S_vals, S_vals[0]))
print("  Are all phi_pred the same? ->", np.allclose(DF_C["phi_pred_num"].values, DF_C["phi_pred_num"].values[0]))
print("  Any NaN in phi_pred? ->", np.isnan(df["phi_pred_num"].values).sum())
print("  Any NaN in stress? ->", np.isnan(stress_vals).sum())


# ---------- plots ----------
plt.figure(figsize=(6,6))
plt.scatter(DF_C["phi_true_num"], DF_C["phi_pred_num"], alpha=0.7, edgecolors='k')
mn = min(DF_C["phi_true_num"].min(), DF_C["phi_pred_num"].min())
mx = max(DF_C["phi_true_num"].max(), DF_C["phi_pred_num"].max())
plt.plot([mn,mx],[mn,mx],"r--")
plt.xlabel("phi true (deg)")
plt.ylabel("phi predicted (deg)")
plt.title("phi: true vs predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"phi_true_vs_pred.png"), dpi=150)
plt.show()

if "C" in df.columns:
    plt.figure(figsize=(6,6))
    plt.scatter(df["C"], df["C_computed"], alpha=0.7, edgecolors='k')
    mn = min(df["C"].min(), df["C_computed"].min())
    mx = max(df["C"].max(), df["C_computed"].max())
    plt.plot([mn, mx], [mn, mx], "r--", label="y = x")
    plt.xlabel("True C")
    plt.ylabel("Computed C (from phi_pred)")
    plt.title("True vs Computed C")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "C_true_vs_computed.png"), dpi=150)
    plt.show()
else:
    print("Column 'C' not found in dataframe — cannot plot True vs Computed C.")

# save dataframe sample and full results
df.to_csv(os.path.join(OUT_DIR, "debug_predictions.csv"), index=False)
print("\nSaved debug CSV to:", os.path.join(OUT_DIR, "debug_predictions.csv"))