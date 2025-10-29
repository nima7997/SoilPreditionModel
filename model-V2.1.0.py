# train_phi_C_with_proper_holdout.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_predict, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import Utils as Utl

# optional XGBoost
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# ---------- CONFIG ----------
DATA_PATH = "NewDF - Sheet1.csv"
OUT_DIR = "./models_phi_C_bootstrap"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42
N_SYNTH = 200       # number of synthetic rows to generate
TEST_SIZE = 0.20     # holdout fraction of the *real* data
FEATURE_COLS = ["S_Prop_num", "qc", "fs", "u2", "Effective Stress", "Fr", "Qt", "Depth"]

# ---------- LOAD ----------
df = pd.read_csv(DATA_PATH)

# ---------- CHECK ----------
required_cols = ["S Propesed", "phi", "Stress"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

# ---------- COERCE NUMERIC ----------
def to_numeric_col(series):
    s = series.astype(str).str.strip()
    has_comma = s.str.contains(',', regex=False)
    has_dot = s.str.contains(r'\.', regex=True)
    mask_both = has_comma & has_dot
    s.loc[mask_both] = s.loc[mask_both].str.replace(',', '')
    mask_comma_only = has_comma & (~has_dot)
    s.loc[mask_comma_only] = s.loc[mask_comma_only].str.replace(',', '.')
    s = s.str.replace(' ', '')
    return pd.to_numeric(s, errors='coerce')

df["S_Prop_num"] = to_numeric_col(df["S Propesed"])
df["stress_num"] = to_numeric_col(df["Stress"])
df["phi_true_num"] = to_numeric_col(df["phi"])
if "C" in df.columns:
    df["C_true_num"] = to_numeric_col(df["C"])

# ---------- DROP INCOMPLETE ROWS (essential features) ----------
before = len(df)
required_for_drop = ["S_Prop_num", "stress_num", "phi_true_num"] + [c for c in FEATURE_COLS if c != "S_Prop_num"]
df = df.dropna(subset=required_for_drop).reset_index(drop=True)
after = len(df)
print(f"Dropped {before - after} rows with missing essential values. Remaining rows: {after}")
if len(df) == 0:
    raise RuntimeError("No data left after dropping missing values.")

# ensure feature cols numeric
for c in FEATURE_COLS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# ---------- SPLIT real data -> train_real / holdout_real (holdout must be kept aside) ----------
real_train, real_holdout = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
real_train = real_train.reset_index(drop=True)
real_holdout = real_holdout.reset_index(drop=True)
print(f"Real train rows: {len(real_train)}, Real holdout rows: {len(real_holdout)}")

# ---------- GENERATE SYNTHETIC ROWS from real_train ONLY ----------
# Prefer Utils.generate_bootstrap_synthetic_rows if available; fallback to Utl.augment_and_retrain behaviour
try:
    synth = Utl.generate_bootstrap_synthetic_rows(real_train, n_synth=N_SYNTH)
    print("Used Utils.generate_bootstrap_synthetic_rows() to create synthetic rows.")
except Exception:
    # fallback: try augment_and_retrain but accept that it may train inside - we only want synthetic rows.
    try:
        combined_tmp, _model, _sclr, _poly = Utl.augment_and_retrain(real_train, n_synth=N_SYNTH, method='bootstrap')
        # augment_and_retrain returns combined (real_train + synth). Extract synth by synthetic_flag
        if "synthetic_flag" in combined_tmp.columns:
            synth = combined_tmp[combined_tmp["synthetic_flag"] == True].reset_index(drop=True)
            print("Used Utl.augment_and_retrain() fallback and extracted synthetic rows.")
        else:
            raise RuntimeError("augment_and_retrain did not return synthetic_flag; cannot extract synthetic rows.")
    except Exception as e:
        raise RuntimeError("Utils does not expose a synthetic-generator function. Error: " + str(e))

# ensure synthetic flag
synth = synth.reset_index(drop=True)
synth["synthetic_flag"] = True

# ---------- BUILD TRAINING DATA = real_train + synthetic ----------
real_train_copy = real_train.copy().reset_index(drop=True)
real_train_copy["synthetic_flag"] = False
train_combined = pd.concat([real_train_copy, synth], ignore_index=True).reset_index(drop=True)
print("Training combined shape (real_train + synthetic):", train_combined.shape)

# ---------- Fill any missing feature columns with median of real_train ----------
for c in FEATURE_COLS:
    if c not in train_combined.columns:
        train_combined[c] = np.nan
    train_combined[c] = pd.to_numeric(train_combined[c], errors='coerce')
    med = real_train_copy[c].median() if c in real_train_copy.columns else 0.0
    train_combined[c] = train_combined[c].fillna(med)

# For targets, ensure numeric
train_combined["phi_true_num"] = pd.to_numeric(train_combined["phi_true_num"], errors='coerce')
if "C_true_num" in train_combined.columns:
    train_combined["C_true_num"] = pd.to_numeric(train_combined["C_true_num"], errors='coerce')

# ---------- PREPROCESSING: polynomial + scaler FIT ON TRAINING COMBINED ONLY ----------
X_train_raw = train_combined[FEATURE_COLS].values
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_train_poly = poly.fit_transform(X_train_raw)
scaler = RobustScaler().fit(X_train_poly)
X_train = scaler.transform(X_train_poly)

# ---------- Helper: train_model (RandomizedSearchCV) ----------
def train_model(X, y, name):
    """Randomized hyperparam search + cross-val diagnostics. Returns fitted estimator."""
    if XGB_AVAILABLE:
        base_model = XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, verbosity=0)
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
        base_model = RandomForestRegressor(random_state=RANDOM_STATE)
        param_dist = {
            "n_estimators": [200, 400],
            "max_features": ["sqrt", 1.0],
            "max_depth": [4, 6, None],
            "min_samples_leaf": [1, 2],
        }

    rs = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=12,
        scoring="neg_mean_absolute_error",
        cv=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    print(f"\nRunning hyperparameter search for {name}...")
    rs.fit(X, y)
    best = rs.best_estimator_
    print(f"Best {name} params:", rs.best_params_)

    # Cross-validated predictions on training combined (for diagnostics)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred_cv = cross_val_predict(best, X, y, cv=cv, n_jobs=-1)

    print(f"{name} CV metrics (on train_combined):")
    print(" MAE:", mean_absolute_error(y, y_pred_cv))
    print(" RMSE:", np.sqrt(mean_squared_error(y, y_pred_cv)))
    print(" R2:", r2_score(y, y_pred_cv))

    best.fit(X, y)
    return best

# ---------- TRAIN φ (phi) MODEL on training combined ----------
y_phi_train = train_combined["phi_true_num"].values
phi_model = train_model(X_train, y_phi_train, "phi")

# ---------- TRAIN C MODEL (if true C present) ----------
if "C_true_num" in train_combined.columns and train_combined["C_true_num"].notna().sum() > 5:
    y_C_train = train_combined["C_true_num"].values
    C_model = train_model(X_train, y_C_train, "C")
else:
    C_model = None
    print("Skipping C model training (no sufficient C in training data).")

# ---------- HOLDOUT (REAL ONLY) EVALUATION ----------
# Prepare holdout features using same poly & scaler (transform only)
for c in FEATURE_COLS:
    if c not in real_holdout.columns:
        real_holdout[c] = real_train_copy[c].median()  # unlikely but safe
    real_holdout[c] = pd.to_numeric(real_holdout[c], errors='coerce')
X_hold_raw = real_holdout[FEATURE_COLS].values
X_hold_poly = poly.transform(X_hold_raw)
X_hold = scaler.transform(X_hold_poly)

# phi holdout eval
y_phi_hold_true = real_holdout["phi_true_num"].values
y_phi_hold_pred = phi_model.predict(X_hold)
print("\n=== HOLDOUT (REAL-ONLY) PHI METRICS ===")
print(" MAE:", mean_absolute_error(y_phi_hold_true, y_phi_hold_pred))
print(" RMSE:", np.sqrt(mean_squared_error(y_phi_hold_true, y_phi_hold_pred)))
print(" R2:", r2_score(y_phi_hold_true, y_phi_hold_pred))

# C holdout eval (if model exists and holdout has C)
if C_model is not None and "C_true_num" in real_holdout.columns and real_holdout["C_true_num"].notna().sum() > 2:
    y_C_hold_true = real_holdout["C_true_num"].values
    y_C_hold_pred = C_model.predict(X_hold)
    print("\n=== HOLDOUT (REAL-ONLY) C METRICS ===")
    print(" MAE:", mean_absolute_error(y_C_hold_true, y_C_hold_pred))
    print(" RMSE:", np.sqrt(mean_squared_error(y_C_hold_true, y_C_hold_pred)))
    print(" R2:", r2_score(y_C_hold_true, y_C_hold_pred))
else:
    y_C_hold_pred = None
    print("\nC holdout evaluation skipped (no C in holdout or no C model).")

# ---------- PREDICT & SAVE results ----------
# Predictions for train_combined (for inspection)
train_combined["phi_pred"] = phi_model.predict(X_train)
if C_model is not None:
    train_combined["C_pred"] = C_model.predict(X_train)

# Predictions for holdout_real
real_holdout = real_holdout.reset_index(drop=True)
real_holdout["phi_pred"] = y_phi_hold_pred
if y_C_hold_pred is not None:
    real_holdout["C_pred"] = y_C_hold_pred

# Save CSVs
train_combined.to_csv(os.path.join(OUT_DIR, "train_combined_with_preds.csv"), index=False)
real_holdout.to_csv(os.path.join(OUT_DIR, "holdout_real_with_preds.csv"), index=False)
print("Saved train_combined_with_preds.csv and holdout_real_with_preds.csv")

# Save models + preprocessors
joblib.dump(phi_model, os.path.join(OUT_DIR, "phi_model.joblib"))
if C_model is not None:
    joblib.dump(C_model, os.path.join(OUT_DIR, "C_model.joblib"))
joblib.dump(poly, os.path.join(OUT_DIR, "poly.joblib"))
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
print("Saved models and preprocessing objects to", OUT_DIR)

# ---------- PLOTTING ----------
def plot_true_vs_pred(true_arr, pred_arr, name, path):
    plt.figure(figsize=(6,6))
    plt.scatter(true_arr, pred_arr, alpha=0.7, edgecolors='k')
    mn = np.nanmin(true_arr)
    mx = np.nanmax(true_arr)
    plt.plot([mn, mx], [mn, mx], "r--", label="y = x")
    plt.xlabel(f"{name} true")
    plt.ylabel(f"{name} predicted")
    plt.title(f"{name}: true vs predicted")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()

plot_true_vs_pred(train_combined["phi_true_num"], train_combined["phi_pred"],
                  "phi_train", os.path.join(OUT_DIR, "phi_train_true_vs_pred.png"))
plot_true_vs_pred(real_holdout["phi_true_num"], real_holdout["phi_pred"],
                  "phi_holdout", os.path.join(OUT_DIR, "phi_holdout_true_vs_pred.png"))

if C_model is not None:
    plot_true_vs_pred(train_combined["C_true_num"], train_combined["C_pred"],
                      "C_train", os.path.join(OUT_DIR, "C_train_true_vs_pred.png"))
    if y_C_hold_pred is not None:
        plot_true_vs_pred(real_holdout["C_true_num"], real_holdout["C_pred"],
                          "C_holdout", os.path.join(OUT_DIR, "C_holdout_true_vs_pred.png"))

print("All done.")
