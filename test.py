import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import RobustScaler, PolynomialFeatures

# ---------- config ----------
MODEL_DIR = "./models_phi_C_bootstrap"
TEST_DATA_PATH = "Test-Data2.csv"

# ---------- load models ----------
phi_model_path = os.path.join(MODEL_DIR, "phi_model.joblib")
C_model_path = os.path.join(MODEL_DIR, "C_model.joblib")

if not os.path.exists(phi_model_path):
    raise FileNotFoundError(f"phi model not found at {phi_model_path}")

phi_model = joblib.load(phi_model_path)
C_model = joblib.load(C_model_path) if os.path.exists(C_model_path) else None

# ---------- load test data ----------
test_df = pd.read_csv(TEST_DATA_PATH)

# ---------- check required columns ----------
required_cols = ["qc", "fs", "u2", "Effective Stress", "Fr", "Qt", "Depth", "S"]
for c in required_cols:
    if c not in test_df.columns:
        raise ValueError(f"Missing column {c} in Test-Data.csv")

# ---------- convert numeric ----------
for c in required_cols:
    test_df[c] = pd.to_numeric(test_df[c], errors='coerce')

# drop NaN rows if any
before = len(test_df)
test_df = test_df.dropna(subset=required_cols)
after = len(test_df)
if before != after:
    print(f"Dropped {before - after} rows with NaN values")

# ---------- prepare features ----------
if "S" in test_df.columns:
    test_df["S_Prop_num"] = pd.to_numeric(test_df["S"], errors="coerce")

feature_cols = ["S_Prop_num", "qc", "fs", "u2", "Effective Stress", "Fr", "Qt", "Depth"]

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
scaler = RobustScaler()

# Note: The polynomial & scaler should be *fitted on the same structure* as during training.
# Here, we re-fit them because the training ones weren’t saved.
# If you saved them before, load them instead of fitting again.
X_raw = test_df[feature_cols].values
X_poly = poly.fit_transform(X_raw)
X_scaled = scaler.fit_transform(X_poly)

# ---------- predict ----------
test_df["phi_predicted"] = phi_model.predict(X_scaled)
if C_model is not None:
    test_df["C_predicted"] = C_model.predict(X_scaled)
else:
    test_df["C_predicted"] = np.nan
    print("⚠️ No C model found — skipping C prediction.")

# ---------- save results ----------
output_path = os.path.join(MODEL_DIR, "TestData_Predictions.csv")
test_df.to_csv(output_path, index=False)

print("\n✅ Prediction complete!")
print("Results saved to:", output_path)
print("\nPredictions:")
print(test_df[["phi_predicted", "C_predicted", "S"]])
