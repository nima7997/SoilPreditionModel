import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor
import joblib

# ---------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------
DATA_PATH = "NewDF - Sheet1.csv"  # ← change this to your CSV file
df = pd.read_csv(DATA_PATH)

# Drop irrelevant or empty columns
df = df.dropna(axis=1, how="all")

# Ensure phi exists
if "phi" not in df.columns:
    raise ValueError("❌ 'phi' column not found in dataset.")

# Define target and features
target_col = "phi"
exclude_cols = ["C", "phi", "S Propesed"]  # drop other labels if present
feature_cols = [c for c in df.columns if c not in exclude_cols]

df = df.dropna(subset=[target_col])  # keep rows where phi exists
X = df[feature_cols].fillna(df.median())
y = df[target_col].values

# ---------------------------------------------------------
# 2. Train/test split (few-shot)
# ---------------------------------------------------------
X_train, X_hold, y_train, y_hold = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Labeled rows: {len(X)}, Train: {len(X_train)}, Holdout: {len(X_hold)}")

# ---------------------------------------------------------
# 3. Standardize features
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_hold_scaled = scaler.transform(X_hold)

# ---------------------------------------------------------
# 4. Train TabPFN for phi
# ---------------------------------------------------------
print("\n=== Training TabPFN model for φ (phi) ===")

model = TabPFNRegressor(device="cpu")  # use device="cuda" if you have a GPU
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------
# 5. Evaluate on holdout
# ---------------------------------------------------------
y_pred = model.predict(X_hold_scaled)

mae = mean_absolute_error(y_hold, y_pred)
rmse = np.sqrt(mean_squared_error(y_hold, y_pred))
r2 = r2_score(y_hold, y_pred)

print(f"\n📊 φ (phi) Holdout Performance:")
print(f" MAE = {mae:.4f}")
print(f" RMSE = {rmse:.4f}")
print(f" R² = {r2:.4f}")

# ---------------------------------------------------------
# 6. Save model and scaler
# ---------------------------------------------------------
os.makedirs("./models_tabpfn_phi", exist_ok=True)
joblib.dump(model, "./models_tabpfn_phi/phi_TabPFN.joblib")
joblib.dump(scaler, "./models_tabpfn_phi/scaler.joblib")

print("\n✅ Model and scaler saved to './models_tabpfn_phi'")

# ---------------------------------------------------------
# 7. Example predictions
# ---------------------------------------------------------
print("\n🔮 Example φ predictions:")
example = X_hold.iloc[:5]
example_scaled = scaler.transform(example)
example_preds = model.predict(example_scaled)

for i, val in enumerate(example_preds):
    print(f" Sample {i+1}: predicted φ = {val:.3f}")
