# ---------- Synthetic augmentation utilities ----------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import os

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# --- helper: sample from empirical distribution with small jitter
def sample_empirical(arr, n, jitter=0.0, method='bootstrap'):
    # arr: 1D numeric array
    if method == 'bootstrap':
        choices = np.random.choice(arr, size=n, replace=True)
    else:
        # sample from KDE/normal fit
        mu, sigma = np.nanmean(arr), np.nanstd(arr)
        choices = np.random.normal(mu, sigma, size=n)
    if jitter > 0:
        choices = choices + np.random.normal(0, jitter, size=n)
    return choices

# ---------- strategy A: pure physics synthetic generation ----------
def generate_physics_synthetic(df,
                               n_synth=500,
                               phi_source='empirical',    # 'empirical' or 'normal'
                               C_source='empirical',
                               stress_source='empirical',
                               noise_phi_deg=0.5,
                               noise_C_kPa=1.0,
                               units='same'):
    """
    Generate synthetic rows by sampling phi (deg), C (kPa), stress (MPa),
    computing S = C + stress_kPa * tan(phi_rad).
    Returns a DataFrame with synthetic S (in same unit as df['S_Prop_num']) and phi, C, stress.
    """
    # get observed arrays
    phi_obs = df['phi_true_num'].values
    C_obs = df['C'].values if 'C' in df.columns else None
    stress_obs = df['stress_num'].values

    # sample
    phi_s = sample_empirical(phi_obs, n_synth, jitter=noise_phi_deg, method='bootstrap' if phi_source=='empirical' else 'normal')
    if C_obs is not None:
        C_s = sample_empirical(C_obs, n_synth, jitter=noise_C_kPa, method='bootstrap' if C_source=='empirical' else 'normal')
    else:
        # if no C in real data, sample small positive values (example)
        C_s = np.abs(np.random.normal(df['C_computed'].mean() if 'C_computed' in df.columns else 10.0, 5.0, n_synth))
    stress_s = sample_empirical(stress_obs, n_synth, jitter=0.0, method='bootstrap' if stress_source=='empirical' else 'normal')

    # convert stress to kPa if needed (assume stress_obs same units as S_Prop_num base)
    # Here we assume stress_s is in MPa and we want kPa inside eqn:
    stress_kPa = stress_s * 1000.0

    # compute S_kPa = C_kPa + stress_kPa * tan(phi_rad)
    phi_rad = np.deg2rad(phi_s)
    S_kPa = C_s + stress_kPa * np.tan(phi_rad)

    # convert back to S_prop units: if df['S_Prop_num'] in MPa, convert kPa -> MPa
    # We'll detect by comparing magnitudes: if S_Prop_num mean < 10, likely in MPa
    s_mean = np.nanmean(df['S_Prop_num'].values)
    if s_mean < 10:
        S_prop_units = S_kPa / 1000.0   # MPa
    else:
        S_prop_units = S_kPa            # already kPa

    synth = pd.DataFrame({
        'S_Prop_num': S_prop_units,
        'stress_num': stress_s,
        'phi_true_num': phi_s,
        'C': C_s
    })
    synth['synthetic_flag'] = True
    return synth

# ---------- strategy B: bootstrap real rows and overwrite physics targets ----------
def generate_bootstrap_synthetic_rows(df, n_synth=500, jitter_phi=0.5, jitter_C=1.0, feature_jitter_frac=0.03):
    rows = df.sample(n=n_synth, replace=True).reset_index(drop=True).copy()

    phi_s = sample_empirical(df['phi_true_num'].values, n_synth, jitter=jitter_phi, method='bootstrap')
    C_s = sample_empirical(df['C'].values if 'C' in df.columns else np.zeros(len(df)),
                           n_synth, jitter=jitter_C, method='bootstrap')

    feature_cols = ["qc", "fs", "u2", "Effective Stress", "Fr", "Qt", "Depth"]

    for col in feature_cols:
        if col in rows.columns:
            col_values = pd.to_numeric(rows[col], errors='coerce')
            std_val = np.nanstd(col_values)
            if std_val > 0:
                noise = np.random.normal(0, std_val * feature_jitter_frac, size=n_synth)
                col_values = col_values + noise
            rows[col] = col_values

    # -----------------------------
    # 3. Recompute S from new φ and C
    # -----------------------------
    stress_s = rows['stress_num'].values
    stress_kPa = stress_s * 1000.0
    phi_rad = np.deg2rad(phi_s)
    S_kPa = C_s + stress_kPa * np.tan(phi_rad)

    s_mean = np.nanmean(df['S_Prop_num'].values)
    if s_mean < 10:
        S_prop_units = S_kPa / 1000.0
    else:
        S_prop_units = S_kPa

    # -----------------------------
    # 4. Store updated values
    # -----------------------------
    rows['S_Prop_num'] = S_prop_units
    rows['phi_true_num'] = phi_s
    rows['C'] = C_s
    rows['synthetic_flag'] = True

    return rows


# ---------- Merge synthetic with real and retrain example ----------
def augment_and_retrain(df, n_synth=500, method='bootstrap'):
    if method == 'physics':
        synth = generate_physics_synthetic(df, n_synth=n_synth)
    else:
        synth = generate_bootstrap_synthetic_rows(df, n_synth=n_synth)
    # mark real rows
    df2 = df.copy().reset_index(drop=True)
    df2['synthetic_flag'] = False
    combined = pd.concat([df2, synth], ignore_index=True).reset_index(drop=True)

    # prepare features as you had (example feature set)
    feature_cols = ["S_Prop_num", "qc", "fs", "u2", "Effective Stress", "Fr", "Qt", "Depth"]
    # ensure numeric; if missing, fill with median of real data
    for c in feature_cols:
        if c not in combined.columns:
            combined[c] = np.nan
        combined[c] = pd.to_numeric(combined[c], errors='coerce')
        # fill NaN with median from real df only (not from synth)
        combined.loc[combined['synthetic_flag']==False, c] = pd.to_numeric(combined.loc[combined['synthetic_flag']==False, c], errors='coerce')
        med = combined.loc[combined['synthetic_flag']==False, c].median()
        combined[c] = combined[c].fillna(med)

    # target and X
    X_raw = combined[feature_cols].values
    y = combined['phi_true_num'].values

    # polynomial + scaling
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    X_poly = poly.fit_transform(X_raw)
    scaler = RobustScaler().fit(X_poly)
    X = scaler.transform(X_poly)

    # Use only real rows for final evaluation (holdout)
    real_mask = ~combined['synthetic_flag'].values
    X_real = X[real_mask]
    y_real = y[real_mask]

    # train model on combined (real + synthetic)
    if XGB_AVAILABLE:
        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42, objective='reg:squarederror')
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42)

    # cross-validated evaluation on real-only (use combined model but CV predictions only on real rows)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # We must do CV using only real rows to get true generalization
    y_pred_cv = cross_val_predict(model, X_real, y_real, cv=cv, n_jobs=1)
    print("Before training on combined data - CV on real only")
    print(" MAE:", mean_absolute_error(y_real, y_pred_cv))
    print(" RMSE:", np.sqrt(mean_squared_error(y_real, y_pred_cv)))
    print(" R2:", r2_score(y_real, y_pred_cv))

    # Now fit model on combined data
    model.fit(X, y)

    # evaluate on real test set (holdout)
    # use the same CV split or single holdout - here we do a quick holdout
    Xtr, Xte, ytr, yte = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
    model.fit(X, y)  # already fit but keep call
    y_pred_real = model.predict(Xte)
    print("\nAfter training on combined (real+synthetic), holdout real metrics:")
    print(" MAE:", mean_absolute_error(yte, y_pred_real))
    print(" RMSE:", np.sqrt(mean_squared_error(yte, y_pred_real)))
    print(" R2:", r2_score(yte, y_pred_real))

    return combined, model, scaler, poly

# ---------------- Example usage ----------------
# combined_df, model, scaler, poly = augment_and_retrain(df, n_synth=500, method='bootstrap')
