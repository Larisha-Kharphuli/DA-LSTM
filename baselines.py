# =============================================================
# Baselines: Persistence, Seasonal Naïve, SARIMA
#   - Same leakage-free pipeline as neural models
#   - Same 70/15/15 chronological split
#   - Same test period
#   - SARIMA order selected on validation RMSE, then refit on train+val
# =============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

RESULTS_DIR = "results"
OUT_DIR = os.path.join(RESULTS_DIR, "baselines")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# LOAD DATA (same as neural scripts)
# -------------------------------------------------------------
DATA_PATH = r"D:\Research\Objective 1\Myhybridmodel1\Price_index_data\Correction\Agricultural_data.xlsx"
df = pd.read_excel(DATA_PATH)
df = df.sort_values(["Year", "Month"]).reset_index(drop=True)

products = ["Onion", "Cumin", "Garlic", "Castor_Seed", "Tomato", "Potato"]

TRAIN_RATIO, VAL_RATIO = 0.70, 0.15
WINDOW = 12
SEASON = 12

# -------------------------------------------------------------
# METRICS
# -------------------------------------------------------------
def rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))
def mae(y, p):  return float(mean_absolute_error(y, p))
def r2(y, p):   return float(r2_score(y, p))

def mase(y, p, y_train, seasonality=12):
    denom = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    return float(np.mean(np.abs(y - p)) / denom) if denom > 0 else np.nan

def theils_u(y, p):
    naive = np.roll(y, 1).astype(float); naive[0] = y[0]
    den = np.sqrt(np.mean((y - naive) ** 2))
    return float(np.sqrt(np.mean((y - p) ** 2)) / den) if den > 0 else np.nan

# -------------------------------------------------------------
# SPLIT (identical to neural scripts)
# -------------------------------------------------------------
def split_indices(n):
    train_end = int(TRAIN_RATIO * n)
    val_end   = int((TRAIN_RATIO + VAL_RATIO) * n)
    return train_end, val_end

# -------------------------------------------------------------
# PERSISTENCE + SEASONAL NAÏVE
#   These are computed on the raw (unscaled) test series.
#   For a fair comparison, we need the same test indices as the
#   neural models: the neural models predict y[t] from x[t-window:t],
#   so their first test prediction corresponds to y at index
#   (val_end), i.e. the first test point after the val split.
# -------------------------------------------------------------
def persistence_predictions(series, test_start, test_end):
    """y_hat[t] = y[t-1] for t in [test_start, test_end)."""
    preds = np.array([series[t - 1] for t in range(test_start, test_end)])
    trues = np.array([series[t] for t in range(test_start, test_end)])
    return trues, preds

def seasonal_naive_predictions(series, test_start, test_end, season=12):
    """y_hat[t] = y[t-season] for t in [test_start, test_end)."""
    preds = np.array([series[t - season] for t in range(test_start, test_end)])
    trues = np.array([series[t] for t in range(test_start, test_end)])
    return trues, preds

# -------------------------------------------------------------
# SARIMA
# -------------------------------------------------------------
# Small, sensible search grid — keeps runtime reasonable.
# (p,d,q) non-seasonal; (P,D,Q,s) seasonal with s = 12.
SARIMA_GRID = [
    # (p,d,q), (P,D,Q,s)
    ((1,1,1), (1,1,1,12)),
    ((1,1,1), (0,1,1,12)),
    ((1,1,1), (1,1,0,12)),
    ((0,1,1), (0,1,1,12)),
    ((1,1,0), (1,1,0,12)),
    ((2,1,1), (1,1,1,12)),
    ((1,1,2), (1,1,1,12)),
    ((0,1,0), (0,1,1,12)),   # seasonal random walk
]

def fit_sarima_forecast(train_series, val_series, test_series,
                        season=12, grid=SARIMA_GRID, verbose=False):
    """
    Fit SARIMA on train, select order by validation RMSE,
    then refit on train+val and forecast test.
    Returns (trues, preds, best_order).
    """
    best_order = None
    best_val_rmse = np.inf

    # ---- order selection on validation ----
    for order, sorder in grid:
        try:
            model = SARIMAX(
                train_series,
                order=order,
                seasonal_order=sorder,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)
            val_pred = res.forecast(steps=len(val_series))
            v = rmse(val_series, val_pred)
            if verbose:
                print(f"    {order} x {sorder}  val RMSE = {v:.3f}")
            if v < best_val_rmse:
                best_val_rmse = v
                best_order = (order, sorder)
        except Exception as e:
            if verbose:
                print(f"    {order} x {sorder}  FAILED ({e})")
            continue

    if best_order is None:
        raise RuntimeError("All SARIMA fits failed")

    # ---- refit on train + val, forecast test ----
    order, sorder = best_order
    full_train = np.concatenate([train_series, val_series])
    model = SARIMAX(
        full_train,
        order=order,
        seasonal_order=sorder,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    preds = np.asarray(res.forecast(steps=len(test_series)))
    trues = np.asarray(test_series)
    return trues, preds, best_order, best_val_rmse

# -------------------------------------------------------------
# RUN ALL BASELINES
# -------------------------------------------------------------
baseline_rows = []

print("\n" + "=" * 70)
print("BASELINES")
print("=" * 70)

for product in products:
    print(f"\n--- {product} ---")
    series = df[product].values.astype(float)
    n = len(series)
    train_end, val_end = split_indices(n)

    train_series = series[:train_end]
    val_series   = series[train_end:val_end]
    test_series  = series[val_end:]

    # ----- persistence + seasonal naïve -----
    # NOTE: alignment.
    # The neural models' first test prediction corresponds to the
    # first point after the val split (index = val_end). For
    # persistence, we need y[val_end - 1] as the input. Both are
    # available because val_end - 1 < val_end <= len(series).
    t_true, p_persist = persistence_predictions(series, val_end, n)
    _,      p_snaive  = seasonal_naive_predictions(series, val_end, n, season=SEASON)

    # ----- SARIMA -----
    t_true_s, p_sarima, best_order, best_val = fit_sarima_forecast(
        train_series, val_series, test_series, season=SEASON, verbose=False
    )
    print(f"  SARIMA order selected: {best_order[0]} x {best_order[1]}  "
          f"(val RMSE = {best_val:.3f})")

    # ----- record -----
    for name, preds in [("Persistence", p_persist),
                        ("SeasonalNaive", p_snaive),
                        ("SARIMA", p_sarima)]:
        baseline_rows.append({
            "product":  product,
            "model":    name,
            "RMSE":     rmse(t_true, preds),
            "MAE":      mae(t_true, preds),
            "R2":       r2(t_true, preds),
            "MASE":     mase(t_true, preds, train_series, seasonality=SEASON),
            "TheilsU":  theils_u(t_true, preds),
        })

    # save predictions for downstream use
    np.savez(
        os.path.join(OUT_DIR, f"{product}_baselines.npz"),
        actual=t_true,
        persistence=p_persist,
        seasonal_naive=p_snaive,
        sarima=p_sarima,
        sarima_order=np.array([str(best_order[0]) + "x" + str(best_order[1])], dtype=object),
    )

baseline_df = pd.DataFrame(baseline_rows)
baseline_df.to_csv(os.path.join(OUT_DIR, "baseline_metrics.csv"), index=False)

print("\n===== Baseline metrics per commodity =====")
print(baseline_df.round(4).to_string(index=False))

# -------------------------------------------------------------
# SIDE-BY-SIDE WITH NEURAL MODELS (if available)
# -------------------------------------------------------------
neural_files = {
    "LSTM":      os.path.join(RESULTS_DIR, "lstm_metrics_perseed.csv"),
    "ATTN_LSTM": os.path.join(RESULTS_DIR, "attention_lstm_metrics_perseed.csv"),
    "DA_LSTM":   os.path.join(RESULTS_DIR, "hybrid_metrics_perseed.csv"),
}

neural_rows = []
for mname, path in neural_files.items():
    if not os.path.exists(path):
        continue
    dfm = pd.read_csv(path)
    # average across seeds
    agg = dfm.groupby("product").agg(
        RMSE=("rmse", "mean"),
        MAE=("mae", "mean"),
        R2=("r2", "mean"),
        MASE=("mase", "mean"),
        TheilsU=("theils_u", "mean"),
    ).reset_index()
    agg["model"] = mname
    neural_rows.append(agg)

neural_df = pd.concat(neural_rows, ignore_index=True)

# add baseline means (they are deterministic, so mean = single value)
baseline_df["model"] = baseline_df["model"]
combined = pd.concat([neural_df, baseline_df], ignore_index=True)
combined = combined[["product", "model", "RMSE", "MAE", "R2", "MASE", "TheilsU"]]
combined.to_csv(os.path.join(OUT_DIR, "combined_all_models.csv"), index=False)

print("\n===== Combined table (neural means + baselines) =====")
pivot = combined.pivot(index="product", columns="model", values="RMSE")
print("\nRMSE by product × model:")
print(pivot.round(3))

pivot_theils = combined.pivot(index="product", columns="model", values="TheilsU")
print("\nTheil's U by product × model:")
print(pivot_theils.round(3))

# -------------------------------------------------------------
# FIGURE: RMSE bar chart incl. baselines
# -------------------------------------------------------------
import seaborn as sns
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=combined, x="product", y="RMSE", hue="model", ax=ax)
ax.set_title("RMSE: neural models vs. classical baselines")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_rmse_all_models.png"), dpi=150)
plt.close()

# -------------------------------------------------------------
# FIGURE: Theil's U with naïve line
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=combined, x="product", y="TheilsU", hue="model", ax=ax)
ax.axhline(1.0, color="red", linestyle="--", label="Naïve benchmark (U = 1)")
ax.set_title("Theil's U: neural models vs. classical baselines")
plt.xticks(rotation=20)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_theils_all_models.png"), dpi=150)
plt.close()

print(f"\nAll baseline outputs written to {OUT_DIR}/")
print("Done.")