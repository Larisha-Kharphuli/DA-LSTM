# =============================================================
# Baseline LSTM (corrected pipeline — matches DA-LSTM.py)
# =============================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------------------------------------
# REPRODUCIBILITY
# -------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
DATA_PATH = r"D:\Research\Objective 1\Myhybridmodel1\Price_index_data\Correction\Agricultural_data.xlsx"
df = pd.read_excel(DATA_PATH)
df = df.sort_values(["Year", "Month"]).reset_index(drop=True)

products = ["Onion", "Cumin", "Garlic", "Castor_Seed", "Tomato", "Potato"]

# -------------------------------------------------------------
# DATASET
# -------------------------------------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, series, window):
        self.X, self.y = [], []
        for i in range(len(series) - window):
            self.X.append(series[i:i + window])
            self.y.append(series[i + window])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)   # FIX: np.array() first
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------------------------------------------
# MODEL: LSTM only
# -------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h, _ = self.lstm(x)              # (B, T, hidden)
        z = h[:, -1, :]                  # last timestep
        y_hat = self.decoder(z).squeeze(-1)
        return y_hat

# -------------------------------------------------------------
# FUTURE FORECAST (recursive)
# -------------------------------------------------------------
def forecast_future(model, last_window, scaler, steps=12):
    model.eval()
    preds = []
    window = last_window.copy()
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            y_hat = model(x)
            pred = y_hat.cpu().numpy().flatten()[0]
            preds.append(pred)
            window = np.vstack([window[1:], [[pred]]])
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    return preds.flatten()

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
    num = np.sqrt(np.mean((y - p) ** 2))
    den = np.sqrt(np.mean((y - naive) ** 2))
    return float(num / den) if den > 0 else np.nan

# -------------------------------------------------------------
# TRAIN ONE MODEL FOR ONE PRODUCT (one seed)
# -------------------------------------------------------------
def train_one_product(product, seed=42, window=12,
                      train_ratio=0.70, val_ratio=0.15,
                      epochs=80, batch_size=32, lr=1e-3):

    set_seed(seed)
    data = df[product].values.reshape(-1, 1)
    n = len(data)

    train_end = int(train_ratio * n)
    val_end   = int((train_ratio + val_ratio) * n)

    train_raw = data[:train_end]
    val_raw   = data[train_end:val_end]
    test_raw  = data[val_end:]

    scaler = StandardScaler().fit(train_raw)
    train_scaled = scaler.transform(train_raw)
    val_scaled   = scaler.transform(val_raw)
    test_scaled  = scaler.transform(test_raw)

    train_ds = TimeSeriesDataset(train_scaled, window)
    val_input  = np.vstack([train_scaled[-window:], val_scaled])
    test_input = np.vstack([val_scaled[-window:],  test_scaled])
    val_ds  = TimeSeriesDataset(val_input,  window)
    test_ds = TimeSeriesDataset(test_input, window)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model = LSTMModel(1, 64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device); y = y.to(device).squeeze(-1)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = F.mse_loss(y_hat, y)
            loss.backward()
            optimizer.step()

    def predict(loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y_hat = model(x)
                preds.extend(y_hat.cpu().numpy().reshape(-1))
                trues.extend(y.cpu().numpy().reshape(-1))
        preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        trues = scaler.inverse_transform(np.array(trues).reshape(-1, 1)).flatten()
        return trues, preds

    val_true,  val_pred  = predict(val_loader)
    test_true, test_pred = predict(test_loader)

    last_window = scaler.transform(data[-window:])
    future_pred = forecast_future(model, last_window, scaler, steps=12)

    metrics = {
        "product":  product, "seed": seed,
        "rmse":     rmse(test_true, test_pred),
        "mae":      mae(test_true, test_pred),
        "r2":       r2(test_true, test_pred),
        "mase":     mase(test_true, test_pred, train_raw.flatten()),
        "theils_u": theils_u(test_true, test_pred),
        "val_rmse": rmse(val_true, val_pred),
    }
    return metrics, test_true, test_pred, future_pred

# -------------------------------------------------------------
# RUN ALL PRODUCTS × SEEDS
# -------------------------------------------------------------
SEEDS = [42, 1, 7, 13, 21]
WINDOW = 12

all_metrics = []
store_preds = {p: {"true": None, "pred": None, "future": None} for p in products}

for product in products:
    for seed in SEEDS:
        print(f"\nTraining {product} | seed {seed}")
        m, tt, tp, fp = train_one_product(product, seed=seed, window=WINDOW)
        all_metrics.append(m)
        if store_preds[product]["true"] is None:
            store_preds[product]["true"]   = tt
            store_preds[product]["pred"]   = tp
            store_preds[product]["future"] = fp

metrics_df = pd.DataFrame(all_metrics)
print("\n===== Per-run metrics =====")
print(metrics_df)

agg = metrics_df.groupby("product").agg(
    rmse_mean=("rmse","mean"),       rmse_std=("rmse","std"),
    mae_mean=("mae","mean"),         mae_std=("mae","std"),
    r2_mean=("r2","mean"),           r2_std=("r2","std"),
    mase_mean=("mase","mean"),       mase_std=("mase","std"),
    theils_mean=("theils_u","mean"), theils_std=("theils_u","std"),
).reset_index()

print("\n===== Aggregated (mean ± std across seeds) =====")
print(agg)

pooled_true = np.concatenate([store_preds[p]["true"] for p in products])
pooled_pred = np.concatenate([store_preds[p]["pred"] for p in products])
print("\n===== Global pooled metrics (seed 42) =====")
print(f"RMSE = {rmse(pooled_true, pooled_pred):.4f}")
print(f"MAE  = {mae(pooled_true, pooled_pred):.4f}")
print(f"R2   = {r2(pooled_true, pooled_pred):.4f}")

# -------------------------------------------------------------
# SAVE
# -------------------------------------------------------------
os.makedirs("results", exist_ok=True)
metrics_df.to_csv("results/lstm_metrics_perseed.csv", index=False)
agg.to_csv("results/lstm_metrics_aggregated.csv", index=False)

min_len = min(len(store_preds[p]["true"]) for p in products)
preds_arr   = np.stack([store_preds[p]["pred"][:min_len] for p in products], axis=1)
actuals_arr = np.stack([store_preds[p]["true"][:min_len] for p in products], axis=1)

np.savez("results/lstm_results.npz",
         predictions=preds_arr, actual=actuals_arr, products=np.array(products))

print("\nSaved results to ./results/")
print("Done.")