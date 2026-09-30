# DA-LSTM (Diffusion-Enhanced Attention LSTM)
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


DATA_PATH = r"D:\Research\Objective 1\Myhybridmodel1\Price_index_data\Correction\Agricultural_data.xlsx"
df = pd.read_excel(DATA_PATH).sort_values(["Year", "Month"]).reset_index(drop=True)

products = ["Onion", "Cumin", "Garlic", "Castor_Seed", "Tomato", "Potato"]

# DATASET
class TimeSeriesDataset(Dataset):
    def __init__(self, series, window):
        X, y = [], []
        for i in range(len(series) - window):
            X.append(series[i:i + window])
            y.append(series[i + window])
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# SHARED MODULES
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        h, _ = self.lstm(x)
        return h

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, h):
        w = torch.softmax(self.attn(h), dim=1)
        return torch.sum(w * h, dim=1)

def linear_beta_schedule(T):
    return torch.linspace(1e-4, 0.02, T)

class DenoiseNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 128), nn.ReLU(), nn.Linear(128, dim)
        )
    def forward(self, z_t, t):
        return self.net(torch.cat([z_t, t], dim=1))

class LatentDDPM(nn.Module):
    def __init__(self, dim, T=100):
        super().__init__()
        self.T = T
        self.denoiser = DenoiseNet(dim)
        betas = linear_beta_schedule(T)
        alphas = 1 - betas
        self.register_buffer("alpha_bar", torch.cumprod(alphas, dim=0))
    def forward(self, z0):
        B = z0.size(0)
        t = torch.randint(0, self.T, (B,), device=z0.device)
        noise = torch.randn_like(z0)
        ab = self.alpha_bar[t].unsqueeze(1)
        z_t = torch.sqrt(ab) * z0 + torch.sqrt(1 - ab) * noise
        t_norm = t.float().unsqueeze(1) / self.T
        noise_pred = self.denoiser(z_t, t_norm)
        return noise, noise_pred

class OneStepDenoiser(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, dim))
    def forward(self, z0):
        return self.net(z0)

# DA-LSTM
class DALSTMModel(nn.Module):
    """
    Final DA-LSTM:
      - LSTM encoder
      - temporal attention
      - concatenate attention context with LSTM last hidden -> project
      - latent diffusion with one-step denoiser used at inference
      - decoder takes denoised latent
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim)
        self.attn = TemporalAttention(hidden_dim)
        self.diffusion = LatentDDPM(hidden_dim)
        self.denoiser1 = OneStepDenoiser(hidden_dim)
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = self.encoder(x)
        z_attn = self.attn(h)
        h_last = h[:, -1, :]
        z = self.proj(torch.cat([z_attn, h_last], dim=-1))
        noise, noise_pred = self.diffusion(z)
        z_hat = self.denoiser1(z)
        y_hat = self.decoder(z_hat).squeeze(-1)
        return y_hat, noise, noise_pred

# FUTURE FORECAST (recursive)
def forecast_future(model, last_window, scaler, steps=12):
    model.eval(); preds = []; window = last_window.copy()
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            out = model(x)
            y_hat = out[0] if isinstance(out, tuple) else out
            pred = y_hat.cpu().numpy().flatten()[0]
            preds.append(pred)
            window = np.vstack([window[1:], [[pred]]])
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

# METRICS
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

# TRAIN ONE MODEL FOR ONE PRODUCT (one seed)
def train_one_product(product, seed=42, window=12,
                      train_ratio=0.70, val_ratio=0.15,
                      epochs=80, warmup=10, batch_size=32, lr=1e-3):

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

    model = DALSTMModel(1, 64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        lam = min(1.0, epoch / warmup)
        for x, y in train_loader:
            x = x.to(device); y = y.to(device).squeeze(-1)
            opt.zero_grad()
            y_hat, noise, noise_pred = model(x)
            loss_pred = F.mse_loss(y_hat, y)
            loss_diff = F.mse_loss(noise_pred, noise)
            loss = (loss_pred / (loss_pred.detach() + 1e-8)
                    + lam * loss_diff / (loss_diff.detach() + 1e-8))
            loss.backward()
            opt.step()

    def predict(loader):
        model.eval(); preds, trues = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                out = model(x)
                y_hat = out[0] if isinstance(out, tuple) else out
                preds.extend(y_hat.cpu().numpy().reshape(-1))
                trues.extend(y.cpu().numpy().reshape(-1))
        preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        trues = scaler.inverse_transform(np.array(trues).reshape(-1, 1)).flatten()
        return trues, preds

    _, _            = predict(val_loader)
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
    }
    return metrics, test_true, test_pred, future_pred

# RUN
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

os.makedirs("results", exist_ok=True)
metrics_df.to_csv("results/hybrid_metrics_perseed.csv", index=False)
agg.to_csv("results/hybrid_metrics_aggregated.csv", index=False)

min_len = min(len(store_preds[p]["true"]) for p in products)
preds_arr   = np.stack([store_preds[p]["pred"][:min_len] for p in products], axis=1)
actuals_arr = np.stack([store_preds[p]["true"][:min_len] for p in products], axis=1)

np.savez("results/hybrid_results.npz",
         predictions=preds_arr, actual=actuals_arr, products=np.array(products))

print("\nSaved results to ./results/")