import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


DATA_PATH = r"D:\Research\Objective 1\Myhybridmodel1\Price_index_data\Correction\Agricultural_data.xlsx"
df = pd.read_excel(DATA_PATH).sort_values(["Year", "Month"]).reset_index(drop=True)
products = ["Onion", "Cumin", "Garlic", "Castor_Seed", "Tomato", "Potato"]

OUT_DIR = "results/sensitivity"
os.makedirs(OUT_DIR, exist_ok=True)


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

class DALSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, T_diffusion=100):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim)
        self.attn = TemporalAttention(hidden_dim)
        self.diffusion = LatentDDPM(hidden_dim, T=T_diffusion)
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




def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


# TRAIN ONE PRODUCT, ONE SEED, ONE HYPERPARAMETER SETTING

def train_eval(product, seed,
               window=12, T_diffusion=100, warmup=10, hidden=64,
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
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model = DALSTMModel(1, hidden, T_diffusion=T_diffusion).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        lam = min(1.0, epoch / warmup) if warmup > 0 else 1.0
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

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y_hat, _, _ = model(x)
            preds.extend(y_hat.cpu().numpy().reshape(-1))
            trues.extend(y.cpu().numpy().reshape(-1))

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    trues = scaler.inverse_transform(np.array(trues).reshape(-1, 1)).flatten()
    return rmse(trues, preds)


# RUN SENSITIVITY GRIDS

SEEDS = [42, 1, 7]

DEFAULT = dict(window=12, T_diffusion=100, warmup=10, hidden=64)

GRIDS = {
    "window":          {"param": "window",       "values": [6, 12, 18]},
    "diffusion_steps": {"param": "T_diffusion",  "values": [50, 100, 200]},
    "warmup":          {"param": "warmup",       "values": [0, 10, 20]},
    "hidden_dim":      {"param": "hidden",       "values": [32, 64, 128]},
}

for grid_name, spec in GRIDS.items():
    print(f"\n===== Sensitivity: {grid_name} =====")
    param = spec["param"]
    rows = []

    for product in products:
        for value in spec["values"]:
            kwargs = dict(DEFAULT)
            kwargs[param] = value
            rmses = []
            for seed in SEEDS:
                r = train_eval(product, seed=seed, **kwargs)
                rmses.append(r)
            rows.append({
                "product":  product,
                param:      value,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std":  float(np.std(rmses, ddof=1)),
            })
            print(f"  {product:12s} {param}={value:>4}  RMSE={np.mean(rmses):.3f} ± {np.std(rmses, ddof=1):.3f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT_DIR, f"{grid_name}.csv"), index=False)


# FIGURE — 2x2 grid of sensitivity curves

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

for idx, grid_name in enumerate(GRIDS.keys()):
    ax = axes[idx]
    df_g = pd.read_csv(os.path.join(OUT_DIR, f"{grid_name}.csv"))
    param = GRIDS[grid_name]["param"]
    for product in products:
        sub = df_g[df_g["product"] == product].sort_values(param)
        ax.plot(sub[param], sub["rmse_mean"], marker="o", label=product)
    ax.set_xlabel(param)
    ax.set_ylabel("RMSE (test set, mean of 3 seeds)")
    ax.set_title(f"Sensitivity to {param}")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_sensitivity.png"), dpi=150)
plt.close()

print(f"\nAll sensitivity outputs written to {OUT_DIR}/")