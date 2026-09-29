# =============================================================
# DA-LSTM component ablation: five configurations C1..C5
#   C1: LSTM only
#   C2: LSTM + temporal attention                (= Attention-LSTM)
#   C3: LSTM + latent diffusion (no attention)
#   C4: LSTM + attention + diffusion (no residual fusion)
#   C5: LSTM + attention + diffusion + residual fusion (= full DA-LSTM)
#
# Fixed pipeline: leakage-free scaling, 70/15/15 split, 5 seeds.
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
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
DATA_PATH = r"D:\Research\Objective 1\Myhybridmodel1\Price_index_data\Correction\Agricultural_data.xlsx"
df = pd.read_excel(DATA_PATH).sort_values(["Year", "Month"]).reset_index(drop=True)
products = ["Onion", "Cumin", "Garlic", "Castor_Seed", "Tomato", "Potato"]

# -------------------------------------------------------------
# DATASET
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# SHARED MODULES
# -------------------------------------------------------------
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
            nn.Linear(dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
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
    """Consistency-style single-step denoiser: z0 -> z_hat_0.
    Supervised by the DDPM objective (predict the noise added to z0)."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, dim)
        )
    def forward(self, z0):
        return self.net(z0)

# -------------------------------------------------------------
# CONFIGURATION C1 — LSTM only
# -------------------------------------------------------------
class ModelC1(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = self.encoder(x)
        z = h[:, -1, :]
        y_hat = self.decoder(z).squeeze(-1)
        # No diffusion in this configuration; return placeholders so the
        # training loop stays uniform across configurations.
        return y_hat, None, None

# -------------------------------------------------------------
# CONFIGURATION C2 — LSTM + temporal attention
# -------------------------------------------------------------
class ModelC2(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim)
        self.attn = TemporalAttention(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = self.encoder(x)
        z = self.attn(h)
        y_hat = self.decoder(z).squeeze(-1)
        return y_hat, None, None

# -------------------------------------------------------------
# CONFIGURATION C3 — LSTM + latent diffusion (no attention)
# -------------------------------------------------------------
class ModelC3(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim)
        self.diffusion = LatentDDPM(hidden_dim)
        self.denoiser1 = OneStepDenoiser(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = self.encoder(x)
        z0 = h[:, -1, :]
        noise, noise_pred = self.diffusion(z0)
        z_hat = self.denoiser1(z0)
        y_hat = self.decoder(z_hat).squeeze(-1)
        return y_hat, noise, noise_pred

# -------------------------------------------------------------
# CONFIGURATION C4 — LSTM + attention + diffusion, no residual fusion
# -------------------------------------------------------------
class ModelC4(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim)
        self.attn = TemporalAttention(hidden_dim)
        self.diffusion = LatentDDPM(hidden_dim)
        self.denoiser1 = OneStepDenoiser(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = self.encoder(x)
        z0 = self.attn(h)
        noise, noise_pred = self.diffusion(z0)
        z_hat = self.denoiser1(z0)
        y_hat = self.decoder(z_hat).squeeze(-1)
        return y_hat, noise, noise_pred

# -------------------------------------------------------------
# CONFIGURATION C5 — full DA-LSTM
#   LSTM + attention + diffusion + residual fusion
# -------------------------------------------------------------
class ModelC5(nn.Module):
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
# FUTURE FORECAST
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# TRAINING LOOP (per product, per seed, per configuration)
# -------------------------------------------------------------
def train_one(product, config, seed=42, window=12,
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

    # pick model class
    ModelCls = {"C1": ModelC1, "C2": ModelC2, "C3": ModelC3,
                "C4": ModelC4, "C5": ModelC5}[config]
    model = ModelCls(1, 64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Configurations that actually use the diffusion module
    uses_diffusion = config in ("C3", "C4", "C5")

    for epoch in range(epochs):
        model.train()
        lambda_diff = min(1.0, epoch / warmup) if warmup > 0 else 1.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).squeeze(-1)
            opt.zero_grad()
            y_hat, noise, noise_pred = model(x)

            loss_pred = F.mse_loss(y_hat, y)

            if uses_diffusion:
                loss_diff = F.mse_loss(noise_pred, noise)
                # loss rescaling (same as variants C/D in the development ablation)
                loss = (loss_pred / (loss_pred.detach() + 1e-8)
                        + lambda_diff * loss_diff / (loss_diff.detach() + 1e-8))
            else:
                loss = loss_pred

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

    _, _            = predict(val_loader)   # kept for parity, not reported
    test_true, test_pred = predict(test_loader)

    last_window = scaler.transform(data[-window:])
    future_pred = forecast_future(model, last_window, scaler, steps=12)

    metrics = {
        "product":  product, "config": config, "seed": seed,
        "rmse":     rmse(test_true, test_pred),
        "mae":      mae(test_true, test_pred),
        "r2":       r2(test_true, test_pred),
        "mase":     mase(test_true, test_pred, train_raw.flatten()),
        "theils_u": theils_u(test_true, test_pred),
    }
    return metrics, test_true, test_pred, future_pred

# -------------------------------------------------------------
# RUN ALL CONFIGS × PRODUCTS × SEEDS
# -------------------------------------------------------------
SEEDS = [42, 1, 7, 13, 21]
WINDOW = 12
CONFIGS = ["C1", "C2", "C3", "C4", "C5"]

os.makedirs("results/component_ablation", exist_ok=True)

all_metrics = []
store_preds = {(c, p): {"true": None, "pred": None, "future": None}
               for c in CONFIGS for p in products}

for config in CONFIGS:
    print(f"\n{'='*60}\nCONFIG {config}\n{'='*60}")
    for product in products:
        for seed in SEEDS:
            print(f"  {product:12s} seed {seed}")
            m, tt, tp, fp = train_one(product, config, seed=seed, window=WINDOW)
            all_metrics.append(m)
            if store_preds[(config, product)]["true"] is None:
                store_preds[(config, product)]["true"]   = tt
                store_preds[(config, product)]["pred"]   = tp
                store_preds[(config, product)]["future"] = fp

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv("results/component_ablation/component_perseed.csv", index=False)

# -------------------------------------------------------------
# AGGREGATED TABLE (mean ± std across seeds)
# -------------------------------------------------------------
agg = metrics_df.groupby(["config", "product"]).agg(
    rmse_mean=("rmse","mean"),       rmse_std=("rmse","std"),
    mae_mean=("mae","mean"),         mae_std=("mae","std"),
    r2_mean=("r2","mean"),           r2_std=("r2","std"),
    mase_mean=("mase","mean"),       mase_std=("mase","std"),
    theils_mean=("theils_u","mean"), theils_std=("theils_u","std"),
).reset_index()
agg.to_csv("results/component_ablation/component_aggregated.csv", index=False)

print("\n===== Component ablation aggregate (mean across seeds) =====")
print(agg.round(3).to_string(index=False))

# -------------------------------------------------------------
# WIDE COMPARISON: RMSE across configurations
# -------------------------------------------------------------
rmse_wide = agg.pivot(index="product", columns="config", values="rmse_mean")
print("\n===== RMSE by product × configuration =====")
print(rmse_wide.round(3))

# Deltas vs full model (C5)
# positive = C5 better (removed component was helping)
# negative = C5 worse (removed component was actually hurting)
print("\n===== Delta RMSE vs full model C5 =====")
print("(positive = removed component helped; negative = removed component hurt)")
delta = rmse_wide.sub(rmse_wide["C5"], axis=0)
print(delta.round(3))
delta.to_csv("results/component_ablation/component_delta_vs_C5.csv")

# Theil's U wide
theils_wide = agg.pivot(index="product", columns="config", values="theils_mean")
print("\n===== Theil's U by product × configuration =====")
print(theils_wide.round(3))

# -------------------------------------------------------------
# SAVE NPZ PER CONFIGURATION (for downstream comparison)
# -------------------------------------------------------------
for config in CONFIGS:
    min_len = min(len(store_preds[(config, p)]["true"]) for p in products)
    preds_arr   = np.stack([store_preds[(config, p)]["pred"][:min_len] for p in products], axis=1)
    actuals_arr = np.stack([store_preds[(config, p)]["true"][:min_len] for p in products], axis=1)
    np.savez(f"results/component_ablation/DA_LSTM_{config}_results.npz",
             predictions=preds_arr, actual=actuals_arr, products=np.array(products))

# -------------------------------------------------------------
# FIGURES
# -------------------------------------------------------------
import seaborn as sns
sns.set_style("whitegrid")

# RMSE per configuration
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=agg, x="product", y="rmse_mean", hue="config",
            hue_order=CONFIGS, ax=ax)
ax.set_title("RMSE by configuration and commodity (mean across 5 seeds)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("results/component_ablation/fig_rmse_by_config.png", dpi=150)
plt.close()

# Theil's U per configuration with naive line
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=agg, x="product", y="theils_mean", hue="config",
            hue_order=CONFIGS, ax=ax)
ax.axhline(1.0, color="red", linestyle="--", label="Naïve (Theil's U = 1)")
ax.set_title("Theil's U by configuration and commodity")
plt.xticks(rotation=20)
plt.legend()
plt.tight_layout()
plt.savefig("results/component_ablation/fig_theils_by_config.png", dpi=150)
plt.close()

# Delta heatmap (config - C5). Positive = removed component was helping.
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(delta, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("ΔRMSE vs full DA-LSTM (C5)\nPositive = removed component was helping")
plt.tight_layout()
plt.savefig("results/component_ablation/fig_delta_heatmap.png", dpi=150)
plt.close()

# Per-product prediction plot (all configurations vs actual) — seed 42
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()
for i, product in enumerate(products):
    ax = axes[i]
    ax.plot(store_preds[("C5", product)]["true"], label="Actual",
            color="black", lw=2)
    for c in CONFIGS:
        ax.plot(store_preds[(c, product)]["pred"], label=f"Config {c}", alpha=0.8)
    ax.set_title(product)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/component_ablation/fig_predictions_grid.png", dpi=150)
plt.close()

print("\nComponent ablation complete. Outputs in results/component_ablation/")