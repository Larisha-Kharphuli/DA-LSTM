import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Reproducibility
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# Data
DATA_PATH = r"D:\Research\Objective 1\Myhybridmodel1\Price_index_data\Correction\Agricultural_data.xlsx"
df = pd.read_excel(DATA_PATH).sort_values(["Year", "Month"]).reset_index(drop=True)
products = ["Onion", "Cumin", "Garlic", "Castor_Seed", "Tomato", "Potato"]


class TimeSeriesDataset(Dataset):
    def __init__(self, series, window):
        X, y = [], []
        for i in range(len(series) - window):
            X.append(series[i:i + window]); y.append(series[i + window])
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# Building blocks
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        h, _ = self.lstm(x); return h

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
    def __init__(self, dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, dim))
    def forward(self, z_t, t):
        return self.net(torch.cat([z_t, t], dim=1))


class LatentDDPM(nn.Module):
    def __init__(self, dim, T=100, hidden=32):
        super().__init__()
        self.T = T
        self.denoiser = DenoiseNet(dim, hidden=hidden)
        betas  = linear_beta_schedule(T)
        alphas = 1 - betas
        self.register_buffer("alpha_bar", torch.cumprod(alphas, dim=0))
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)

    def forward(self, z0):
        B = z0.size(0)
        t = torch.randint(0, self.T, (B,), device=z0.device)
        noise = torch.randn_like(z0)
        ab = self.alpha_bar[t].unsqueeze(1)
        z_t = torch.sqrt(ab) * z0 + torch.sqrt(1 - ab) * noise
        t_norm = t.float().unsqueeze(1) / self.T
        noise_pred = self.denoiser(z_t, t_norm)
        return noise, noise_pred

    @torch.no_grad()
    def reconstruct(self, z0, t_start=10, deterministic=True):
        self.eval()
        T = self.T
        t_start = int(min(max(t_start, 0), T - 1))
        steps = max(t_start, 1)
        ab0 = self.alpha_bar[t_start]
        eps = torch.randn_like(z0)
        z = torch.sqrt(ab0) * z0 + torch.sqrt(1 - ab0) * eps
        ts = torch.linspace(t_start, 0, steps).long().to(z0.device)
        for i, t in enumerate(ts):
            t_batch = t.expand(z.size(0))
            t_norm  = t_batch.float().unsqueeze(1) / T
            eps_hat = self.denoiser(z, t_norm)
            ab = self.alpha_bar[t]
            z0_hat = (z - torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(ab)
            if i == len(ts) - 1 or int(t) == 0:
                z = z0_hat
            else:
                t_next = ts[i + 1]; ab_next = self.alpha_bar[t_next]
                beta_t, alpha_t = self.betas[t], self.alphas[t]
                var  = beta_t * (1 - ab_next) / (1 - ab)
                mean = (torch.sqrt(ab_next) * beta_t / (1 - ab)) * z0_hat \
                     + (torch.sqrt(alpha_t) * (1 - ab_next) / (1 - ab)) * z
                z = mean if deterministic else mean + torch.sqrt(var) * torch.randn_like(z)
        return z


class OneStepDenoiser(nn.Module):
    def __init__(self, dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, dim))
    def forward(self, z0):
        return self.net(z0)


# Configurations (step-by-step ladder)
class ModelC1(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = self.encoder(x); z = h[:, -1, :]
        return self.decoder(z).squeeze(-1), None, None

class ModelC2(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim)
        self.attn = TemporalAttention(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = self.encoder(x); z = self.attn(h)
        return self.decoder(z).squeeze(-1), None, None

class ModelC3(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim)
        self.diffusion = LatentDDPM(hidden_dim)
        self.denoiser1 = OneStepDenoiser(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = self.encoder(x); z0 = h[:, -1, :]
        noise, noise_pred = self.diffusion(z0)
        z_hat = self.denoiser1(z0)
        return self.decoder(z_hat).squeeze(-1), noise, noise_pred

class ModelC4(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim)
        self.attn = TemporalAttention(hidden_dim)
        self.diffusion = LatentDDPM(hidden_dim)
        self.denoiser1 = OneStepDenoiser(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = self.encoder(x); z0 = self.attn(h)
        noise, noise_pred = self.diffusion(z0)
        z_hat = self.denoiser1(z0)
        return self.decoder(z_hat).squeeze(-1), noise, noise_pred

class ModelC5(nn.Module):
    """Ladder rung C5: attention + fusion, NO diffusion."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim)
        self.attn    = TemporalAttention(hidden_dim)
        self.proj    = nn.Linear(2 * hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = self.encoder(x)
        z_attn = self.attn(h); h_last = h[:, -1, :]
        z = self.proj(torch.cat([z_attn, h_last], dim=-1))
        return self.decoder(z).squeeze(-1), None, None

class ModelC6(nn.Module):
    """Ladder rung C6: attention + fusion + reverse-chain diffusion."""
    def __init__(self, input_dim, hidden_dim, t_start=10):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim)
        self.attn    = TemporalAttention(hidden_dim)
        self.diffusion = LatentDDPM(hidden_dim, hidden=32)
        self.proj    = nn.Linear(2 * hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)
        self.t_start = t_start
        self.use_recon_in_training = False
    def forward(self, x):
        h = self.encoder(x)
        z_attn = self.attn(h); h_last = h[:, -1, :]
        z = self.proj(torch.cat([z_attn, h_last], dim=-1))
        noise, noise_pred = self.diffusion(z)
        if self.training and not self.use_recon_in_training:
            z_hat = z
        else:
            z_hat = self.diffusion.reconstruct(z, t_start=self.t_start,
                                               deterministic=True)
        return self.decoder(z_hat).squeeze(-1), noise, noise_pred

class ModelC7(nn.Module):
    """Ladder rung C7 (proposed DA-LSTM). Inference uses denoiser1(z) only."""
    def __init__(self, input_dim, hidden_dim, t_start=10):
        super().__init__()
        self.encoder   = LSTMEncoder(input_dim, hidden_dim)
        self.attn      = TemporalAttention(hidden_dim)
        self.diffusion = LatentDDPM(hidden_dim, hidden=32)
        self.denoiser1 = OneStepDenoiser(hidden_dim, hidden=32)
        self.proj      = nn.Linear(2 * hidden_dim, hidden_dim)
        self.decoder   = nn.Linear(hidden_dim, 1)
        self.t_start   = t_start
    def forward(self, x):
        h = self.encoder(x)
        z_attn = self.attn(h); h_last = h[:, -1, :]
        z = self.proj(torch.cat([z_attn, h_last], dim=-1))
        noise, noise_pred = self.diffusion(z)
        z_hat = self.denoiser1(z)
        return self.decoder(z_hat).squeeze(-1), noise, noise_pred

# Metrics
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


def forecast_future(model, last_window, scaler, steps=12):
    model.eval(); preds = []; window = last_window.copy()
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            out = model(x); y_hat = out[0] if isinstance(out, tuple) else out
            pred = y_hat.cpu().numpy().flatten()[0]
            preds.append(pred)
            window = np.vstack([window[1:], [[pred]]])
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


# C7 (DA-LSTM) hyperparameters
C7_BEST = {
    "Onion":       dict(lr=5e-4, t_start=10),
    "Cumin":       dict(lr=1e-3, t_start=10),
    "Garlic":      dict(lr=1e-3, t_start=10),
    "Castor_Seed": dict(lr=5e-4, t_start=10),
    "Tomato":      dict(lr=5e-4, t_start=10),
    "Potato":      dict(lr=5e-4, t_start=10),
}

_TUNING_PROVENANCE = pd.DataFrame([
    {"product": "Onion",       "lr": 5e-4, "t_start": 10, "selected": True},
    {"product": "Cumin",       "lr": 1e-3, "t_start": 10, "selected": True},
    {"product": "Garlic",      "lr": 1e-3, "t_start": 10, "selected": True},
    {"product": "Castor_Seed", "lr": 5e-4, "t_start": 10, "selected": True},
    {"product": "Tomato",      "lr": 5e-4, "t_start": 10, "selected": True},
    {"product": "Potato",      "lr": 5e-4, "t_start": 10, "selected": True},
])


# Training
def train_one(product, config, seed=42, window=12,
              train_ratio=0.70, val_ratio=0.15,
              epochs=80, warmup=10, batch_size=32, lr=1e-3,
              t_start=10, finetune_epochs=10):

    set_seed(seed)
    data = df[product].values.reshape(-1, 1)
    n = len(data)
    train_end = int(train_ratio * n)
    val_end   = int((train_ratio + val_ratio) * n)

    train_raw = data[:train_end]; val_raw = data[train_end:val_end]; test_raw = data[val_end:]
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

    hidden_dim = 64
    if config == "C7":
        model = ModelC7(1, hidden_dim, t_start=t_start).to(device)
    elif config == "C6":
        model = ModelC6(1, hidden_dim, t_start=t_start).to(device)
    elif config == "C5":
        model = ModelC5(1, hidden_dim).to(device)
    else:
        ModelCls = {"C1": ModelC1, "C2": ModelC2, "C3": ModelC3,
                    "C4": ModelC4}[config]
        model = ModelCls(1, hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    uses_diffusion = config in ("C3", "C4", "C6", "C7")
    finetune_start = epochs - finetune_epochs

    for epoch in range(epochs):
        model.train()
        if config == "C6":
            model.use_recon_in_training = (epoch >= finetune_start)
        lambda_diff = 0.5 + 0.5 * min(1.0, epoch / warmup) if warmup > 0 else 1.0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device).squeeze(-1)
            opt.zero_grad()
            y_hat, noise, noise_pred = model(x)
            loss_pred = F.mse_loss(y_hat, y)
            if uses_diffusion:
                loss_diff = F.mse_loss(noise_pred, noise)
                loss = loss_pred + lambda_diff * loss_diff
            else:
                loss = loss_pred
            loss.backward(); opt.step()

    def predict(loader):
        model.eval(); preds, trues = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device); out = model(x)
                y_hat = out[0] if isinstance(out, tuple) else out
                preds.extend(y_hat.cpu().numpy().reshape(-1))
                trues.extend(y.cpu().numpy().reshape(-1))
        preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        trues = scaler.inverse_transform(np.array(trues).reshape(-1, 1)).flatten()
        return trues, preds

    _, _ = predict(val_loader)
    test_true, test_pred = predict(test_loader)
    last_window = scaler.transform(data[-window:])
    future_pred = forecast_future(model, last_window, scaler, steps=12)

    metrics = {
        "product": product, "config": config, "seed": seed,
        "rmse": rmse(test_true, test_pred), "mae": mae(test_true, test_pred),
        "r2":   r2(test_true, test_pred),
        "mase": mase(test_true, test_pred, train_raw.flatten()),
        "theils_u": theils_u(test_true, test_pred),
    }
    residuals = np.asarray(test_true) - np.asarray(test_pred)
    return metrics, test_true, test_pred, future_pred, residuals


# Main ablation
SEEDS  = [42, 1, 7, 13, 21]
WINDOW = 12
CONFIGS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
DEFAULT_T_START = 10

OUTDIR = "results/component_ablation"
os.makedirs(OUTDIR, exist_ok=True)

all_metrics = []
store_preds = {(c, p): {"true": None, "pred": None, "future": None}
               for c in CONFIGS for p in products}

print("\n===== C7 (DA-LSTM) hyperparameters: frozen from validation tuning =====")
for product in products:
    cfg = C7_BEST[product]
    print(f"  [C7 frozen]  {product:12s}  ->  lr={cfg['lr']}, t_start={cfg['t_start']}")
_TUNING_PROVENANCE.to_csv(f"{OUTDIR}/c7_validation_tuning.csv", index=False)

print("\n===== Main ablation (5 seeds per product per config) =====")
for config in CONFIGS:
    print(f"\n{'='*60}\nCONFIG {config}\n{'='*60}")
    for product in products:
        for seed in SEEDS:
            print(f"  {product:12s} seed {seed}")
            if config == "C7":
                cfg = C7_BEST[product]
                m, tt, tp, fp, res = train_one(
                    product, config, seed=seed, window=WINDOW,
                    lr=cfg["lr"], t_start=cfg["t_start"])
            elif config == "C6":
                ts = C7_BEST[product]["t_start"]
                m, tt, tp, fp, res = train_one(
                    product, config, seed=seed, window=WINDOW, t_start=ts)
            else:
                m, tt, tp, fp, res = train_one(
                    product, config, seed=seed, window=WINDOW,
                    t_start=DEFAULT_T_START)
            all_metrics.append(m)
            if store_preds[(config, product)]["true"] is None:
                store_preds[(config, product)]["true"]   = tt
                store_preds[(config, product)]["pred"]   = tp
                store_preds[(config, product)]["future"] = fp

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(f"{OUTDIR}/component_perseed.csv", index=False)

agg = metrics_df.groupby(["config", "product"]).agg(
    rmse_mean=("rmse","mean"),       rmse_std=("rmse","std"),
    mae_mean=("mae","mean"),         mae_std=("mae","std"),
    r2_mean=("r2","mean"),           r2_std=("r2","std"),
    mase_mean=("mase","mean"),       mase_std=("mase","std"),
    theils_mean=("theils_u","mean"), theils_std=("theils_u","std"),
).reset_index()
agg.to_csv(f"{OUTDIR}/component_aggregated.csv", index=False)

rmse_wide = agg.pivot(index="product", columns="config", values="rmse_mean")[CONFIGS]
rmse_wide.to_csv(f"{OUTDIR}/rmse_all7.csv")

rmse_std  = agg.pivot(index="product", columns="config", values="rmse_std")[CONFIGS]
rmse_pm   = rmse_wide.round(2).astype(str) + " +/- " + rmse_std.round(2).astype(str)
rmse_pm.to_csv(f"{OUTDIR}/rmse_all7_mean_pm_std.csv")

best_table = pd.concat([rmse_wide.idxmin(axis=1).rename("best_config"),
                        rmse_wide.min(axis=1).rename("best_rmse")], axis=1)
best_table.to_csv(f"{OUTDIR}/rmse_best_per_commodity.csv")

theils_wide = agg.pivot(index="product", columns="config", values="theils_mean")[CONFIGS]
theils_wide.to_csv(f"{OUTDIR}/theils_all7.csv")

contrasts = pd.DataFrame({
    "C2-C5 fusion":                         rmse_wide["C2"] - rmse_wide["C5"],
    "C5-C6 reverse diffusion on fusion":    rmse_wide["C5"] - rmse_wide["C6"],
    "C2-C4 diffusion with attention":       rmse_wide["C2"] - rmse_wide["C4"],
    "C6-C7 one-step on top of reverse":     rmse_wide["C6"] - rmse_wide["C7"],
    "C5-C7 diffusion+one-step on fusion":   rmse_wide["C5"] - rmse_wide["C7"],
})
contrasts.to_csv(f"{OUTDIR}/contrasts.csv")

print("\n===== RMSE (mean +/- std) all 7 configs =====")
print(rmse_pm.to_string())
print("\n===== RMSE by product x configuration (mean) =====")
print(rmse_wide.round(3).to_string())
print("\n===== Best config per commodity =====")
print(best_table.round(3).to_string())
print("\n===== Contrasts (positive = second model better) =====")
print(contrasts.round(3).to_string())
print("\n===== Theil's U (all 7) =====")
print(theils_wide.round(3).to_string())

# C6 t_start sweep (C6 uses t_start at inference; C7 does not)
print("\n===== [C6] t_start sweep (seed 42) =====")
sweep_rows = []
for ts_ in [5, 10, 20, 99]:
    for product in products:
        m, *_ = train_one(product, "C6", seed=42, window=WINDOW, t_start=ts_)
        m["t_start"] = ts_; sweep_rows.append(m)
        print(f"  t_start={ts_:3d}  {product:12s}  RMSE = {m['rmse']:.3f}")
sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(f"{OUTDIR}/c6_tstart_sweep.csv", index=False)
print("\n===== C6 RMSE vs t_start =====")
print(sweep_df.pivot(index="product", columns="t_start", values="rmse").round(3).to_string())


# Save predictions
for config in CONFIGS:
    min_len = min(len(store_preds[(config, p)]["true"]) for p in products)
    np.savez(f"{OUTDIR}/DA_LSTM_{config}_results.npz",
             predictions=np.stack([store_preds[(config, p)]["pred"][:min_len]
                                   for p in products], axis=1),
             actual=np.stack([store_preds[(config, p)]["true"][:min_len]
                              for p in products], axis=1),
             products=np.array(products))


# Plots
import seaborn as sns
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(13, 5))
sns.barplot(data=agg, x="product", y="rmse_mean", hue="config", hue_order=CONFIGS, ax=ax)
ax.set_title("RMSE by configuration and commodity (mean across 5 seeds)")
plt.xticks(rotation=20); plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_rmse_by_config.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(rmse_wide, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("RMSE: commodity x configuration")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/fig_rmse_heatmap.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(contrasts, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Contrasts (positive = second model better)")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/fig_contrasts.png", dpi=150); plt.close()

fig, axes = plt.subplots(3, 2, figsize=(14, 12)); axes = axes.flatten()
for i, product in enumerate(products):
    ax = axes[i]
    ax.plot(store_preds[("C5", product)]["true"], label="Actual", color="black", lw=2)
    for c in CONFIGS:
        ax.plot(store_preds[(c, product)]["pred"], label=f"{c}", alpha=0.8)
    ax.set_title(product); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/fig_predictions_grid.png", dpi=150); plt.close()

print(f"\nDone. All outputs in {OUTDIR}/")