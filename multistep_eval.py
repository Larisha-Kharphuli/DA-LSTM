import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

DATA_PATH = r"D:\Research\Objective 1\Myhybridmodel1\Price_index_data\Correction\Agricultural_data.xlsx"
df = pd.read_excel(DATA_PATH).sort_values(["Year", "Month"]).reset_index(drop=True)
products = ["Onion", "Cumin", "Garlic", "Castor_Seed", "Tomato", "Potato"]

WINDOW = 12
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15
HORIZON = 12
SEASON = 12

OUT_DIR = "results/multistep"
os.makedirs(OUT_DIR, exist_ok=True)


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

def linear_beta_schedule(T): return torch.linspace(1e-4, 0.02, T)

class DenoiseNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + 1, 128), nn.ReLU(), nn.Linear(128, dim))
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

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.decoder(h[:, -1, :]).squeeze(-1)

class AttentionLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attn = TemporalAttention(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.decoder(self.attn(h)).squeeze(-1)

class DALSTMModel(nn.Module):
    """Variant D — matches the architecture saved by retrain_save.py."""
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

CLS_MAP = {"LSTM": LSTMModel,
           "Attention-LSTM": AttentionLSTMModel,
           "DA-LSTM": DALSTMModel}


def load_model(product, model_name, seed=42):
    path = f"results/weights/{product}_{model_name}_seed{seed}.pt"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CLS_MAP[model_name](1, 64).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scaler = StandardScaler()
    scaler.mean_  = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_   = scaler.scale_ ** 2
    scaler.n_features_in_ = 1
    return model, scaler, ckpt

# RECURSIVE MULTI-STEP FORECAST (neural)
def recursive_forecast(model, window_scaled, steps=12):
    """window_scaled: (WINDOW, 1) numpy array. Returns (steps, 1) scaled preds."""
    preds = []
    w = window_scaled.copy()
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(device)
            out = model(x)
            y_hat = out[0] if isinstance(out, tuple) else out
            p = y_hat.cpu().numpy().flatten()[0]
            preds.append(p)
            w = np.vstack([w[1:], [[p]]])
    return np.array(preds).reshape(-1, 1)


# BASELINE FORECASTS
def persistence_forecast(raw_window, steps=12):
    return np.full((steps,), raw_window[-1])

def seasonal_naive_forecast(raw_history, origin_idx, steps=12, season=12):
    preds = []
    for h in range(1, steps + 1):
        idx = origin_idx + h - season
        preds.append(raw_history[idx] if idx >= 0 else raw_history[-1])
    return np.array(preds)


# ROLLING-ORIGIN EVALUATION

records = []

for product in products:
    print(f"\n=== {product} ===")
    raw = df[product].values.astype(float)
    n = len(raw)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * n)

    origins = list(range(val_end, n - HORIZON + 1))
    print(f"  origins: {len(origins)} (from idx {val_end} to {n - HORIZON})")

    neural_models = {}
    scalers = {}
    for m in ["LSTM", "Attention-LSTM", "DA-LSTM"]:
        try:
            mdl, scl, _ = load_model(product, m)
            neural_models[m] = mdl
            scalers[m] = scl
        except FileNotFoundError:
            print(f"    [skip] no weights for {m}")

    full_train = raw[:val_end]
    try:
        sarima = SARIMAX(
            full_train,
            order=(1,1,1),
            seasonal_order=(0,1,1,12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    except Exception as e:
        print(f"    SARIMA fit failed: {e}")
        sarima = None

    for origin in origins:
        actual_future = raw[origin:origin + HORIZON]

        pers   = persistence_forecast(raw[:origin], HORIZON)
        snaive = seasonal_naive_forecast(raw, origin, HORIZON, SEASON)

        if sarima is not None:
            try:
                fc = sarima.get_forecast(steps=(origin - val_end) + HORIZON).predicted_mean
                sarima_pred = np.array(fc[origin - val_end:])
            except Exception:
                sarima_pred = np.full(HORIZON, raw[origin - 1])
        else:
            sarima_pred = np.full(HORIZON, np.nan)

        for m, mdl in neural_models.items():
            scl = scalers[m]
            win_raw = raw[origin - WINDOW:origin].reshape(-1, 1)
            win_scaled = scl.transform(win_raw)
            pred_scaled = recursive_forecast(mdl, win_scaled, HORIZON)
            pred = scl.inverse_transform(pred_scaled).flatten()
            for h in range(1, HORIZON + 1):
                records.append({
                    "product": product, "model": m, "origin": origin,
                    "horizon": h,
                    "actual": actual_future[h-1],
                    "pred":   pred[h-1],
                    "error":  actual_future[h-1] - pred[h-1],
                })

        for name, p in [("Persistence", pers),
                        ("SeasonalNaive", snaive),
                        ("SARIMA", sarima_pred)]:
            for h in range(1, HORIZON + 1):
                records.append({
                    "product": product, "model": name, "origin": origin,
                    "horizon": h,
                    "actual": actual_future[h-1],
                    "pred":   p[h-1],
                    "error":  actual_future[h-1] - p[h-1],
                })

err_df = pd.DataFrame(records)
err_df.to_csv(os.path.join(OUT_DIR, "multistep_errors_raw.csv"), index=False)
print(f"\nTotal predictions recorded: {len(err_df)}")




def agg_metrics(df):
    rows = []
    for (prod, mdl, h), g in df.groupby(["product", "model", "horizon"]):
        if g["actual"].isna().any():
            continue
        y, p = g["actual"].values, g["pred"].values
        rows.append({
            "product": prod, "model": mdl, "horizon": h,
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae":  float(mean_absolute_error(y, p)),
            "n_origins": len(g),
        })
    return pd.DataFrame(rows)

metrics_h = agg_metrics(err_df)
metrics_h.to_csv(os.path.join(OUT_DIR, "multistep_metrics_by_horizon.csv"), index=False)


summary = metrics_h.groupby(["product", "model"]).agg(
    rmse_h1=("rmse", lambda s: s.iloc[0]),
    rmse_mean=("rmse", "mean"),
    rmse_h12=("rmse", lambda s: s.iloc[-1]),
).reset_index()
summary.to_csv(os.path.join(OUT_DIR, "multistep_summary.csv"), index=False)

print("\n===== Multi-step summary (RMSE) =====")
print(summary.round(3).to_string(index=False))




sns.set_style("whitegrid")

# 1) RMSE vs horizon, per commodity
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()
for i, product in enumerate(products):
    ax = axes[i]
    sub = metrics_h[metrics_h["product"] == product]
    for m in sub["model"].unique():
        s = sub[sub["model"] == m].sort_values("horizon")
        ax.plot(s["horizon"], s["rmse"], marker="o", label=m)
    ax.set_title(product)
    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_rmse_vs_horizon.png"), dpi=150)
plt.close()

# 2) Mean RMSE across horizons
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=summary, x="product", y="rmse_mean", hue="model", ax=ax)
ax.set_title("Mean RMSE across horizons 1–12 (rolling-origin)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_rmse_mean_by_model.png"), dpi=150)
plt.close()

# 3) RMSE at h=12
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=summary, x="product", y="rmse_h12", hue="model", ax=ax)
ax.set_title("RMSE at horizon = 12 (rolling-origin)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_rmse_h12.png"), dpi=150)
plt.close()

print(f"\nMulti-step evaluation complete. Outputs in {OUT_DIR}/")