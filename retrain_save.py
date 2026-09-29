# =============================================================
# retrain_save.py
#   Retrains LSTM, Attention-LSTM, and DA-LSTM (Variant D).
#   Saves model weights + scaler for multi-step rolling-origin
#   evaluation in multistep_eval.py.
#
#   Pipeline:
#     - leakage-free (scaler fit on train only)
#     - 70/15/15 chronological split
#     - seed 42 only (multi-step eval is over origins, not seeds)
# =============================================================

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

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

os.makedirs("results/weights", exist_ok=True)

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

# -------------------------------------------------------------
# BASELINE MODELS
# -------------------------------------------------------------
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
        z = self.attn(h)
        return self.decoder(z).squeeze(-1)

# -------------------------------------------------------------
# DA-LSTM (Variant D) — final architecture
# -------------------------------------------------------------
class DALSTMModel(nn.Module):
    """
    Variant D:
      - LSTM encoder
      - temporal attention
      - concat attention context + LSTM last hidden -> projection
      - latent diffusion with one-step denoiser used at inference
      - decoder consumes denoised latent
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

# -------------------------------------------------------------
# TRAIN + SAVE
# -------------------------------------------------------------
def train_and_save(product, model_name, seed=42, window=12,
                   train_ratio=0.70, val_ratio=0.15,
                   epochs=80, warmup=10, batch_size=32, lr=1e-3):

    set_seed(seed)
    data = df[product].values.reshape(-1, 1)
    n = len(data)
    train_end = int(train_ratio * n)
    val_end   = int((train_ratio + val_ratio) * n)

    train_raw = data[:train_end]
    val_raw   = data[train_end:val_end]

    scaler = StandardScaler().fit(train_raw)
    train_scaled = scaler.transform(train_raw)
    val_scaled   = scaler.transform(val_raw)

    train_ds = TimeSeriesDataset(train_scaled, window)
    val_input = np.vstack([train_scaled[-window:], val_scaled])
    val_ds    = TimeSeriesDataset(val_input, window)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    cls_map = {"LSTM": LSTMModel,
               "Attention-LSTM": AttentionLSTMModel,
               "DA-LSTM": DALSTMModel}
    model = cls_map[model_name](1, 64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        lam = min(1.0, epoch / warmup)
        for x, y in train_loader:
            x = x.to(device); y = y.to(device).squeeze(-1)
            opt.zero_grad()
            out = model(x)
            if isinstance(out, tuple):
                y_hat, noise, noise_pred = out
                loss_pred = F.mse_loss(y_hat, y)
                loss_diff = F.mse_loss(noise_pred, noise)
                # Variant D uses rescaled loss
                loss = (loss_pred / (loss_pred.detach() + 1e-8)
                        + lam * loss_diff / (loss_diff.detach() + 1e-8))
            else:
                loss = F.mse_loss(out, y)
            loss.backward()
            opt.step()

    fname = f"results/weights/{product}_{model_name}_seed{seed}.pt"
    torch.save({
        "state_dict":   model.state_dict(),
        "model_name":   model_name,
        "product":      product,
        "scaler_mean":  scaler.mean_,
        "scaler_scale": scaler.scale_,
        "window":       window,
        "train_end":    train_end,
        "val_end":      val_end,
        "n":            n,
        "seed":         seed,
    }, fname)
    print(f"  saved {fname}")

# -------------------------------------------------------------
# RUN
# -------------------------------------------------------------
MODELS_TO_TRAIN = ["LSTM", "Attention-LSTM", "DA-LSTM"]

for product in products:
    print(f"\n=== {product} ===")
    for model_name in MODELS_TO_TRAIN:
        train_and_save(product, model_name, seed=42)

print("\nAll model weights saved to results/weights/")