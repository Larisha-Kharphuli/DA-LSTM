# IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# LOAD DATA
df = pd.read_excel(r"D:\Research\Objective 1\Myhybridmodel1\Price_index_data\Agricultural_data.xlsx")
df = df.sort_values(["Year","Month"]).reset_index(drop=True)

products = ["Onion","Cumin","Garlic","Castor_Seed","Tomato","Potato"]


# DATASET
class TimeSeriesDataset(Dataset):
    def __init__(self, series, window):
        self.X, self.y = [], []

        for i in range(len(series) - window):
            self.X.append(series[i:i+window])
            self.y.append(series[i+window])

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# MODEL COMPONENTS
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
        weights = torch.softmax(self.attn(h), dim=1)
        z = torch.sum(weights * h, dim=1)
        return z


# DIFFUSION
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
        alpha_bar_t = self.alpha_bar[t].unsqueeze(1)

        z_t = torch.sqrt(alpha_bar_t) * z0 + torch.sqrt(1 - alpha_bar_t) * noise
        t_norm = t.float().unsqueeze(1) / self.T

        noise_pred = self.denoiser(z_t, t_norm)

        return noise, noise_pred


class ForecastHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, z):
        return self.fc(z)


# HYBRID MODEL
class HybridModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim)
        self.attn = TemporalAttention(hidden_dim)
        self.diffusion = LatentDDPM(hidden_dim)
        self.decoder = ForecastHead(hidden_dim)

    def forward(self, x):
        h = self.encoder(x)
        z0 = self.attn(h)
        noise, noise_pred = self.diffusion(z0)
        y_hat = self.decoder(z0).squeeze(-1)
        return y_hat, noise, noise_pred


# FUTURE FORECAST FUNCTION
def forecast_future(model, last_window, scaler, steps=12):

    model.eval()
    preds = []
    window = last_window.copy()

    with torch.no_grad():
        for _ in range(steps):

            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            y_hat, _, _ = model(x)

            pred = y_hat.cpu().numpy().flatten()[0]
            preds.append(pred)

            window = np.vstack([window[1:], [[pred]]])

    preds = np.array(preds).reshape(-1,1)
    preds = scaler.inverse_transform(preds)

    return preds.flatten()


# TRAINING SETTINGS
window = 12
results = []

all_predictions = []
all_actual = []


# TRAIN LOOP
for product in products:

    print("\nTraining:", product)

    data = df[product].values.reshape(-1,1)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    dataset = TimeSeriesDataset(data_scaled, window)

    train_size = int(0.7 * len(dataset))

    train_ds = torch.utils.data.Subset(dataset, range(train_size))
    val_ds   = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = HybridModel(1, 64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 80
    warmup = 10

    #TRAIN
    for epoch in range(epochs):

        model.train()
        lambda_diff = min(1.0, epoch / warmup)

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device).squeeze(-1)

            optimizer.zero_grad()

            y_hat, noise, noise_pred = model(x)

            loss = F.mse_loss(y_hat, y) + lambda_diff * F.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()

    #VALIDATION
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device).squeeze(-1)

            y_hat, _, _ = model(x)

            preds.extend(y_hat.cpu().numpy().reshape(-1))
            trues.extend(y.cpu().numpy().reshape(-1))

    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1))
    trues = scaler.inverse_transform(np.array(trues).reshape(-1,1))

    #FORECAST
    future_steps = 12
    last_window = data_scaled[-window:]
    future_preds = forecast_future(model, last_window, scaler, future_steps)

    #PLOT
    plt.figure(figsize=(10,5))
    plt.plot(data.flatten(), label="Historical", linewidth=2)

    future_index = range(len(data), len(data)+future_steps)
    plt.plot(future_index, future_preds, linestyle="--", label="Forecast")

    plt.axvline(x=len(data)-1, linestyle=":", label="Forecast Start")

    plt.title(f"{product} Forecast")
    plt.legend()
    plt.grid(True)
    plt.show()

    #STORE
    all_predictions.append(preds.flatten())
    all_actual.append(trues.flatten())

    #METRICS
    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae  = mean_absolute_error(trues, preds)
    r2   = r2_score(trues, preds)

    results.append([product, rmse, mae, r2])


# FINAL SHAPE FIX
all_predictions = np.array(all_predictions).T
all_actual = np.array(all_actual).T

print("Final shape:", all_predictions.shape)


# SAVE RESULTS
results_df = pd.DataFrame(results, columns=["Product","RMSE","MAE","R2"])

np.savez(
    "Hybrid_results.npz",
    predictions=all_predictions,
    actual=all_actual,
    metrics=results_df.values
)

print("\nHybrid results saved correctly.")


# FINAL RESULTS
print("\nFinal Results")
print(results_df)