# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import mean_squared_error

# LOAD RESULTS
lstm   = np.load("lstm_results.npz", allow_pickle=True)
attn   = np.load("attention_lstm_results.npz", allow_pickle=True)
hybrid = np.load("Hybrid_results.npz", allow_pickle=True)

lstm_pred   = lstm["predictions"]     # (T,6)
attn_pred   = attn["predictions"]     # (T,6)
hybrid_pred = hybrid["predictions"]   # (T,6)


y_true = lstm["actual"]

products = ["Onion","Cumin","Garlic","Castor_Seed","Tomato","Potato"]



def rmse(y, p):
    return np.sqrt(mean_squared_error(y, p))

def align_all(y, p1, p2, p3):
    n = min(len(y), len(p1), len(p2), len(p3))
    return y[:n], p1[:n], p2[:n], p3[:n]


# RMSE PER PRODUCT
rmse_df = pd.DataFrame(index=products)

for i, product in enumerate(products):

    y = y_true[:, i]
    p1 = lstm_pred[:, i]
    p2 = attn_pred[:, i]
    p3 = hybrid_pred[:, i]

    y, p1, p2, p3 = align_all(y, p1, p2, p3)

    rmse_df.loc[product, "LSTM"] = rmse(y, p1)
    rmse_df.loc[product, "ATTN_LSTM"] = rmse(y, p2)
    rmse_df.loc[product, "HYBRID"] = rmse(y, p3)

print("\nRMSE per product:")
print(rmse_df)

print("\nBest model per product:")
print(rmse_df.idxmin(axis=1))


# PLOT PER PRODUCT
for i, product in enumerate(products):

    y = y_true[:, i]
    p1 = lstm_pred[:, i]
    p2 = attn_pred[:, i]
    p3 = hybrid_pred[:, i]

    y, p1, p2, p3 = align_all(y, p1, p2, p3)

    plt.figure(figsize=(10,5))
    plt.plot(y, label="Actual", linewidth=2)
    plt.plot(p1, label="LSTM")
    plt.plot(p2, label="ATTN-LSTM")
    plt.plot(p3, label="HYBRID")

    plt.title(product)
    plt.legend()
    plt.grid(True)
    plt.show()


# FLATTEN DATA
y_all = []
lstm_all = []
attn_all = []
hybrid_all = []

for i in range(len(products)):

    y = y_true[:, i]
    p1 = lstm_pred[:, i]
    p2 = attn_pred[:, i]
    p3 = hybrid_pred[:, i]

    y, p1, p2, p3 = align_all(y, p1, p2, p3)

    y_all.extend(y)
    lstm_all.extend(p1)
    attn_all.extend(p2)
    hybrid_all.extend(p3)

y_all = np.array(y_all)
lstm_all = np.array(lstm_all)
attn_all = np.array(attn_all)
hybrid_all = np.array(hybrid_all)

print("\nAligned sample size:", len(y_all))


# GLOBAL METRICS
print("\n===== GLOBAL PERFORMANCE =====")

def print_metrics(name, y, p):
    print(f"{name} RMSE:", rmse(y,p))

print_metrics("LSTM", y_all, lstm_all)
print_metrics("ATTN_LSTM", y_all, attn_all)
print_metrics("HYBRID", y_all, hybrid_all)


# FRIEDMAN TEST
print("\n===== FRIEDMAN TEST =====")

friedman = friedmanchisquare(
    np.abs(y_all - lstm_all),
    np.abs(y_all - attn_all),
    np.abs(y_all - hybrid_all)
)

print("Statistic:", friedman.statistic)
print("p-value :", friedman.pvalue)


# WILCOXON TEST
print("\n===== WILCOXON TEST =====")

pairs = [
    ("LSTM", lstm_all),
    ("ATTN_LSTM", attn_all),
    ("HYBRID", hybrid_all)
]

for i in range(len(pairs)):
    for j in range(i+1, len(pairs)):

        name1, p1 = pairs[i]
        name2, p2 = pairs[j]

        stat, p = wilcoxon(
            np.abs(y_all - p1),
            np.abs(y_all - p2)
        )

        print(f"{name1} vs {name2}: p-value = {p}")


# DIEBOLD-MARIANO TEST
print("\n===== DIEBOLD-MARIANO TEST =====")

def dm_test(e1, e2):
    d = e1**2 - e2**2
    mean_d = np.mean(d)
    var_d  = np.var(d, ddof=1)
    return mean_d / np.sqrt(var_d / len(d))

for i in range(len(pairs)):
    for j in range(i+1, len(pairs)):

        name1, p1 = pairs[i]
        name2, p2 = pairs[j]

        e1 = y_all - p1
        e2 = y_all - p2

        dm = dm_test(e1, e2)

        print(f"{name1} vs {name2}: DM = {dm}")


# DURBIN-WATSON TEST
print("\n===== DURBIN-WATSON =====")

for name, pred in pairs:

    resid = y_all - pred
    dw = durbin_watson(resid)

    print(name, "DW:", dw)


# MODEL STABILITY
print("\n===== MODEL STABILITY =====")

rmse_values = {}

for name, pred in pairs:

    resid = y_all - pred
    rmse_series = np.sqrt(resid**2)   # absolute error

    rmse_values[name] = rmse_series

rmse_df_stability = pd.DataFrame(rmse_values)

print("\nMean Error")
print(rmse_df_stability.mean())

print("\nStd Error")
print(rmse_df_stability.std())