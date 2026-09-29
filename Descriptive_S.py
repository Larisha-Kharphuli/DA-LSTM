import pandas as pd

DATA_PATH = r"D:\Research\Objective 1\Myhybridmodel1\Price_index_data\Correction\Agricultural_data.xlsx"
df = pd.read_excel(DATA_PATH)
df = df.sort_values(["Year", "Month"]).reset_index(drop=True)

cols = ["Onion", "Cumin", "Garlic", "Castor_Seed", "Tomato", "Potato"]

# Split ratios — must match the rest of the pipeline
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

rows = []
for c in cols:
    series = df[c].values.astype(float)
    n = len(series)

    train_end = int(TRAIN_RATIO * n)
    val_end   = int((TRAIN_RATIO + VAL_RATIO) * n)

    train = series[:train_end]
    val   = series[train_end:val_end]
    test  = series[val_end:]

    rows.append({
        "Commodity":       c.replace("_", " "),
        "N":               n,
        "Full-sample mean": round(series.mean(), 2),
        "Full-sample S.D":  round(series.std(ddof=1), 2),
        "Train mean":       round(train.mean(), 2),
        "Validation mean":  round(val.mean(), 2),
        "Test mean":        round(test.mean(), 2),
    })

stats = pd.DataFrame(rows)
print(stats.to_string(index=False))

# Optional: save to CSV for the manuscript
stats.to_csv("results/descriptive_stats.csv", index=False)