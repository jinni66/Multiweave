import pandas as pd
import numpy as np

def build_trace(csv_path, bin_us=1000, use_col="pkt"):
    df = pd.read_csv(csv_path)

    ts = df["ts"].values
    values = df[use_col].values

    ts = ts - ts.min()
    bins = (ts // bin_us).astype(int)

    series = np.zeros(bins.max() + 1)

    for b, v in zip(bins, values):
        series[b] += v

    return series


def normalize(x, target_mean=50):
    return x / (x.mean() + 1e-8) * target_mean


# ===== load =====
real = build_trace("raw.csv")
gen  = build_trace("syn.csv")

min_len = min(len(real), len(gen))
real = real[:min_len]
gen  = gen[:min_len]

# ===== normalize =====
real = normalize(real)
gen  = normalize(gen)

np.save("traces/real.npy", real)
np.save("traces/gen.npy", gen)