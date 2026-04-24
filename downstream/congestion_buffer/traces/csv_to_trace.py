import pandas as pd
import numpy as np


# =========================
# 1. Build trace (time series)
# =========================
def build_trace(csv_path,
                bin_us=1000,
                use_col="pkt",
                max_len=8000,
                scale_target_mean=1.0):

    df = pd.read_csv(csv_path)
    if "ts" not in df.columns:
        if "time" in df.columns:
            df = df.rename(columns={"time": "ts"})
        else:
            raise ValueError("CSV must contain 'ts' or 'time' column for real trace.")

    ts = df["ts"].values.astype(np.float64)
    values = df[use_col].values.astype(np.float64)

    ts = ts - ts.min()

    bins = (ts // bin_us).astype(int)

    max_bin = min(bins.max() + 1, max_len)

    series = np.zeros(max_bin)

    for b, v in zip(bins, values):
        if b < max_bin:
            series[b] += v

    if np.sum(series) == 0:
        raise ValueError("Empty trace detected. Check bin size or CSV format.")

    series = series / (series.mean() + 1e-8) * scale_target_mean

    return series

def load_generated(csv_path,
                   use_col="pkt",
                   scale_target_mean=1.0):
    """
    Generated trace may NOT have timestamps.
    """

    df = pd.read_csv(csv_path)

    if use_col not in df.columns:
        raise ValueError(f"Missing column: {use_col}")

    series = df[use_col].values.astype(np.float64)

    if len(series) == 0:
        raise ValueError("Empty generated trace.")

    series = series / (series.mean() + 1e-8) * scale_target_mean

    return series


def align(real, gen):
    min_len = min(len(real), len(gen))
    return real[:min_len], gen[:min_len]

if __name__ == "__main__":
    real_csv = "raw.csv"
    gen_csv  = "syn.csv"

    BIN_US = 1000
    USE_COL = "pkt"
    TARGET_MEAN = 50

    print("Building real trace...")
    real = build_trace(real_csv,
                       bin_us=BIN_US,
                       use_col=USE_COL,
                       scale_target_mean=TARGET_MEAN)

    print("Building generated trace...")
    gen = load_generated(gen_csv,
                         use_col=USE_COL,
                         scale_target_mean=TARGET_MEAN)

    real, gen = align(real, gen)

    print("\n=== TRACE CHECK ===")
    print("Real length:", len(real))
    print("Gen  length:", len(gen))
    print("Real mean/std:", np.mean(real), np.std(real))
    print("Gen  mean/std:", np.mean(gen), np.std(gen))

    util_est = np.mean(real) / 50.0
    print("\nEstimated utilization (real):", util_est)

    if util_est < 0.3:
        print("UNDERLOAD WARNING: increase TARGET_MEAN")
    elif util_est > 2.0:
        print("OVERLOAD WARNING: decrease TARGET_MEAN")
    else:
        print("✔ VALID CONGESTION REGIME")

    print("===================\n")
    np.save("traces/real.npy", real)
    np.save("traces/gen.npy", gen)

    print("Done. Trace length =", len(real))