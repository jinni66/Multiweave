import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def compute_emd(real_csv, synth_csv, cols=("pkt", "byt")):

    real = pd.read_csv(real_csv, usecols=cols).values
    synth = pd.read_csv(synth_csv, usecols=cols).values

    T = min(len(real), len(synth))
    if len(real) != len(synth):
        print(f"[Align] real={len(real)}, synth={len(synth)} -> {T}")

    real = real[:T]
    synth = synth[:T]

    results = {}

    for i, col in enumerate(cols):
        emd = wasserstein_distance(real[:, i], synth[:, i])

        combined = np.concatenate([real[:, i], synth[:, i]])
        rng = combined.max() - combined.min()
        rng = rng if rng > 0 else 1e-9

        emd_norm = emd / rng

        results[col] = {
            "emd": emd,
            "emd_norm": emd_norm
        }

        print(f"{col}: EMD = {emd:.6f}, Norm = {emd_norm:.6f}")

    return results


if __name__ == "__main__":

    real_csv = "raw.csv"
    synth_csv = "syn.csv"

    print("=== EMD Evaluation ===")

    results = compute_emd(real_csv, synth_csv)

    print("\n=== Summary ===")
    for k, v in results.items():
        print(f"{k}: {v}")