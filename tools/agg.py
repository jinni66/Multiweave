import pandas as pd
import numpy as np
import os

def aggregate_one_scale(df, k):
    df = df.copy()
    df["group"] = np.arange(len(df)) // k

    base_td = df["td"].iloc[0]

    out = df.groupby("group", sort=False).apply(lambda x: pd.Series({
        "ts": x["ts"].iloc[0],
        "td": base_td * k,
        "pkt": x["pkt"].sum(),
        "byt": x["byt"].sum()
    })).reset_index(drop=True)

    step = base_td * k
    out["ts"] = out["ts"].iloc[0] + np.arange(len(out)) * step

    return out


def generate_multiscale_csv(input_path, output_dir, ks=(1,2,4,8)):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)

    results = {}

    for k in ks:
        out = aggregate_one_scale(df, k)

        save_path = os.path.join(output_dir, f"cidds_k{k}.csv")
        out.to_csv(save_path, index=False)

        results[k] = out
        print(f"[OK] k={k} saved -> {save_path}")

    return results


input_file = "agg.csv"
output_dir = "multiscale_output/"

results = generate_multiscale_csv(
    input_file,
    output_dir,
    ks=(1, 10, 100, 1000)
)