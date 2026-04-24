import pandas as pd
import numpy as np
import os
from typing import List
from collections import defaultdict

def aggregate_csv_by_time_interval(
        input_csvs: List[str],
        output_dir: str,
        time_steps: List[int] = [100000],
        columns: List[str] = ["pkt", "byt"],
        chunksize: int = 200_000
):

    os.makedirs(output_dir, exist_ok=True)

    for csv_path in input_csvs:
        file_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\nProcessing: {csv_path}")

        print("Scanning min/max ts ...")
        min_ts = None
        max_ts = None

        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            chunk_ts = chunk["ts"].astype(float)
            cmin = chunk_ts.min()
            cmax = chunk_ts.max()
            min_ts = cmin if min_ts is None else min(min_ts, cmin)
            max_ts = cmax if max_ts is None else max(max_ts, cmax)

        print(f"ts range: {min_ts} ~ {max_ts}")

        for step in time_steps:
            step = int(step)
            print(f"\nAggregating step={step} ...")

            ts_bins = np.arange(min_ts, max_ts + step, step)

            bucket_sum = defaultdict(lambda: {col: 0 for col in columns})

            for chunk in pd.read_csv(csv_path, chunksize=chunksize):
                chunk["ts"] = chunk["ts"].astype(float)

                bin_idx = ((chunk["ts"] - min_ts) // step).astype(int)

                chunk["ts_bin"] = min_ts + bin_idx * step

                for col in columns:
                    sums = chunk.groupby("ts_bin")[col].sum()
                    for b, v in sums.items():
                        bucket_sum[b][col] += int(v)

            left_edges = ts_bins[:-1]

            result = pd.DataFrame({
                "ts": left_edges,
                "td": step
            })

            result["pkt"] = [bucket_sum[t]["pkt"] if t in bucket_sum else 0 for t in left_edges]
            result["byt"] = [bucket_sum[t]["byt"] if t in bucket_sum else 0 for t in left_edges]

            output_path = os.path.join(output_dir, f"{file_name}_agg_{step}us.csv")
            result.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")

if __name__ == "__main__":
    input_files = [
        "raw.csv"
    ]
    output_directory = "agg/"
    time_steps_us = [100, 1000, 10000]
    columns_to_aggregate = ["pkt", "byt"]

    aggregate_csv_by_time_interval(
        input_csvs=input_files,
        output_dir=output_directory,
        time_steps=time_steps_us,
        columns=columns_to_aggregate
    )
