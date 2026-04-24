import pandas as pd
import numpy as np
import os
from typing import List
from collections import defaultdict

def aggregate_csv_by_time_interval(
        input_csvs: List[str],
        output_dir: str,
        time_steps: List[int] = [100000],
        columns: List[str] = ["pkt_len"],
        chunksize: int = 200_000 
):

    os.makedirs(output_dir, exist_ok=True)

    for csv_path in input_csvs:
        file_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\nProcessing: {csv_path}")

        print("Scanning min/max time ...")
        min_time = None
        max_time = None

        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            chunk_time = chunk["time"].astype(float)
            cmin = chunk_time.min()
            cmax = chunk_time.max()
            min_time = cmin if min_time is None else min(min_time, cmin)
            max_time = cmax if max_time is None else max(max_time, cmax)

        print(f"time range: {min_time} ~ {max_time}")

        for step in time_steps:
            step = int(step)
            print(f"\nAggregating step={step} ...")

            time_bins = np.arange(min_time, max_time + step, step)

            bucket_sum = defaultdict(lambda: {col: 0 for col in columns} | {"packet_count": 0})

            for chunk in pd.read_csv(csv_path, chunksize=chunksize):
                chunk["time"] = chunk["time"].astype(float)

                bin_idx = ((chunk["time"] - min_time) // step).astype(int)
                chunk["time_bin"] = min_time + bin_idx * step

                for col in columns:
                    sums = chunk.groupby("time_bin")[col].sum()
                    for b, v in sums.items():
                        bucket_sum[b][col] += int(v)

                counts = chunk.groupby("time_bin")["time"].count()
                for b, v in counts.items():
                    bucket_sum[b]["packet_count"] += int(v)

            left_edges = time_bins[:-1]

            result = pd.DataFrame({
                "time": left_edges,
                "td": step
            })

            result["byt"] = [bucket_sum[t]["pkt_len"] if t in bucket_sum else 0 for t in left_edges]
            result["pkt"] = [bucket_sum[t]["packet_count"] if t in bucket_sum else 0
                                      for t in left_edges]
            output_path = os.path.join(output_dir, f"{file_name}_agg_{step}us.csv")
            result.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    input_files = [
        "raw.csv"
    ]
    output_directory = "agg/"
    time_steps_us = [100, 1000, 10000]
    columns_to_aggregate = ["pkt_len"]

    aggregate_csv_by_time_interval(
        input_csvs=input_files,
        output_dir=output_directory,
        time_steps=time_steps_us,
        columns=columns_to_aggregate
    )