import torch
import pandas as pd
import numpy as np
import os
import time
import argparse
from datetime import datetime

from models.multiscale_ed3k import MultiScaleED3

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)

parser.add_argument("--exp_id", type=str, default="exp")
parser.add_argument("--seed_len", type=int, default=200)
parser.add_argument("--seq", type=int, default=100)
parser.add_argument("--gpu", type=int, default=0)

args = parser.parse_args()

device = torch.device(
    f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_id = f"{args.exp_id}_{timestamp}"

output_dir = os.path.join("post_dataset", run_id)
log_dir = "logs"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

log_path = os.path.join(log_dir, "infer_log.txt")

ckpt = torch.load(args.model_path, map_location=device, weights_only=False)

model = MultiScaleED3().to(device)
model.load_state_dict(ckpt["model"])
model.eval()

mean = torch.tensor(ckpt["mean"], device=device)
std = torch.tensor(ckpt["std"], device=device)

data = pd.read_csv(args.data_path)[["pkt", "byt"]].to_numpy(np.float32)

seed = torch.tensor(data[:args.seed_len], device=device).unsqueeze(0)
seed = (seed - mean) / std

generated = seed.clone()


start_time = time.time()

with torch.no_grad():
    while generated.size(1) < len(data):

        inp = generated[:, -args.seq:, :]

        y1, yk, yk2 = model(inp)

        # ===== safe residual smoothing =====
        y1 = 0.9 * y1 + 0.1 * inp[:, -y1.size(1):, :]

        generated = torch.cat([generated, y1[:, -args.seq:, :]], dim=1)

y = generated[:, :len(data), :]

y = y * std + mean
y = torch.clamp(y, min=0)


output_file = os.path.join(output_dir, "output_fine.csv")

pd.DataFrame(
    y.squeeze(0).cpu().numpy(),
    columns=["pkt", "byt"]
).to_csv(output_file, index=False)


infer_time = time.time() - start_time


log_line = (
    f"[{timestamp}] "
    f"EXP={args.exp_id} | "
    f"Time={infer_time:.4f}s | "
    f"Seed={args.seed_len} | "
    f"Seq={args.seq} | "
    f"Model={args.model_path}\n"
)

print(log_line)

with open(log_path, "a") as f:
    f.write(log_line)


print("\n====================")
print("Generation Done")
print(f"Output: {output_file}")
print(f"Time: {infer_time:.4f} sec")
print(f"Log: {log_path}")
print("====================\n")
