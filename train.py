import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
import argparse
from geomloss import SamplesLoss

from models.multiscale_ed3k import MultiScaleED3

loss_fn = SamplesLoss(loss="sinkhorn", p=1, blur=1e-5)


def sinkhorn_normalized_emd(x, y, alpha_dim=None):
    B, T, D = x.shape

    if alpha_dim is None:
        alpha_dim = torch.ones(D, device=x.device)

    emd_total = 0.0

    for d in range(D):
        xd = x[:, :, d:d + 1].contiguous()
        yd = y[:, :, d:d + 1].contiguous()

        rng = torch.clamp(
            torch.max(torch.cat([xd, yd])) - torch.min(torch.cat([xd, yd])),
            min=1e-6
        )

        emd = loss_fn(xd, yd).mean()
        emd_total += emd / rng * alpha_dim[d]

    return emd_total / D


def windowed_emd(x, y, win=20, alpha_dim=None):
    B, T, D = x.shape
    total = 0.0
    cnt = 0

    for i in range(0, T - win + 1, win):
        total += sinkhorn_normalized_emd(
            x[:, i:i + win].contiguous(),
            y[:, i:i + win].contiguous(),
            alpha_dim
        )
        cnt += 1

    return total / max(cnt, 1)


def diff(x):
    return (x[:, 1:] - x[:, :-1]).contiguous()


def total_loss(y1, y1_true,
               yk, yk_true,
               yk2, yk2_true,
               factor,
               phase=1):

    alpha_dim = torch.tensor([1.0, 1.0], device=y1.device)

    emd_f = (
        0.5 * sinkhorn_normalized_emd(y1, y1_true, alpha_dim) +
        0.5 * windowed_emd(y1, y1_true, win=20, alpha_dim=alpha_dim)
    )

    emd_diff = sinkhorn_normalized_emd(diff(y1), diff(y1_true), alpha_dim)

    if phase == 1:
        loss = 300 * emd_f + 0.5 * emd_diff
        return loss, {
            "emd_f": emd_f.item(),
            "emd_diff": emd_diff.item()
        }

    emd_m = sinkhorn_normalized_emd(yk, yk_true, alpha_dim)
    emd_c = sinkhorn_normalized_emd(yk2, yk2_true, alpha_dim)

    f = factor

    y1_k = y1.view(y1.size(0), y1.size(1)//f, f, y1.size(2)).sum(dim=2)
    y1_k2 = y1_k.view(y1_k.size(0), y1_k.size(1)//f, f, y1.size(2)).sum(dim=2)

    cons = F.l1_loss(y1_k, yk_true) + F.l1_loss(y1_k2, yk2_true)

    loss = (
        20 * emd_f +
        0.05 * emd_m +
        0.05 * emd_c +
        0.2 * cons +
        0.2 * emd_diff
    )

    return loss, {
        "emd_f": emd_f.item(),
        "emd_m": emd_m.item(),
        "emd_c": emd_c.item(),
        "cons": cons.item()
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--x1", type=str, required=True)
    parser.add_argument("--xk", type=str, required=True)
    parser.add_argument("--xk2", type=str, required=True)

    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--seq", type=int, default=99)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    factor = args.factor

    model_path = f"models/cidds_best_k{factor}.pth"
    time_log_path = "logs/train_time.txt"
    gpu_log_path = "logs/train_gpu.txt"

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nTraining factor={factor}\n")


    train_start_time = time.time()

    torch.cuda.reset_peak_memory_stats()

    def load(csv):
        return pd.read_csv(csv)[["pkt", "byt"]].to_numpy(np.float32)

    x1 = load(args.x1)
    xk = load(args.xk)
    xk2 = load(args.xk2)

    def norm(x):
        m = x.mean(0, keepdims=True)
        s = x.std(0, keepdims=True) + 1e-6
        return (x - m) / s, m, s

    x1, m1, s1 = norm(x1)
    xk, _, _ = norm(xk)
    xk2, _, _ = norm(xk2)

    seq = args.seq

    assert seq % factor == 0
    assert (seq // factor) % factor == 0

    X1, Xk, Xk2 = [], [], []

    for i in range(0, len(x1) - seq + 1, seq):
        X1.append(x1[i:i + seq])
        Xk.append(xk[i // factor:(i + seq) // factor])
        Xk2.append(xk2[i // (factor * factor):(i + seq) // (factor * factor)])

    X1 = torch.tensor(np.stack(X1), device=device)
    Xk = torch.tensor(np.stack(Xk), device=device)
    Xk2 = torch.tensor(np.stack(Xk2), device=device)

    idx = np.arange(len(X1))
    np.random.shuffle(idx)

    split = int(0.8 * len(idx))

    train_idx, val_idx = idx[:split], idx[split:]

    X1_tr, X1_val = X1[train_idx], X1[val_idx]
    Xk_tr, Xk_val = Xk[train_idx], Xk[val_idx]
    Xk2_tr, Xk2_val = Xk2[train_idx], Xk2[val_idx]


    model = MultiScaleED3(factor=factor).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")

    for epoch in range(args.epochs):

        phase = 1 if epoch < int(0.85 * args.epochs) else 2

        model.train()

        y1, yk, yk2 = model(X1_tr)

        loss, _ = total_loss(
            y1, X1_tr,
            yk, Xk_tr,
            yk2, Xk2_tr,
            factor=factor,
            phase=phase
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()

        with torch.no_grad():
            y1v, ykv, yk2v = model(X1_val)

            val_loss, _ = total_loss(
                y1v, X1_val,
                ykv, Xk_val,
                yk2v, Xk2_val,
                factor=factor,
                phase=phase
            )

        print(f"Epoch {epoch:03d} | Train {loss.item():.4f} | Val {val_loss.item():.4f}")


        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "mean": m1,
                "std": s1
            }, model_path)

    total_time = time.time() - train_start_time

    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    else:
        max_mem = 0.0

    time_log = f"factor={factor} | time={total_time:.2f}s | best_val={best_val:.6f}\n"
    gpu_log = f"factor={factor} | peak_gpu={max_mem:.2f}MB\n"

    with open(time_log_path, "a") as f:
        f.write(time_log)

    with open(gpu_log_path, "a") as f:
        f.write(gpu_log)

    print("\n DONE")
    print(time_log)
    print(gpu_log)