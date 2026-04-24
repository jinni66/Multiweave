import json
import numpy as np
import matplotlib.pyplot as plt


# ==============================
# Load results (SAFE VERSION)
# ==============================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


cong = load_json("results/congestion.json")
buff = load_json("results/buffer.json")

real = cong["real"]
gen = cong["gen"]


# ==============================
# helper
# ==============================
def safe_mean(x):
    return np.mean(x) if len(x) > 0 else 0.0

def safe_std(x):
    return np.std(x) if len(x) > 0 else 1e-8


# ==============================
# 1. Congestion Control Metrics (ROBUST)
# ==============================
def congestion_metrics(real, gen):

    eps = 1e-8
    metrics = {}

    # throughput
    metrics["throughput_rel_err"] = (
        safe_mean(real["rate"]) - safe_mean(gen["rate"])
    ) / (safe_mean(real["rate"]) + eps)


    # loss
    metrics["loss_rel_err"] = real["loss_rate"] - gen["loss_rate"]

    return metrics


# ==============================
# 2. Buffer Metrics (PAPER-GRADE)
# ==============================
def buffer_metrics(buff):

    # -------- compatibility fix
    real_curve = buff.get("real_curve", buff.get("real", None))
    gen_curve  = buff.get("gen_curve", buff.get("gen", None))

    if real_curve is None or gen_curve is None:
        raise ValueError("Buffer JSON format error!")

    buffers = sorted(map(int, real_curve.keys()))

    real_loss = np.array([real_curve[str(b)] for b in buffers])
    gen_loss  = np.array([gen_curve[str(b)] for b in buffers])

    # -------- optimal buffer
    def find_opt(loss_curve):
        for b, l in zip(buffers, loss_curve):
            if l < 0.01:
                return b
        return buffers[-1]

    B_real = find_opt(real_loss)
    B_gen  = find_opt(gen_loss)

    rel_err = abs(B_real - B_gen) / (B_real + 1e-8)

    l2 = np.sqrt(np.mean((real_loss - gen_loss) ** 2))

    corr = np.corrcoef(real_loss, gen_loss)[0, 1]
    corr = 0.0 if np.isnan(corr) else corr

    return {
        "B_real": B_real,
        "B_gen": B_gen,
        "buffers": buffers,
        "real_loss": real_loss.tolist(),
        "gen_loss": gen_loss.tolist()
    }


# ==============================
# 3. Unified Score (STABLE + INTERPRETABLE)
# ==============================
def control_fidelity_score(cong_m, buff_m):

    score = 0.0

    # congestion (all normalized errors)
    for k, v in cong_m.items():
        score += abs(v)

    # buffer (decision + curve)
    score += 2.0 * buff_m["relative_error"]
    score += 3.0 * buff_m["l2_distance"]
    score += 1.0 * (1 - buff_m["correlation"])

    return np.log1p(score)


# ==============================
# 4. Plotting
# ==============================
def plot_buffer(buff_m):

    plt.figure(figsize=(6,4))

    plt.plot(buff_m["buffers"], buff_m["real_loss"], marker="o", label="Real")
    plt.plot(buff_m["buffers"], buff_m["gen_loss"], marker="s", label="Generated")

    plt.xlabel("Buffer Size")
    plt.ylabel("Loss Rate")
    plt.title("Buffer Provisioning Fidelity")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/buffer_analysis.png", dpi=300)
    plt.close()


def plot_congestion(cong_m):

    plt.figure(figsize=(6,4))

    labels = list(cong_m.keys())
    values = list(cong_m.values())

    plt.bar(labels, values)

    plt.ylabel("Relative Error")
    plt.title("Congestion Control Fidelity")

    plt.xticks(rotation=20)

    plt.tight_layout()
    plt.savefig("results/congestion_analysis.png", dpi=300)
    plt.close()


# ==============================
# Main
# ==============================
def main():

    cong_m = congestion_metrics(real, gen)
    buff_m = buffer_metrics(buff)

    score = control_fidelity_score(cong_m, buff_m)

    print("\n==================== RESULTS ====================")

    print("\n[Congestion Control]")
    for k, v in cong_m.items():
        print(f"{k}: {v:.4f}")

    print("\n[Buffer Provisioning]")
    print(f"B_real: {buff_m['B_real']}")
    print(f"B_gen : {buff_m['B_gen']}")
    print(f"Relative Error: {buff_m['relative_error']:.4f}")

    print("\n[Unified Control Fidelity Score]")
    print(score)

    print("\n=================================================\n")

    plot_buffer(buff_m)
    plot_congestion(cong_m)


if __name__ == "__main__":
    main()