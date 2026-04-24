import json
import matplotlib.pyplot as plt

def plot():

    data = json.load(open("results/buffer.json"))

    real_curve = data["real_curve"]
    gen_curve  = data["gen_curve"]

    buffers = sorted([int(k) for k in real_curve.keys()])

    real = [real_curve[str(b)] for b in buffers]
    gen  = [gen_curve[str(b)] for b in buffers]

    plt.plot(buffers, real, label="Real")
    plt.plot(buffers, gen, label="Gen")

    plt.xlabel("Buffer Size")
    plt.ylabel("Loss Rate")
    plt.legend()

    plt.savefig("results/buffer_curve.png")