import numpy as np

np.random.seed(0)

T = 2000

# =========================
# Real traffic (self-similar bursty)
# =========================
real = np.random.pareto(1.5, T)
real = real / real.mean() * 50  # normalize

# =========================
# Generated traffic (too smooth)
# =========================
gen = np.random.normal(50, 5, T)
gen = np.clip(gen, 0, None)

np.save("traces/real.npy", real)
np.save("traces/gen.npy", gen)

print("Traces generated.")