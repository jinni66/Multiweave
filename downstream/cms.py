import numpy as np
import pandas as pd
import hashlib

WIDTH = 2000
DEPTH = 5
HH_RATIO = 0.1
REPEAT = 10

class CountMinSketch:
    def __init__(self, width=2000, depth=5, seed=0):
        self.width = width
        self.depth = depth
        self.seed = seed
        self.table = np.zeros((depth, width))

    def _hash(self, key, i):
        s = str(key) + str(i) + str(self.seed)
        return int(hashlib.md5(s.encode()).hexdigest(), 16) % self.width

    def add(self, key, value):
        for i in range(self.depth):
            idx = self._hash(key, i)
            self.table[i, idx] += value

    def query(self, key):
        return min(self.table[i, self._hash(key, i)] for i in range(self.depth))


def compute_sketch_error(df, col='pkt', seed=0):
    cms = CountMinSketch(WIDTH, DEPTH, seed=seed)

    values = df[col].values
    n = len(values)

    for i, v in enumerate(values):
        cms.add(i, v)

    k = max(1, int(n * HH_RATIO))
    top_idx = np.argsort(values)[-k:]

    error = 0.0
    for idx in top_idx:
        est = cms.query(idx)
        true = values[idx]
        error += abs(est - true)

    return error

def avg_error(df, col):
    errors = []
    for seed in range(REPEAT):
        err = compute_sketch_error(df, col, seed)
        errors.append(err)
    return np.mean(errors)

def evaluate(real_df, syn_df, name="Model"):
    results = {}

    for col in ['pkt', 'byt']:
        error_real = avg_error(real_df, col)
        error_syn = avg_error(syn_df, col)

        rel_error = abs(error_syn - error_real) / (error_real + 1e-6)

        results[col] = {
            'real': error_real,
            'syn': error_syn,
            'rel': rel_error
        }

        print(f"\n===== {name} | {col.upper()} =====")
        print(f"E_real: {error_real:.4f}")
        print(f"E_syn : {error_syn:.4f}")
        print(f"Relative Error: {rel_error:.6f}")

    return results


real_df = pd.read_csv("raw.csv")
syn_df = pd.read_csv("syn.csv")

print("\n================ REAL vs SYN ================")
res = evaluate(real_df, syn_df, "Baseline")

summary = []

for method, res in [('Baseline', res)]:
    summary.append({
        'Method': method,
        'pkt_rel_error': res['pkt']['rel'],
        'byt_rel_error': res['byt']['rel']
    })

summary_df = pd.DataFrame(summary)

print("\n=========== FINAL RESULT TABLE ===========")
print(summary_df)
