import numpy as np

def load_trace(path):
    return np.load(path)

def pkt_to_rate(pkt_series, bin_ms=10, pkt_size=1500):
    return (pkt_series * pkt_size * 8) / (bin_ms / 1000) / 1e6