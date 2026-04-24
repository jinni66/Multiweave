import numpy as np
from core.queue_sim import QueueSimulator
from core.congestion_ctrl import RateController
from core.sanity_check import check

def run(trace, service_rate):

    sim = QueueSimulator(capacity=1000, service_rate=service_rate)
    ctrl = RateController()

    rates, queues = [], []

    for t in range(len(trace)):

        qlen = len(sim.q)
        rate = ctrl.update(qlen)

        arrivals = int(trace[t] * (rate / 1000.0))

        sim.step(arrivals, t)

        rates.append(rate)
        queues.append(len(sim.q))

    return {
        "rate": rates,
        "queue": queues,
        **sim.stats()
    }


if __name__ == "__main__":

    real = np.load("traces/real.npy")
    gen  = np.load("traces/gen.npy")

    service_rate = int(np.mean(real) * 0.8)

    check(real, service_rate, "real")
    check(gen, service_rate, "gen")

    r1 = run(real, service_rate)
    r2 = run(gen, service_rate)

    import json
    with open("results/congestion.json", "w") as f:
        json.dump({"real": r1, "gen": r2}, f, indent=2)