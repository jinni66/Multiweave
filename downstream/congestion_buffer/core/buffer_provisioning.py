import numpy as np

class BufferProvisioningEvaluator:

    def __init__(self, simulator_class, service_rate=500):
        self.simulator_class = simulator_class
        self.service_rate = service_rate

    def run_curve(self, trace, buffers):
        results = {}

        for B in buffers:
            sim = self.simulator_class(capacity=B, service_rate=self.service_rate)

            for t in range(len(trace)):
                sim.step(int(trace[t]), t)

            results[B] = sim.stats()["loss_rate"]

        return results

    def optimal_buffer(self, loss_curve):
        for B, loss in loss_curve.items():
            if loss < 0.01:
                return B
        return max(loss_curve.keys())

    def curve_distance(self, real_curve, gen_curve):

        buffers = sorted(real_curve.keys())

        real = np.array([real_curve[b] for b in buffers])
        gen  = np.array([gen_curve[b] for b in buffers])

        return np.sum(np.abs(real - gen))

    def evaluate(self, real_trace, gen_trace, buffers):

        real_curve = self.run_curve(real_trace, buffers)
        gen_curve  = self.run_curve(gen_trace, buffers)

        B_real = self.optimal_buffer(real_curve)
        B_gen  = self.optimal_buffer(gen_curve)

        rel_error = abs(B_real - B_gen) / (B_real + 1e-8)

        dist = self.curve_distance(real_curve, gen_curve)

        return {
            "real_curve": real_curve,
            "gen_curve": gen_curve,
            "B_real": B_real,
            "B_gen": B_gen,
            "relative_error": rel_error,
            "curve_distance": dist
        }