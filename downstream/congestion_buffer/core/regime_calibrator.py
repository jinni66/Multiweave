import numpy as np

class RegimeCalibrator:
    def __init__(self, target_util=0.85):
        self.target_util = target_util

    def calibrate(self, trace, service_rate):

        mean_arrival = np.mean(trace)

        current_util = mean_arrival / (service_rate + 1e-8)

        scale = self.target_util / (current_util + 1e-8)

        return trace * scale