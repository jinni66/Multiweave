import numpy as np

def check(trace, service_rate, name="trace"):
    util = np.mean(trace) / service_rate

    print(f"[{name}] utilization = {util:.4f}")

    if util < 0.3:
        print("UNDERLOAD → results not valid")
    elif util > 1.5:
        print("OVERLOAD → unstable regime")
    else:
        print("VALID REGIME")