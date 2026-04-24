import numpy as np
import json
from core.queue_sim import QueueSimulator
from core.buffer_provisioning import BufferProvisioningEvaluator

real = np.load("traces/real.npy")
gen  = np.load("traces/gen.npy")

buffers = [8, 16, 32, 64, 128, 256, 512]

service_rate = int(max(1, np.mean(real) * 100))

evaluator = BufferProvisioningEvaluator(
    QueueSimulator,
    service_rate=service_rate
)

result = evaluator.evaluate(real, gen, buffers)

with open("results/buffer.json", "w") as f:
    json.dump(result, f, indent=2)

print("B_real:", result["B_real"])
print("B_gen :", result["B_gen"])
print("distance:", result["curve_distance"])