from collections import deque

class QueueSimulator:
    def __init__(self, capacity=1000, service_rate=2000):
        self.q = deque()
        self.capacity = capacity
        self.service_rate = service_rate
        self.last = 0

        self.loss = 0
        self.total = 0
        self.delays = []

    def step(self, arrivals, t):

        for _ in range(arrivals):
            self.total += 1
            if len(self.q) < self.capacity:
                self.q.append(t)
            else:
                self.loss += 1

        if self.last == 0:
            self.last = t

        dt = t - self.last
        self.last = t

        serve = int(dt * self.service_rate)

        for _ in range(min(serve, len(self.q))):
            t_in = self.q.popleft()
            self.delays.append(t - t_in)

    def stats(self):
        return {
            "loss_rate": self.loss / max(1, self.total),
            "avg_delay": sum(self.delays)/len(self.delays) if self.delays else 0,
            "queue_len": len(self.q)
        }