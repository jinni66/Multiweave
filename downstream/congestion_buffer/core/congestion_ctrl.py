class RateController:
    def __init__(self):
        self.rate = 1000

    def update(self, qlen):
        if qlen > 200:
            self.rate *= 0.8
        else:
            self.rate *= 1.05

        self.rate = max(10, min(self.rate, 10000))
        return self.rate