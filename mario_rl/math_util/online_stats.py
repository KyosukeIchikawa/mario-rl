import math


class OnlineStats:
    """Online statistics class that computes mean and standard deviation."""
    def __init__(self):
        self._sum = 0
        self._sum_sq = 0
        self._count = 0

    def add(self, value):
        self._sum += value
        self._sum_sq += value * value
        self._count += 1

    def mean(self):
        return self._sum / self._count if self._count > 0 else 0

    def std(self):
        return math.sqrt(self._sum_sq / self._count - self.mean() * self.mean())

    def sum(self):
        return self._sum

    def count(self):
        return self._count

    def reset(self):
        self._sum = 0
        self._sum_sq = 0
        self._count = 0
