import random
from collections import deque

from .experience import Experience


class ReplayBuffer:
    """A simple FIFO experience replay buffer for agents."""
    def __init__(self, size: int):
        self._buffer = deque(maxlen=size)

    def __len__(self):
        return len(self._buffer)

    def append(self, exp: Experience):
        self._buffer.append(exp)

    def sample(self, batch_size: int) -> Experience:
        return random.sample(self._buffer, batch_size)
