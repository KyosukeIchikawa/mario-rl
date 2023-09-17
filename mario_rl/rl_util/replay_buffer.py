import pickle
import random
import zlib
from collections import deque
from typing import List

from .experience import Experience


class ReplayBuffer:
    """A simple FIFO experience replay buffer for agents."""
    def __init__(self, size: int, compress=True):
        self._compress = compress
        self._buffer = deque(maxlen=size)

    def __len__(self):
        return len(self._buffer)

    def append(self, exp: Experience):
        """Append an experience to the buffer.

        :param exp: an experience to append
        """
        if self._compress:
            exp = zlib.compress(pickle.dumps(exp))
        self._buffer.append(exp)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences from the buffer.

        :param batch_size: the number of experiences to sample
        :return: a batch of experiences
        """
        experiences = random.sample(self._buffer, batch_size)
        if self._compress:
            experiences = [pickle.loads(zlib.decompress(exp)) for exp in experiences]
        return experiences
