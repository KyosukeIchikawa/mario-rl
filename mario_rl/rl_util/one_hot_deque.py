from collections import deque

import numpy as np


class OneHotDeque(deque):
    def __init__(self, dim: int, maxlen: int):
        self._dim = dim
        zeros = np.zeros(shape=(dim,), dtype=np.float32)
        super().__init__([zeros] * maxlen, maxlen=maxlen)

    def as_ndarray(self) -> np.ndarray:
        return np.array(self)

    def append_one_hot(self, id_: int):
        one_hot_actions = np.zeros(shape=(self._dim,), dtype=np.float32)
        one_hot_actions[id_] = 1.0
        self.append(one_hot_actions)
