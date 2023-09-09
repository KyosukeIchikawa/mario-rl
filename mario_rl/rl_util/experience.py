from typing import NamedTuple

import numpy as np


class Experience(NamedTuple):
    state: np.ndarray
    next_state: np.ndarray
    action: int
    reward: float
    done: bool
