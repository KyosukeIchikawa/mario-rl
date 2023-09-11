from typing import Any, NamedTuple


class Experience(NamedTuple):
    state: Any
    next_state: Any
    action: int
    reward: float
    done: bool
