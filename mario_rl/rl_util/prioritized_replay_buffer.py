import pickle
import random
import zlib
from collections import deque
from typing import List, Tuple, Sequence

from .experience import Experience
from .sum_tree import SumTree


class PrioritizedReplayBuffer:
    """A prioritized experience replay buffer for agents."""
    def __init__(self, size: int, compress=True):
        self._compress = compress
        self._buffer_index = 0
        self._buffer = deque(maxlen=size)
        self._priorities = SumTree(capacity=size)
        self._max_priority = 1.0

    def __len__(self):
        return len(self._buffer)

    def append(self, exp: Experience):
        """Append an experience to the buffer with the maximum priority.

        :param exp: an experience to append
        """
        if self._compress:
            exp = zlib.compress(pickle.dumps(exp))
        self._buffer.append(exp)
        # set the maximum priority to sample the new experience with the highest priority
        self._priorities[self._buffer_index] = self._max_priority
        self._buffer_index += 1
        if self._buffer_index >= self._buffer.maxlen:
            self._buffer_index = 0

    @staticmethod
    def _decompress(exp: bytes) -> Experience:
        return pickle.loads(zlib.decompress(exp))

    def sample(self, batch_size: int) -> Tuple[List[int], List[float], List[Experience]]:
        """Sample experiences from the buffer with priority weights.

        :param batch_size: the number of experiences to sample
        :return: a tuple of (indices, probabilities, experiences)
        """
        # TODO: Perhaps, we can search all the indices at once efficiently.
        indices = [self._priorities.weighted_sample_index() for _ in range(batch_size)]
        coe = 1.0 / self._priorities.sum()
        probabilities = [self._priorities[index] * coe for index in indices]
        decompress_if_need = self._decompress if self._compress else lambda _: _
        experiences = [decompress_if_need(self._buffer[index]) for index in indices]
        return indices, probabilities, experiences

    def update_priorities(self, indices: Sequence[int], priorities: Sequence[float]):
        """Update the priorities of the experiences at the given indices.

        :param indices: the indices of the experiences to update
        :param priorities: the new priorities of the experiences
        """
        for index, priority in zip(indices, priorities):
            self._priorities[index] = priority
            self._max_priority = max(self._max_priority, priority)


class PrioritizedReplayBufferWithoutSumTree:
    """A prioritized experience replay buffer for agents without using a sum tree.

    WARNING: If buffer size is large, this class is very slow. So, use PrioritizedReplayBuffer instead.
    """
    def __init__(self, size: int, compress=True):
        self._compress = compress
        self._buffer = deque(maxlen=size)
        self._priorities = deque(maxlen=size)
        self._sum_priorities = 0.0
        self._max_priority = 1.0

    def __len__(self):
        return len(self._buffer)

    def append(self, exp: Experience):
        """Append an experience to the buffer with the maximum priority.

        :param exp: an experience to append
        """
        if self._compress:
            exp = zlib.compress(pickle.dumps(exp))
        self._buffer.append(exp)
        # set the maximum priority to sample the new experience with the highest priority
        if len(self._priorities) == self._priorities.maxlen:
            self._sum_priorities -= self._priorities.popleft()
        self._sum_priorities += self._max_priority
        self._priorities.append(self._max_priority)

    @staticmethod
    def _decompress(exp: bytes) -> Experience:
        return pickle.loads(zlib.decompress(exp))

    def sample(self, batch_size: int) -> Tuple[List[int], List[float], List[Experience]]:
        """Sample experiences from the buffer with priority weights without duplicates.

        :param batch_size: the number of experiences to sample
        :return: a tuple of (indices, probabilities, experiences)
        """
        decompress_if_need = self._decompress if self._compress else lambda _: _
        len_buffer = len(self._buffer)
        indices = random.choices(range(len_buffer), weights=self._priorities, k=batch_size)
        probabilities = [self._priorities[index] / self._sum_priorities for index in indices]
        experiences = [decompress_if_need(self._buffer[index]) for index in indices]
        return indices, probabilities, experiences

    def update_priorities(self, indices: Sequence[int], priorities: Sequence[float]):
        """Update the priorities of the experiences at the given indices.

        :param indices: the indices of the experiences to update
        :param priorities: the new priorities of the experiences
        """
        for index, priority in zip(indices, priorities):
            self._priorities[index] = priority
            self._max_priority = max(self._max_priority, priority)
