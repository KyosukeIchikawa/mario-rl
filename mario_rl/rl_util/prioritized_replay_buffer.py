import pickle
import random
import zlib
from collections import deque
from typing import List, Tuple, Sequence, Union

from .experience import Experience


class _SumTree:
    """A binary sum tree data structure."""
    def __init__(self, capacity: int):
        # check if capacity is a power of 2
        # NOTE: For example, 5 & 4 = 4, because 0101 & 0100 = 0100.
        #       Also, 4 & 3 = 0, because 0100 & 0011 = 0000.
        #       Also, 8 & 7 = 0, because 1000 & 0111 = 0000.
        #       So if capacity is a power of 2, capacity is 0010, 0100, 1000, etc. and
        #       (capacity - 1) is 0001, 0011, 0111, etc. and capacity & (capacity - 1) is 0000.
        assert capacity & (capacity - 1) == 0
        self._capacity = capacity
        # This tree has sum of the values of its children at each node.
        # tree[0]: sum of all values
        # tree[1]: sum of left values of tree[0]
        # tree[2]: sum of right values of tree[0]
        # tree[3]: sum of left values of tree[1]
        # ...
        # tree[capacity - 1]: biggest value
        # tree[capacity]: 2nd biggest value
        # ...
        # tree[2 * capacity - 2]: smallest value
        self._tree = [0.0] * (2 * capacity - 1)

    def sum(self) -> float:
        return self._tree[0]

    def weighted_sample_index(self) -> int:
        """Sample the leaf node with weight defined by each node's value.
        This method returns the index of the sampled leaf node.

        :return: the index of the sampled leaf node
        """
        value = random.uniform(0.0, self.sum())
        # loop until we reach a leaf node
        index = 0
        while index < self._capacity - 1:
            left = 2 * index + 1
            left_value = self._tree[left]
            if value <= left_value:
                # go to the left
                index = left
            else:
                # go to the right
                index = left + 1
                value -= left_value
        return index - (self._capacity - 1)

    def __setitem__(self, index, value):
        """Update the value of the leaf node at the given index."""
        assert 0 <= index < self._capacity
        # update the leaf node
        index += self._capacity - 1
        self._tree[index] = value
        # propagate the change through the tree
        while index > 0:
            # go to the parent
            index = (index - 1) // 2
            # update the parent
            left = 2 * index + 1
            right = left + 1
            self._tree[index] = self._tree[left] + self._tree[right]

    def __getitem__(self, index):
        """Return the value of the leaf node at the given index."""
        return self._tree[index + self._capacity - 1]


class PrioritizedReplayBuffer:
    """A prioritized experience replay buffer for agents."""
    def __init__(self, size: int, compress=True):
        self._compress = compress
        self._buffer_index = 0
        self._buffer = deque(maxlen=size)
        self._priorities = _SumTree(capacity=size)
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
