import pickle
import random
import zlib
from collections import deque
from typing import List, Tuple, Sequence

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

    def weighted_sample_indices(self, n: int) -> List[int]:
        """Sample the leaf nodes with weight defined by each node's value.
        This method returns the indices of the sampled leaf nodes.

        :param n: the number of leaf nodes to sample
        :return: the indices of the sampled leaf nodes
        """
        return [self.weighted_sample_index() for _ in range(n)]

    def _get_sorted_random_values(self, n: int) -> List[float]:
        """Return n random values sorted in ascending order."""
        if n == 1:
            return [random.uniform(0, self.sum())]

        max_ = self.sum()
        delta = max_ / n
        return [random.uniform(delta * i, delta * (i + 1)) for i in range(n)]

    def weighted_sample_indices_for_spiky_values(self, n: int) -> List[int]:
        """Sample the leaf nodes with weight defined by each node's value.
        This method returns the indices of the sampled leaf nodes.

        NOTE: This method is deprecated because it is generally slower than weighted_sample_indices().
              However, this method is faster when the shape of priority distribution is very spiky.

        :param n: the number of leaf nodes to sample
        :return: the indices of the sampled leaf nodes
        """
        sorted_values = self._get_sorted_random_values(n)
        indices = []
        index = 0
        floor_value = 0.0
        ceil_value = self._tree[0]
        floor_and_ceil_values = deque()
        for value in sorted_values:
            # go to the parent node until the value is in the range
            while value > ceil_value:
                index = (index - 1) // 2
                floor_value, ceil_value = floor_and_ceil_values.pop()
            value -= floor_value

            # go to the leaf node
            while index < self._capacity - 1:
                floor_and_ceil_values.append((floor_value, ceil_value))

                left = 2 * index + 1
                left_value = self._tree[left]
                right_value = self._tree[left + 1]
                if value <= left_value:
                    # go to the left
                    index = left
                    ceil_value -= right_value
                else:
                    # go to the right
                    index = left + 1
                    value -= left_value
                    floor_value += left_value

            index_ = index - (self._capacity - 1)
            indices.append(index_)
        return indices

    def __setitem__(self, index, value):
        """Update the value of the leaf node at the given index."""
        assert 0 <= index < self._capacity
        assert value >= 0.0
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
