import random
from collections import deque
from typing import List

from .sum_tree import SumTree


class SumTreeForSpiky(SumTree):
    """A binary sum tree data structure.

    NOTE: This class is deprecated because it is generally slower than SumTree.
            However, this class is faster when the shape of priority distribution is very spiky.
    """
    def _get_sorted_random_values(self, n: int) -> List[float]:
        """Return n random values sorted in ascending order."""
        if n == 1:
            return [random.uniform(0, self.sum())]

        max_ = self.sum()
        delta = max_ / n
        return [random.uniform(delta * i, delta * (i + 1)) for i in range(n)]

    def weighted_sample_indices(self, n: int) -> List[int]:
        """Sample the leaf nodes with weight defined by each node's value.
        This method returns the indices of the sampled leaf nodes.

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
