from unittest.mock import patch

import pytest

from mario_rl.rl_util.prioritized_replay_buffer import _SumTree


@pytest.mark.parametrize("items, expected", [
    [[], [0, 0, 0, 0, 0, 0, 0]],
    [[4, 1, 100], [105, 5, 100, 4, 1, 100, 0]],
    [[4, 1, 100, 1], [106, 5, 101, 4, 1, 100, 1]],
])
def test_sum_tree_setitem(items, expected):
    sum_tree = _SumTree(capacity=4)
    for i, item in enumerate(items):
        sum_tree[i] = item
    assert sum_tree._tree == expected


@pytest.mark.parametrize("random_value, expected_index", [
    [0, 0], [4, 0], [4.1, 1], [5, 1], [5.1, 2], [105, 2], [106, 3],
])
def test_sum_tree_weighted_sample_index(random_value, expected_index):
    sum_tree = _SumTree(capacity=4)
    sum_tree[0] = 4
    sum_tree[1] = 1
    sum_tree[2] = 100
    sum_tree[3] = 1

    with patch('random.uniform', return_value=random_value):
        index = sum_tree.weighted_sample_index()
    assert index == expected_index


@pytest.mark.parametrize("random_values, expected_indices", [
    [[0], [0]],
    [[0] * 10, [0] * 10],
    [[0, 4, 4.1, 5, 5, 5.1, 105, 106], [0, 0, 1, 1, 1, 2, 2, 3]],
])
def test_sum_tree_weighted_sample_indices_for_spiky_values(random_values, expected_indices):
    sum_tree = _SumTree(capacity=4)
    sum_tree[0] = 4
    sum_tree[1] = 1
    sum_tree[2] = 100
    sum_tree[3] = 1

    sum_tree._get_sorted_random_values = lambda _: random_values
    index = sum_tree.weighted_sample_indices_for_spiky_values(len(random_values))
    assert index == expected_indices
