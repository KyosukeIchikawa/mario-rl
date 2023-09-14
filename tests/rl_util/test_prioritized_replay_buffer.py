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
