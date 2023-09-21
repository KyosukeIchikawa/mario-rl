import pytest

from mario_rl.rl_util.sum_tree_for_spiky import SumTreeForSpiky


@pytest.mark.parametrize("random_values, expected_indices", [
    [[0], [0]],
    [[0] * 10, [0] * 10],
    [[0, 4, 4.1, 5, 5, 5.1, 105, 106], [0, 0, 1, 1, 1, 2, 2, 3]],
])
def test_sum_tree_for_spiky_weighted_sample_indices(random_values, expected_indices):
    sum_tree = SumTreeForSpiky(capacity=4)
    sum_tree[0] = 4
    sum_tree[1] = 1
    sum_tree[2] = 100
    sum_tree[3] = 1

    sum_tree._get_sorted_random_values = lambda _: random_values
    index = sum_tree.weighted_sample_indices_for_spiky_values(len(random_values))
    assert index == expected_indices
