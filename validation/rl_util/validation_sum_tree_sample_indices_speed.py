import sys
import os
import time

current_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(root_directory)

import numpy as np

from mario_rl.math_util import OnlineStats
from mario_rl.rl_util.sum_tree import SumTree
from mario_rl.rl_util.sum_tree_for_spiky import SumTreeForSpiky


_TEST_ITERATIONS = 100
_BUFFER_SIZE = 2 ** 17  # 131072
_BATCH_SIZE = 64
_MAX_PRIORITY = 100.0
_RANDOM_NORMAL_LOC = 0
_RANDOM_NORMAL_SCALE = _MAX_PRIORITY / 4


def validation_sum_tree_sample_indices_speed():
    spiky_values = np.random.uniform(0.0, _MAX_PRIORITY / (_BUFFER_SIZE - 1), size=_BUFFER_SIZE)
    spiky_values[-1] = _MAX_PRIORITY
    very_spiky_values = np.random.uniform(0.0, _MAX_PRIORITY / (_BUFFER_SIZE - 1) / 10, size=_BUFFER_SIZE)
    very_spiky_values[-1] = _MAX_PRIORITY

    priorities_by_method = {
        "uniform random": np.random.uniform(0.0, _MAX_PRIORITY, size=_BUFFER_SIZE),
        "normal random": np.clip(np.random.normal(loc=_RANDOM_NORMAL_LOC, scale=_RANDOM_NORMAL_SCALE, size=_BUFFER_SIZE),
                                 a_min=0.0, a_max=_MAX_PRIORITY),
        "spiky": spiky_values,
        "very spiky": very_spiky_values,
    }

    for method, priorities in priorities_by_method.items():
        print(f"Validation sampling speed ({method} priorities, iter={_TEST_ITERATIONS}, batch={_BATCH_SIZE}) ...")
        for sum_tree_class in [SumTree, SumTreeForSpiky]:
            sum_tree = sum_tree_class(capacity=_BUFFER_SIZE)
            for i in range(_BUFFER_SIZE):
                sum_tree[i] = priorities[i]

            time_stats = OnlineStats()
            for _ in range(_TEST_ITERATIONS):
                start_time = time.time()
                sum_tree.weighted_sample_indices(_BATCH_SIZE)
                elapsed_time = time.time() - start_time
                time_stats.add(elapsed_time)
            print(f"{sum_tree_class.__name__:20s} : "
                  f"sum={time_stats.sum():.4f}s, mean={time_stats.mean():.4f}s, std={time_stats.std():.4f}s")


if __name__ == '__main__':
    validation_sum_tree_sample_indices_speed()
