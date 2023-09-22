import time

import numpy as np

from mario_rl.math_util import OnlineStats
from mario_rl.rl_util import Experience, PrioritizedReplayBuffer, PrioritizedReplayBufferWithoutSumTree

_TEST_ITERATIONS = 100
_BUFFER_SIZE = 2 ** 17  # 131072
_BATCH_SIZE = 64


def validation_prioritized_replay_buffer_speed():
    dummy_exp = Experience(0, 0, 0, 0, False)
    indices = list(range(_BUFFER_SIZE))
    priorities = np.random.uniform(0.0, 100.0, size=_BUFFER_SIZE)

    replay_buffers = {}
    for replay_buffer_class in [PrioritizedReplayBuffer, PrioritizedReplayBufferWithoutSumTree]:
        replay_buffer = replay_buffer_class(size=_BUFFER_SIZE, compress=False)
        for i in range(_BUFFER_SIZE):
            replay_buffer.append(dummy_exp)
        replay_buffer.update_priorities(indices, priorities)
        replay_buffers[replay_buffer_class.__name__] = replay_buffer

    print(f"Validation sampling speed (iter={_TEST_ITERATIONS}, batch={_BATCH_SIZE}) ...")
    for name, buffer in replay_buffers.items():
        time_stats = OnlineStats()
        for _ in range(_TEST_ITERATIONS):
            start_time = time.time()
            buffer.sample(_BATCH_SIZE)
            elapsed_time = time.time() - start_time
            time_stats.add(elapsed_time)
        print(f"{name:38s} : sum={time_stats.sum():.4f}s, mean={time_stats.mean():.4f}s, std={time_stats.std():.4f}s")


if __name__ == '__main__':
    validation_prioritized_replay_buffer_speed()
