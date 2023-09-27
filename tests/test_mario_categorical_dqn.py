import pytest

import numpy as np

from mario_rl.mario_categorical_dqn import Mario


@pytest.mark.parametrize("action_indices, expected_mask", [
    ([0], [[[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
    ([0, 2], [[[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
              [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]]),
    ([2, 0, 1], [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
                 [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                 [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]]),
])
def test_mario_make_mask(action_indices, expected_mask):
    action_indices = np.array(action_indices, dtype=np.float32)
    expected_mask = np.array(expected_mask, dtype=np.float32)

    Mario._NUM_ATOMS = 5
    mario = Mario(state_shape=((2, 64, 64), (2, 3)), action_dim=3)
    masks = mario._make_masks(action_indices=action_indices)
    assert np.array_equal(masks, expected_mask)


def test_extract_action_probs():
    Mario._NUM_ATOMS = 5
    mario = Mario(state_shape=((2, 64, 64), (2, 3)), action_dim=3)
    probs = np.array([[[0.0, 0.2, 0.8, 0.0, 0.0], [0.0, 0.3, 0.3, 0.0, 0.4], [1.0, 0.0, 0.0, 0.0, 0.0]],
                      [[0.6, 0.1, 0.1, 0.1, 0.1], [0.0, 0.5, 0.5, 0.0, 0.0], [0.0, 0.2, 0.0, 0.8, 0.0]]])
    action_indices = np.array([1, 2])
    expected_action_probs = []
    for prob, action_index in zip(probs, action_indices):
        expected_action_probs.append(prob[action_index])
    expected_action_probs = np.array(expected_action_probs, dtype=np.float32)
    action_probs = mario._extract_q_probs(q_probs=probs, action_indices=action_indices)
    assert np.array_equal(action_probs, expected_action_probs)


@pytest.mark.parametrize("rewards, dones, next_q_probs, expected_target_probs", [
    ([-100], [False],
     [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
     [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    ([0], [False],
     [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
     [[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    ([0], [False],
     [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]],
     [[0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25]]),
    ([9], [False],
     [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
     [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    ([0], [True],
     [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
     [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    ([100], [True],
     [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
     [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
    ([0, 9, 0], [False, False, True],
     [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
     [[0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25],
      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
])
def test_mario_calc_target_probs(rewards, dones, next_q_probs, expected_target_probs):
    rewards = np.array(rewards)
    dones = np.array(dones)
    next_q_probs = np.array(next_q_probs)
    expected_target_probs = np.array(expected_target_probs)
    # num_atoms = 11, vmax = 10, vmin = -10, gamma = 0.9
    # support = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    # gamma * support = [-9, -7.2, -5.4, -3.6, -1.8, 0, 1.8, 3.6, 5.4, 7.2, 9]
    # delta_support = 2
    Mario._NUM_ATOMS = 11
    mario = Mario(state_shape=((2, 64, 64), (2, 3)), action_dim=3)
    target_probs = mario._calc_target_probs(rewards=rewards, dones=dones, next_q_probs=next_q_probs)
    np.array_equal(target_probs, expected_target_probs)
