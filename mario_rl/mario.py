import copy
from typing import Optional

import keras
import numpy as np

from rl_util import Experience, ReplayBuffer


class Mario:
    _GAMMA = 0.9  # discount factor for future rewards
    _LEARNING_RATE = 0.00025  # learning rate for q-network
    _BATCH_SIZE = 32  # no. of experiences to sample in each training update
    _SYNC_EVERY = 1000  # no. of experiences between each sync of online and target network
    _FREQ_LEARN = 4  # no. of experiences between each training update
    _EXPLORATION_RATE_MAX = 1.0  # initial exploration rate
    _EXPLORATION_RATE_MIN = 0.1  # final exploration rate
    _EXPLORATION_RATE_DECAY = 0.99999975  # rate of exponential decay of exploration rate per action in training

    def __init__(self, state_shape: tuple, action_dim: int, save_dir: str):
        self._state_shape = state_shape
        self._action_dim = action_dim
        self._save_dir = save_dir

        self._q_online = keras.Sequential([
            keras.layers.Permute((2, 3, 1), input_shape=self._state_shape),
            keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(units=512, activation='relu'),
            keras.layers.Dense(units=action_dim, activation=None),
        ])
        self._q_target = copy.deepcopy(self._q_online)
        self._q_target.trainable = False

        optimizer = keras.optimizers.Adam(learning_rate=self._LEARNING_RATE, clipnorm=1.0)
        self._q_online.compile(optimizer=optimizer, loss='mse')

        self.n_learn = 0
        self.exploration_rate = self._EXPLORATION_RATE_MAX

        self.memory = ReplayBuffer(size=1000000)

    def act(self, state: np.ndarray, train=False) -> int:
        """Acting Policy of the Mario Agent given an observation."""
        # epsilon-greedy exploration strategy
        if train and np.random.rand() < self.exploration_rate:
            # explore - do random action
            action_idx = np.random.randint(self._action_dim)
        else:
            # exploit
            action_values = self._q_online(state[np.newaxis])
            action_idx = np.argmax(action_values, axis=1)

        if train:
            # decrease exploration_rate
            self.exploration_rate *= self._EXPLORATION_RATE_DECAY
            self.exploration_rate = max(self._EXPLORATION_RATE_MIN, self.exploration_rate)

        return int(action_idx)

    def cache(self, exp: Experience):
        """Cache the experience into memory buffer"""
        self.memory.append(exp)

    def learn(self) -> Optional[list]:
        """Sample experiences from memory and run one iteration of gradient descent.
        If memory is not yet full enough to sample a batch, no learning is done and None is returned.

        :return: The loss on this gradient step if learning was done, else None.
        """
        self.n_learn += 1

        if self.n_learn % self._SYNC_EVERY == 0:
            self._q_target.set_weights(self._q_online.get_weights())

        if self.n_learn % self._FREQ_LEARN != 0 or len(self.memory) < self._BATCH_SIZE:
            return None

        experiences = self.memory.sample(self._BATCH_SIZE)
        states = np.array([exp.state for exp in experiences])  # [batch_size, width, height, channels]
        next_states = np.array([exp.next_state for exp in experiences])  # [batch_size, width, height, channels]
        actions = np.array([exp.action for exp in experiences])  # [batch_size]
        q_values = self._q_online(states)  # [batch_size, action_dim]
        next_q_values = self._q_target(next_states)  # [batch_size, action_dim]
        td_targets = np.array([exp.reward + (1 - exp.done) * self._GAMMA * np.max(next_q)
                               for next_q, exp in zip(next_q_values, experiences)])  # [batch_size]
        # Update online network
        q_target = q_values.numpy()
        q_target[np.arange(self._BATCH_SIZE), actions] = td_targets
        loss = self._q_online.train_on_batch(states, q_target)
        return loss
