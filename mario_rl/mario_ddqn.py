import copy
from typing import Optional

import keras
import numpy as np
import tensorflow as tf

from rl_util import Experience, ReplayBuffer


class Mario:
    """Agent that learns to play Super Mario Bros using Double Deep Q-Networks (DDQN)."""
    _GAMMA = 0.9  # discount factor for future rewards
    _LEARNING_RATE = 0.00025  # learning rate for q-network
    _BATCH_SIZE = 32  # no. of experiences to sample in each training update
    _SYNC_EVERY = 10000  # no. of calls to learn() before syncing target network with online network
    _FREQ_LEARN = 1  # no. of calls to learn() before updating online network
    _LEARN_START = 1000  # no. of experiences in replay buffer before learning starts
    _EXPLORATION_RATE_INIT = 1.0  # initial exploration rate
    _EXPLORATION_RATE_MIN = 0.1  # final exploration rate
    _EXPLORATION_RATE_DECAY = 0.99999  # rate of exponential decay of exploration rate per call to act() with train=True
    _REPLAY_BUFFER_SIZE = 100000  # no. of experiences to store in replay buffer

    def __init__(self, state_shape: tuple, action_dim: int):
        """
        :param state_shape: Shape of the state space (image and last action).
        :param action_dim: Shape of the action space.
        """
        self._action_dim = action_dim

        # online network
        # input image and last action
        input_img = keras.layers.Input(shape=state_shape[0], dtype='float32')
        input_last_action = keras.layers.Input(shape=state_shape[1], dtype='float32')
        # network for image
        output_img = keras.layers.Permute((2, 3, 1))(input_img)
        output_img = keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu')(output_img)
        output_img = keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu')(output_img)
        output_img = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(output_img)
        output_img = keras.layers.Flatten()(output_img)
        # network for last action
        output_last_action = keras.layers.Flatten()(input_last_action)
        output_last_action = keras.layers.Dense(units=32, activation='relu')(output_last_action)
        # concatenate networks
        outputs = keras.layers.Concatenate()([output_img, output_last_action])
        outputs = keras.layers.Dense(units=512, activation='relu')(outputs)
        q_values = keras.layers.Dense(units=self._action_dim, activation='linear')(outputs)
        self._q_online = keras.Model(inputs=[input_img, input_last_action], outputs=q_values)

        # target network
        self._q_target = copy.deepcopy(self._q_online)
        self._q_target.trainable = False

        self._optimizer = keras.optimizers.Adam(learning_rate=self._LEARNING_RATE, epsilon=0.01/self._BATCH_SIZE)

        self.exploration_rate = self._EXPLORATION_RATE_INIT
        self.memory = ReplayBuffer(size=self._REPLAY_BUFFER_SIZE)
        self._cnt_called_learn = 0

    def act(self, state, train=False) -> (int, Optional[np.ndarray]):
        """Acting Policy of the Mario Agent given an observation.
        Decreases the exploration_rate linearly over time.
        """
        action_values = None
        # epsilon-greedy exploration strategy
        if train and np.random.rand() < self.exploration_rate:
            # explore - do random action
            action_idx = np.random.randint(self._action_dim)
        else:
            # exploit
            action_idx, action_values = self.greedy_act(state)

        if train:
            # decrease exploration_rate
            self.exploration_rate *= self._EXPLORATION_RATE_DECAY
            self.exploration_rate = max(self._EXPLORATION_RATE_MIN, self.exploration_rate)

        return int(action_idx), action_values

    def greedy_act(self, state) -> (int, np.ndarray):
        """Acting Policy of the Mario Agent given an observation."""
        action_values = self._q_online((np.array([state[0]]), np.array([state[1]])))
        action_idx = np.argmax(action_values, axis=1)
        return int(action_idx), action_values[0]

    def cache(self, exp: Experience):
        """Cache the experience into memory buffer"""
        self.memory.append(exp)

    def learn(self) -> Optional[list]:
        """Sample experiences from memory and run one iteration of gradient descent.
        If memory is not yet full enough to sample a batch, no learning is done and None is returned.

        :return: The loss on this gradient step if learning was done, else None.
        """
        self._cnt_called_learn += 1

        if self._cnt_called_learn % self._SYNC_EVERY == 0:
            self._q_target.set_weights(self._q_online.get_weights())

        if (self._cnt_called_learn % self._FREQ_LEARN != 0 or
                self._cnt_called_learn < self._LEARN_START or
                len(self.memory) < self._BATCH_SIZE):
            return None

        experiences = self.memory.sample(self._BATCH_SIZE)
        states_img = np.array([exp.state[0] for exp in experiences])  # [batch_size, steps, width, height]
        states_last_action = np.array([exp.state[1] for exp in experiences])  # [batch_size, steps, action_dim]
        states = [states_img, states_last_action]
        next_states_img = np.array([exp.next_state[0] for exp in experiences])  # [batch_size, width, height, steps]
        next_states_last_action = np.array([exp.next_state[1] for exp in experiences])  # [batch_size, steps, action_dim]
        next_states = [next_states_img, next_states_last_action]
        actions = np.array([exp.action for exp in experiences])  # [batch_size,]
        next_q_online_values = self._q_online(next_states)  # [batch_size, action_dim]
        best_next_actions = np.argmax(next_q_online_values, axis=1)  # [batch_size,]
        next_q_target_values = self._q_target(next_states)  # [batch_size, action_dim]
        td_targets = np.array([exp.reward + (1 - exp.done) * self._GAMMA * next_q_target[best_next_action]
                               for exp, next_q_target, best_next_action
                               in zip(experiences, next_q_target_values, best_next_actions)])  # [batch_size,]
        with tf.GradientTape() as tape:
            q_values = self._q_online(states)  # [batch_size, action_dim]
            one_hot_actions = tf.one_hot(actions, self._action_dim)  # [batch_size, action_dim]
            q_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)  # [batch_size,]
            td_errors = td_targets - q_values  # [batch_size,]
            loss = tf.reduce_mean(tf.square(td_errors))
        grads = tape.gradient(loss, self._q_online.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._q_online.trainable_variables))
        return loss
