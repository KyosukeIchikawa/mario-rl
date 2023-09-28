import copy
from typing import Optional

import keras
import numpy as np
import tensorflow as tf

from mario_rl.rl_util import Experience, ReplayBuffer


class Mario:
    """Agent that learns to play Super Mario Bros using Categorical Deep Q-Networks.

    Reference: https://arxiv.org/abs/1707.06887
    """
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
    _V_MIN = -20.0  # min value of value distribution
    _V_MAX = 20.0  # max value of value distribution
    _NUM_ATOMS = 51  # no. of atoms in value distribution

    def __init__(self, state_shape: tuple, action_dim: int):
        """
        :param state_shape: Shape of the state space (image and last action).
        :param action_dim: Shape of the action space.
        """
        self._action_dim = action_dim
        # Q-values are represented as a value distribution over a discrete range of values
        # support_points defines the discrete range of values
        self._support_points = np.linspace(self._V_MIN, self._V_MAX, self._NUM_ATOMS)
        self._delta_support = (self._V_MAX - self._V_MIN) / (self._NUM_ATOMS - 1)

        # online network
        # input image and last action
        input_img = keras.layers.Input(shape=state_shape[0], dtype='float32')
        input_last_action = keras.layers.Input(shape=state_shape[1], dtype='float32')
        # network for image
        output_img = keras.layers.Permute((2, 3, 1))(input_img)
        output_img = keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu',
                                         kernel_initializer="he_normal")(output_img)
        output_img = keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu',
                                         kernel_initializer="he_normal")(output_img)
        output_img = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu',
                                         kernel_initializer="he_normal")(output_img)
        output_img = keras.layers.Flatten()(output_img)
        # network for last action
        output_last_action = keras.layers.Flatten()(input_last_action)
        output_last_action = keras.layers.Dense(units=32, activation='relu',
                                                kernel_initializer="he_normal")(output_last_action)
        # concatenate networks
        outputs = keras.layers.Concatenate()([output_img, output_last_action])
        outputs = keras.layers.Dense(units=512, activation='relu',
                                     kernel_initializer="he_normal")(outputs)
        outputs = keras.layers.Dense(units=action_dim * self._NUM_ATOMS,
                                     kernel_initializer="he_normal")(outputs)  # [batch_size, action_dim * num_atoms]
        outputs = keras.layers.Reshape((action_dim, self._NUM_ATOMS))(outputs)  # [batch_size, action_dim, num_atoms]
        outputs = keras.layers.Softmax()(outputs)  # [batch_size, action_dim, num_atoms]
        self._q_online = keras.Model(inputs=[input_img, input_last_action], outputs=outputs)

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
        """Acting Policy of the Mario Agent given an observation.
        Q-values are represented as a value distribution over a discrete range of values.

        :param state: The state to get the greedy action for.
        :return: The action index and the Q-values for each action.
        """
        action_indices, q_values, _ = self._greedy_actions((np.array([state[0]]), np.array([state[1]])), self._q_online)
        return action_indices.numpy()[0], q_values.numpy()[0]

    def _greedy_actions(self, states, q_network: keras.Model) -> (tf.Tensor, tf.Tensor, tf.Tensor):
        """Returns the greedy action for a given state.
        Q-values are represented as a value distribution over a discrete range of values.

        :param states: Batch of states to get the greedy action for.
        :return: The action indices, Q-values and Q-value distributions for each state.
        """
        q_probs = q_network(states)  # [batch_size, action_dim, num_atoms]
        # Q-values are represented as a value distribution over a discrete range of values
        q_means = tf.reduce_sum(q_probs * self._support_points, axis=-1)  # [batch_size, action_dim]
        action_indices = tf.argmax(q_means, axis=-1)  # [batch_size]
        return action_indices, q_means, q_probs

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
        next_states_last_action = np.array(
            [exp.next_state[1] for exp in experiences])  # [batch_size, steps, action_dim]
        next_states = [next_states_img, next_states_last_action]
        actions = np.array([exp.action for exp in experiences])  # [batch_size,]
        rewards = np.array([exp.reward for exp in experiences])  # [batch_size,]
        dones = np.array([exp.done for exp in experiences])  # [batch_size,]
        # best_next_actions: [batch_size,]
        # next_q_online_values: [batch_size, action_dim]
        # next_q_probs: [batch_size, action_dim, num_atoms]
        best_next_actions, _, _ = self._greedy_actions(next_states, self._q_online)
        _, next_q_online_values, next_q_probs = self._greedy_actions(next_states, self._q_target)
        target_probs = self._calc_target_probs_for_actions(q_probs=next_q_probs, action_indices=best_next_actions,
                                                           rewards=rewards, dones=dones)  # [batch_size, num_atoms]
        with tf.GradientTape() as tape:
            q_probs = self._q_online(states)  # [batch_size, action_dim, num_atoms]
            masks = self._make_masks(action_indices=actions)  # [batch_size, action_dim, num_atoms]
            q_probs = tf.reduce_sum(q_probs * masks, axis=1)  # [batch_size, num_atoms]
            q_probs = tf.clip_by_value(q_probs, 1e-8, 1.0)  # clip to avoid log(0)
            # cross entropy loss
            loss = - tf.reduce_mean(tf.reduce_sum(target_probs * tf.math.log(q_probs), axis=-1))
        grads = tape.gradient(loss, self._q_online.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._q_online.trainable_variables))
        return loss

    def _make_masks(self, action_indices) -> tf.Tensor:
        """Create masks for the action indices.
        Shape of masks is [batch_size, action_dim, num_atoms].
        (batch_size is the number of action indices)

        For example, if action_indices = [0, 2] and action_dim = 3 and num_atoms = 5, then
        masks = [
          [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], # action 0
          [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]  # action 2
        ]
        """
        actions_one_hot = tf.one_hot(action_indices, self._action_dim, dtype=tf.float32)  # [batch_size, action_dim]
        actions_one_hot = tf.expand_dims(actions_one_hot, axis=-1)  # [batch_size, action_dim, 1]
        masks = tf.tile(actions_one_hot, [1, 1, self._NUM_ATOMS])  # [batch_size, action_dim, num_atoms]
        return masks

    def _extract_q_probs(self, q_probs: tf.Tensor, action_indices: tf.Tensor) -> tf.Tensor:
        """Extract the Q-value probabilities for the given action indices.

        :param q_probs: The Q-value probabilities. Shape is [batch_size, action_dim, num_atoms]
        :param action_indices: The action indices. Shape is [batch_size,]
        :return: The Q-value probabilities for the given action indices. Shape is [batch_size, num_atoms]
        """
        masks = self._make_masks(action_indices)  # [batch_size, action_dim, num_atoms]
        q_probs = q_probs * masks  # [batch_size, action_dim, num_atoms]
        return tf.reduce_sum(q_probs, axis=1)  # [batch_size, num_atoms]

    def _calc_target_probs(self, rewards: np.ndarray, dones: np.ndarray, next_q_probs: np.ndarray) -> np.ndarray:
        """Calculate the target probabilities.

        :param rewards: The rewards of the experiences. Shape is [batch_size,]
        :param dones: The dones of the experiences. Shape is [batch_size,]
        :param next_q_probs: The distribution of the next states. Shape is [batch_size, num_atoms]
        :return: The target distribution. Shape is [batch_size, num_atoms]
        """
        batch_size = len(rewards)
        target_probs = np.zeros((batch_size, self._NUM_ATOMS))
        for j, support_point in enumerate(self._support_points):
            # Calculate the target value for each support point
            targets = rewards + (1 - dones) * self._GAMMA * support_point
            targets = np.clip(targets, self._V_MIN, self._V_MAX)
            # Calculate indices of each target value in the support points
            target_indices = (targets - self._V_MIN) / self._delta_support  # [batch_size,]
            # Get the lower and upper indices of the probability
            lower_indices = np.floor(target_indices).astype(np.int16)  # [batch_size,]
            upper_indices = np.ceil(target_indices).astype(np.int16)  # [batch_size,]
            # Calculate the probability of the lower and upper indices
            lower_ratios = upper_indices - target_indices  # [batch_size,]
            upper_ratios = 1 - lower_ratios  # [batch_size,]
            next_q_prob = next_q_probs[:, j]  # [batch_size,]
            # Add the probability to the target distribution
            # NOTE: Because reduce_sum(nexq_q_probs, axis=1) == 1.0
            #       target_dist[:, lower_indices] + target_dist[:, upper_indices] == next_q_prob
            #       So, reduce_sum(target_dist, axis=1) == 1.0
            target_probs[:, lower_indices] += lower_ratios * next_q_prob
            target_probs[:, upper_indices] += upper_ratios * next_q_prob
        return target_probs

    def _calc_target_probs_for_actions(self, q_probs: tf.Tensor, action_indices: tf.Tensor,
                                       rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Calculate the target probabilities for the best actions.

        :param q_probs: The Q-value probabilities. Shape is [batch_size, action_dim, num_atoms]
        :param action_indices: The action indices. Shape is [batch_size,]
        :param rewards: The rewards of the experiences. Shape is [batch_size,]
        :param dones: The dones of the experiences. Shape is [batch_size,]
        :return: The target distribution. Shape is [batch_size, num_atoms]
        """
        next_q_probs = self._extract_q_probs(q_probs=q_probs,
                                             action_indices=action_indices)  # [batch_size, action_dim, num_atoms]
        target_probs = self._calc_target_probs(rewards=rewards, dones=dones, next_q_probs=next_q_probs.numpy())
        return target_probs
