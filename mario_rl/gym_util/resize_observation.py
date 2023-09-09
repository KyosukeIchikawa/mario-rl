import cv2
import gym
import numpy as np


class ResizeObservation(gym.ObservationWrapper):
    """Resize observation images.

    This class is based on the following code.
    https://github.com/pytorch/tutorials/blob/main/intermediate_source/mario_rl_tutorial.py
    """
    def __init__(self, env, size: tuple):
        super().__init__(env)
        n_channel = self.observation_space.shape[2:]
        new_shape = size + n_channel
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=np.uint8,
        )

    def observation(self, observation):
        frame = cv2.resize(observation, self.observation_space.shape[:2])
        frame = frame.astype(np.float32) / 255.0
        return frame
