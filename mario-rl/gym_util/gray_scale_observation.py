import cv2
import gym
import numpy as np


class GrayScaleObservation(gym.ObservationWrapper):
    """Convert observation images to grayscale.

    This class is based on the following code.
    https://github.com/pytorch/tutorials/blob/main/intermediate_source/mario_rl_tutorial.py
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.observation_space.shape[:2],
            dtype=np.uint8,
        )

    def observation(self, observation):
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return frame
