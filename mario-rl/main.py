import time

import gym_super_mario_bros
import numpy as np
from gym.wrappers import FrameStack
from matplotlib import pyplot
from nes_py.wrappers import JoypadSpace

import gym_util


def main():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

    # Limit the action-space to
    #   0. walk right
    #   1. jump right
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = gym_util.SkipFrame(env, skip=4)
    env = gym_util.GrayScaleObservation(env)
    env = gym_util.ResizeObservation(env, size=(84, 84))
    env = FrameStack(env, num_stack=4)

    env.reset()
    next_state, reward, done, info = env.step(action=0)
    rgb_array = env.render(mode="rgb_array")
    # pyplotで画像を表示する
    pyplot.imshow(rgb_array, cmap="gray")
    pyplot.show()


if __name__ == '__main__':
    main()
