import gym_super_mario_bros
import numpy as np
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

from mario import Mario
from gym_util import SkipFrame, GrayScaleObservation, ResizeObservation
from math_util import OnlineStats
from rl_util import Experience

_EPISODES = 1000


def main():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

    # Limit the action-space to
    #   0. walk right
    #   1. jump right
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, size=(84, 84))
    env = FrameStack(env, num_stack=4)

    mario = Mario(state_shape=env.observation_space.shape,
                  action_dim=env.action_space.n,
                  save_dir=None)

    total_experiences = 0
    for e in range(1, _EPISODES + 1):
        state = env.reset()
        state = np.array(state)
        done = False
        total_reward = 0
        loss_stats = OnlineStats()

        while not done:
            total_experiences += 1
            action = mario.act(state=state, train=True)
            next_state, reward, done, info = env.step(action=action)
            next_state = np.array(next_state)
            total_reward += reward
            mario.cache(exp=Experience(state, next_state, action, reward, done))

            loss = mario.learn()
            if loss is not None:
                loss_stats.add(loss)

            state = next_state

        print(f"Episode: {e: 5.0f}/{_EPISODES}, "
              f"Total Reward: {total_reward: 5.0f}, "
              f"Average Loss: {loss_stats.mean(): 8.4f}, "
              f"Explore P: {mario.exploration_rate: 8.4f},",
              f"Reply Memory Size: {len(mario.memory): 5.0f},")


if __name__ == '__main__':
    main()
