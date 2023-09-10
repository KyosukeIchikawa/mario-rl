import copy

import gym_super_mario_bros
import numpy as np
from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from mario import Mario
from gym_util import SkipFrame, GrayScaleObservation, ResizeObservation
from math_util import OnlineStats
from rl_util import Experience
from visual import draw_horizontal_bar_graph, make_video

_ACTION_LABELS = tuple("+".join(x) for x in SIMPLE_MOVEMENT)
_ACTION_COLORS = ((222, 222, 222), (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255))
_EPISODES = 1001
_TEST_FREQUENCY = 10
_VIDEO_FREQUENCY = 1


def run_episode(episode, env, mario, train: bool, need_video: bool):
    state = env.reset()
    state = np.array(state)
    done = False
    total_reward = 0
    loss_stats = OnlineStats()
    frames = [copy.deepcopy(env.render(mode='rgb_array'))] if need_video else []

    while not done:
        action, action_values = mario.act(state=state, train=train)
        next_state, reward, done, info = env.step(action=action)
        next_state = np.array(next_state)
        total_reward += reward

        if train:
            mario.cache(exp=Experience(state, next_state, action, reward, done))
            loss = mario.learn()
            if loss is not None:
                loss_stats.add(loss)

        if need_video:
            frame = copy.deepcopy(env.render(mode='rgb_array'))
            from matplotlib import pyplot as plt
            if action_values is not None:
                # Draw a bar graph of the action value with legend in the upper left corner of the game screen.
                alphas = [0.5] * len(_ACTION_LABELS)
                alphas[action] = 1.0
                frame = draw_horizontal_bar_graph(
                    img=frame,
                    values=action_values,
                    max_value=100,
                    labels=_ACTION_LABELS,
                    colors=_ACTION_COLORS,
                    x=5, y=5, width=len(frame[0]) - 10, label_width=100,
                    alphas=alphas,
                )

            frames.append(frame)

        state = next_state

    if need_video:
        make_video(frames, f'./dat/ep_{episode: 05d}_reward_{total_reward: 05.0f}.mp4')

    prefix = "[TRAIN]" if train else "[TEST]"
    print(f"{prefix} Episode: {episode: 5.0f}/{_EPISODES}, "
          f"Total Reward: {total_reward: 5.0f}, "
          f"Average Loss: {loss_stats.mean(): 8.4f}, "
          f"Explore P: {mario.exploration_rate: 8.4f},",
          f"Reply Memory Size: {len(mario.memory): 5.0f},")


def main():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)  # Limit action space
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, size=(84, 84))
    env = FrameStack(env, num_stack=4)

    mario = Mario(state_shape=env.observation_space.shape,
                  action_dim=env.action_space.n,
                  save_dir=None)

    for episode in range(_EPISODES):
        run_episode(episode, env, mario, train=True, need_video=False)

        if episode % _TEST_FREQUENCY == 0:
            need_video = episode % (_TEST_FREQUENCY * _VIDEO_FREQUENCY) == 0
            run_episode(episode, env, mario, train=False, need_video=need_video)


if __name__ == '__main__':
    main()
