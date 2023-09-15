import argparse
import copy
import os
from datetime import datetime

import gym_super_mario_bros
import numpy as np
from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from gym_util import GrayScaleObservation, ResizeObservation, SkipFrame
from math_util import OnlineStats
from rl_util import Experience, OneHotDeque
from visual import draw_horizontal_bar_graph, make_video

_ACTIONS = SIMPLE_MOVEMENT
_ACTION_LABELS = tuple("+".join(x) for x in _ACTIONS)
_ACTION_COLORS = ((222, 222, 222), (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255))
_NUM_STACKED_FRAMES = 4


def _get_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--rl', type=str, default='ddqn',
                                 choices=['ddqn', 'ddqn_per'], help='RL algorithm')
    argument_parser.add_argument('--world_stage', type=str, default='1-1',
                                 help='World and stage number of Super Mario Bros (e.g. 1-1, 4-2)')
    argument_parser.add_argument('--episodes', type=int, default=1000,
                                 help='Number of episodes to run')
    argument_parser.add_argument('--test_frequency', type=int, default=10,
                                 help='Frequency of running test episodes')
    argument_parser.add_argument('--video_frequency', type=int, default=1,
                                 help='Frequency of recording videos')
    argument_parser.add_argument('--use_test_experience', type=bool, default=True,
                                 help='Whether to use test experience')
    return argument_parser.parse_args()


class Main:
    def __init__(self):
        args = _get_args()
        print(f'Start RL algorithm: {args.rl}, World and stage: {args.world_stage}')

        self._args = args
        self._episodes = args.episodes
        self._test_frequency = args.test_frequency
        self._video_frequency = args.video_frequency
        self._use_test_experience = args.use_test_experience
        self._save_dir = f'./data/{args.world_stage}_{args.rl}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
        os.makedirs(self._save_dir, exist_ok=True)

        args = self._args
        env_name = f"SuperMarioBros-{args.world_stage}-v0"
        env = gym_super_mario_bros.make(env_name)
        env = JoypadSpace(env, _ACTIONS)  # Limit action space
        env = SkipFrame(env, skip=4)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, size=(84, 84))
        env = FrameStack(env, num_stack=_NUM_STACKED_FRAMES)
        self._env = env

        if args.rl == 'ddqn':
            from mario_ddqn import Mario
        elif args.rl == 'ddqn_per':
            from mario_ddqn_per import Mario
        else:
            raise ValueError(f'Unknown RL algorithm: {args.rl}')
        state_img_shape = env.observation_space.shape
        state_last_action_shape = (_NUM_STACKED_FRAMES, len(SIMPLE_MOVEMENT))
        self._mario = Mario(state_shape=(state_img_shape, state_last_action_shape),
                            action_dim=env.action_space.n)

    def _run_episode(self, episode, train=True):
        env = self._env
        mario = self._mario

        state = env.reset()
        action_deque = OneHotDeque(dim=len(_ACTIONS), maxlen=_NUM_STACKED_FRAMES)
        state = (np.array(state), action_deque.as_ndarray())
        done = False
        total_reward = 0
        loss_stats = OnlineStats()

        is_need_video = not train and (episode % self._video_frequency == 0)
        frames = [copy.deepcopy(env.render(mode='rgb_array'))] if is_need_video else []

        while not done:
            action, action_values = mario.act(state=state, train=train)
            next_state, reward, done, info = env.step(action=action)
            action_deque.append_one_hot(action)
            next_state = (np.array(next_state), action_deque.as_ndarray())
            total_reward += reward

            if train or self._use_test_experience:
                mario.cache(exp=Experience(state, next_state, action, reward, done))

            if train:
                loss = mario.learn()
                if loss is not None:
                    loss_stats.add(loss)

            if is_need_video:
                frame = copy.deepcopy(env.render(mode='rgb_array'))
                if action_values is not None:
                    # Draw a bar graph of the action values
                    alphas = [0.5] * len(_ACTIONS)
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

        if is_need_video:
            make_video(frames, f'{self._save_dir}/ep_{episode: 05d}_r_{total_reward: 05.0f}.mp4')

        prefix = "[TRAIN]" if train else "[TEST]"
        print(f"{prefix} Episode: {episode: 5.0f}/{self._episodes}, "
              f"Total Reward: {total_reward: 5.0f}, "
              f"Average Loss: {loss_stats.mean(): 8.4f}, "
              f"Explore Prob: {mario.exploration_rate: 8.4f},",
              f"Reply Memory Size: {len(mario.memory): 5.0f},")

    def run(self):
        for episode in range(1, self._episodes + 1):
            self._run_episode(episode=episode, train=True)

            if episode % self._test_frequency == 0:
                self._run_episode(episode=episode, train=False)

        self._env.close()


if __name__ == '__main__':
    Main().run()
