import argparse
import copy
import os
from datetime import datetime
from typing import List

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


class EpisodeLogger:
    """Episode logger for RL algorithms"""
    def __init__(self, *, save_dir: str, episode: int, train: bool):
        self._save_dir = save_dir
        self._episode = episode
        self._train = train
        self.loss_stats = OnlineStats()
        self.total_reward = 0.0
        self._experience_history: List[Experience] = []
        self._action_values_history = []
        self._exploration_rate = 0.0
        self._replay_memory_size = 0

    def append(self, *, loss: float, exp: Experience, action_values: np.ndarray,
               exploration_rate: float, replay_memory_size: int):
        self.loss_stats.add(loss)
        self.total_reward += exp.reward
        self._experience_history.append(exp)
        self._action_values_history.append(action_values)
        self._exploration_rate = exploration_rate
        self._replay_memory_size = replay_memory_size

    def print(self):
        prefix = "[TRAIN]" if self._train else "[TEST]"
        print(f"{prefix} Episode {self._episode:5d}, "
              f"Reward {self.total_reward:5.0f}, "
              f"Loss {self.loss_stats.mean():10.5f}, "
              f"Explore Prob: {self._exploration_rate:8.5f}, "
              f"Reply Memory Size: {self._replay_memory_size:5.0f}")

    def save_experiences(self):
        """Save experiences as a compressed numpy file"""
        prefix = "train" if self._train else "test"
        file_name = f"{prefix}_ep_{self._episode:05d}_r_{self.total_reward:.0f}"
        file_path = os.path.join(self._save_dir, file_name)
        states_img = np.array([exp.state[0] for exp in self._experience_history])
        states_action = np.array([exp.state[1] for exp in self._experience_history])
        actions = np.array([exp.action for exp in self._experience_history])
        rewards = np.array([exp.reward for exp in self._experience_history])
        dones = np.array([exp.done for exp in self._experience_history])
        action_values = np.array(self._action_values_history)
        np.savez_compressed(file_path, states_img=states_img, states_action=states_action, actions=actions,
                            rewards=rewards, dones=dones, action_values=action_values)


def _get_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--rl", type=str, default="ddqn",
                                 choices=["categorical_dqn", "ddqn", "ddqn_per"], help="RL algorithm")
    argument_parser.add_argument("--world_stage", type=str, default="1-1",
                                 help="World and stage number of Super Mario Bros (e.g. 1-1, 4-2)")
    argument_parser.add_argument("--episodes", type=int, default=1000,
                                 help="Number of episodes to run")
    argument_parser.add_argument("--test_frequency", type=int, default=10,
                                 help="Frequency of running test episodes")
    argument_parser.add_argument("--video_frequency", type=int, default=1,
                                 help="Frequency of recording videos")
    argument_parser.add_argument("--use_test_experience", type=bool, default=True,
                                 help="Whether to use test experience")
    return argument_parser.parse_args()


class Main:
    def __init__(self):
        args = _get_args()
        print(f"Start RL algorithm: {args.rl}, World and stage: {args.world_stage}")

        self._args = args
        self._episodes = args.episodes
        self._test_frequency = args.test_frequency
        self._video_frequency = args.video_frequency
        self._use_test_experience = args.use_test_experience
        self._save_dir = f"./data/{args.world_stage}_{args.rl}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
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
        self._video_max_value = 100

        if args.rl == "categorical_dqn":
            from mario_categorical_dqn import Mario
            self._video_max_value = Mario._V_MAX
        elif args.rl == "ddqn":
            from mario_ddqn import Mario
        elif args.rl == "ddqn_per":
            from mario_ddqn_per import Mario
        else:
            raise ValueError(f"Unknown RL algorithm: {args.rl}")
        state_img_shape = env.observation_space.shape
        state_last_action_shape = (_NUM_STACKED_FRAMES, len(SIMPLE_MOVEMENT))
        self._mario = Mario(state_shape=(state_img_shape, state_last_action_shape),
                            action_dim=env.action_space.n)

    def _run_episode(self, episode, train=True) -> EpisodeLogger:
        episode_logger = EpisodeLogger(save_dir=self._save_dir, episode=episode, train=train)
        env = self._env
        mario = self._mario

        state = env.reset()
        action_deque = OneHotDeque(dim=len(_ACTIONS), maxlen=_NUM_STACKED_FRAMES)
        state = (np.array(state), action_deque.as_ndarray())
        done = False

        is_need_video = not train and (episode % self._video_frequency == 0)
        frames = [copy.deepcopy(env.render(mode="rgb_array"))] if is_need_video else []

        while not done:
            action, action_values = mario.act(state=state, train=train)
            next_state, reward, done, info = env.step(action=action)
            action_deque.append_one_hot(action)
            next_state = (np.array(next_state), action_deque.as_ndarray())

            exp = Experience(state, next_state, action, reward, done)
            if train or self._use_test_experience:
                mario.cache(exp)

            loss = 0
            if train:
                loss = mario.learn() or 0
            episode_logger.append(loss=loss, exp=exp, action_values=action_values,
                                  exploration_rate=mario.exploration_rate, replay_memory_size=len(mario.memory))

            if is_need_video:
                frame = copy.deepcopy(env.render(mode="rgb_array"))
                if action_values is not None:
                    # Draw a bar graph of the action values
                    alphas = [0.5] * len(_ACTIONS)
                    alphas[action] = 1.0
                    frame = draw_horizontal_bar_graph(
                        img=frame,
                        values=action_values,
                        max_value=self._video_max_value,
                        labels=_ACTION_LABELS,
                        colors=_ACTION_COLORS,
                        x=5, y=5, width=len(frame[0]) - 10, label_width=100,
                        alphas=alphas,
                    )
                frames.append(frame)

            state = next_state

        if is_need_video:
            prefix = "train" if train else "test"
            total_reward = episode_logger.total_reward
            make_video(frames, f"{self._save_dir}/{prefix}_ep_{episode:05d}_r_{total_reward:.0f}.mp4")

        return episode_logger

    def run(self):
        try:
            with open(f"{self._save_dir}/train.log", "w") as train_log, \
                    open(f"{self._save_dir}/test.log", "w") as test_log:
                train_log.write("episode,train_step,loss,total_reward\n")
                test_log.write("episode,total_reward\n")

                for episode in range(1, self._episodes + 1):
                    episode_logger = self._run_episode(episode=episode, train=True)
                    episode_logger.print()
                    train_step = self._mario.cnt_called_learn
                    loss = episode_logger.loss_stats.mean()
                    total_reward = episode_logger.total_reward
                    train_log.write(f"{episode},{train_step},{loss:.5f},{total_reward:.0f}\n")

                    if episode % self._test_frequency == 0:
                        episode_logger = self._run_episode(episode=episode, train=False)
                        episode_logger.print()
                        episode_logger.save_experiences()
                        test_log.write(f"{episode},{total_reward:.0f}\n")
        finally:
            self._env.close()


if __name__ == "__main__":
    Main().run()
