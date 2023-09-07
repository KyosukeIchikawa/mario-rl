import gym


class SkipFrame(gym.Wrapper):
    """Return only every `skip`-th frame

    This class is based on the following code.
    https://github.com/pytorch/tutorials/blob/main/intermediate_source/mario_rl_tutorial.py
    """
    def __init__(self, env, skip: int):
        assert skip >= 1
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
