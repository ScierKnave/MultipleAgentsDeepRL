import collections
import numpy as np
from gymnasium import Wrapper, spaces


class HistoryWrapper(Wrapper):
    def __init__(self, env, length):
        super().__init__(env)
        self.length = length
        low, high = env.observation_space.low, env.observation_space.high
        low = np.repeat(low[np.newaxis, :], length, axis=0).flatten()
        high = np.repeat(high[np.newaxis, :], length, axis=0).flatten()

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self._reset_buf()

    def _reset_buf(self):
        self.history = collections.deque(maxlen=self.length)
        for _ in range(self.length): 
            self.history.append(np.zeros(self.env.observation_space.shape))

    def _make_observation(self):
        array = np.concatenate(list(self.history), axis=0)
        array = array.reshape(array.shape[0], -1).flatten()
        return array

    def reset(self):
        self._reset_buf()
        obs = super(HistoryWrapper, self).reset()
        self.history.append(obs)      
        return self._make_observation()

    def step(self, actions):
        obs, reward1, reward2, done, info = super(HistoryWrapper, self).step(actions)
        self.history.append(obs)
        return self._make_observation(), reward1, reward2, done, info