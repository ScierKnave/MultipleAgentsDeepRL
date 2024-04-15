import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RedBlueCoinGame(gym.Env):
    def __init__(self, grid_size=(5, 5), num_coins=5):
        super(RedBlueCoinGame, self).__init__()
        self.grid_size = grid_size
        self.num_coins = num_coins
        self.state = None
        self.red_position = None
        self.blue_position = None
        self.red_score = 0
        self.blue_score = 0
        self.action_space = spaces.Discrete(4)  # 4 possible actions: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size[0], grid_size[1], 3), dtype=np.float32)

    def reset(self):
        self.state = np.zeros((self.grid_size[0], self.grid_size[1], 3))
        self.red_position = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
        self.blue_position = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
        self.state[self.red_position[0], self.red_position[1], 0] = 1  # Red player
        self.state[self.blue_position[0], self.blue_position[1], 1] = 1  # Blue player
        for _ in range(self.num_coins):
            coin_position = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
            self.state[coin_position[0], coin_position[1], 2] = 1  # Coins
        return self.state

    def step(self, action_red, action_blue):
        reward_red = 0
        reward_blue = 0

        # Update red player position
        new_red_position = self._move(self.red_position, action_red)
        if self._is_valid_move(new_red_position):
            self.state[self.red_position[0], self.red_position[1], 0] = 0
            self.red_position = new_red_position
            self.state[self.red_position[0], self.red_position[1], 0] = 1
            if self.state[self.red_position[0], self.red_position[1], 2] == 1:  # Red player collects a coin
                reward_red += 1
                self.state[self.red_position[0], self.red_position[1], 2] = 0

        # Update blue player position
        new_blue_position = self._move(self.blue_position, action_blue)
        if self._is_valid_move(new_blue_position):
            self.state[self.blue_position[0], self.blue_position[1], 1] = 0
            self.blue_position = new_blue_position
            self.state[self.blue_position[0], self.blue_position[1], 1] = 1
            if self.state[self.blue_position[0], self.blue_position[1], 2] == 1:  # Blue player collects a coin
                reward_blue += 1
                self.state[self.blue_position[0], self.blue_position[1], 2] = 0
                
        done = False
        if np.sum(self.state[:, :, 2]) == 0:  # No more coins left
            done = True
        return self.state, reward_red, reward_blue, done, {}

    def _move(self, position, action):
        if action == 0:  # Up
            return (max(position[0] - 1, 0), position[1])
        elif action == 1:  # Down
            return (min(position[0] + 1, self.grid_size[0] - 1), position[1])
        elif action == 2:  # Left
            return (position[0], max(position[1] - 1, 0))
        elif action == 3:  # Right
            return (position[0], min(position[1] + 1, self.grid_size[1] - 1))

    def _is_valid_move(self, position):
        if position[0] < 0 or position[0] >= self.grid_size[0] or \
           position[1] < 0 or position[1] >= self.grid_size[1]:
            return False
        return True
