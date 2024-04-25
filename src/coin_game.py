import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RedBlueCoinGame(gym.Env):
    def __init__(self, steps, grid_size=(2, 2)):
        super(RedBlueCoinGame, self).__init__()
        self.grid_size = grid_size
        self.max_steps = steps
        self.steps = 0
        self.state = None
        self.red_position = None
        self.blue_position = None
        self.coin_position = None
        self.coin_color = None  # 0 for red, 1 for blue
        self.red_score = 0
        self.blue_score = 0
        self.action_space = spaces.Discrete(4)  # 4 possible actions: up, down, left, right
        # Update observation space for distinct coin positions
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size[0], grid_size[1], 4), dtype=np.float32) 

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.state = np.zeros(self.observation_space.shape)
        self.red_position = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
        self.blue_position = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
        # Ensure unique start positions for both players
        while self.blue_position == self.red_position:
            self.blue_position = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
        self.coin_color = np.random.choice([0, 1])  # Randomly choose initial color
        self._spawn_coin()
        self._update_state()
        return self.state

    def step(self, actions):
        action_red, action_blue = actions
        self.steps += 1
        self.red_position = self._move(self.red_position, action_red)
        self.blue_position = self._move(self.blue_position, action_blue)
        
        info = {
            'p_own_coin_red': None,
            'p_own_coin_blue': None,
            'p_coin_red': 0,
            'p_coin_blue': 0,
        }
        
        reward_red, reward_blue = 0, 0
        if self.red_position == self.coin_position:
            reward_red += 1
            info['p_own_coin_red'] = 1
            info['p_coin_red'] = 1
            if self.coin_color == 1:  # Red agent picks up a blue coin
                info['p_own_coin_red'] = 0
                reward_blue -= 2

        if self.blue_position == self.coin_position:
            reward_blue += 1
            info['p_own_coin_blue'] = 1
            info['p_coin_blue'] = 1
            if self.coin_color == 0:  # Blue agent picks up a red coin
                info['p_own_coin_blue'] = 0
                reward_red -= 2

        if self.red_position == self.coin_position or self.blue_position == self.coin_position:
            self._toggle_coin_color()
            self._spawn_coin()

        self.red_score += reward_red
        self.blue_score += reward_blue
        self._update_state()

        done = self.steps >= self.max_steps
        info['reward_red'] = reward_red
        info['reward_blue'] = reward_blue
        return self.state, reward_red, reward_blue, done, info

    def _spawn_coin(self):
        # Spawn a coin in a position that is not occupied by any player
        self.coin_position = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
        while self.coin_position == self.red_position or self.coin_position == self.blue_position:
            self.coin_position = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))

    def _toggle_coin_color(self):
        self.coin_color = 1 - self.coin_color  # Toggle color: 0 becomes 1, and 1 becomes 0

    def _update_state(self):
        self.state = np.zeros(self.observation_space.shape)
        self.state[self.red_position[0], self.red_position[1], 0] = 1  # Red player position
        self.state[self.blue_position[0], self.blue_position[1], 1] = 1  # Blue player position
        if self.coin_color == 0:
            self.state[self.coin_position[0], self.coin_position[1], 2] = 1  # Red coin position
        else:
            self.state[self.coin_position[0], self.coin_position[1], 3] = 1  # Blue coin position

    def _move(self, position, action):
        if action == 0:  # Up
            return (max(position[0] - 1, 0), position[1])
        elif action == 1:  # Down
            return (min(position[0] + 1, self.grid_size[0] - 1), position[1])
        elif action == 2:  # Left
            return (position[0], max(position[1] - 1, 0))
        elif action == 3:  # Right
            return (position[0], min(position[1] + 1, self.grid_size[1] - 1))