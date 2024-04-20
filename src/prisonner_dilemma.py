import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PrisonersDilemma(gym.Env):
    def __init__(self, steps):
        super(PrisonersDilemma, self).__init__()
        self.action_space = spaces.Discrete(2)  # Two possible actions: cooperate or defect
        self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2)))  # Observations of the opponent's previous actions
        self.state = None
        self.steps = 0
        self.max_steps = steps
        # Action Cooperate: 0
        # Action Defect: 1
        self.reward_matrix = np.array([[3, 0], [5, 1]])  # Reward matrix for (Agent1, Agent2): (Cooperate, Cooperate), (Cooperate, Defect), (Defect, Cooperate), (Defect, Defect)

    def reset(self):
        self.state = np.array([0, 0])  # Initial state where both agents start with cooperating
        self.steps = 0
        return self.state

    def step(self, action1, action2):
        self.steps += 1
        reward1 = self.reward_matrix[action1, action2]
        reward2 = self.reward_matrix[action2, action1]
        # self.state = (action1, action2)
        self.state = np.array([action1, action2])
        info = {
        'cooperate_a': 1-action1,
        'cooperate_b': 1-action2,
        'reward_a': reward1,
        'reward_b': reward2,
        }
        
        done = self.steps >= self.max_steps
        return self.state, reward1, reward2, done, info

    def render(self, mode='human'):
        print("Current state:", self.state)
