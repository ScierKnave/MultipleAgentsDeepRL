import gym
from gym import spaces
import numpy as np

class PrisonersDilemma(gym.Env):
    def __init__(self):
        super(PrisonersDilemma, self).__init__()
        self.action_space = spaces.Discrete(2)  # Two possible actions: cooperate or defect
        self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2)))  # Observations of the opponent's previous actions
        self.state = None
        self.reward_matrix = np.array([[3, 0], [5, 1]])  # Reward matrix for (Agent1, Agent2): (Cooperate, Cooperate), (Cooperate, Defect), (Defect, Cooperate), (Defect, Defect)

    def reset(self):
        self.state = (0, 0)  # Initial state where both agents start with cooperating
        return self.state

    def step(self, action1, action2):
        reward1 = self.reward_matrix[action1, action2]
        reward2 = self.reward_matrix[action2, action1]
        self.state = (action1, action2)
        return self.state, reward1, reward2, {}

    def render(self, mode='human'):
        print("Current state:", self.state)
