import gym
from gym import spaces
import numpy as np


class FiniteMDP(gym.Env):
    MAX_STEPS = 100

    def __init__(self):
        self.transition = np.array([[1, 2],
                                    [0, 3],
                                    [2, 2],
                                    [3, 3]])
        self.reward = np.array([[0, 1000],
                                [0, -1000],
                                [0, 0],
                                [0, 0]])
        self.state = 0
        self.steps = 0
        self.reset()

    def _deterministic_to_stochastic(self):
        shape = np.shape(self.transition)
        if np.size(shape) == 2:
            transition = np.zeros((shape[0], shape[1], shape[0]))
            for s in range(shape[0]):
                for a in range(shape[1]):
                    transition[s, a, self.transition[s, a]] = 1
            self.transition = transition

    def reset(self):
        self._deterministic_to_stochastic()
        self.observation_space = spaces.Discrete(np.shape(self.transition)[0])
        self.action_space = spaces.Discrete(np.shape(self.transition)[1])
        self.steps = 0
        return self.state

    def step(self, action):
        reward = self.reward[self.state, action]
        probs = self.transition[self.state, action, :]
        self.state = np.random.choice(np.arange(np.shape(self.transition)[0]), p=probs)
        self.steps += 1
        return self.state, reward, self.steps > self.MAX_STEPS, None

    def render(self, mode='human'):
        pass

    def seed(self, seed):
        return [seed]

