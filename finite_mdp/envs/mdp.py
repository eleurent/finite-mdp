import numpy as np


class MDP(object):
    def __init__(self, state=0):
        self.state = state
        self.transition = None
        self.reward = None
        self.terminal = None

    def step(self, action, np_random=np.random):
        raise NotImplementedError()

    def reset(self):
        self.state = 0
        return self.state

    @staticmethod
    def from_config(config, np_random=np.random):
        mode = config.get("mode", None)
        transition = np.array(config.get("transition", []))
        reward = np.array(config.get("reward", []))
        terminal = np.array(config.get("terminal", []))
        if mode == "deterministic":
            mdp = DeterministicMDP(transition, reward, terminal)
        elif mode == "stochastic":
            mdp = StochasticMDP(transition, reward, terminal)
        else:
            raise ValueError("Unknown MDP mode in configuration")
        if config.get("random", False):
            mdp.randomize(np_random)
        return mdp

    def to_config(self):
        raise NotImplementedError()


class DeterministicMDP(MDP):
    mode = "deterministic"

    def __init__(self, transition, reward, terminal=None, state=0):
        """
        :param transition: array of shape S x A
        :param reward: array of shape S x A
        :param terminal: array of shape S
        :param int state: initial state
        """
        super(DeterministicMDP, self).__init__(state)
        self.transition = transition
        self.reward = reward
        self.terminal = terminal
        if terminal is None or not np.size(terminal):
            self.terminal = np.zeros(np.shape(transition)[0])
        self.terminal = self.terminal.astype(bool)

    def step(self, action, np_random=np.random):
        reward = self.reward[self.state, action]
        done = self.terminal[self.state]
        self.state = self.next_state(self.state, action)
        return self.state, reward, done, self.to_config()

    def next_state(self, state, action):
        return self.transition[state, action]

    def randomize(self, np_random):
        self.transition = np_random.choice(range(np.shape(self.transition)[0]), size=np.shape(self.transition))
        self.reward = np_random.rand(*np.shape(self.reward))

    def to_config(self):
        return dict(
            mode=self.mode,
            transition=self.transition.tolist(),
            reward=self.reward.tolist(),
            terminal=self.terminal.tolist()
        )

    def update(self, config):
        if "transition" in config:
            self.transition = np.array(config["transition"])
        if "reward" in config:
            self.reward = np.array(config["reward"])
        if "terminal" in config:
            self.terminal = np.array(config["terminal"])


class StochasticMDP(DeterministicMDP):
    mode = "stochastic"

    def __init__(self, transition, reward, terminal=None, state=0):
        """
        :param transition: array of size S x A x S
        :param reward:  array of shape S x A
        :param terminal:  array of shape S
        :param int state: initial state
        """
        super(StochasticMDP, self).__init__(transition, reward, terminal, state)

    def step(self, action, np_random=np.random):
        reward = self.reward[self.state, action]
        done = self.terminal[self.state]
        self.state = self.next_state(self.state, action)
        return self.state, reward, done, self.to_config()

    def next_state(self, state, action, np_random=np.random):
        probs = self.transition[state, action, :]
        return np_random.choice(np.arange(np.shape(self.transition)[0]), p=probs)

    @staticmethod
    def from_deterministic(mdp: DeterministicMDP):
        shape = np.shape(mdp.transition)
        new_transition = np.zeros((shape[0], shape[1], shape[0]))
        for s in range(shape[0]):
            for a in range(shape[1]):
                new_transition[s, a, int(mdp.transition[s, a])] = 1
        return StochasticMDP(new_transition, mdp.reward, mdp.terminal)

    def randomize(self, np_random=np.random):
        self.transition = np_random.rand(*np.shape(self.transition))
        self.reward = np_random.rand(*np.shape(self.reward))