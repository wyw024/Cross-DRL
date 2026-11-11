"""Double DQN: online network selects, target network evaluates."""

from algorithms.dqn import DQNAgent


class DoubleDQNAgent(DQNAgent):
    def __init__(self, **kwargs):
        super().__init__(double_dqn=True, **kwargs)
