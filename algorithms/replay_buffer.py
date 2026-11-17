"""Uniform experience replay with reproducible sampling."""

from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=20_000, seed=42):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.items = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def add(self, observation, action, reward, next_observation, terminated):
        self.items.append((
            np.asarray(observation, dtype=np.float32).copy(), int(action),
            float(reward), np.asarray(next_observation, dtype=np.float32).copy(),
            bool(terminated),
        ))

    def sample(self, batch_size):
        if batch_size <= 0 or batch_size > len(self):
            raise ValueError("invalid batch size")
        batch = [self.items[i] for i in self.rng.choice(len(self), batch_size, replace=False)]
        observations, actions, rewards, next_observations, terminated = zip(*batch)
        return (
            np.stack(observations), np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32), np.stack(next_observations),
            np.asarray(terminated, dtype=np.float32),
        )
