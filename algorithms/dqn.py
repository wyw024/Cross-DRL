"""DQN agent with replay and target network; Double DQN is a target option."""

from pathlib import Path
import numpy as np
import torch
from torch import nn

from algorithms.network import QNetwork
from algorithms.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, observation_dim=7, action_count=5, hidden_size=64,
                 learning_rate=1e-3, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995, buffer_capacity=20_000,
                 batch_size=64, target_update_interval=200, seed=42,
                 double_dqn=False):
        if observation_dim <= 0 or action_count <= 0 or batch_size <= 0:
            raise ValueError("dimensions and batch_size must be positive")
        if target_update_interval <= 0:
            raise ValueError("target_update_interval must be positive")
        self.observation_dim = observation_dim
        self.action_count = action_count
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.double_dqn = double_dqn
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.device = torch.device("cpu")
        self.online = QNetwork(observation_dim, action_count, hidden_size).to(self.device)
        self.target = QNetwork(observation_dim, action_count, hidden_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(buffer_capacity, seed)
        self.update_count = 0

    def _observation(self, observation):
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (self.observation_dim,) or not np.all(np.isfinite(value)):
            raise ValueError(f"expected finite observation of shape ({self.observation_dim},)")
        return value

    def select_action(self, observation, info=None, training=False):
        obs = self._observation(observation)
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_count))
        with torch.no_grad():
            values = self.online(torch.as_tensor(obs, device=self.device).unsqueeze(0))
            return int(values.argmax(dim=1).item())

    def learn(self, observation, action, reward, next_observation,
              terminated, truncated=False):
        if not 0 <= int(action) < self.action_count:
            raise ValueError("invalid action")
        self.buffer.add(self._observation(observation), action, reward,
                        self._observation(next_observation), terminated)
        if len(self.buffer) < self.batch_size:
            return None
        obs, actions, rewards, next_obs, terminals = self.buffer.sample(self.batch_size)
        obs = torch.as_tensor(obs, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        next_obs = torch.as_tensor(next_obs, device=self.device)
        terminals = torch.as_tensor(terminals, device=self.device)
        chosen_q = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.online(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target(next_obs).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - terminals) * next_q
        loss = nn.functional.smooth_l1_loss(chosen_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"algorithm": "double_dqn" if self.double_dqn else "dqn",
                    "observation_dim": self.observation_dim,
                    "action_count": self.action_count, "hidden_size": self.hidden_size,
                    "online": self.online.state_dict(), "target": self.target.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "epsilon": self.epsilon,
                    "update_count": self.update_count}, path)

    def load(self, path, resume=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        expected = "double_dqn" if self.double_dqn else "dqn"
        if checkpoint["algorithm"] != expected or any(
            checkpoint[key] != getattr(self, key)
            for key in ("observation_dim", "action_count", "hidden_size")
        ):
            raise ValueError("checkpoint algorithm or architecture does not match")
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint["target"])
        self.target.eval()
        if resume:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = float(checkpoint["epsilon"])
            self.update_count = int(checkpoint["update_count"])
