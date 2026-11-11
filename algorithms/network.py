"""Small CPU-friendly Q network for the seven-dimensional SUMO observation."""

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, observation_dim=7, action_count=5, hidden_size=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, observations):
        return self.layers(observations)
