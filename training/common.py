"""Helpers shared by the three training scripts (no SUMO / PyTorch import here)."""

import math

import numpy as np

DENSITIES = ("low", "medium", "high")
ROUTES = ("straight", "left", "right")
DENSITY_CHOICES = (*DENSITIES, "extreme", "mixed")

# Training environment seeds live far away from the test seeds (default 1000..1049):
#   env_seed = TRAIN_SEED_BASE + train_seed * TRAIN_SEED_STRIDE + episode
# so that no training scenario can reappear in the evaluation.
TRAIN_SEED_BASE = 100_000
TRAIN_SEED_STRIDE = 10_000


def env_seed(train_seed, episode):
    if episode >= TRAIN_SEED_STRIDE:
        raise ValueError(f"episodes must stay below {TRAIN_SEED_STRIDE}")
    return TRAIN_SEED_BASE + int(train_seed) * TRAIN_SEED_STRIDE + int(episode)


def auto_epsilon_decay(start, end, episodes, reach_fraction=0.7):
    """Per-episode multiplicative decay that brings epsilon from `start` to `end`
    after reach_fraction * episodes episodes (then it stays at `end`)."""
    if not 0 < end <= start:
        raise ValueError("need 0 < epsilon_min <= epsilon_start")
    if not 0 < reach_fraction <= 1:
        raise ValueError("--epsilon-reach must be in (0, 1]")
    steps = max(1, int(round(episodes * reach_fraction)))
    return (end / start) ** (1.0 / steps)


def resolve_epsilon_decay(value, start, end, episodes, reach_fraction):
    """`value` is 'auto' or a float string."""
    if str(value).lower() == "auto":
        return auto_epsilon_decay(start, end, episodes, reach_fraction)
    decay = float(value)
    if not 0 < decay <= 1:
        raise ValueError("--epsilon-decay must be 'auto' or in (0, 1]")
    return decay


class ScenarioSampler:
    """Chooses (density, route) for every training episode, reproducibly.

    density: one of low / medium / high / extreme, or 'mixed' = uniform over low/medium/high
    routes : subset of straight / left / right; several routes are sampled uniformly
    """

    def __init__(self, density, routes, seed):
        if density not in DENSITY_CHOICES:
            raise ValueError(f"density must be one of {DENSITY_CHOICES}")
        routes = tuple(routes)
        if not routes or any(r not in ROUTES for r in routes):
            raise ValueError(f"routes must be a non-empty subset of {ROUTES}")
        self.density = density
        self.routes = routes
        self.rng = np.random.default_rng(10_007 + int(seed))

    def sample(self):
        density = (str(self.rng.choice(DENSITIES)) if self.density == "mixed"
                   else self.density)
        route = (self.routes[0] if len(self.routes) == 1
                 else str(self.rng.choice(self.routes)))
        return density, route


def moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    out = np.empty_like(values)
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out[i] = total / min(i + 1, window)
    return out


def describe_schedule(decay, start, end, episodes):
    """Human-readable epsilon schedule for the console."""
    if decay >= 1:
        return f"epsilon fixed at {start:g}"
    reach = math.log(end / start) / math.log(decay)
    shown = "never" if reach > episodes else f"episode {int(math.ceil(reach))}"
    return f"epsilon {start:g} -> {end:g} (decay {decay:.5f}/episode, reaches the floor at {shown})"
