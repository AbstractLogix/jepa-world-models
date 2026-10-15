from __future__ import annotations

from typing import Tuple

import gymnasium as gym
import numpy as np


def make_env(env_id: str = "CartPole-v1", seed: int = 42) -> gym.Env:
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def collect_random_rollout(env: gym.Env, steps: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs, _ = env.reset()
    observations, actions, rewards = [], [], []

    for _ in range(steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    return np.array(observations), np.array(actions), np.array(rewards)
