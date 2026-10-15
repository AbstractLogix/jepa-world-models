from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_episode_rewards(rewards: np.ndarray, title: str = "Episode rewards") -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(rewards, marker="o", linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()


def plot_action_sequence(actions: np.ndarray, title: str = "Planned action sequence") -> None:
    plt.figure(figsize=(7, 3))
    plt.plot(actions)
    plt.xlabel("Timestep")
    plt.ylabel("Action")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
