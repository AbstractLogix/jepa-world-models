"""Minimal demo script for value-guided planning scaffolding.

Run from repo root after installing dependencies.
"""

from __future__ import annotations

from planning.envs_utils import make_env, collect_random_rollout


if __name__ == "__main__":
    env = make_env("CartPole-v1")
    obs, acts, rews = collect_random_rollout(env, steps=200)
    print(f"Collected rollout: obs={obs.shape}, actions={acts.shape}, rewards={rews.shape}")
