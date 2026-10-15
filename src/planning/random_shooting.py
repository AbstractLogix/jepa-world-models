from __future__ import annotations

import numpy as np


def random_shooting(
    score_fn,
    action_low: float,
    action_high: float,
    horizon: int,
    n_candidates: int,
    action_dim: int,
    rng: np.random.Generator | None = None,
):
    """Sample candidate action sequences and return the best by score_fn."""
    rng = rng or np.random.default_rng(42)
    candidates = rng.uniform(
        low=action_low,
        high=action_high,
        size=(n_candidates, horizon, action_dim),
    )
    scores = score_fn(candidates)
    best_idx = int(np.argmax(scores))
    return candidates[best_idx], float(scores[best_idx])
