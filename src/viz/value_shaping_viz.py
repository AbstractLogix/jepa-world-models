from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def plot_value_distance_correlation(distances: np.ndarray, values: np.ndarray) -> float:
    corr, _ = pearsonr(distances, values)

    plt.figure(figsize=(6, 4))
    plt.scatter(distances, values, s=10, alpha=0.7)
    plt.xlabel("Latent distance")
    plt.ylabel("Value / return")
    plt.title(f"Value-distance relation (r={corr:.3f})")
    plt.tight_layout()

    return float(corr)
