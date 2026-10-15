from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, title: str = "t-SNE") -> None:
    z2 = TSNE(n_components=2, random_state=42, init="pca").fit_transform(embeddings)
    plt.figure(figsize=(6, 5))
    plt.scatter(z2[:, 0], z2[:, 1], c=labels, s=8, cmap="tab10", alpha=0.8)
    plt.title(title)
    plt.tight_layout()


def plot_pca(embeddings: np.ndarray, labels: np.ndarray, title: str = "PCA") -> None:
    z2 = PCA(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(6, 5))
    plt.scatter(z2[:, 0], z2[:, 1], c=labels, s=8, cmap="tab10", alpha=0.8)
    plt.title(title)
    plt.tight_layout()


def plot_activation_hist(embeddings: np.ndarray, bins: int = 50) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(embeddings.ravel(), bins=bins)
    plt.title("Embedding activation distribution")
    plt.tight_layout()
