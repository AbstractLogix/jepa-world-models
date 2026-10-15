from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score


def linear_probe_accuracy(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    max_iter: int = 1000,
) -> float:
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(train_x, train_y)
    preds = clf.predict(test_x)
    return float(accuracy_score(test_y, preds))


def tsne_2d(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    return TSNE(n_components=2, random_state=random_state, init="pca").fit_transform(embeddings)


def embedding_stats(embeddings: np.ndarray) -> Dict[str, float]:
    cov = np.cov(embeddings, rowvar=False)
    diag = np.diag(cov)
    off = cov - np.diag(diag)
    return {
        "mean_std": float(np.std(embeddings, axis=0).mean()),
        "trace_cov": float(np.trace(cov)),
        "mean_abs_offdiag_cov": float(np.mean(np.abs(off))),
    }
