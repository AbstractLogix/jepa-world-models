"""JEPA tutorial helper package.

This namespace collects lightweight utility modules used throughout the
notebooks. The old ``code`` package name conflicted with Python's stdlib
and caused import errors; we now use ``jepa_tutorial`` and expose a
few common helpers here for convenience.
"""

from __future__ import annotations

# make frequently used symbols available at package level
from .eval_utils import embedding_stats, linear_probe_accuracy, tsne_2d
from .lejepa_loss import lejepa_total_loss, predictive_loss
from .rdmreg import rectified_distribution_matching_loss
from .sigreg import sigreg_loss
from .value_shaping import ValueHead, value_loss, value_shaped_distance

__all__ = [
    "embedding_stats",
    "linear_probe_accuracy",
    "tsne_2d",
    "sigreg_loss",
    "rectified_distribution_matching_loss",
    "ValueHead",
    "value_loss",
    "value_shaped_distance",
    "predictive_loss",
    "lejepa_total_loss",
]
