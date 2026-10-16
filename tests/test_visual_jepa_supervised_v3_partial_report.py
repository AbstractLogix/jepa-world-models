"""Tests for the validation-only figures from the incomplete v3 run."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_visual_jepa_supervised_positive_control_v3_observed import (  # noqa: E402
    EVIDENCE_DIR,
    generate_figures,
)


class PartialRunFigureTests(unittest.TestCase):
    def test_observed_control_figures_regenerate_deterministically(self) -> None:
        e_root = Path("/mnt/e/codex-research/jepa-world-models-tutorial")
        with tempfile.TemporaryDirectory(prefix="v3-partial-a-", dir=e_root) as first_dir:
            with tempfile.TemporaryDirectory(prefix="v3-partial-b-", dir=e_root) as second_dir:
                first = generate_figures(
                    evidence_dir=EVIDENCE_DIR, output_dir=Path(first_dir) / "figures"
                )
                second = generate_figures(
                    evidence_dir=EVIDENCE_DIR, output_dir=Path(second_dir) / "figures"
                )

        self.assertEqual(first["evidence_status"], "incomplete")
        self.assertFalse(first["heldout_open"])
        self.assertEqual(
            [figure["id"] for figure in first["figures"]],
            ["figure-a-observed-action-fidelity", "figure-b-observed-positive-controls"],
        )
        self.assertEqual(
            [figure["sha256"] for figure in first["figures"]],
            [figure["sha256"] for figure in second["figures"]],
        )
        for figure in first["figures"]:
            self.assertTrue(figure["alt_text"])
            self.assertTrue(figure["caption"])
            self.assertTrue(figure["sample_size"])
            self.assertTrue(figure["limitations"])


if __name__ == "__main__":
    unittest.main()
