"""Focused checks for portfolio figures built from retained experiment records."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_portfolio_figures import (  # noqa: E402
    summarize_action_video,
    summarize_calibration,
    summarize_decision_usefulness,
    wilson_interval,
)


class PortfolioEvidenceTests(unittest.TestCase):
    def test_wilson_interval_known_edges(self) -> None:
        lower, upper = wilson_interval(0, 20)
        self.assertEqual(lower, 0.0)
        self.assertAlmostEqual(upper, 0.161125, places=5)
        lower, upper = wilson_interval(20, 20)
        self.assertAlmostEqual(lower, 0.838875, places=5)
        self.assertEqual(upper, 1.0)

    def test_notebook_14_episode_table_matches_reported_outcomes(self) -> None:
        summary = summarize_decision_usefulness()
        self.assertEqual({key: (row["successes"], row["total"]) for key, row in summary.items()}, {
            "learned_model_mpc": (0, 20),
            "direct_momentum": (20, 20),
        })

    def test_notebook_16_csv_matches_reported_outcomes(self) -> None:
        summary = summarize_action_video()
        self.assertEqual({key: (row["successes"], row["total"]) for key, row in summary.items()}, {
            "uniform_cem": (0, 20),
            "momentum_prior_cem": (20, 20),
            "direct_momentum": (20, 20),
        })

    def test_notebook_17_calibration_summary_matches_report(self) -> None:
        summary = summarize_calibration()
        self.assertAlmostEqual(summary[("in_distribution_test", "position")]["mean"], 0.9072, places=4)
        self.assertAlmostEqual(summary[("policy_shift_test", "position")]["mean"], 0.8897, places=4)
        self.assertAlmostEqual(summary[("in_distribution_test", "velocity")]["mean"], 0.9034, places=4)
        self.assertAlmostEqual(summary[("policy_shift_test", "velocity")]["mean"], 0.8919, places=4)
        self.assertAlmostEqual(summary[("policy_shift_test", "position")]["lower_95"], 0.8224, places=4)

    def test_manifest_paths_and_accessibility_fields_exist(self) -> None:
        manifest = json.loads((ROOT / "docs/portfolio/project.json").read_text(encoding="utf-8"))
        for field in ("schemaVersion", "slug", "title", "summary", "description", "categories", "technologies", "status", "verifiedFindings", "pendingWork", "repository", "links", "media"):
            self.assertIn(field, manifest)
        self.assertTrue(manifest["categories"])
        self.assertTrue(manifest["technologies"])
        self.assertTrue(manifest["status"]["label"])
        self.assertTrue(manifest["status"]["summary"])
        self.assertGreaterEqual(len(manifest["verifiedFindings"]), 3)
        self.assertLessEqual(len(manifest["verifiedFindings"]), 5)
        verified_ids = {finding["id"] for finding in manifest["verifiedFindings"]}
        for finding in manifest["verifiedFindings"]:
            self.assertTrue(finding["claimBoundary"])
            for evidence_path in finding["evidence"]:
                self.assertTrue((ROOT / evidence_path).is_file(), evidence_path)
        for work in manifest["pendingWork"]:
            self.assertTrue(work["status"])
            self.assertTrue(work["claimBoundary"])
            self.assertNotIn(work["id"], verified_ids)
            for evidence_path in work["evidence"]:
                self.assertTrue((ROOT / evidence_path).is_file(), evidence_path)
        for link in manifest["links"]:
            if "path" in link:
                self.assertTrue((ROOT / link["path"]).exists(), link["path"])
        for media in manifest["media"]:
            self.assertTrue(media["alt"])
            self.assertTrue(media["caption"])
            self.assertTrue((ROOT / media["path"]).is_file(), media["path"])


if __name__ == "__main__":
    unittest.main()
