from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.analyze_visual_jepa_decision_study import (
    action_effect_diagnostics,
    paired_seed_start_bootstrap_interval,
    primary_comparison,
)
from scripts.generate_visual_jepa_decision_study_figures import require_complete_evidence
from scripts.validate_visual_jepa_decision_study import (
    FINAL_SEEDS,
    validate_oracle_gate_rows,
)


class PairedSeedStartBootstrapTests(unittest.TestCase):
    def test_resamples_shared_starts_deterministically(self) -> None:
        deltas = {
            "seed-a": {"start-1": 0.0, "start-2": 10.0},
            "seed-b": {"start-1": 10.0, "start-2": 0.0},
        }
        first = paired_seed_start_bootstrap_interval(deltas, replicates=200)
        second = paired_seed_start_bootstrap_interval(deltas, replicates=200)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], 5.0)
        self.assertGreaterEqual(first[1], 5.0)

    def test_rejects_unpaired_or_nonfinite_start_deltas(self) -> None:
        with self.assertRaisesRegex(ValueError, "same non-empty evaluation starts"):
            paired_seed_start_bootstrap_interval(
                {"seed-a": {"start-1": 1.0}, "seed-b": {"start-2": 1.0}}
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            paired_seed_start_bootstrap_interval(
                {"seed-a": {"start-1": float("nan")}, "seed-b": {"start-1": 1.0}}
            )

    def test_primary_comparison_pairs_each_model_seed_to_direct_starts(self) -> None:
        rows = []
        for seed, offset in (("m1", 1.0), ("m2", 3.0)):
            for start, direct_return in (("e1", 10.0), ("e2", 20.0)):
                rows.append(
                    {
                        "method": "jepa_cem",
                        "model_seed": seed,
                        "eval_seed": start,
                        "initialization": "momentum_prior",
                        "status": "complete",
                        "return": str(direct_return + offset),
                    }
                )
        for start, direct_return in (("e1", 10.0), ("e2", 20.0)):
            rows.append(
                {
                    "method": "direct_visual_momentum",
                    "model_seed": "",
                    "eval_seed": start,
                    "initialization": "direct",
                    "status": "complete",
                    "return": str(direct_return),
                }
            )
        result = primary_comparison(rows, replicates=200)
        self.assertEqual(result["per_seed_pair_mean_return_delta"], {"m1": 1.0, "m2": 3.0})
        self.assertEqual(result["mean_across_seed_pair_means"], 2.0)
        self.assertEqual(result["shared_evaluation_starts"], 2)


class ActionEffectDiagnosticTests(unittest.TestCase):
    def test_figure_generator_fails_closed_without_retained_evidence(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                require_complete_evidence(Path(directory))

    def test_reports_predicted_and_observed_action_spans_per_seed(self) -> None:
        rows = [
            {
                "status": "complete",
                "method": "jepa",
                "model_seed": "m1",
                "start_id": "held-out-1",
                "action": str(action),
                "predicted_position": str(0.1 * action),
                "actual_position": str(0.2 * action),
                "predicted_velocity": str(0.01 * action),
                "actual_velocity": str(0.02 * action),
            }
            for action in (-1.0, 0.0, 1.0)
        ]
        result = action_effect_diagnostics(rows)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["predicted_position_span"], 0.2)
        self.assertAlmostEqual(result[0]["observed_velocity_span"], 0.04)

    def test_rejects_missing_action_in_a_complete_group(self) -> None:
        rows = [
            {
                "status": "complete",
                "method": "jepa",
                "model_seed": "m1",
                "start_id": "held-out-1",
                "action": str(action),
                "predicted_position": "0",
                "actual_position": "0",
                "predicted_velocity": "0",
                "actual_velocity": "0",
            }
            for action in (-1.0, 1.0)
        ]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            action_effect_diagnostics(rows)


class OracleGateValidationTests(unittest.TestCase):
    def test_matches_recorded_gate_to_exact_final_start_rows(self) -> None:
        rows = [
            {
                "method": "oracle_cem",
                "model_seed": "",
                "eval_seed": str(seed),
                "initialization": "momentum_prior",
                "privileged_state": "True",
                "status": "complete",
                "success": "1",
                "return": "1.0",
                "steps": "1",
                "transition_calls": "100",
                "forward_batches": "1",
                "planning_seconds": "0.1",
                "elapsed_seconds": "0.1",
                "device": "cpu",
                "error": "",
                "horizon_steps": "60",
                "candidates": "48",
                "elites": "8",
                "updates": "4",
                "score_name": "cumulative_return",
                "replan_interval": "5",
            }
            for seed in FINAL_SEEDS
        ]
        gate = {
            "successes": len(FINAL_SEEDS),
            "transition_calls": 100 * len(FINAL_SEEDS),
            "threshold": 18,
        }
        planner_config = {
            "horizon_steps": 60,
            "candidates": 48,
            "elites": 8,
            "updates": 4,
            "score_name": "cumulative_return",
            "initialization": "momentum_prior",
            "replan_interval": 5,
        }
        validate_oracle_gate_rows(rows, gate, planner_config)

        tampered = [*rows[:-1], {**rows[-1], "eval_seed": rows[0]["eval_seed"]}]
        with self.assertRaisesRegex(ValueError, "duplicate episode case"):
            validate_oracle_gate_rows(tampered, gate, planner_config)

        with self.assertRaisesRegex(ValueError, "success total"):
            validate_oracle_gate_rows(rows, {**gate, "successes": 19}, planner_config)

        with self.assertRaisesRegex(ValueError, "locked setting"):
            validate_oracle_gate_rows(
                rows,
                gate,
                {**planner_config, "horizon_steps": 59},
            )


if __name__ == "__main__":
    unittest.main()
