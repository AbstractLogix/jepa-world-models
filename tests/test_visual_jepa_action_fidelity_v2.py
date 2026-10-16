from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from jepa_tutorial.action_conditioned_visual import ActionPredictor
from jepa_tutorial.action_fidelity_v2 import (
    action_effect_metrics,
    effective_rank_diagnostics,
    fit_stable_probe,
    probe_bootstrap_rmse,
    split_episode_indices,
    stable_seed,
    unclipped_state_diagnostics,
    validate_action_triplet_records,
    validate_experiment_completeness,
    validate_file_manifest,
    validate_rollout_records,
)
from jepa_tutorial.visual_mountaincar import render_frame, resize_frame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_visual_jepa_action_fidelity_v2 as figure_analyzer  # noqa: E402
import run_visual_jepa_action_fidelity_v2 as study_runner  # noqa: E402


class ActionFidelityDiagnosticsTests(unittest.TestCase):
    def test_seed_pair_training_forwards_both_seed_values(self) -> None:
        data = object()
        output_dir = Path("/tmp/v2-seed-forwarding")
        device = torch.device("cpu")
        expected = ({}, {}, [], [])
        with patch.object(study_runner, "train_one", return_value=expected) as train_one:
            result = study_runner.fit_seed_pair(
                "jepa",
                "sigreg",
                data,
                9401,
                9301,
                device,
                output_dir,
                epochs=1,
            )
        self.assertIs(result, expected)
        train_one.assert_called_once_with(
            "jepa", "sigreg", data, 9401, 9301, device, output_dir, epochs=1
        )

    def test_action_predictor_responds_to_its_action_input(self) -> None:
        torch.manual_seed(17)
        predictor = ActionPredictor(embedding_dim=32)
        embedding = torch.zeros((1, 32))
        actions = torch.tensor([-1.0, 0.0, 1.0])
        predictions = predictor(embedding.expand(3, -1), actions)
        self.assertGreater(float(torch.linalg.vector_norm(predictions[0] - predictions[2]).detach()), 1e-4)

    def test_constant_and_non_finite_embeddings_fail_representation_health(self) -> None:
        constant = effective_rank_diagnostics(torch.ones(12, 8))
        self.assertTrue(constant["finite"])
        self.assertEqual(constant["low_variance_dims"], 8)
        self.assertEqual(constant["effective_rank"], 0.0)

        non_finite = effective_rank_diagnostics(torch.tensor([[0.0, 1.0], [float("nan"), 2.0]]))
        self.assertFalse(non_finite["finite"])

    def test_effective_rank_uses_covariance_spectrum(self) -> None:
        x = torch.arange(20, dtype=torch.float32).unsqueeze(1)
        embeddings = torch.cat((x, x * 2.0, x * -1.0, torch.ones_like(x)), dim=1)
        diagnostics = effective_rank_diagnostics(embeddings)
        self.assertEqual(diagnostics["dimension"], 4)
        self.assertAlmostEqual(diagnostics["effective_rank"], 1.0, places=5)
        self.assertGreater(diagnostics["covariance_trace"], 0.0)

    def test_probe_uses_positive_alpha_and_stable_feature_floor(self) -> None:
        rng = np.random.default_rng(41)
        x = rng.normal(size=(200, 4))
        x[:, 3] = 1e-7 * rng.normal(size=200)
        y = np.column_stack((x[:, 0] + 0.2 * x[:, 1], x[:, 1] - x[:, 2]))
        probe = fit_stable_probe(x[:150], y[:150], x[150:], y[150:])
        self.assertGreater(probe["alpha"], 0.0)
        self.assertGreaterEqual(float(np.min(probe["feature_scale"])), 1e-3)
        self.assertTrue(np.isfinite(probe["raw_coef_l2"]))
        self.assertLessEqual(probe["gram_condition"], 1e6)

    def test_probe_bootstrap_is_deterministic_and_episode_blocked(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(120, 3))
        groups = np.repeat(np.arange(6), 20)
        y = np.column_stack((x[:, 0], x[:, 1] + x[:, 2]))
        first = probe_bootstrap_rmse(x, y, groups, x[:12], alpha=0.1, seed=99, replicates=5)
        second = probe_bootstrap_rmse(x, y, groups, x[:12], alpha=0.1, seed=99, replicates=5)
        self.assertEqual(first, second)
        self.assertEqual(first["replicates"], 5)

    def test_action_effect_metrics_flag_action_insensitivity(self) -> None:
        rows = []
        for action in (-1.0, 0.0, 1.0):
            rows.append(
                {
                    "method": "jepa",
                    "start_id": "s1",
                    "action": action,
                    "actual_position": 0.1 + 0.0015 * action,
                    "actual_velocity": 0.01 + 0.0015 * action,
                    "predicted_position_raw": 0.2,
                    "predicted_velocity_raw": 0.0,
                }
            )
        summary = action_effect_metrics(rows)
        self.assertEqual(summary["starts"], 1)
        self.assertEqual(summary["position_order_accuracy"], 0.0)
        self.assertAlmostEqual(summary["position_median_normalized_contrast_error"], 0.5)

    def test_action_contrast_error_excludes_tied_contexts(self) -> None:
        rows = []
        for start_id, actual_values, predicted_values in (
            ("tied", (0.0, 0.0, 0.0), (1.0, 0.0, -1.0)),
            ("moving", (-1.0, 0.0, 1.0), (-1.0, 0.0, 1.0)),
        ):
            for action, actual, predicted in zip((-1.0, 0.0, 1.0), actual_values, predicted_values, strict=True):
                rows.append(
                    {
                        "method": "visual",
                        "start_id": start_id,
                        "action": action,
                        "actual_position": actual,
                        "actual_velocity": actual,
                        "predicted_position_raw": predicted,
                        "predicted_velocity_raw": predicted,
                    }
                )
        summary = action_effect_metrics(rows)
        self.assertEqual(summary["position_valid_starts"], 1)
        self.assertEqual(summary["velocity_valid_starts"], 1)
        self.assertEqual(summary["position_median_normalized_contrast_error"], 0.0)
        self.assertEqual(summary["velocity_median_normalized_contrast_error"], 0.0)

    def test_validation_gate_uses_bootstrap_fields_and_clipping_counts(self) -> None:
        record = {
            "finite": True,
            "low_variance_dims": 8,
            "effective_rank": 16.0,
            "target_finite": True,
            "target_low_variance_dims": 8,
            "target_effective_rank": 15.0,
            "probe_gram_condition": 100.0,
            "position_rmse": 0.005,
            "velocity_rmse": 0.002,
            "position_r2": 0.95,
            "velocity_r2": 0.75,
            "position_mae": 0.01,
            "velocity_mae": 0.005,
        }
        action_summary = {
            "position_median_normalized_contrast_error": 0.1,
            "velocity_median_normalized_contrast_error": 0.1,
            "position_order_accuracy": 1.0,
            "velocity_order_accuracy": 1.0,
            "position_valid_starts": 20,
            "velocity_valid_starts": 20,
            "one_step_position_mae": 0.01,
            "one_step_velocity_mae": 0.005,
            "unbounded_prediction_count": 3,
            "action_prediction_count": 60,
        }
        self.assertTrue(study_runner._fit_validation_gates(record, action_summary, method="jepa")["validation_pass"])
        unstable_probe = dict(record, position_rmse=0.011)
        self.assertFalse(study_runner._fit_validation_gates(unstable_probe, action_summary, method="jepa")["validation_pass"])
        collapsed_target = dict(record, target_low_variance_dims=17)
        self.assertFalse(study_runner._fit_validation_gates(collapsed_target, action_summary, method="jepa")["representation_pass"])
        clipped = dict(action_summary, unbounded_prediction_count=4)
        self.assertFalse(study_runner._fit_validation_gates(record, clipped, method="jepa")["clipping_pass"])

    def test_supervised_visual_planner_uses_rendered_context_not_privileged_state(self) -> None:
        class ConstantVisualDynamics(torch.nn.Module):
            def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
                return torch.zeros((observations.shape[0], 2), device=observations.device)

        frames = SimpleNamespace(
            previous_frame=resize_frame(render_frame(-0.5)),
            current_frame=resize_frame(render_frame(-0.48)),
        )
        candidates = np.asarray([[0.0, 0.0], [0.5, -0.5]], dtype=np.float32)
        config = SimpleNamespace(
            horizon_steps=10,
            replan_interval=5,
            action_blocks=2,
            score_name="cumulative_return",
        )
        model = ConstantVisualDynamics()
        scores = study_runner._score_supervised_visual_candidates(
            model, frames, candidates, config, torch.device("cpu")
        )
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(np.isfinite(scores).all())
        prior, calls = study_runner._supervised_visual_prior_mean(
            model, frames, config, torch.device("cpu")
        )
        self.assertEqual(prior.shape, (2,))
        self.assertEqual(calls, 10)

    def test_action_input_sensitivity_changes_with_predicted_effect(self) -> None:
        rows = []
        for action in (-1.0, 0.0, 1.0):
            rows.append(
                {
                    "method": "visual",
                    "start_id": "s2",
                    "action": action,
                    "actual_position": action * 0.0015,
                    "actual_velocity": action * 0.0015,
                    "predicted_position_raw": action * 0.0015,
                    "predicted_velocity_raw": action * 0.0015,
                }
            )
        summary = action_effect_metrics(rows)
        self.assertEqual(summary["position_order_accuracy"], 1.0)
        self.assertAlmostEqual(summary["position_median_normalized_contrast_error"], 0.0)

    def test_unclipped_state_reports_clipping_and_resets_wall_velocity(self) -> None:
        raw = torch.tensor([[-1.3, -0.04], [0.2, 0.08]])
        clipped, mask = unclipped_state_diagnostics(raw)
        self.assertTrue(bool(mask[0, 0]))
        self.assertTrue(bool(mask[1, 1]))
        self.assertAlmostEqual(float(clipped[0, 0]), -1.2, places=5)
        self.assertEqual(float(clipped[0, 1]), 0.0)
        self.assertAlmostEqual(float(clipped[1, 1]), 0.07, places=5)
        wall_reset, wall_mask = unclipped_state_diagnostics(torch.tensor([[-1.2, 0.03]]))
        self.assertEqual(float(wall_reset[0, 1]), 0.0)
        self.assertTrue(bool(wall_mask[0, 1]))

    def test_episode_split_keeps_whole_episodes_disjoint(self) -> None:
        episode_ids = np.repeat(np.arange(20), 3)
        train, validation = split_episode_indices(episode_ids, train_episode_split=16)
        self.assertEqual(set(np.unique(episode_ids[train])), set(range(16)))
        self.assertEqual(set(np.unique(episode_ids[validation])), set(range(16, 20)))
        self.assertFalse(set(train).intersection(validation))

    def test_stable_seed_is_repeatable_and_role_sensitive(self) -> None:
        first = stable_seed("v2", "test", 9501, "start-1")
        second = stable_seed("v2", "test", 9501, "start-1")
        other_role = stable_seed("v2", "train", 9501, "start-1")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other_role)

    def test_rollout_validation_rejects_duplicate_or_non_finite_records(self) -> None:
        row = {
            "method": "jepa",
            "model_seed": "9401",
            "start_id": "test-1",
            "horizon": 5,
            "actual_position": 0.1,
            "actual_velocity": 0.01,
            "predicted_position": 0.11,
            "predicted_velocity": 0.02,
            "status": "complete",
        }
        validate_rollout_records([row])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_rollout_records([row, row])
        broken = dict(row, predicted_velocity=float("nan"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_rollout_records([broken])

    def test_action_triplets_require_all_three_finite_actions(self) -> None:
        rows = [
            {
                "method": "visual",
                "objective": "control",
                "model_seed": 1,
                "data_seed": 2,
                "start_id": "s1",
                "action": action,
                "actual_position": .1,
                "actual_velocity": .01,
                "predicted_position_raw": .1,
                "predicted_velocity_raw": .01,
            }
            for action in (-1.0, 0.0, 1.0)
        ]
        validate_action_triplet_records(rows)
        with self.assertRaisesRegex(ValueError, "one row for each"):
            validate_action_triplet_records(rows[:2])
        broken = [dict(row) for row in rows]
        broken[0]["predicted_position_raw"] = float("inf")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_action_triplet_records(broken)

    def test_experiment_completeness_detects_missing_seed_pairs(self) -> None:
        methods = (("jepa", "sigreg"),)
        model_rows = []
        action_rows = []
        for data_seed, model_seed in ((9301, 9401), (9302, 9402)):
            model_rows.append(
                {
                    "method": "jepa",
                    "objective": "sigreg",
                    "model_seed": model_seed,
                    "data_seed": data_seed,
                    "status": "complete",
                }
            )
            for start in range(20):
                for action in (-1.0, 0.0, 1.0):
                    action_rows.append(
                        {
                            "method": "jepa",
                            "objective": "sigreg",
                            "model_seed": model_seed,
                            "data_seed": data_seed,
                            "start_id": f"{data_seed}-{start}",
                            "action": action,
                            "actual_position": .1,
                            "actual_velocity": .01,
                            "predicted_position_raw": .1,
                            "predicted_velocity_raw": .01,
                        }
                    )
        # The exact and handcrafted rows are shared positive controls per data seed.
        for data_seed in (9301, 9302):
            for method, objective in (("exact", "simulator_oracle"), ("handcrafted", "visual_state_plus_oracle")):
                for start in range(20):
                    for action in (-1.0, 0.0, 1.0):
                        action_rows.append(
                            {
                                "method": method,
                                "objective": objective,
                                "model_seed": 0,
                                "data_seed": data_seed,
                                "start_id": f"{data_seed}-{start}",
                                "action": action,
                                "actual_position": .1,
                                "actual_velocity": .01,
                                "predicted_position_raw": .1,
                                "predicted_velocity_raw": .01,
                            }
                        )
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            validate_experiment_completeness(
                model_rows[:-1],
                action_rows,
                data_seeds=(9301, 9302),
                model_seeds=(9401, 9402),
                methods=methods,
            )

    def test_complete_seed_pairs_and_positive_controls_validate(self) -> None:
        methods = (
            ("jepa", "sigreg"),
            ("jepa", "visreg_style"),
            ("jepa", "variance_cov"),
            ("supervised_visual", "supervised_state_mse"),
            ("state_model", "privileged_state_mse"),
        )
        pairs = ((9301, 9401), (9302, 9402), (9303, 9403))
        models = []
        actions = []
        for data_seed, model_seed in pairs:
            for method, objective in methods:
                models.append(
                    {
                        "method": method,
                        "objective": objective,
                        "model_seed": model_seed,
                        "data_seed": data_seed,
                        "status": "complete",
                    }
                )
                for start in range(20):
                    for action in (-1.0, 0.0, 1.0):
                        actions.append(
                            {
                                "method": method,
                                "objective": objective,
                                "model_seed": model_seed,
                                "data_seed": data_seed,
                                "start_id": f"{data_seed}-{start}",
                                "action": action,
                                "actual_position": .1,
                                "actual_velocity": .01,
                                "predicted_position_raw": .1,
                                "predicted_velocity_raw": .01,
                            }
                        )
            for method, objective in (("exact", "simulator_oracle"), ("handcrafted", "visual_state_plus_oracle")):
                for start in range(20):
                    for action in (-1.0, 0.0, 1.0):
                        actions.append(
                            {
                                "method": method,
                                "objective": objective,
                                "model_seed": 0,
                                "data_seed": data_seed,
                                "start_id": f"{data_seed}-{start}",
                                "action": action,
                                "actual_position": .1,
                                "actual_velocity": .01,
                                "predicted_position_raw": .1,
                                "predicted_velocity_raw": .01,
                            }
                        )
        validate_experiment_completeness(models, actions)

    def test_evidence_manifest_detects_changed_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence.csv"
            evidence.write_text("a,b\n1,2\n", encoding="utf-8")
            import hashlib

            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "files": [
                            {
                                "path": "evidence.csv",
                                "size_bytes": evidence.stat().st_size,
                                "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            validate_file_manifest(manifest, root)
            evidence.write_text("a,b\n3,4\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "size changed|hash changed"):
                validate_file_manifest(manifest, root)

    def test_public_figures_regenerate_deterministically_from_retained_records(self) -> None:
        required = (
            figure_analyzer.V1_ACTIONS,
            figure_analyzer.V2_ACTIONS,
            figure_analyzer.V1_AUDIT,
            figure_analyzer.V2_MODELS,
        )
        if not all(path.is_file() for path in required):
            self.skipTest("retained v1/v2 records are not present in this checkout")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = figure_analyzer.verify_regeneration(root / "first", root / "second")
            self.assertEqual(result["status"], "pass")
            self.assertGreaterEqual(result["figure_count"], 5)


if __name__ == "__main__":
    unittest.main()
