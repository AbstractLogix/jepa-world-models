"""Contract tests for the supervised visual positive-control study v3."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from jepa_tutorial.action_fidelity_v2 import (
    effective_rank_diagnostics,
    fit_stable_probe,
    split_episode_indices,
    stable_seed,
    validate_action_triplet_records,
    validate_rollout_records,
)
from jepa_tutorial.supervised_visual_v3 import (
    METHOD_ABS_RAW,
    METHOD_ABS_STANDARDIZED,
    METHOD_ACTION_SHUFFLE,
    METHOD_RESIDUAL,
    SupervisedVisualDynamics,
    VisualStateEstimator,
    can_open_heldout,
    choose_candidate_method,
    context_tensor,
    dynamics_seed_gate,
    fit_target_standardizer,
    residual_state,
    residual_target,
    shuffle_training_actions,
    standardize_targets,
    unstandardize_targets,
    validate_method_run_coverage,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_visual_jepa_supervised_positive_control_v3 as runner_v3  # noqa: E402
from analyze_visual_jepa_supervised_positive_control_v3 import generate_figures  # noqa: E402
from run_visual_jepa_supervised_positive_control_v3 import (  # noqa: E402
    StudyError,
    _clip_rollout_state,
)
from validate_visual_jepa_supervised_positive_control_v3 import validate_figures, validate_manifest  # noqa: E402


class VisualPositiveControlV3Tests(unittest.TestCase):
    def test_context_keeps_frame_order_and_uses_zero_one_scale(self) -> None:
        frames = np.zeros((2, 3, 48, 128, 3), dtype=np.uint8)
        frames[:, 0, ..., 0] = 64
        frames[:, 1, ..., 0] = 192

        result = context_tensor(frames, device="cpu")

        self.assertEqual(tuple(result.shape), (2, 6, 48, 128))
        self.assertAlmostEqual(float(result[0, 0, 0, 0]), 64 / 255, places=6)
        self.assertAlmostEqual(float(result[0, 3, 0, 0]), 192 / 255, places=6)

    def test_state_estimator_and_dynamics_shapes(self) -> None:
        context = torch.zeros((2, 6, 48, 128), dtype=torch.float32)
        actions = torch.tensor([-1.0, 1.0])

        self.assertEqual(tuple(VisualStateEstimator()(context).shape), (2, 2))
        self.assertEqual(tuple(SupervisedVisualDynamics()(context, actions).shape), (2, 2))

    def test_supervised_dynamics_responds_to_action_input(self) -> None:
        model = SupervisedVisualDynamics()
        with torch.no_grad():
            for parameter in model.encoder.parameters():
                parameter.zero_()
            for parameter in model.predictor.parameters():
                parameter.zero_()
            model.predictor[0].weight[0, -1] = 1.0
            model.predictor[2].weight[0, 0] = 1.0
        context = torch.zeros((3, 6, 48, 128), dtype=torch.float32)

        prediction = model(context, torch.tensor([-1.0, 0.0, 1.0]))

        self.assertGreater(float((prediction[2, 0] - prediction[0, 0]).detach()), 0.5)
        self.assertTrue(torch.isfinite(prediction).all())

    def test_episode_split_is_disjoint_and_by_episode(self) -> None:
        episode_ids = np.asarray([0, 0, 1, 15, 16, 16, 19], dtype=np.int16)
        train, validation = split_episode_indices(episode_ids, train_episode_split=16)

        self.assertEqual(set(episode_ids[train]), {0, 1, 15})
        self.assertEqual(set(episode_ids[validation]), {16, 19})
        self.assertFalse(set(train).intersection(validation))

    def test_target_standardizer_is_invertible_and_uses_train_values(self) -> None:
        train = np.asarray([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]], dtype=np.float32)
        validation = np.asarray([[100.0, -100.0]], dtype=np.float32)
        mean, scale = fit_target_standardizer(train)
        normalized = standardize_targets(train, mean, scale)

        np.testing.assert_allclose(mean, [3.0, 14.0])
        np.testing.assert_allclose(scale, [np.std(train[:, 0]), np.std(train[:, 1])])
        np.testing.assert_allclose(unstandardize_targets(normalized, mean, scale), train)
        self.assertFalse(np.allclose(mean, validation.mean(axis=0)))

    def test_residual_target_and_state_reconstruction(self) -> None:
        next_state = np.asarray([[0.2, -0.01]], dtype=np.float32)
        visual_state = np.asarray([[0.19, -0.012]], dtype=np.float32)
        residual = residual_target(next_state, visual_state)

        np.testing.assert_allclose(residual, [[0.01, 0.002]], atol=1e-8)
        np.testing.assert_allclose(residual_state(visual_state, residual), next_state)

    def test_training_action_shuffle_is_deterministic_and_preserves_values(self) -> None:
        actions = np.asarray([-0.9, -0.1, 0.2, 0.8], dtype=np.float32)

        first = shuffle_training_actions(actions, seed=1234)
        second = shuffle_training_actions(actions, seed=1234)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.sort(first), np.sort(actions))
        self.assertFalse(np.array_equal(first, actions))

    def test_dynamics_gate_checks_each_coordinate_and_clipping(self) -> None:
        record = {
            "finite": True,
            "position_mae": 0.01,
            "velocity_mae": 0.002,
            "position_median_normalized_contrast_error": 0.25,
            "velocity_median_normalized_contrast_error": 0.24,
            "position_order_accuracy": 0.8,
            "velocity_order_accuracy": 0.85,
            "position_valid_starts": 12,
            "velocity_valid_starts": 20,
            "unbounded_prediction_fraction": 0.05,
            "low_variance_dims": 30,
            "effective_rank": 2.0,
            "position_probe_r2": 0.8,
            "velocity_probe_r2": 0.8,
        }

        self.assertTrue(dynamics_seed_gate(record)["validation_pass"])
        self.assertFalse(dynamics_seed_gate(dict(record, velocity_mae=0.0021))["validation_pass"])
        self.assertFalse(dynamics_seed_gate(dict(record, unbounded_prediction_fraction=0.051))["validation_pass"])
        self.assertFalse(dynamics_seed_gate(dict(record, effective_rank=1.99))["representation_pass"])
        self.assertFalse(dynamics_seed_gate(dict(record, velocity_probe_r2=0.79))["validation_pass"])

    def test_candidate_selection_requires_two_seeds_and_valid_shuffle_control(self) -> None:
        records = []
        for method in (METHOD_ABS_RAW, METHOD_ABS_STANDARDIZED, METHOD_RESIDUAL):
            for seed, passed in ((9801, True), (9802, True), (9803, False)):
                records.append(
                    {
                        "method": method,
                        "model_seed": seed,
                        "validation_pass": passed,
                        "mean_normalized_contrast_error": 0.2 if method == METHOD_ABS_STANDARDIZED else 0.3,
                        "normalized_state_error": 1.0,
                    }
                )
        records.extend(
            {
                "method": METHOD_ACTION_SHUFFLE,
                "model_seed": seed,
                "validation_pass": False,
                "action_fidelity_pass": False,
                "mean_normalized_contrast_error": 0.9,
                "normalized_state_error": 3.0,
            }
            for seed in (9801, 9802, 9803)
        )

        decision = choose_candidate_method(records, data_path_pass=True, records_complete=True)

        self.assertEqual(decision["selected_method"], METHOD_ABS_STANDARDIZED)
        self.assertTrue(decision["action_shuffle_valid"])
        self.assertTrue(decision["heldout_open"])

    def test_heldout_is_fail_closed(self) -> None:
        passing = {
            "data_path_pass": True,
            "records_complete": True,
            "selected_method": METHOD_ABS_RAW,
            "action_shuffle_valid": True,
            "heldout_open": True,
        }
        self.assertTrue(can_open_heldout(passing))
        for field, value in (
            ("data_path_pass", False),
            ("records_complete", False),
            ("selected_method", None),
            ("action_shuffle_valid", False),
            ("heldout_open", False),
        ):
            self.assertFalse(can_open_heldout(dict(passing, **{field: value})))

    def test_heldout_runner_does_not_create_test_data_when_gate_is_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            (evidence / "run-summary.json").write_text(
                json.dumps(
                    {
                        "data_path_pass": True,
                        "records_complete": True,
                        "action_shuffle_valid": False,
                        "selected_method": None,
                        "heldout_open": False,
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(runner_v3, "verify_protocol_lock"),
                patch.object(runner_v3, "EVIDENCE_DIR", evidence),
                patch.object(
                    runner_v3,
                    "_collect_test_contexts",
                    side_effect=AssertionError("test data access must stay closed"),
                ),
            ):
                with self.assertRaises(StudyError):
                    runner_v3.run_heldout(Path("/mnt/e/codex-research/jepa-world-models-tutorial/test"))
            self.assertFalse((evidence / "heldout-starts.csv").exists())

    def test_near_constant_embedding_is_detected(self) -> None:
        diagnostics = effective_rank_diagnostics(np.ones((20, 32), dtype=np.float32))

        self.assertTrue(diagnostics["finite"])
        self.assertEqual(diagnostics["low_variance_dims"], 32)
        self.assertEqual(diagnostics["effective_rank"], 0.0)

    def test_non_finite_embedding_is_rejected_by_diagnostic(self) -> None:
        values = np.zeros((4, 2), dtype=np.float32)
        values[0, 0] = np.nan

        diagnostics = effective_rank_diagnostics(values)

        self.assertFalse(diagnostics["finite"])

    def test_action_triplet_requires_all_three_actions(self) -> None:
        rows = [
            {
                "method": "m",
                "objective": "o",
                "model_seed": 1,
                "data_seed": 2,
                "start_id": "s",
                "action": action,
                "actual_position": 0.1,
                "actual_velocity": 0.01,
                "predicted_position_raw": 0.1,
                "predicted_velocity_raw": 0.01,
            }
            for action in (-1.0, 0.0, 1.0)
        ]
        validate_action_triplet_records(rows)
        with self.assertRaises(ValueError):
            validate_action_triplet_records(rows[:-1])

    def test_rollout_record_requires_failure_reason(self) -> None:
        row = {
            "method": "m",
            "model_seed": 1,
            "start_id": "s",
            "horizon": 1,
            "actual_position": 0.1,
            "actual_velocity": 0.01,
            "predicted_position": 0.1,
            "predicted_velocity": 0.01,
            "status": "complete",
        }
        validate_rollout_records([row])
        with self.assertRaises(ValueError):
            validate_rollout_records([{**row, "status": "failed"}])

    def test_probe_stays_finite_with_constant_features(self) -> None:
        train_x = np.column_stack((np.linspace(-1, 1, 30), np.ones(30))).astype(np.float32)
        train_y = np.column_stack((train_x[:, 0], 0.5 * train_x[:, 0])).astype(np.float32)
        result = fit_stable_probe(train_x, train_y, train_x[:5], train_y[:5])

        self.assertTrue(np.isfinite(result["gram_condition"]))
        self.assertTrue(np.isfinite(result["standardized_coef_l2"]))
        self.assertEqual(result["minimum_observed_feature_scale"], 0.0)

    def test_predicted_state_clipping_is_reported(self) -> None:
        clipped, changed = _clip_rollout_state(np.asarray([-1.3, 0.08], dtype=np.float32))

        self.assertTrue(changed)
        np.testing.assert_allclose(clipped, [-1.2, 0.0], atol=1e-6)

    def test_seed_derivation_is_stable(self) -> None:
        self.assertEqual(stable_seed("v3", "pair", 9701), stable_seed("v3", "pair", 9701))

    def test_run_coverage_rejects_missing_seed_pair(self) -> None:
        rows = [
            {"method": method, "data_seed": data_seed, "model_seed": model_seed}
            for data_seed, model_seed in zip((9701, 9702, 9703), (9801, 9802, 9803), strict=True)
            for method in (
                "visual_state_estimator",
                "supervised_visual_absolute_raw",
                "supervised_visual_absolute_standardized",
                "supervised_visual_residual",
                "supervised_visual_action_shuffle",
            )
        ]
        rows.extend(
            {"method": method, "data_seed": data_seed, "model_seed": 0}
            for data_seed in (9701, 9702, 9703)
            for method in ("handcrafted_feature_ridge", "handcrafted_state_exact_dynamics")
        )
        validate_method_run_coverage(rows)
        with self.assertRaises(ValueError):
            validate_method_run_coverage(rows[:-1])

    def test_evidence_manifest_detects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root / "record.csv"
            record.write_text("value\n1\n", encoding="utf-8")
            digest = hashlib.sha256(record.read_bytes()).hexdigest()
            (root / "sha256-manifest.json").write_text(
                json.dumps(
                    {
                        "protocol_version": "visual-jepa-supervised-positive-control-v3",
                        "files": {
                            "record.csv": {"bytes": record.stat().st_size, "sha256": digest}
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(validate_manifest(root), 1)
            record.write_text("value\n2\n", encoding="utf-8")
            with self.assertRaises(StudyError):
                validate_manifest(root)

    def test_figure_regeneration_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence"
            first_output = root / "figures-a"
            second_output = root / "figures-b"
            evidence.mkdir()
            _write_v3_figure_fixture(evidence)

            first = generate_figures(evidence_dir=evidence, output_dir=first_output)
            second = generate_figures(evidence_dir=evidence, output_dir=second_output)

            first_hashes = [row["sha256"] for row in first["figures"]]
            second_hashes = [row["sha256"] for row in second["figures"]]
            self.assertEqual(first_hashes, second_hashes)
            self.assertGreater(validate_figures(first_output, evidence_dir=evidence), 0)


def _write_v3_figure_fixture(evidence: Path) -> None:
    methods = (
        "handcrafted_state_exact_dynamics",
        "handcrafted_feature_ridge",
        "visual_state_estimator",
        "supervised_visual_absolute_raw",
        "supervised_visual_absolute_standardized",
        "supervised_visual_residual",
        "supervised_visual_action_shuffle",
    )
    data_seeds = ("9701", "9702", "9703")
    model_seeds = ("9801", "9802", "9803")
    action_methods = tuple(method for method in methods if method != "visual_state_estimator")
    triplets = []
    action_summaries = []
    for method in action_methods:
        for index, (data_seed, model_seed) in enumerate(zip(data_seeds, model_seeds, strict=True)):
            if method in ("handcrafted_state_exact_dynamics", "handcrafted_feature_ridge"):
                model_seed = "0"
            start_id = f"d{data_seed}-row1"
            for action in (-1.0, 0.0, 1.0):
                actual_position = 0.2 + 0.001 * action
                actual_velocity = 0.01 + 0.0005 * action
                triplets.append(
                    {
                        "stage": "validation",
                        "method": method,
                        "model_seed": model_seed,
                        "data_seed": data_seed,
                        "start_id": start_id,
                        "action": action,
                        "actual_position": actual_position,
                        "actual_velocity": actual_velocity,
                        "predicted_position_raw": actual_position * (1.0 + 0.01 * index),
                        "predicted_velocity_raw": actual_velocity * (1.0 + 0.01 * index),
                    }
                )
            action_summaries.append(
                {
                    "stage": "validation",
                    "method": method,
                    "model_seed": model_seed,
                    "data_seed": data_seed,
                    "start_id": start_id,
                    "position_normalized_contrast_error": 0.1,
                    "velocity_normalized_contrast_error": 0.1,
                    "position_order_correct": 3,
                    "position_order_total": 3,
                    "position_order_accuracy": 1.0,
                    "velocity_order_correct": 3,
                    "velocity_order_total": 3,
                    "velocity_order_accuracy": 1.0,
                    "position_clipped_fraction": 0.0,
                    "velocity_clipped_fraction": 0.0,
                }
            )
    runs = []
    for method in methods:
        for data_seed, model_seed in zip(data_seeds, model_seeds, strict=True):
            runs.append(
                {
                    "method": method,
                    "model_seed": "0" if method in ("handcrafted_state_exact_dynamics", "handcrafted_feature_ridge") else model_seed,
                    "data_seed": data_seed,
                    "status": "complete",
                    "position_mae": 0.005,
                    "velocity_mae": 0.001,
                    "effective_rank": 2.5,
                }
            )
    gates = []
    for method in action_methods[2:]:
        for data_seed, model_seed in zip(data_seeds, model_seeds, strict=True):
            gates.append(
                {
                    "method": method,
                    "model_seed": model_seed,
                    "data_seed": data_seed,
                    "position_median_normalized_contrast_error": 0.1,
                    "velocity_median_normalized_contrast_error": 0.1,
                    "position_order_accuracy": 1.0,
                    "velocity_order_accuracy": 1.0,
                    "position_clipped_fraction": 0.0,
                    "velocity_clipped_fraction": 0.0,
                }
            )
    curves = []
    gradients = []
    embeddings = []
    probes = []
    for method in (
        "visual_state_estimator",
        "supervised_visual_absolute_raw",
        "supervised_visual_absolute_standardized",
        "supervised_visual_residual",
        "supervised_visual_action_shuffle",
    ):
        for data_seed, model_seed in zip(data_seeds, model_seeds, strict=True):
            probes.append(
                {
                    "method": method,
                    "data_seed": data_seed,
                    "model_seed": model_seed,
                    "probe_position_r2": 0.9,
                    "probe_velocity_r2": 0.85,
                }
            )
            for dimension in range(32):
                embeddings.append(
                    {
                        "method": method,
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "dimension": dimension,
                        "std": 0.2 if dimension < 3 else 0.0,
                        "low_spread": str(dimension >= 3),
                        "rows": 60,
                    }
                )
            for epoch in (1, 35):
                curves.append(
                    {
                        "method": method,
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "epoch": epoch,
                        "train_position_mse_raw": 0.01 / epoch,
                        "validation_position_mse_raw": 0.02 / epoch,
                        "train_velocity_mse_raw": 0.002 / epoch,
                        "validation_velocity_mse_raw": 0.003 / epoch,
                    }
                )
            for coordinate in ("position", "velocity"):
                for epoch in (1, 5, 10, 15, 20, 25, 30, 35):
                    gradients.append(
                        {
                            "method": method,
                            "data_seed": data_seed,
                            "model_seed": model_seed,
                            "coordinate": coordinate,
                            "epoch": epoch,
                            "encoder_gradient_norm": 0.1 / epoch,
                            "prediction_head_gradient_norm": 0.2 / epoch,
                        }
                    )
    for filename, rows in (
        ("action-triplets.csv", triplets),
        ("action-summary.csv", action_summaries),
        ("model-runs.csv", runs),
        ("validation-gates.csv", gates),
        ("learning-curves.csv", curves),
        ("gradient-diagnostics.csv", gradients),
        ("embedding-diagnostics.csv", embeddings),
        ("linear-probes.csv", probes),
    ):
        path = evidence / filename
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
