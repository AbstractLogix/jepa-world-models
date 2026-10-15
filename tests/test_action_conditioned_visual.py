from __future__ import annotations

import unittest
import warnings
from unittest.mock import patch

import torch

from jepa_tutorial.action_conditioned_visual import (
    ActionConditionedJEPA,
    ImageReconstructionDynamics,
    StateDynamics,
    embedding_collapse_diagnostics,
    select_device,
)
from jepa_tutorial.evidence_validation import validate_intervention_records
from jepa_tutorial.visreg_style import visreg_style_regularization_loss


def _complete_intervention_record(method: str, model_seed: int | str, action: float) -> dict[str, object]:
    record: dict[str, object] = {
        "method": method,
        "prediction_space": {
            "oracle": "simulator_state",
            "state_model": "simulator_state",
            "image_reconstruction": "pixels_and_latent",
            "jepa": "latent",
        }[method],
        "scoring_space": {
            "oracle": "simulator_state_goal_distance",
            "state_model": "simulator_state_goal_distance",
            "image_reconstruction": "pixel_goal_l1",
            "jepa": "latent_goal_mse",
        }[method],
        "device": "cpu",
        "model_seed": model_seed,
        "start_id": "2900-40",
        "action": action,
        "actual_position": -0.6,
        "observed_score": -1.05,
        "predicted_score": -1.0,
        "status": "complete",
        "error": "",
    }
    if method in {"oracle", "state_model"}:
        record.update(
            predicted_position=-0.55,
            position_abs_error=0.05,
            velocity_abs_error=0.01,
        )
    elif method == "image_reconstruction":
        record.update(pixel_l1_error=0.1, latent_mse=0.02)
    else:
        record["latent_mse"] = 0.02
    return record


def _complete_intervention_rows() -> list[dict[str, object]]:
    rows = []
    for method, model_seed in (
        ("oracle", ""),
        ("state_model", 9021),
        ("image_reconstruction", 9021),
        ("jepa", 9021),
    ):
        rows.extend(
            _complete_intervention_record(method, model_seed, action)
            for action in (-1.0, 0.0, 1.0)
        )
    return rows


class VisualDynamicsTests(unittest.TestCase):
    def test_jepa_predicts_next_context_embedding_from_image_pair_and_action(self) -> None:
        model = ActionConditionedJEPA(embedding_dim=8)
        observations = torch.rand(2, 6, 48, 128)
        actions = torch.tensor([-1.0, 1.0])

        prediction, target = model(observations, actions, observations)

        self.assertEqual(prediction.shape, (2, 8))
        self.assertEqual(target.shape, (2, 8))
        self.assertTrue(torch.isfinite(prediction).all())
        loss = torch.nn.functional.mse_loss(prediction, target)
        self.assertTrue(torch.isfinite(loss))

    def test_reconstruction_model_outputs_future_frame(self) -> None:
        model = ImageReconstructionDynamics(embedding_dim=8)
        observations = torch.rand(2, 6, 96, 256)
        actions = torch.tensor([0.0, 0.5])

        prediction, next_image = model(observations, actions)

        self.assertEqual(prediction.shape, (2, 8))
        self.assertEqual(next_image.shape, (2, 3, 48, 128))
        self.assertTrue(torch.isfinite(next_image).all())

    def test_state_model_requires_position_and_velocity(self) -> None:
        model = StateDynamics()
        state = torch.zeros(3, 2)
        action = torch.zeros(3, 1)

        self.assertEqual(model(state, action).shape, (3, 2))
        with self.assertRaisesRegex(ValueError, "position and velocity"):
            model(torch.zeros(3, 3), action)

    def test_collapse_diagnostics_flag_constant_embeddings(self) -> None:
        constant = torch.ones(4, 8)
        varied = torch.randn(128, 8)

        self.assertTrue(embedding_collapse_diagnostics(constant)["collapsed"])
        self.assertFalse(embedding_collapse_diagnostics(varied)["collapsed"])

    def test_device_selection_uses_cpu_when_cuda_is_not_available(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(select_device(), torch.device("cpu"))

    def test_device_selection_falls_back_when_cuda_probe_fails(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.ones", side_effect=RuntimeError("unsupported device")),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            device = select_device()

        self.assertEqual(device, torch.device("cpu"))
        self.assertTrue(any("using CPU" in str(item.message) for item in caught))


class VisRegStyleLossTests(unittest.TestCase):
    def test_returns_finite_components_and_gradients(self) -> None:
        embeddings = torch.randn(16, 8, requires_grad=True)

        total, components = visreg_style_regularization_loss(
            embeddings,
            generator=torch.Generator().manual_seed(17),
        )
        total.backward()

        self.assertEqual(set(components), {"scale_loss", "shape_loss", "center_loss"})
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(all(torch.isfinite(value) for value in components.values()))
        self.assertTrue(torch.isfinite(embeddings.grad).all())

    def test_shape_term_breaks_an_exact_constant_batch(self) -> None:
        embeddings = torch.zeros(16, 8, requires_grad=True)

        _, components = visreg_style_regularization_loss(
            embeddings,
            generator=torch.Generator().manual_seed(23),
        )
        components["shape_loss"].backward()

        self.assertTrue(torch.isfinite(embeddings.grad).all())
        self.assertGreater(float(embeddings.grad.norm()), 0.0)

    def test_seeded_projection_directions_are_reproducible(self) -> None:
        embeddings = torch.randn(12, 5)
        first, first_components = visreg_style_regularization_loss(
            embeddings,
            generator=torch.Generator().manual_seed(29),
        )
        second, second_components = visreg_style_regularization_loss(
            embeddings,
            generator=torch.Generator().manual_seed(29),
        )

        self.assertTrue(torch.equal(first, second))
        for name in first_components:
            self.assertTrue(torch.equal(first_components[name], second_components[name]))

    def test_rejects_invalid_inputs_and_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2 rows"):
            visreg_style_regularization_loss(torch.ones(1, 4))
        with self.assertRaisesRegex(ValueError, "floating-point"):
            visreg_style_regularization_loss(torch.ones(4, 4, dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "finite values"):
            visreg_style_regularization_loss(torch.tensor([[0.0], [float("nan")]]))
        with self.assertRaisesRegex(ValueError, "num_projections"):
            visreg_style_regularization_loss(torch.ones(4, 4), num_projections=0)
        with self.assertRaisesRegex(ValueError, "at least one"):
            visreg_style_regularization_loss(
                torch.ones(4, 4), scale_weight=0, shape_weight=0, center_weight=0
            )


class InterventionEvidenceValidationTests(unittest.TestCase):
    def test_accepts_finite_complete_records_with_paired_actions(self) -> None:
        validate_intervention_records(_complete_intervention_rows())

    def test_accepts_empty_pilot_intervention_records(self) -> None:
        validate_intervention_records([])

    def test_accepts_paired_incomplete_rows_with_status_and_error_reason(self) -> None:
        rows: list[dict[str, object]] = []
        for method, status, error in (
            ("image_reconstruction", "missing_model", ""),
            ("jepa", "error", "inference failed"),
        ):
            for action in (-1.0, 0.0, 1.0):
                rows.append(
                    {
                        "method": method,
                        "prediction_space": {
                            "image_reconstruction": "pixels_and_latent",
                            "jepa": "latent",
                        }[method],
                        "scoring_space": {
                            "image_reconstruction": "pixel_goal_l1",
                            "jepa": "latent_goal_mse",
                        }[method],
                        "device": "cpu",
                        "model_seed": 9021,
                        "start_id": "2900-40",
                        "action": action,
                        "actual_position": -0.6,
                        "observed_score": -1.05,
                        "predicted_score": "",
                        "status": status,
                        "error": error,
                    }
                )

        validate_intervention_records(rows)

    def test_rejects_error_status_without_reason(self) -> None:
        rows = _complete_intervention_rows()
        for row in rows[-3:]:
            row["status"] = "error"
            row["error"] = ""
        with self.assertRaisesRegex(ValueError, "requires a non-empty error description"):
            validate_intervention_records(rows)

    def test_rejects_invalid_method_seed_start_action_status_and_metric(self) -> None:
        invalid_records = (
            ("method", "unknown method", "invalid method"),
            ("model_seed", "", "model_seed must be an integer"),
            ("start_id", "", "start_id must be non-empty"),
            ("action", 0.25, "action must be one of"),
            ("status", "done", "invalid status"),
            ("prediction_space", "unknown", "prediction_space must be"),
            ("scoring_space", "unknown", "scoring_space must be"),
            ("device", "tpu", "device must be 'cpu' or 'cuda'"),
            ("latent_mse", float("nan"), "latent_mse must be finite"),
        )
        for field, value, message in invalid_records:
            with self.subTest(field=field):
                rows = _complete_intervention_rows()
                target = rows[0] if field == "method" else rows[-1]
                target[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_intervention_records(rows)

    def test_rejects_duplicate_cases_and_missing_action_pairs(self) -> None:
        rows = _complete_intervention_rows()
        with self.assertRaisesRegex(ValueError, "duplicate case"):
            validate_intervention_records(rows + [rows[0]])

        with self.assertRaisesRegex(ValueError, "expected actions -1, 0, and 1"):
            validate_intervention_records(rows[:-1])


if __name__ == "__main__":
    unittest.main()
