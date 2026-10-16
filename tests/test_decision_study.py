from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
import torch

from jepa_tutorial.decision_study import (
    PlannerConfig,
    cem_proposal,
    gate_passes,
    make_momentum_prior,
    mountaincar_reward,
    mountaincar_step,
    oracle_gate_is_valid,
    protocol_lock_is_valid,
    stable_seed,
    terminal_potential,
    validate_episode_records,
    validate_intervention_records,
)
from jepa_tutorial.visual_mountaincar import VisualStart
from scripts.run_visual_jepa_decision_study import (
    actual_intervention_return,
    predicted_intervention,
    run_direct_episode,
    scalar_momentum_action,
)


class MountainCarDynamicsTests(unittest.TestCase):
    def test_step_matches_installed_gymnasium_dynamics(self) -> None:
        states = np.asarray(
            [(-0.7, 0.03), (-1.2, -0.02), (0.1, 0.04), (0.449, 0.06)],
            dtype=np.float32,
        )
        actions = (-1.0, 0.0, 0.5, 1.0)
        env = gym.make("MountainCarContinuous-v0", max_episode_steps=300)
        try:
            for state, action in zip(states, actions, strict=True):
                env.reset(seed=17)
                env.unwrapped.state = state.astype(np.float64)
                env.step(np.asarray([action], dtype=np.float32))
                expected = np.asarray(env.unwrapped.state, dtype=np.float32)
                actual = mountaincar_step(state, action)
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)
        finally:
            env.close()

    def test_batch_shapes_rewards_and_terminal_potential_are_finite(self) -> None:
        states = mountaincar_step(np.zeros((4, 2), dtype=np.float32), np.asarray([-2, -1, 0, 2]))
        self.assertEqual(states.shape, (4, 2))
        self.assertTrue(np.isfinite(states).all())
        rewards = mountaincar_reward(states, np.asarray([-2, -1, 0, 2]))
        self.assertTrue(np.isfinite(rewards).all())
        self.assertTrue(np.isfinite(terminal_potential(states)).all())

    def test_rejects_invalid_states_and_actions(self) -> None:
        with self.assertRaises(ValueError):
            mountaincar_step(np.zeros((2, 3)), 0.0)
        with self.assertRaises(ValueError):
            mountaincar_step(np.asarray([np.nan, 0.0]), 0.0)
        with self.assertRaises(ValueError):
            mountaincar_step(np.zeros((2, 2)), np.zeros(3))

    def test_intervention_continuation_handles_single_state_momentum_action(self) -> None:
        frame = np.zeros((48, 128, 3), dtype=np.uint8)
        start = VisualStart(
            start_id="test-start",
            seed=123,
            step=40,
            state=np.asarray([-0.5, 0.01], dtype=np.float32),
            previous_frame=frame,
            current_frame=frame,
        )

        actual_state, actual_return = actual_intervention_return(start, 0.0)
        predicted_state, predicted_return = predicted_intervention(
            "oracle", None, None, start, 0.0, torch.device("cpu")
        )

        self.assertTrue(np.isfinite(actual_state).all())
        self.assertTrue(np.isfinite(actual_return))
        self.assertTrue(np.isfinite(predicted_state).all())
        self.assertTrue(np.isfinite(predicted_return))

    def test_scalar_momentum_helper_and_direct_episode_accept_single_states(self) -> None:
        self.assertEqual(scalar_momentum_action(np.asarray([-0.5, 0.01])), 1.0)
        self.assertEqual(scalar_momentum_action(np.asarray([-0.5, 0.01]).reshape(1, 2)), 1.0)

        row = run_direct_episode(456)
        self.assertEqual(row["status"], "complete")
        self.assertIn(row["success"], (0, 1))


class CEMPlannerTests(unittest.TestCase):
    def test_seeded_search_is_deterministic_bounded_and_counts_transitions(self) -> None:
        config = PlannerConfig(
            horizon_steps=11,
            candidates=12,
            elites=3,
            updates=2,
            score_name="cumulative_return",
            initialization="uniform",
        )

        def score(candidates: np.ndarray) -> np.ndarray:
            return -np.square(candidates - 0.4).sum(axis=1)

        first = cem_proposal(config, score, np.random.default_rng(91))
        second = cem_proposal(config, score, np.random.default_rng(91))
        np.testing.assert_array_equal(first.action_blocks, second.action_blocks)
        self.assertEqual(first.action_blocks.shape, (3,))
        self.assertTrue(np.all(first.action_blocks >= -1.0))
        self.assertTrue(np.all(first.action_blocks <= 1.0))
        self.assertEqual(first.transition_calls, 12 * 11 * 2)
        self.assertEqual(first.forward_batches, 22)

    def test_prior_initialization_requires_valid_prior_mean(self) -> None:
        config = PlannerConfig(
            horizon_steps=10,
            candidates=8,
            elites=2,
            updates=1,
            score_name="terminal_potential",
            initialization="momentum_prior",
        )
        def scorer(values: np.ndarray) -> np.ndarray:
            return -np.square(values).sum(axis=1)

        rng = np.random.default_rng(3)
        with self.assertRaisesRegex(ValueError, "requires prior_mean"):
            cem_proposal(config, scorer, rng)
        with self.assertRaisesRegex(ValueError, "length"):
            cem_proposal(config, scorer, rng, prior_mean=np.zeros(1))

    def test_rejects_bad_scores(self) -> None:
        config = PlannerConfig(5, 8, 2, 1, "cumulative_return", "uniform")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            cem_proposal(
                config,
                lambda values: np.full(len(values), np.nan),
                np.random.default_rng(2),
            )

    def test_momentum_prior_uses_bounded_model_rollouts(self) -> None:
        config = PlannerConfig(11, 8, 2, 1, "cumulative_return", "momentum_prior")
        seen: list[float] = []

        def step(state: np.ndarray, action: np.ndarray) -> np.ndarray:
            seen.extend(action.tolist())
            return mountaincar_step(state, action)

        prior, calls = make_momentum_prior(np.asarray([-0.5, 0.0]), config, step)
        self.assertEqual(prior.shape, (3,))
        self.assertTrue(np.all(np.abs(prior) <= 1.0))
        self.assertEqual(calls, 11)
        self.assertEqual(len(seen), 11)

    def test_planner_config_rejects_invalid_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "elites"):
            PlannerConfig(10, 8, 8, 2, "cumulative_return", "uniform")


class StudyGateAndEvidenceTests(unittest.TestCase):
    def test_seed_derivation_is_stable_and_role_sensitive(self) -> None:
        self.assertEqual(stable_seed("model", 1), stable_seed("model", 1))
        self.assertNotEqual(stable_seed("model", 1), stable_seed("data", 1))

    def test_gate_threshold(self) -> None:
        self.assertTrue(gate_passes(18, 20, 18))
        self.assertFalse(gate_passes(17, 20, 18))
        with self.assertRaises(ValueError):
            gate_passes(21, 20, 18)

    def test_protocol_and_oracle_gates_fail_closed_on_wrong_hashes(self) -> None:
        lock = {
            "schema_version": 1,
            "experiment_version": "visual-jepa-decision-study-v1",
            "status": "locked",
            "preregistration_hash": "pre",
            "code_hash": "code",
            "planner_config": {"horizon_steps": 120},
        }
        self.assertTrue(protocol_lock_is_valid(lock, preregistration_hash="pre", code_hash="code"))
        self.assertFalse(protocol_lock_is_valid(lock, preregistration_hash="other", code_hash="code"))
        gate = {
            "schema_version": 1,
            "experiment_version": "visual-jepa-decision-study-v1",
            "status": "passed",
            "protocol_lock_hash": "lock",
            "total": 20,
            "successes": 18,
        }
        self.assertTrue(oracle_gate_is_valid(gate, protocol_lock_hash="lock"))
        self.assertFalse(oracle_gate_is_valid(gate, protocol_lock_hash="wrong"))
        malformed_gate = {**gate, "successes": "not-a-count"}
        self.assertFalse(oracle_gate_is_valid(malformed_gate, protocol_lock_hash="lock"))

    def test_episode_records_preserve_valid_failures_and_reject_duplicates(self) -> None:
        base = {
            "method": "direct_visual_momentum",
            "model_seed": "",
            "eval_seed": 8700,
            "initialization": "direct",
            "privileged_state": False,
            "status": "complete",
            "success": 1,
            "return": 99.0,
            "steps": 100,
            "transition_calls": 0,
            "forward_batches": 0,
            "planning_seconds": 0.0,
            "elapsed_seconds": 0.1,
            "device": "cpu",
            "error": "",
        }
        validate_episode_records([base])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_episode_records([base, base])
        failed = {**base, "status": "error", "error": "model failed"}
        validate_episode_records([failed])

    def test_intervention_rows_require_paired_actions_and_finite_outcomes(self) -> None:
        rows = []
        for action in (-1.0, 0.0, 1.0):
            rows.append(
                {
                    "method": "jepa",
                    "model_seed": 7401,
                    "start_id": "8600-40",
                    "action": action,
                    "status": "complete",
                    "error": "",
                    "actual_position": -0.5,
                    "actual_velocity": 0.01,
                    "predicted_position": -0.49,
                    "predicted_velocity": 0.02,
                    "observed_return": 90.0,
                    "predicted_return": 89.0,
                }
            )
        validate_intervention_records(rows)
        with self.assertRaisesRegex(ValueError, "actions"):
            validate_intervention_records(rows[:2])
        rows[0]["observed_return"] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_intervention_records(rows)


if __name__ == "__main__":
    unittest.main()
