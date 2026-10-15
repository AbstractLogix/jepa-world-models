# Implementation Plan: Action-Conditioned Video Planning Lesson

**Branch**: `016-action-video-planning` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

## Summary

Add Notebook 16 as a CPU-sized simulator analogue. It renders Gymnasium MountainCar observations with Pillow, extracts position from image color moments and velocity from two frames, fits an action-conditioned polynomial dynamics model, and compares uniform-start CEM, controller-prior-start CEM, and the controller itself. Use matched held-out starts and report prediction, intervention, closed-loop, and compute evidence. The notebook must say that it is not JEPA and does not reproduce V-JEPA 2-AC or PiJEPA.

## Technical Context

**Language/Version**: Python 3.10–3.13

**Primary Dependencies**: Existing NumPy, Gymnasium, Pillow, scikit-learn, and Matplotlib dependencies; no new project dependency

**Storage**: No runtime store; small checked-in episode and intervention CSVs are summarized in Markdown

**Testing**: Execute all notebook code cells in order in a supported local Python environment; inspect notebook JSON and compare the run record with the committed evidence. No separate test suite is needed for this teaching artifact.

**Target Platform**: Local CPU, Linux or Windows with WSL; no GPU, external dataset, or checkpoint required

**Project Type**: Jupyter notebook lesson with documentation and evidence

**Run Bounds**: Limit data collection to 20 exploration episodes, interventions to 20 held-out states, and closed-loop evaluation to 20 paired starts. Stop either planner after 60 seconds per method and preserve incomplete or missing episodes in the report. This is one fitted model; under the project evaluation protocol, report the result as exploratory and do not present episode intervals as training-seed uncertainty.

**Constraints**: The predictor and planner receive only features derived from rendered frames. Simulator truth is available only to generate frames and score realized outcomes. Both planners use identical candidate counts, horizon, iterations, episode seeds, and step limit. The prior is a simple direct controller, not a learned policy.

**Scale/Scope**: One notebook, one local run report, two evidence CSVs (episodes and interventions), and curriculum / roadmap / reproduction / project-alignment updates

## Constitution Check

- Learner value and measurable acceptance criteria are stated before implementation: **pass**.
- Local simulation, generated observations, and paper results remain separate: **pass**.
- No data or weights are downloaded; external sources and limitations are documented: **pass**.
- Use existing dependencies and a small CPU run; do not add an abstraction or training framework: **pass**.
- Report exact seeds, versions, methods, and incomplete evidence: **pass**.

**Post-design check**: pass. The design adds one self-contained notebook and evidence files. It does not alter package APIs or add project dependencies.

## Design and Decisions

1. **Simulator and rendered observations**: Reuse `MountainCarContinuous-v0`; draw the car and target into deterministic RGB frames with Pillow. This avoids the simulator's optional video/window dependencies.
2. **Visual features**: Locate the red car and green target by channel moments. Estimate velocity from the car's position across two adjacent frames. Keep ground-truth position and velocity out of the predictor and planner.
3. **Predictor**: Fit a degree-three polynomial Ridge model from `[visual_position, estimated_velocity, visual_goal, action]` to the next position and velocity. Report held-out error and the limits of the fixed encoder.
4. **Intervention check**: On 20 held-out states, apply all three discrete evaluation actions in the simulator and model. Report state error and action-ranking agreement separately.
5. **Planner comparison**: Use CEM with equal horizon, candidates, elites, iterations, action bounds, and replanning. One planner starts with a zero mean; the other starts with a sequence from the direct momentum controller. Include that controller as a standalone method.
6. **Evaluation**: Use the same 20 reset seeds for all methods. Report success counts with Wilson intervals, returns, steps, predictor calls, and planning wall time. Preserve every episode row.
7. **Claims**: Compare the local design with V-JEPA 2-AC and PiJEPA in the discussion. Describe algorithmic similarities and explicit differences; make no paper reproduction claim.

## Research Basis

- V-JEPA 2-AC uses action-conditioned latent predictions and image-goal planning; the official implementation has a large model and robot-data pipeline: [paper](https://arxiv.org/abs/2506.09985), [code](https://github.com/facebookresearch/vjepa2).
- PiJEPA uses a learned Octo policy prior to warm-start MPPI over a separate JEPA world model. Its paper is published in the CVPR 2026 workshop; its public implementation and CAST/GPU requirements make a small local analogue the appropriate first lesson: [paper](https://arxiv.org/abs/2603.25981), [code](https://github.com/AmirhoseinCh/PiJEPA).
- V-JEPA Policy and JEPA-WAM are newer examples of world-action models that couple predictive visual latents with action generation. They remain roadmap candidates, not this feature's implementation: [V-JEPA Policy](https://arxiv.org/abs/2609.37250), [JEPA-WAM](https://arxiv.org/abs/2608.09381).

## Project Structure

```text
notebooks/16_action_conditioned_video_planning.ipynb
docs/reproductions/action-video-planning.md
docs/evidence/action-video-planning-episodes.csv
docs/evidence/action-video-planning-interventions.csv
docs/ROADMAP.md
docs/CURRICULUM_COVERAGE.md
docs/REPRODUCTION_MATRIX.md
README.md
CURRICULUM_GUIDE.md
specs/001-project-vision/tasks.md
specs/001-project-vision/alignment.md
specs/001-project-vision/remaining-work.md
specs/016-action-video-planning/
```

**Structure Decision**: Keep implementation in one notebook, as with existing tutorial lessons. Do not add a package module for code used by one lesson. Store the small per-episode and action-intervention evidence; do not store generated frames or models.

## Risks and Mitigations

- **Quantized visual position**: Use smooth subpixel markers and calibrate the color-moment encoder before collecting trajectories.
- **Velocity from frame differencing**: Start each episode from a rendered reset frame and pair adjacent rendered frames; include reset and boundary cases in the check.
- **Model exploitation**: Measure held-out action interventions and score planner behavior in the real simulator, not in imagined rollouts.
- **Short-horizon bias**: State the horizon and target utility. Keep paper-specific horizon claims out of the local result.
- **Method cost mismatch**: Fix candidate budgets across planners and report direct-policy cost separately rather than implying equal compute.

## Complexity Tracking

No constitution violations.
