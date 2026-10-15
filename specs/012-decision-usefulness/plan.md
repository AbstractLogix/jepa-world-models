# Implementation Plan: Decision-Usefulness Evaluation Lesson

**Branch**: `012-decision-usefulness` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

## Summary

Add a runnable MountainCar lesson that fits a small state-transition model from simulator-generated data, measures action-intervention accuracy on held-out states, and compares model-based MPC with a direct momentum policy on paired starts. Report prediction, intervention, closed-loop success, and compute as separate evidence.

## Technical Context

**Language/Version**: Python 3.10–3.13, matching the project  
**Primary Dependencies**: Existing NumPy, scikit-learn, Gymnasium, Matplotlib, and Jupyter dependencies  
**Storage**: Notebook outputs and compact report only; no downloaded data or weights  
**Testing**: Execute the notebook end to end; verify disjoint seed groups, at least 60 intervention outcomes, 20 paired episodes, and complete raw outcomes  
**Target Platform**: Local CPU on Linux, macOS, or Windows  
**Project Type**: Educational notebook and reproduction-style local evidence report  
**Performance Goals**: Finish the evaluation in under 60 CPU seconds for each policy, excluding notebook startup and data collection  
**Constraints**: No new project dependencies; no robot claim; no paper-reproduction claim; keep all 20 evaluation starts paired; give both policy evaluations the same 60-second elapsed-time cap

## Constitution Check

- Teach from evidence: pass. Link the primary evaluation papers and distinguish their claims from local measurements.
- Label teaching examples honestly: pass. The fitted state-space model is an illustrative analogue, not a JEPA model or paper reproduction.
- Make the learning path usable: pass. The lesson uses the existing simulator and project dependencies, with no asset download.
- Measure before making performance claims: pass. Fix data splits, intervention actions, episodes, success rule, and compute caps before final evaluation.
- Keep changes small and reviewable: pass. Add one notebook, one report, one bounded feature, and alignment updates; reuse existing dependencies.

## Research Decisions

1. Follow the decision-making evaluation position paper's separation of counterfactual action fidelity, closed-loop validity, and decision utility. The paper proposes several measures; this lesson uses only the measures needed for a small local example.
2. Follow the benchmark survey's recommendation to make a direct-policy comparison explicit. The survey reports 11 such comparisons among 160 reviewed benchmarks; this is motivation, not a result from this project.
3. Use Gymnasium MountainCarContinuous so the lesson has no external data or hardware requirement and can reuse the project's documented environment.
4. Fit a deterministic polynomial state-transition regressor from random-action simulator episodes with seeds 1000–1019. Keep those seeds separate from intervention seeds 2000–2004 and evaluation seeds 3000–3019.
5. Use the goal position 0.25 and the momentum rule from Notebook 10 as the direct policy. Success means the car reaches the goal with non-negative velocity. Compare it with a learned-model CEM planner using a 10-step horizon, 64 candidates, 8 elites, and 2 iterations on the same 20 evaluation starts and 300-step episode limit.
6. Give both policy evaluations the same 60-second elapsed-time cap on the local CPU. Record actual runtime and model-call counts. If either exceeds the cap, or if the cap cannot be enforced, keep the comparison partial and make no matched-budget claim.
7. Evaluate actions (-1, 0, 1) from 20 held-out states sampled from the intervention seeds. Measure next-state prediction error and pairwise agreement in the utility U(x, v) = -((x - 0.25)^2 + 0.05v^2), excluding and counting ties.
8. Save the report at `docs/reproductions/decision-usefulness.md`; update the curriculum coverage, roadmap, project alignment, and remaining-work records only after the notebook run is recorded.

## Project Structure

### Feature documentation

```text
specs/012-decision-usefulness/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── tasks.md
```

### Delivered lesson and evidence

```text
notebooks/14_decision_usefulness.ipynb
docs/reproductions/decision-usefulness.md
docs/CURRICULUM_COVERAGE.md
docs/ROADMAP.md
specs/001-project-vision/alignment.md
specs/001-project-vision/remaining-work.md
specs/001-project-vision/tasks.md
```

**Structure Decision**: Keep the implementation in one Jupyter lesson. The lesson does not add a general-purpose model or evaluation library.

## Constitution Check After Design

Pass. The design uses existing dependencies and local data, fixes the paired protocol before evaluation, and labels the work as an illustrative state-space analogue.

## Complexity Tracking

No constitution violations.
