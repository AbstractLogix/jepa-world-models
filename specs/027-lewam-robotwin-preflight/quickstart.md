# Quickstart: Review LeWAM RoboTwin Readiness

This is a documentation review. It does not start LeWAM, RoboTwin, or an asset transfer.

1. Read the [preflight report](../../docs/reproductions/lewam-robotwin-preflight.md).
2. Check the pinned [paper](https://arxiv.org/abs/2609.27455) and [official code revision](https://github.com/XuejiFang/LeWAM/tree/290458796c2e0725b71cc191b93531a177f97ca7).
3. Confirm the [model card](https://huggingface.co/XuejiFang/LeWAM) license applies only to the bundled I-JEPA encoder, and check terms for the predictor, code, and simulator assets.
4. Confirm the target host supports RoboTwin GPU simulation and rendering. The current WSL2 host does not.
5. Before a future run, pin the model and code, choose one task and setting, create and hash the official fixed case list, and record the case count and deviations.
6. Do not report that one task reproduces the paper's 50-task average.

## Expected outcome

The current status is `not run`. No source checkout, model, or simulator asset was transferred for this preflight.
