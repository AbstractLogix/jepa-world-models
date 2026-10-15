# Quickstart: Action-Conditioned Video Planning

## Prerequisites

- Python 3.10–3.13
- The repository's default dependencies: NumPy, Gymnasium, Pillow, scikit-learn, and Matplotlib
- A CPU; no GPU, model checkpoint, or dataset download is needed

## Run

1. Install the project dependencies by following the repository README.
2. Start Jupyter from the project environment.
3. Open `notebooks/16_action_conditioned_video_planning.ipynb`.
4. Run all cells in order.

## Expected evidence

- The renderer's color-moment encoder returns finite car and goal coordinates.
- The notebook prints package versions and separate training, intervention, and evaluation seed ranges.
- The notebook reports held-out prediction error and action-intervention error.
- The notebook reports paired outcomes for uniform-start CEM, controller-prior-start CEM, and the direct controller.
- The notebook displays an illustrative closed-loop episode replay with sampled frames and a position trace; this replay is outside the paired evaluation metrics.
- The notebook writes completed or stopped episodes to `docs/evidence/action-video-planning-episodes.csv` and all held-out action interventions to `docs/evidence/action-video-planning-interventions.csv`.
- The run report distinguishes local simulator evidence from the V-JEPA 2-AC and PiJEPA paper results.

## Limits

The local fixed image-moment features are not pretrained JEPA features. The direct controller is not a trained policy. The local CEM comparison is not the papers' MPPI or robot evaluation.
