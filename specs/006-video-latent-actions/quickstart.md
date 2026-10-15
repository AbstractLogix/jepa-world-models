# Quickstart: Continuous Video-Derived Latent Actions

## Requirements

- Python 3.10–3.13 and the project environment.
- PyTorch, NumPy, and Matplotlib installed as described in the root README.
- No network access, external dataset, or pretrained checkpoint is required.
- CPU is supported. The run uses a small generated dataset and five seeds per method.

## Run

Start Jupyter from the repository root and run `notebooks/12_video_latent_actions.ipynb` from top to bottom.

## Expected output

- Generated example clips for the training, new-shape, and new-camera cases.
- Frame-feature validity and split counts.
- Five runs for each method, with per-seed one-step and transfer errors.
- Mean, median, standard deviation, and uncertain 95% intervals across seeds.
- Continuous-action range and use summary, plus discrete code usage.
- A local run report that separates this synthetic analogue from the paper's in-the-wild results and lists remaining paper-reproduction gates.
