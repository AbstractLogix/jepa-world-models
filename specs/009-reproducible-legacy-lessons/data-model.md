# Data Model: Reproducible Legacy Lesson Evidence

## Run record

Each notebook's local evidence record contains:

- Notebook name and claim label.
- Python, PyTorch, TorchVision, Gymnasium, NumPy, timm, and pandas versions that apply to the run.
- Device, run duration, and whether public data or weights were fetched or cached.
- Data, model, training, and evaluation seed values.
- Dataset identifier and license, plus model repository, revision, weight hash, and license when pretrained weights are used.
- Measured results and a clear error state if a cell did not finish.
- Limits that prevent the result from supporting a broader or paper-level claim.

## Evaluation episode

Each CartPole comparison episode contains:

- Method name.
- Initial-state seed.
- Episode return.
- Whether the environment terminated or reached its time limit.

The methods in a comparison use the same initial-state seed list. Episode values describe the evaluated policy on those cases. They are not independent training runs.

## Ablation result

Each notebook 08 setting contains its latent dimension, predictor depth, value weight, one training seed, fixed evaluation seed list, and all episode returns. Its episode mean and standard deviation describe those ten episodes only.

## Artifact storage

Dataset archives stay in ignored `notebooks/data/`. Optional model checkpoints stay in ignored `notebooks/checkpoints/`. Executed notebook copies used for evidence collection stay in `/tmp`; only a compact Markdown result report is retained in the repository.
