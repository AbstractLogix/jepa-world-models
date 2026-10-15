# Quickstart: LeWAM Push-T Checkpoint Reproduction

## Requirements

- Linux x86-64 and Python 3.11, as used by the official installer.
- The official LeWAM environment and its pinned research dependencies.
- About 13.8 GB for the compressed Push-T archive and 43.1 GiB for the expanded HDF5 file.
- A released checkpoint and network access for the first download.
- A local CPU or supported NVIDIA GPU. The pinned evaluator has a CPU fallback; check it with a short preflight before starting the full protocol.

## Run

1. Clone `https://github.com/MinghaoFu/lewam` into a temporary directory and check out commit `60732e25ba172b117d0d0a476d3a9dcd5d30c03b`.
2. Create an isolated research environment using the official `install.sh` instructions. Do not change the tutorial project's dependency files.
3. Set `STABLEWM_HOME` to an external directory. Download model revision `d8e7039c6e3d8eaec320ba0d0122a5e9c9ce23c2` and dataset revision `63346c94f4dbe62b4744261c6c99d63cbff3f6e6`.
4. Verify the checkpoint and dataset hashes listed in `research.md` before evaluation.
5. Run the official Push-T evaluator for seeds 42, 0, and 1, with 50 episodes per run, for each mode:
   - Reactive policy: `eval.mode=lewam_policy`.
   - Best-of-32 planner: `eval.mode=lewam_plan eval.plan_mode=best_of_k`.
   - Gradient planner: `eval.mode=lewam_plan eval.plan_mode=grad eval.grad_all_k=true eval.grad_tr=0.01`.
6. Preserve raw episode results and compare each mode with the official paper values in `docs/reproductions/lewam-pusht.md`.

The official guide and fixed task settings are at [`docs/REPRODUCE.md`](https://github.com/MinghaoFu/lewam/blob/60732e25ba172b117d0d0a476d3a9dcd5d30c03b/docs/REPRODUCE.md). The evaluation samples from training data; it is not a held-out generalization test.
