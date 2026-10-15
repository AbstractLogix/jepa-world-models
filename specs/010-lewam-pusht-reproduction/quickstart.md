# Quickstart: LeWAM Push-T Checkpoint Reproduction

## Requirements

- Linux x86-64 and Python 3.11, as used by the official installer.
- The official LeWAM environment and its pinned research dependencies.
- About 13.8 GB for the compressed Push-T archive and 43.1 GiB for the expanded HDF5 file.
- A released checkpoint and network access for the first download.
- A local CPU or supported NVIDIA GPU. The pinned evaluator has a CPU fallback; check it with a short preflight before starting the full protocol.

## Run

1. Choose a run root on E: or D:. For WSL, `E:\codex-research\jepa-lewam-pusht` maps to `/mnt/e/codex-research/jepa-lewam-pusht`. For D:, use the matching `/mnt/d/...` path. Example:

   ```bash
   export LEWAM_RUN_ROOT=/mnt/e/codex-research/jepa-lewam-pusht
   export STABLEWM_HOME="$LEWAM_RUN_ROOT"
   export CONDA_ENVS_PATH="$LEWAM_RUN_ROOT/conda-envs"
   export CONDA_PKGS_DIRS="$LEWAM_RUN_ROOT/conda-pkgs"
   export PIP_CACHE_DIR="$LEWAM_RUN_ROOT/cache/pip"
   export HF_HOME="$LEWAM_RUN_ROOT/cache/huggingface"
   export XDG_CACHE_HOME="$LEWAM_RUN_ROOT/cache/xdg"
   export TMPDIR="$LEWAM_RUN_ROOT/tmp"
   mkdir -p "$LEWAM_RUN_ROOT" "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" \
       "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"
   ```

   Keep all these paths on the selected drive. The Conda executable itself may already be installed elsewhere; this setup puts the new environment, packages, caches, data, checkpoint, and outputs on E: or D:.

   `CONDA_ENVS_PATH` and `CONDA_PKGS_DIRS` set the Conda environment and package locations. Keep both on the same drive so Conda can link packages where supported. See [Conda's custom environment and package locations](https://docs.conda.io/projects/conda/en/stable/user-guide/configuration/custom-env-and-pkg-locations.html).
2. Clone `https://github.com/MinghaoFu/lewam` into `$LEWAM_RUN_ROOT/code` and check out commit `60732e25ba172b117d0d0a476d3a9dcd5d30c03b`.
3. Create the isolated research environment with the pinned checkout's official `install.sh`, using the Conda environment path above. Do not change the tutorial project's dependency files.
4. Download model revision `d8e7039c6e3d8eaec320ba0d0122a5e9c9ce23c2` under `$STABLEWM_HOME/checkpoints` and dataset revision `63346c94f4dbe62b4744261c6c99d63cbff3f6e6` under `$STABLEWM_HOME/datasets`.
5. Verify the checkpoint and dataset hashes listed in `research.md` before evaluation.
6. Run the official Push-T evaluator for seeds 42, 0, and 1, with 50 episodes per run, for each mode:
   - Reactive policy: `eval.mode=lewam_policy`.
   - Best-of-32 planner: `eval.mode=lewam_plan eval.plan_mode=best_of_k`.
   - Gradient planner: `eval.mode=lewam_plan eval.plan_mode=grad eval.grad_all_k=true eval.grad_tr=0.01`.
7. Preserve raw episode results under `$STABLEWM_HOME/eval/` and compare each mode with the official paper values in `docs/reproductions/lewam-pusht.md`.

The official guide and fixed task settings are at [`docs/REPRODUCE.md`](https://github.com/MinghaoFu/lewam/blob/60732e25ba172b117d0d0a476d3a9dcd5d30c03b/docs/REPRODUCE.md). The evaluation samples from training data; it is not a held-out generalization test.
