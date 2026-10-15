# Quickstart: Planning Limits Lesson

## Prerequisites

- Python 3.10 through 3.13.
- The project environment with NumPy, Matplotlib, Gymnasium, and Jupyter installed.
- CPU. No download, checkpoint, or robot is needed.
- Each episode stops at success or 300 environment steps.
- Success means reaching the goal position with non-negative velocity.

## Run

1. Open `notebooks/10_planning_limits.ipynb` in JupyterLab.
2. Select **Run All**.
3. Confirm that the transition check passes before the evaluation starts.
4. Read the paper results and local results in their separate sections.

## Expected output

- A fixed-seed evaluation table and plots for goal position, rollout horizon, and planner mode. Subgoal MPC uses a same-seed momentum-reference trajectory with a target five steps ahead.
- Success rate with a Wilson 95% interval, successful-episode step count, and planning calls.
- The Python and Gymnasium versions, hardware type, seed list, and CEM settings.
- An explicit note that the result is a MountainCar teaching analogue, not a Meta-World or robot reproduction.

Failed episodes must appear in the tables. The notebook must not replace failures with successful-only measurements.
