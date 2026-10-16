# Action-fidelity study figures

All measurement figures use retained CSV records. SVG files include title and description text; PNG files are rendered at 260 dpi.

## Historical v1 action-effect fidelity

**Alt text:** Action-effect fidelity for Retained Notebook 23 checkpoint replay; pre-clamp probe outputs.. Two rows compare position and velocity changes from action zero. Each panel shows actual simulator changes as dashed lines and model predictions as colored lines for actions minus one, zero, and plus one. The plot uses 100 model-seed and context groups, with every seed shown separately.

Retained Notebook 23 checkpoint replay; pre-clamp probe outputs.. Each point is a mean over matched fixed contexts for one model seed. Changes are measured relative to action zero. Predictions use pre-clip values; simulator trajectories follow MountainCarContinuous-v0. This synthetic environment does not test transfer to physical systems.

## V2 validation action-effect fidelity

**Alt text:** Action-effect fidelity for New validation-only v2 records; All recorded fits completed. Held-out evaluation remains separate.. Two rows compare position and velocity changes from action zero. Each panel shows actual simulator changes as dashed lines and model predictions as colored lines for actions minus one, zero, and plus one. The plot uses 420 model-seed and context groups, with every seed shown separately.

New validation-only v2 records; All recorded fits completed. Held-out evaluation remains separate.. Each point is a mean over matched fixed contexts for one model seed. Changes are measured relative to action zero. Predictions use pre-clip values; simulator trajectories follow MountainCarContinuous-v0. This synthetic environment does not test transfer to physical systems.

## Historical v1 representation health

**Alt text:** Six panels show context and target embedding mean standard deviation, the worse low-spread dimension count, the worse effective rank, and position and velocity probe R-squared. Every completed model seed appears as a labeled point, and failed fits are listed below their method. These are historical v1 checkpoint replays and v1 test records.

Representation health is reported per seed. The low-spread count and effective rank show the worse of the context and target embeddings; low spread uses the per-dimension 1e-3 rule. Effective rank is computed from the validation covariance spectrum. Historical v1 values are derived from retained v1 checkpoints and the original v1 test split.

## V2 validation representation health

**Alt text:** Six panels show context and target embedding mean standard deviation, the worse low-spread dimension count, the worse effective rank, and position and velocity probe R-squared. Every completed model seed appears as a labeled point, and failed fits are listed below their method. These are v2 validation-only records; no held-out data are shown.

Representation health is reported per seed. The low-spread count and effective rank show the worse of the context and target embeddings; low spread uses the per-dimension 1e-3 rule. Effective rank is computed from the validation covariance spectrum. V2 uses the frozen train/validation split and the 20-refit episode-block probe diagnostic.

## V2 validation objective comparison

**Alt text:** Four panels compare effective rank, velocity-probe R-squared, position action-contrast error, and velocity action-contrast error for each v2 objective and the supervised visual control. Every completed seed is shown. The selected objective is marked only when validation gates select one.

Validation-only data are used. Error metrics are lower-is-better; rank and probe R-squared are higher-is-better. A negative or failed seed remains visible. No held-out test data are used for model selection.

## Action-conditioned visual JEPA architecture

**Alt text:** A left-to-right diagram shows rendered frame pairs entering a visual encoder, then an action-conditioned predictor, a predicted latent, a frozen state readout, a planner, and environment feedback. The environment sends new rendered frames back to the encoder. The action enters the predictor.

Architecture diagram for the local synthetic MountainCar study. The state readout is a diagnostic linear probe, not a learned pixel decoder.

## FIGURE_C not generated

held-out stage did not complete or contain supported rollouts

## FIGURE_E not generated

planning gate stayed closed or no planning rows were recorded

## Data and regeneration

The figure generator is `scripts/analyze_visual_jepa_action_fidelity_v2.py` with SHA-256 `740e5772a06d45ae8de7f29479257bd8a3b526bb12df14ec8c9c23c846182bf3`.
Measurement inputs and their SHA-256 values are listed in `figure-manifest.json`.
