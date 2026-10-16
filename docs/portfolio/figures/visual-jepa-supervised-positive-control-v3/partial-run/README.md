# v3 Observed Positive-Control Figures

## figure-a-observed-action-fidelity

![Six panels show position and velocity changes across actions minus one, zero, and plus one for each of three validation data seeds. Solid lines show exact and ridge predictions. Dashed lines show simulator effects. Whiskers show the min–max range across 20 contexts.](figure-a-observed-action-fidelity.svg)

**Caption:** Simulator, handcrafted state plus exact dynamics, and handcrafted ridge action effects. The ridge uses the same handcrafted state features but a fixed linear map.

**Sample size:** 20 shared validation action contexts for each of 3 independent data seeds

**Limitations:** No visual neural fit completed. This figure measures one-step prediction only and does not show planning performance.

## figure-b-observed-positive-controls

![The first two panels show position and velocity one-step MAE for exact dynamics and handcrafted ridge across three seeds. The third panel shows next-state training variance ratios from 138.1 to 142.0. The title states that no neural fit completed.](figure-b-observed-positive-controls.svg)

**Caption:** Raw validation MAE for handcrafted state plus exact dynamics and the fixed handcrafted-feature ridge model, alongside the training next-state position-to-velocity variance ratios.

**Sample size:** Full validation partition per seed for MAE; 20 training episodes per seed for target scales

**Limitations:** The ridge model is a fixed linear baseline and cannot establish what a learned visual encoder can achieve. The neural study stopped before training.
