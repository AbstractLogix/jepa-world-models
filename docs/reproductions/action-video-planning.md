# Action-Conditioned Video Planning: Local Run

**Run date**: 2026-10-15  
**Status**: Completed one-fit exploratory simulator analogue; no paper reproduction claim

## Scope

Notebook 16 links rendered MountainCar observations, action-conditioned prediction, and receding-horizon planning. It uses a fixed color-moment feature extractor and a degree-three polynomial Ridge model. One fitted model was evaluated, so this is an exploratory result under the project [evaluation protocol](../EVALUATION_PROTOCOL.md), not an estimate across training runs. It does not load JEPA weights, use public robot data, or reproduce V-JEPA 2-AC or PiJEPA.

The run used `MountainCarContinuous-v0`, goal position 0.45, and a 300-step episode limit. The renderer drew frames locally with Pillow. The planner only received position and velocity estimated from two rendered frames and the target position estimated from the current frame. It did not receive Gymnasium's hidden state.

## Run conditions

- Python 3.13.14; NumPy 2.5.4; Gymnasium 0.29.1; scikit-learn 1.9.1; Pillow 12.3.0; Matplotlib 3.11.2.
- Linux WSL2, kernel `6.18.54.1-microsoft-standard-WSL2`, `x86_64`; CPU only.
- Training: 20 random-action episodes, seeds 1000–1019, 6,000 transitions.
- Interventions: 20 held-out states from seeds 2000–2004, with actions -1, 0, and 1 (60 cases).
- Closed-loop evaluation: seeds 3000–3019, paired across all methods.
- The notebook also replays the momentum-prior method on seed 3000 to display sampled rendered frames and the estimated position trace. This teaching figure is outside the 20-start summary and CSV.
- CEM: horizon 10, 24 candidates, 4 elites, 2 updates, replan after each observed simulator step; 60-second total cap per method. No method hit its cap.
- Neutral-start CEM and momentum-prior CEM used the same candidate budget. The momentum prior is the direct feedback controller rolled forward through the learned model.

The full protocol and code are in [Notebook 16](../../notebooks/16_action_conditioned_video_planning.ipynb). Per-episode results are in [action-video-planning-episodes.csv](../evidence/action-video-planning-episodes.csv); the 60 raw action interventions are in [action-video-planning-interventions.csv](../evidence/action-video-planning-interventions.csv).

## Prediction and intervention results

| Measure | Result |
| --- | ---: |
| Training next-position MAE | 0.0000513 |
| Training next-velocity MAE | 0.0000513 |
| Held-out intervention next-position MAE | 0.0000213 |
| Held-out intervention next-velocity MAE | 0.0000258 |
| Non-tied one-step utility action pairs ranked correctly | 60 / 60 |
| Tied pairs excluded | 0 |

The values are MountainCar visual-feature errors in simulator units. They measure this fixed renderer and feature extractor. They do not measure a learned JEPA representation. Since only one model was fitted, no interval across training runs is available.

## Closed-loop results

Success means Gymnasium terminated at the goal. The Wilson intervals describe evaluation-episode success only. There is one fitted deterministic model, so these are not training-seed intervals.

| Method | Success | 95% Wilson interval | Mean return | Median return | Mean steps | Median steps | Model calls | CEM planning time | Full method time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Neutral-start CEM | 0 / 20 | [0.000, 0.161] | -18.400 | -18.498 | 300.0 | 300.0 | 2,880,000 | 9.058 s | 13.889 s |
| Momentum-prior CEM | 20 / 20 | [0.839, 1.000] | 91.405 | 91.742 | 131.2 | 125.5 | 1,286,250 | 4.745 s | 6.921 s |
| Direct momentum controller | 20 / 20 | [0.839, 1.000] | 89.285 | 89.400 | 107.2 | 106.0 | 0 | 0.000 s | 1.664 s |

Paired mean differences over the same 20 starts:

- Neutral-start CEM minus direct control: success -1.000, return -107.685, steps +192.8.
- Momentum-prior CEM minus direct control: success 0.000, return +2.120, steps +24.1.

Both the direct controller and momentum-prior CEM reached the goal on every start. The direct controller reached it sooner on average and used no model calls. The prior planner did not improve success over this controller. Neutral-start CEM failed on all starts despite low one-step and intervention errors. This local result shows that one-step prediction and action ranking do not ensure useful planning.

## Claim limits

- The local action prior is a hand-written momentum controller. PiJEPA uses a learned Octo policy to initialize MPPI. This notebook uses CEM and a different task, model, observation, action space, and evaluation.
- V-JEPA 2-AC uses a pretrained video encoder, robot trajectories, and image-goal planning. This notebook uses no pretrained representation, external data, or robot.
- The local results apply only to this MountainCar renderer, model, utility, horizon, seed set, and planner budget. They do not establish that policy priors help all world-model planners.
- Per-method wall time includes simulator steps, rendering, and feature extraction. Planning time counts only CEM search. Different methods do different amounts of work, so the elapsed times are descriptive, not a matched-compute speed claim.

## Research reviewed

- V-JEPA 2-AC predicts action-conditioned latent futures and plans toward image goals. See [the paper](https://arxiv.org/abs/2506.09985) and [official code](https://github.com/facebookresearch/vjepa2).
- PiJEPA uses a learned policy distribution to warm-start MPPI over a JEPA world model. The work appeared in the [CVPR 2026 workshop](https://openaccess.thecvf.com/content/CVPR2026W/WDFM-EAI/papers/Chahe_Policy-Guided_World_Model_Planning_for_Language-Conditioned_Visual_Navigation_CVPRW_2026_paper.pdf); see also the [arXiv record](https://arxiv.org/abs/2603.25981) and [authors' MIT-licensed code](https://github.com/AmirhoseinCh/PiJEPA). Its CAST, Octo, JEPA-model, and GPU setup is outside this local run.
- Newer world-action model directions include [V-JEPA Policy](https://arxiv.org/abs/2609.37250) (v2 dated 2026-10-10) and [JEPA-WAM](https://arxiv.org/abs/2608.09381). These methods couple predictive visual latents with action generation. The local notebook does not implement those methods.
