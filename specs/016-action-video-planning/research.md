# Research: Action-Conditioned Video Planning

**Reviewed**: 2026-10-15

## Findings

### V-JEPA 2-AC: the closest first-party reference

- **Decision**: Teach the loop from visual observations to action-conditioned prediction and image-goal planning, but use a small local simulator task.
- **Rationale**: V-JEPA 2-AC post-trains a latent action-conditioned predictor from robot video and actions, then plans toward image goals. The paper reports zero-shot Franka manipulation in two labs after using less than 62 hours of DROID video. The official repository contains model code, an action-conditioned checkpoint entry, and an energy-landscape example. Its standard model is too large for a small default CPU lesson, and the paper's robot data, model, and deployment setup are not the local experiment.
- **Alternatives considered**: Load the official model and reproduce a robot result. This needs large pretrained weights, robot-trajectory data, and a compatible inference path. Keep that as a separate reproduction gate.
- **Sources**: [paper](https://arxiv.org/abs/2506.09985); [official repository](https://github.com/facebookresearch/vjepa2). The repository lists MIT and Apache-2.0 code licenses; the README does not state separate checkpoint terms.

### Policy-guided planning: local comparison target

- **Decision**: Compare a uniform-start planner with a planner initialized by a simple goal-directed action prior. Also report the prior as a direct feedback policy.
- **Rationale**: PiJEPA trains an Octo policy on CAST visual navigation data, then uses its action distribution to warm-start MPPI over a separate JEPA world model. The paper reports better real-world goal-reaching than standalone policy execution or uninformed MPPI. Its public repository is MIT-licensed and links the research code, but the described pipeline uses CAST, large visual models, Octo, distributed GPU training, and real navigation data. A compact MountainCar comparison can exercise the design question but cannot reproduce this result.
- **Alternatives considered**: Recreate the policy and robot navigation pipeline. That is outside this bounded lesson and depends on large assets and hardware.
- **Sources**: [CVPR 2026 workshop paper](https://openaccess.thecvf.com/content/CVPR2026W/WDFM-EAI/papers/Chahe_Policy-Guided_World_Model_Planning_for_Language-Conditioned_Visual_Navigation_CVPRW_2026_paper.pdf); [arXiv record](https://arxiv.org/abs/2603.25981); [authors' code](https://github.com/AmirhoseinCh/PiJEPA).

### Newer world-action models: roadmap context

- **V-JEPA Policy (v2 dated 2026-10-10)** trains an instruction-conditioned future-latent predictor and flow-matching action expert on a frozen V-JEPA 2.1 visual encoder. The authors report results across LIBERO, LIBERO-Plus, and RoboCasa-GR1. The author repository is MIT-licensed, but its README says task checkpoints and the DROID-pretrained predictor are planned; the setup still needs model weights, datasets, simulator assets, and GPU training.
- **JEPA-WAM (2026-08)** couples joint-embedding transition prediction with continuous action generation through a shared predictor and reports robot manipulation results. Its paper is useful as a future lesson candidate; this feature does not implement action generation.
- **GRASP (2026-02)** changes the planner using virtual states and soft dynamics constraints. The official author repository is MIT-licensed and has adapters for several world models. It motivates future longer-horizon tests, not a claim about this simple CEM comparison.
- **Sources**: [V-JEPA Policy paper](https://arxiv.org/abs/2609.37250) and [author code](https://github.com/breez3young/VJEPA-Policy); [JEPA-WAM paper](https://arxiv.org/abs/2608.09381); [GRASP paper](https://arxiv.org/abs/2602.00475) and [author code](https://github.com/michael-psenka/grasp).

## Local design decisions

| Decision | Rationale | Alternatives considered |
| --- | --- | --- |
| Use Gymnasium MountainCarContinuous with a small Pillow renderer. | The project already uses this simulator. A custom renderer avoids a windowing or video-data dependency and creates repeatable frames. | Real robot data or a large public video corpus would add transfer, access, and licensing gates without improving this introductory comparison. |
| Extract position from the red car and green goal in the rendered frames; estimate velocity from two adjacent frames. | It makes the observation boundary visible and keeps the feature extractor small enough to inspect. | Loading a pretrained V-JEPA encoder would add a large checkpoint and model terms. It remains an optional future lesson. |
| Fit a compact polynomial transition model on action-feature pairs. | This isolates the role of action conditioning and keeps training deterministic and CPU-sized. | A deep action-conditioned transformer would copy scale without testing the core planning loop. |
| Use CEM with the same horizon, candidates, update count, and evaluation starts for both initial distributions. | This gives a controlled local test of whether an informed prior helps search under a fixed budget. | The paper uses MPPI and a learned robot policy. The CEM/controller-prior setup is an explicit deviation, not an exact algorithm reproduction. |
| Include a direct controller and held-out action interventions. | Prediction and planning outcomes need separate evidence. The direct policy also shows the prior's behavior without a learned model. | Comparing only planner success would hide whether the policy itself is sufficient. |

## Claim boundary

This notebook is an action-conditioned visual-model planning analogue. Its fixed image-moment features are not learned JEPA features. Its action prior is a small controller, not a trained VLA policy. A local result does not reproduce V-JEPA 2-AC or PiJEPA, and it does not validate robot behavior.
