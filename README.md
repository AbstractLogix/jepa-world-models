# JEPA World Models – Educational Repo (2026 edition)

Hands-on exploration of **Joint Embedding Predictive Architecture (JEPA)** as the foundation for non-generative world models — Yann LeCun's path to predictive understanding, planning, and reasoning without pixel-level generation.

Focus: From core JEPA → latest extensions (LeJEPA, Rectified LpJEPA, VL-JEPA, value-guided planning, gradient-based planners, latent actions, reproducible ecosystems) → simple planning & robotics demos.

## Core Concepts (start here – updated reading list)

- Why JEPA > generative autoregressive world models (energy-based latent prediction, collapse prevention, efficiency)
- Theoretical foundations: isotropic Gaussians, sparsity, maximum-entropy representations
- Scaling to video, language, actions, planning in latent space
- Latest advances: value shaping, gradient-based MPC, latent action models, reproducible baselines

**Must-read recent papers (2025–2026)** – prioritize these for cutting-edge understanding:

- **[LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics](https://arxiv.org/abs/2511.08544)** (Balestriero & LeCun, Nov 2025)  
  Comprehensive theory of JEPA + Sketched Isotropic Gaussian Regularization (SIGReg) → stable, heuristics-free training.

- **[Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations](https://arxiv.org/abs/2602.01456)** (Kuang et al., Feb 2026)  
  Extends JEPA to sparse, rectified representations via Rectified Distribution Matching → better efficiency & biological plausibility.

- **[Value-guided action planning with JEPA world models](https://arxiv.org/abs/2601.00844)** (Destrade et al., Dec 2025)  
  Shapes JEPA latent space so distances ≈ negative value functions → dramatically better planning on control tasks.

- **[What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?](https://arxiv.org/abs/2512.24497)** (Terver et al., Dec 2025/Jan 2026)  
  Large-scale ablation study of JEPA-WM components for navigation/manipulation → best practices for physical planning.

- **[VL-JEPA: Joint Embedding Predictive Architecture for Vision-language](https://arxiv.org/abs/2512.10942)** (Chen et al., Dec 2025)  
  JEPA for vision + language → predicts text embeddings (not tokens) → efficient multimodal world understanding.

- **[Parallel Stochastic Gradient-Based Planning for World Models](https://arxiv.org/abs/2602.00475)** (Psenka et al., Jan 2026)  
  GRASP planner: fast, parallel gradient-based optimization in world-model latent space.

Other strong recent ones:

- [stable-worldmodel-v1: Reproducible World Modeling Research and Evaluation](https://arxiv.org/abs/2602.08968) (Maes et al., Feb 2026) → modular ecosystem for fair comparisons.
- [Learning Latent Action World Models In The Wild](https://arxiv.org/abs/2601.05230) (Garrido et al., Jan 2026) → action-free latent actions from raw video.
- [JEPA as a Neural Tokenizer...](https://arxiv.org/abs/2512.07168) (Ioannides et al., Dec 2025) → JEPA for speech/audio representations.
- [Closing the Train-Test Gap in World Models for Gradient-Based Planning](https://arxiv.org/abs/2512.09929) (Parthasarathy et al., Dec 2025) → data synthesis tricks for better planning.

Classic starters:

- LeCun's 2022 position paper: [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- I-JEPA (CVPR 2023), V-JEPA (Meta), etc.

## Notebooks (progressive – now with latest integrations)

1. `01_intuition.ipynb`          – JEPA vs generative models + latest theory (LeJEPA isotropic Gaussians, sparsity)
2. `02_lejepa_from_scratch.ipynb` – Minimal LeJEPA implementation (SIGReg + predictive loss)
3. `03_rectified_lpjepa.ipynb`    – Sparse Rectified LpJEPA (RDMReg) demo on images
4. `04_vjepa_plus_vl_jepa.ipynb`  – Using official V-JEPA → VL-JEPA features for vision-language tasks
5. `05_value_guided_planning.ipynb` – Implement value-shaped JEPA latents + simple planning (from 2601.00844)
6. `06_gradient_based_planning.ipynb` – GRASP-style stochastic gradient planning in latent space
7. `07_latent_action_world_models.ipynb` – Latent actions from video (no labels) + basic controller
8. `08_physical_planning_ablation.ipynb` – Reproduce key ablations from "What Drives Success..." paper

## Folders

```bash
notebooks/                # main tutorials
code/                     # modules: lejepa_loss.py, sigreg.py, rdmreg.py, value_shaping.py, grasp_planner.py
configs/                  # experiment configs (include LeJEPA, LpJEPA, planning hyperparameters)
planning/                 # MPC/GRASP/value-guided search in JEPA space + gymnasium/dm_control examples
viz/                      # latent space viz, value-distance correlations, planning rollouts
reproducibility/          # scripts inspired by stable-worldmodel-v1 for fair eval
```

## Quick Start

```bash
git clone https://github.com/yourname/jepa-world-models-tutorial
cd jepa-world-models-tutorial
pip install -r requirements.txt
```

Start with notebooks/01_intuition.ipynb or the LeJEPA one for the freshest theory.

## References / Starters

- Official repos: facebookresearch/ijepa, facebookresearch/vjepa (check for V-JEPA 2 updates)
- Latest theory & extensions: LeJEPA, Rectified LpJEPA, VL-JEPA papers above
- Planning-focused: value-guided JEPA, GRASP planner, "What Drives Success..." ablations
- Reproducibility: stable-worldmodel-v1 codebase
- Community forks/experiments welcome: add multimodal, robotics, audio JEPA variants

PRs encouraged: implementations of missing papers, better visualizations of value-distance shaping, or integrations with real robot data.

**Minimal tech stack**  

- PyTorch  
- torchvision / torchcodec (video)  
- lightning or bare training loops  
- matplotlib/seaborn + scikit-learn (viz)  
- gymnasium / dm_control (simple envs)
