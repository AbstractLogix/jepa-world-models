# JEPA World Models – Educational Repo

Hands-on introduction to **Joint Embedding Predictive Architecture (JEPA)** as predictive world models (Yann LeCun's non-generative approach).

Goal: Understand → implement → extend JEPA for images/videos → simple planning.

## Core Concepts (start here)

- Why JEPA > generative world models
- Energy-based prediction in latent space
- Avoiding collapse + multi-modality
- From I-JEPA → V-JEPA → planning

Read: [LeCun 2022 position paper](https://openreview.net/pdf?id=BZ5a1r-kVsf)

## Notebooks (progressive)

1. `01_intuition.ipynb`          – JEPA vs autoencoders vs diffusion (diagrams + toy 2D)
2. `02_i-jepa_from_scratch.ipynb` – Minimal image JEPA (PyTorch)
3. `03_using_official_ijepa.ipynb` – facebookresearch/ijepa + feature viz
4. `04_vjepa_video_world_model.ipynb` – V-JEPA features + frame prediction
5. `05_action_conditioned_jepa.ipynb` – Simple grid-world / dm_control planning demo
6. `06_hierarchical_jepa_ideas.ipynb` – H-JEPA sketch + open challenges

## Folders

```bash
notebooks/          # main educational content
code/               # reusable modules (encoders, predictors, vicreg loss, etc.)
configs/            # YAML configs for experiments
data/               # small demo datasets or processing scripts
planning/           # basic search / MPC with JEPA latents
viz/                # t-SNE / probing / prediction visualizations
```

## Quick Start

```bash
git clone https://github.com/yourname/jepa-world-models-tutorial
cd jepa-world-models-tutorial
pip install -r requirements.txt
```

Start with notebooks/01_intuition.ipynb

## References / Starters

- Official: facebookresearch/ijepa, facebookresearch/jepa (V-JEPA)
- Community: yunusskeete/jepas, eb_jepa
- Key papers: I-JEPA (CVPR 2023), V-JEPA, Value-guided planning with JEPA (arXiv 2601.00844)

Contributions welcome: new modalities, robotics examples, better planning demos.

**Minimal tech stack**  

- PyTorch  
- torchvision / torchcodec (video)  
- lightning or bare training loops  
- matplotlib/seaborn + scikit-learn (viz)  
- gymnasium / dm_control (simple envs)

This layout balances theory, code-from-scratch, official models, and world-model usage while staying beginner-accessible yet deep enough for researchers.
