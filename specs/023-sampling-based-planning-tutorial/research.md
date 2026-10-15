# Research: Sampling-Based Planning with World Models

**Reviewed**: 2026-10-15

## Primary source

- Gao, Yang, and Kingston, [Sampling-based Kinodynamic Planning with World Models](https://arxiv.org/abs/2610.12910), v1 submitted 2026-10-09 and listed as under review for ICRA 2027. The paper uses a frozen DINO-WM visual model as a latent transition function. Its tree methods use nearest-neighbor RRT or projected-cell KPIECE selection; an action prior proposes controls; expansions are batched; and the planner replans after executing a short action prefix. The tree is reset at each replan. Optional CEM polishing refines a sampled tree path.
- The paper reports four simulated domains with 50 problems each and a real robot evaluation. In point_maze, RRT solved 49/50 and CEM 50/50; median planning time was 0.61 s versus 16.07 s and reported transition nodes were 2,307 versus 45,000. In Push-T, RRT solved 40/50 and CEM 46/50; median time was 0.72 s versus 13.33 s and nodes were 2,578 versus 45,000. These results show a trade-off, not a universal success-rate advantage. Table I calculates time, cost, and nodes only over solved trials.
- The abstract reports 29/30 versus 23/30 solved robot tasks, with 9/10 versus 5/10 on its detour tier. These are paper-reported values on the paper's hardware, tasks, model, and protocol.
- The [CoMMA Lab publication entry](https://commalab.org/publications/) links the paper PDF and video but no code. The paper page currently links its PDF, HTML, and TeX source. This review found no author-linked planner repository or checkpoint.

## Decision

Create a local teaching analogy with a hand-authored point-mass transition model, obstacle map, and common starts. Compare simplified RRT and KPIECE tree selection with a categorical CEM shooting planner. Give each planner the same maximum number of one-step transition evaluations per planning decision. Execute a fixed short prefix and replan from the current toy state.

## Rationale and limits

- The example shows why a tree can preserve intermediate detour states while shooting methods score complete trajectories. It requires no downloaded assets or new package.
- A hand-authored state and transition function remove image encoding, learned latent dynamics, DINO-WM, model error, GPU batching, the paper's action prior, its task sets, and its robot system.
- Equal transition-call caps in the tutorial are a teaching control. They do not reproduce the paper's compute settings or its GPU batching.
- Report success, executed steps, planner transition calls, runtime, and variation by evaluation group. Do not interpret the five groups as training seeds or as evidence for the paper's claims.
- The paper remains a reproduction candidate. Revisit it if authors release the exact code, model and task artifacts, evaluation protocol, and license details needed for a scoped run.

## References

- [arXiv abstract and source links](https://arxiv.org/abs/2610.12910)
- [Version 1 full text](https://arxiv.org/html/2610.12910v1)
- [CoMMA Lab publications](https://commalab.org/publications/)
