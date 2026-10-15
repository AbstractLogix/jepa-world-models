# Research: Qantara Multi-Paradigm Inference

**Reviewed**: 2026-10-15

## Decision

Use Qantara as a future tutorial for one latent world model queried by three inference paths. Target the Push-T rows of Table 2 in the current arXiv v2 paper for an optional released-checkpoint evaluation. Keep the local CPU lesson separate from the checkpoint result.

## Primary sources

- Paper: [Qantara: Bridge-Flow Training for Multi-Paradigm JEPA Inference](https://arxiv.org/abs/2607.04978), v2 dated 2026-10-10. The authors report Push-T success of 90.1 ± 1.1 for CEM, 82.1 ± 1.0 for BC, and 83.2 ± 0.6 for video–inverse. The table uses three training seeds, five evaluation seeds, and 50 episodes, with a 50-step episode budget. The standard deviation is across the three training-seed means.
- Code: [corl-team/qantara](https://github.com/corl-team/qantara/tree/12429541011e8576eb4fe2bc589d059b0d29dd2e), revision `12429541011e8576eb4fe2bc589d059b0d29dd2e`; repository license is MIT. Its README says the released checkpoints cover seeds 11, 22, and 33. The official Push-T evaluation config defaults to 50 episodes and a 50-step budget. CEM uses K=4; BC and video–inverse use K=2.
- Checkpoints: [t-tech/qantara-checkpoints](https://huggingface.co/t-tech/qantara-checkpoints/tree/de5f7575c9db4a564382e4b2b681fa534dcb3b62), revision `de5f7575c9db4a564382e4b2b681fa534dcb3b62`; the card declares MIT. Each checkpoint is a full `torch.save` model object. The three Qantara Push-T files total about 275 MB. Hub metadata advertises the following sizes and SHA-256 values; verify the bytes locally after any approved transfer.

| Checkpoint | Size (bytes) | Hub SHA-256 |
| --- | ---: | --- |
| qantara-pusht-s11.ckpt | 91,694,043 | `9c0177685b9bf15cae299ca7357ee0bf685fb8dacc122fbdce4e76757e15fb9d` |
| qantara-pusht-s22.ckpt | 91,694,043 | `9d1360a982be5793c6e92b387de43dac1e0b599ac4b3e20eba4a2c7aa7558a88` |
| qantara-pusht-s33.ckpt | 91,694,043 | `37ab6c1539084e33974887a79dbc29617ef53a5f8ff8e4b5903440c19dcdca9b` |

- Data: The paper Appendix A says Push-T uses the DINO-WM-released corpus with 20,000 expert demonstrations. The [DINO-WM dataset card](https://huggingface.co/datasets/gaoyuezhou/dino_wm/tree/d5c69e8fe60c3ed6b01d2f6e88fbfd66be078e57) declares MIT; its `pusht_noise.zip` is 2,785,304,515 bytes with Hub SHA-256 `442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08`. A matching raw archive was already staged at `/tmp/lpwm-data/pusht_noise.zip`; its local hash matches, but this Qantara task did not transfer or unpack it. The current [Qantara README at the pinned revision](https://github.com/corl-team/qantara/blob/12429541011e8576eb4fe2bc589d059b0d29dd2e/README.md) instead labels Push-T data as LeRobot / D4RL expert data, while its config requests `pusht_expert_train` in HDF5 form. A pinned [H-JEPA conversion script](https://github.com/kevinghst/H-JEPA/blob/97054ecaff83e13839195ec4b92d74c08a0127c5/scripts/data/convert_pusht_noise_to_h5.py) maps DINO-WM inputs into a compatible HDF5 schema, but no first-party record ties this candidate conversion to Qantara's actual training/evaluation file or source episode identities. The [LeWM dataset](https://huggingface.co/datasets/quentinll/lewm-pusht/tree/655cd446b9929369d7d406001da85c15d1457850) declares MIT and contains `pusht_expert_train.h5.zst`, size 13,136,247,974 bytes, Hub SHA-256 `7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318`. Matching the paper corpus to Qantara's actual HDF5 file is still unresolved. Do not substitute either data source without validating episode identity and conversion.
- Runtime: The official Qantara project requires Python `>=3.10,<3.14`, PyTorch `>=2.7,<2.8`, and MuJoCo 3.2.0. Its README recommends EGL and lists OSMesa as a fallback. The existing Python 3.11 H-JEPA runtime has CUDA 12.8 and PyTorch 2.7.1, but uses MuJoCo 3.14. EGL initialization exits with code 139 on this host. A direct `MUJOCO_GL=osmesa` import fails inside PyOpenGL before the environment starts. Qantara's evaluation runtime is not qualified.

## Alternatives considered

- **Report Qantara as a general state-of-the-art winner**: Rejected. In the paper's matched pipeline, Qantara trails the LeWM reproduction on Push-T (90.1 vs 92.3) and only the Cube gain is strong across the reported controls. The central tutorial contribution is multi-paradigm inference, not universal task superiority.
- **Call a one-seed result a reproduction of the paper's mean**: Rejected. The paper aggregates three training checkpoints and five evaluation seeds.
- **Treat the HDF5 and DINO-WM archive as interchangeable**: Rejected. Their relationship and exact evaluation episodes are not yet verified.
- **Transfer data or serialized checkpoints now**: Rejected. The user selected a pending scope for Push-T assets, the corpus mapping is unresolved, and the renderer fails before an episode.

## Limits

The paper evaluates simulated tasks on offline data at one approximately 21M-parameter scale. It does not establish real-robot transfer. The five evaluation-seed identities are not specified in the paper or current official run scripts. If a local run uses a fixed documented seed set, report it as a protocol replication and disclose that the exact episode sample differs.
