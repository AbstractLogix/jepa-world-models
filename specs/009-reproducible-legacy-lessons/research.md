# Research: Reproducible Legacy Lesson Evidence

**Reviewed**: 2026-10-14

## Decisions

### Use the environment and action-space seed APIs separately

Gymnasium documents `env.reset(seed=...)` for the environment random number generator and `env.action_space.seed(...)` for reproducible sampled actions. The lessons use random actions to collect offline data and use NumPy random candidates in shooting planners. The implementation will seed each source at the point where it is used. See the [Gymnasium environment API](https://gymnasium.farama.org/v1.0.0/api/env/).

### Use the symmetric eigensolver for covariance ellipses

Notebook 01 passes real symmetric covariance matrices to NumPy. The general `numpy.linalg.eig` solver may return complex arrays for real inputs. `numpy.linalg.eigh` is defined for real symmetric matrices, returns real eigenvalues in ascending order, and returns the eigenvectors in columns. The notebook will sort the eigenpairs before it draws the major axis. See the [NumPy `eigh` reference](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html) and [NumPy `eig` reference](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html).

### Pin and label notebook 04's visual backbone

Notebook 04 uses `timm/resnet18.a1_in1k`, not a V-JEPA checkpoint. The model card identifies it as a ResNet-18 image-classification backbone trained on ImageNet-1K with an Apache-2.0 license. The model was fetched at revision `491b427b45c94c7fb0e78b5474cc919aff584bbf`; its safetensors file SHA-256 is `80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612`. Pin this revision in the notebook and state that its two-dimensional class descriptors are synthetic labels, not language embeddings. See the [model card](https://huggingface.co/timm/resnet18.a1_in1k/blob/491b427b45c94c7fb0e78b5474cc919aff584bbf/README.md).

### Treat episode spread separately from training-run spread

The shared [evaluation protocol](../../docs/EVALUATION_PROTOCOL.md) requires matching evaluation cases for method comparisons. Episodes from one trained model are not independent training runs. Notebooks 05 and 06 will report each paired episode return and describe the summary as an episode-level result. Notebook 08 has one training seed per setting, so it will remain an exploratory ablation without a training-seed interval.

### Batch candidate scoring without changing the seeded draws

The first CPU execution of notebook 05 spent more than six minutes in the scalar candidate-scoring loop and was stopped before the planned-policy episodes completed. The planner now evaluates the same NumPy-generated candidate sequences as a batch. For notebook 08, vectorized `np.random.randint` draws were checked against the previous per-action draw order for seeds 42, 200, and 999. The new run must still verify the notebook outputs; batching can change floating-point details even when candidate sequences match.

### Keep outputs out of version control

The project `.gitignore` already excludes `notebooks/data/` and `notebooks/checkpoints/`. Notebook checkpoints will use the ignored `notebooks/checkpoints/` path. Executed notebook copies used for validation will remain in `/tmp`.

## Environment used for the local run

The validation environment uses Python 3.13.14, PyTorch 2.10.0 CPU, TorchVision 0.25.0 CPU, Gymnasium 0.29.1, timm 1.0.30, and pandas 2.3.3. The host has an RTX 5070, but the installed PyTorch build cannot use CUDA. All recorded runs therefore use CPU. These versions describe this local evidence and do not replace the project's supported Python 3.10–3.13 range.

## Alternatives considered

- **Use only global NumPy and PyTorch seeds**: rejected because Gymnasium action spaces maintain a separate random generator.
- **Keep different evaluation starts for each planner**: rejected because a return difference would mix method and initial-state effects.
- **Add a shared evaluation library**: rejected because the changes are small and limited to existing notebook examples.
- **Treat the one-seed ablation's episode standard deviation as training uncertainty**: rejected because episodes from a single trained model do not estimate variation across model training runs.
