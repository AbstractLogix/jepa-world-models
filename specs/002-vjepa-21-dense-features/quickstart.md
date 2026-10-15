# Quickstart: V-JEPA 2.1 Dense Video Features Tutorial

## No-download path

Use the repository environment and open `notebooks/09_vjepa21_dense_features.ipynb`. Run the setup and synthetic-clip sections. They must generate and show the 16-frame, 384×384 clip without downloading video or model weights.

## Optional official checkpoint path

The checkpoint is about 1.66 GB. Check its current terms on the official repository before downloading. The notebook does not download it unless `JEPA_LOAD_VJEPA21=1` is set.

Install the project with the model extra, then install PyTorch using the suitable wheel index for the host:

```bash
pip install -e '.[vjepa21]'
pip install -r requirements-torch.txt
```

For CPU-only PyTorch wheels, use the index documented in `requirements-torch.txt`:

```bash
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
```

Run the notebook from the repository root with the opt-in flag:

```bash
JEPA_LOAD_VJEPA21=1 jupyter nbconvert --to notebook --execute --inplace notebooks/09_vjepa21_dense_features.ipynb
```

The expected output is an official encoder feature grid and both V-JEPA and mean-RGB patch correspondence measurements. Runtime and memory depend on the host. The notebook prints the selected device, input shape, code revision, preprocessing, elapsed time, and available peak-memory measurement. A successful run is a local measurement on this synthetic clip, not a paper reproduction.

For a clean verification without modifying the source notebook, copy it to a temporary output location and use `--output` instead of `--inplace`.
