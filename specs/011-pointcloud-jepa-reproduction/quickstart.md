# Quickstart: Point-Cloud JEPA Push-T Reproduction

## Requirements

- Linux x86-64, an NVIDIA GPU with at least 24 GB memory, a CUDA 12 driver, and EGL.
- Pixi with the upstream locked environment. It uses Python 3.10 and PyTorch 2.7 with CUDA 12.6.
- About 4.62 GB for the compressed v1 dataset, plus space for the extracted dataset and logs.
- Network access to the official code and Hugging Face assets.

If the host fails the hardware check, stop before downloading the dataset.

## Run

1. Use the upstream code at commit `1b5a90ebf63e8ef165b1dfe100b81b12f4fa1424` in a temporary checkout.
2. Run `pixi install --locked`, then `pixi run build-kernels`.
3. Fetch `fafraob/point-cloud-pusht` dataset revision `d3391d60371b6e9cbcd1c5737bdfc87b4af309ed` and verify the archive hash in `research.md`.
4. Fetch only `point-lewm/` and `point-delta-jepa/` from model revision `664278a4c0b4c5f607d0993744986b2d10b4168e`. Verify both `weights.pt` hashes.
5. Follow the pinned upstream `docs/data.md` to unpack the dataset and set `PLWM_DATA_ROOT` if needed.
6. For each seed `1 25 35 36 40 42 65 80 83 99`, run `eval_lidar.py --config-name=lidar_pusht` once with `policy=pusht/point-lewm/weights.pt` and once with `policy=pusht/point-delta-jepa/weights.pt`. Set `++output.video=false` to avoid video files.
7. Record all 50 episode outcomes per method and seed. Calculate the mean and standard deviation across the ten seed rates.
8. Write the results and limits to `docs/reproductions/point-cloud-pusht.md`.

## Expected comparison

The paper reports **83.6% ± 3.4** for Point-LeWM and **70.8% ± 7.7** for Point-Delta-JEPA on Push-T. Report whether each local mean falls inside its reported one-standard-deviation range. This is not a statistical test. Planning is stochastic because point samples change during encoder calls.

## Claim boundary

This evaluates released checkpoints in simulation. It does not reproduce training, the other environments, real-robot behavior, or the paper's complete set of experiments.
