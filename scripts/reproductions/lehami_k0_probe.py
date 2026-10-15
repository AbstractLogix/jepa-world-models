#!/usr/bin/env python3
"""Run the official LeHami K=0, 25-NN angular-velocity probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Run label, for example a0q_s0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    source_dir = repo_root / "lehami" / "experiments" / "src"
    if not (source_dir / "run_eval.py").is_file():
        parser.error(f"no official run_eval.py found under {source_dir}")
    sys.path.insert(0, str(source_dir))

    import run_eval
    from train_probe_physics import train_probe_knn_gt

    checkpoint = args.checkpoint.resolve()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())

    model, cfg = run_eval.load_model(
        checkpoint,
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        ),
    )
    device = next(model.parameters()).device.type
    h5_path = run_eval._h5_path(cfg)
    expected_hash = manifest.get("dataset_sha256")
    if not expected_hash:
        raise ValueError(f"manifest has no dataset_sha256: {manifest_path}")
    actual_hash = sha256(h5_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"dataset hash mismatch: expected {expected_hash}, got {actual_hash}"
        )

    train_episodes, test_episodes = run_eval._load_manifest(str(manifest_path))
    overlap = set(train_episodes) & set(test_episodes)
    if overlap:
        raise ValueError(f"train and test episode lists overlap at {len(overlap)} indices")

    started = time.perf_counter()
    probe = train_probe_knn_gt(
        model,
        cfg,
        h5_path,
        train_episodes,
        device,
        enc_batch=50,
        n_neighbors=25,
    )
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    features, targets = run_eval._collect_probe_episode_data(
        model, cfg, h5_path, test_episodes, device, enc_batch=50
    )
    if not len(targets) or features.shape[0] != targets.shape[0]:
        raise ValueError("test feature and target arrays are empty or have different row counts")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("test features or targets contain non-finite values")
    prediction = probe.ridge.predict(features)
    if not np.isfinite(prediction).all():
        raise ValueError("KNN probe produced non-finite predictions")
    mae_omega = float(np.mean(np.abs(prediction[:, 2] - targets[:, 2])))
    evaluation_seconds = time.perf_counter() - started

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "label": args.label,
        "status": "measured",
        "metric": "K=0 held-out angular-velocity MAE",
        "mae_omega_rad_s": mae_omega,
        "n_neighbors": 25,
        "distance_weighting": "official train_probe_knn_gt helper",
        "train_episodes": len(train_episodes),
        "train_rows": int(probe.ridge._fit_X.shape[0]),
        "test_episodes": len(test_episodes),
        "test_rows": int(len(targets)),
        "device": device,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "manifest": str(manifest_path),
        "dataset": str(h5_path),
        "dataset_sha256": actual_hash,
        "fit_and_train_encode_seconds": fit_seconds,
        "test_encode_and_score_seconds": evaluation_seconds,
    }
    with (output_dir / "metrics.json").open("w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with (output_dir / "probe_knn.pkl").open("wb") as stream:
        pickle.dump(probe, stream)
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        omega_true=targets[:, 2],
        omega_pred=prediction[:, 2],
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
