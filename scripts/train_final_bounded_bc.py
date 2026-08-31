#!/usr/bin/env python3
"""Train only the bounded behavioral-cloning controller retained by the paper."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from bounded_waypoint_student import BoundedWaypointStudent


ROOT = Path(__file__).resolve().parents[1]
PAPER_RESULT_ROOT = ROOT / "results/final_formal_multiseed/training/proposed_bc"
SOURCE_DATA_ROOT = ROOT / "results/revision_horizon7_distilled_expert_20260826/datasets"
PACKAGE_DATA_ROOT = ROOT / "data/training"
EPOCHS = 60
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
TRAINING_SEED = 171001


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    observations = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError(f"Unexpected dataset shape in {path}")
    if len(observations) != len(actions):
        raise ValueError(f"Observation/action count mismatch in {path}")
    return {"observations": observations, "actions": actions}


def validation_metrics(
    model: BoundedWaypointStudent,
    dataset: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    squared: list[np.ndarray] = []
    absolute: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(dataset["observations"]), BATCH_SIZE):
            observations = torch.as_tensor(
                dataset["observations"][offset : offset + BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )
            predicted = model(observations).cpu().numpy()
            error = predicted - dataset["actions"][offset : offset + BATCH_SIZE]
            squared.append(np.square(error))
            absolute.append(np.abs(error))
    squared_values = np.concatenate(squared)
    absolute_values = np.concatenate(absolute)
    return {
        "mse": float(np.mean(squared_values)),
        "mae": float(np.mean(absolute_values)),
        "p95": float(np.quantile(absolute_values, 0.95)),
    }


def train(
    training: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    out_dir: Path,
    device: torch.device,
    training_seed: int,
) -> Path:
    random.seed(training_seed)
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)

    observations = training["observations"]
    observation_mean64 = observations.mean(axis=0, dtype=np.float64)
    second_moment = np.mean(
        np.square(observations, dtype=np.float64), axis=0
    )
    observation_std = np.sqrt(
        np.maximum(second_moment - np.square(observation_mean64), 1e-8)
    ).astype(np.float32)
    observation_mean = observation_mean64.astype(np.float32)
    model = BoundedWaypointStudent(observation_mean, observation_std).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.default_rng(training_seed)
    steps_per_epoch = max(1, math.ceil(len(observations) / BATCH_SIZE))
    history: list[dict[str, float]] = []
    best_score = math.inf
    best_payload = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for _step in range(steps_per_epoch):
            indices = rng.integers(0, len(observations), size=BATCH_SIZE)
            batch_observations = torch.as_tensor(
                observations[indices], dtype=torch.float32, device=device
            )
            batch_actions = torch.as_tensor(
                training["actions"][indices], dtype=torch.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            predicted = model(batch_observations)
            loss = torch.mean(torch.square(predicted - batch_actions))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_observations)
            sample_count += len(batch_observations)

        metrics = validation_metrics(model, validation, device)
        row = {
            "epoch": float(epoch),
            "train_mse": loss_sum / sample_count,
            "validation_mse": metrics["mse"],
            "validation_mae": metrics["mae"],
            "validation_p95": metrics["p95"],
            "selection_score": metrics["mse"],
        }
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(json.dumps(row, sort_keys=True), flush=True)
        if metrics["mse"] < best_score:
            best_score = metrics["mse"]
            best_payload = model.checkpoint_payload()

    if best_payload is None:
        raise RuntimeError("No checkpoint was selected")
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "student.pt"
    torch.save(best_payload, checkpoint)
    with (out_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return checkpoint


def default_data_root() -> Path:
    return PACKAGE_DATA_ROOT if PACKAGE_DATA_ROOT.is_dir() else SOURCE_DATA_ROOT


def main() -> int:
    data_root = default_data_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=Path,
        default=data_root / "teacher_train_160000_160031.npz",
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=data_root / "teacher_validation_161000_161003.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PAPER_RESULT_ROOT / f"seed{TRAINING_SEED}",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--training-seed", type=int, default=TRAINING_SEED)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    training = load_dataset(args.train_data)
    validation = load_dataset(args.validation_data)
    device = torch.device(args.device)
    checkpoint = train(
        training,
        validation,
        args.out_dir,
        device,
        args.training_seed,
    )
    manifest = {
        "protocol": "final_bounded_bc_v1",
        "training_seed": args.training_seed,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_data": str(args.train_data),
        "train_data_sha256": sha256(args.train_data),
        "train_samples": int(len(training["observations"])),
        "validation_data": str(args.validation_data),
        "validation_data_sha256": sha256(args.validation_data),
        "validation_samples": int(len(validation["observations"])),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    (args.out_dir / "final_bc_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
