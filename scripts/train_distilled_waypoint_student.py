#!/usr/bin/env python3
"""Collect synchronized-waypoint demonstrations and distill an IPPO actor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from argparse import Namespace
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch

from project_paths import ONPOLICY_REPO, SCRIPTS, add_to_syspath

add_to_syspath(SCRIPTS, ONPOLICY_REPO)

from evaluate_model_based_waypoint_teacher import (  # noqa: E402
    make_env,
    nonlinear_position_action,
    quadrotor_jacobian,
)
from evaluate_sa_rb_gca_expert_pool import (  # noqa: E402
    get_base_env,
    post_step_swarm_pos_vel,
    seed_everything,
    swarm_goals,
)
from quad_swarm_goal_flow_teacher import (  # noqa: E402
    SynchronizedStageEgressCoordinator,
)
from quad_swarm_obstacle_waypoint_router import ObstacleWaypointRouter  # noqa: E402
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = (
    ROOT
    / "results/revision_horizon7_warmstart_pilot_20260826/"
    "milestones/ippo/step125000"
)
DEFAULT_OUT = ROOT / "results/revision_horizon7_distilled_expert_20260826"
TRAIN_SEEDS = tuple(range(160000, 160032))
VALIDATION_SEEDS = tuple(range(161000, 161004))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_router() -> ObstacleWaypointRouter:
    return ObstacleWaypointRouter(
        clearance_buffer=0.35,
        grid_resolution=0.25,
        room_margin=0.15,
        reached_radius=0.30,
        replan_interval=25,
    )


def make_coordinator() -> SynchronizedStageEgressCoordinator:
    return SynchronizedStageEgressCoordinator(
        staging_radius=1.20,
        staging_ready_radius=0.30,
        egress_radius=1.20,
        max_staging_frames=350,
        waypoint_router=make_router(),
    )


def waypoint_conditioned_observations(
    observations: np.ndarray,
    targets: np.ndarray,
    env,
) -> np.ndarray:
    """Replace only the policy relative-goal vector with the active target."""

    transformed = np.asarray(observations, dtype=np.float32).copy()
    goals = np.asarray(swarm_goals(env), dtype=np.float32)
    offsets = np.asarray(env.policy_goal_slot_offsets, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if transformed.ndim != 2 or transformed.shape[0] != len(targets):
        raise ValueError("Unexpected observation matrix")
    if goals.shape != targets.shape or offsets.shape != targets.shape:
        raise ValueError("Target, goal, and slot arrays must have shape (N, 3)")
    transformed[:, :3] += offsets
    transformed[:, :3] -= targets - goals
    return transformed


def collect_seed(seed: int) -> dict[str, np.ndarray]:
    seed_everything(seed)
    env = make_env(seed)
    try:
        env.seed(seed)
        observations = env.reset()
        coordinator = make_coordinator()
        coordinator.reset(env)
        base = get_base_env(env)
        inverse_jacobians = [
            np.linalg.inv(quadrotor_jacobian(single_env.dynamics))
            for single_env in base.envs
        ]
        dwell = np.zeros(env.n_agents, dtype=np.int64)
        reached = np.zeros(env.n_agents, dtype=bool)
        obs_rows: list[np.ndarray] = []
        action_rows: list[np.ndarray] = []
        seed_rows: list[np.ndarray] = []
        frame_rows: list[np.ndarray] = []

        for frame in range(900):
            targets = coordinator.active_targets(env, reached)
            policy_observations = waypoint_conditioned_observations(
                observations,
                targets,
                env,
            )
            actions = np.stack(
                [
                    nonlinear_position_action(
                        single_env.dynamics,
                        targets[agent_id],
                        inverse_jacobians[agent_id],
                    )
                    for agent_id, single_env in enumerate(base.envs)
                ]
            )
            obs_rows.append(policy_observations)
            action_rows.append(actions)
            seed_rows.append(np.full(env.n_agents, seed, dtype=np.int32))
            frame_rows.append(np.full(env.n_agents, frame, dtype=np.int16))

            observations, _rewards, dones, _infos = env.step(actions)
            done_flags = np.asarray(dones, dtype=bool)
            positions, velocities, _terminal = post_step_swarm_pos_vel(
                env,
                done_flags,
            )
            goals = np.asarray(swarm_goals(env), dtype=np.float32)
            goal_distance = np.linalg.norm(positions - goals, axis=1)
            speed = np.linalg.norm(velocities, axis=1)
            inside = (goal_distance <= 0.5) & (speed <= 0.5)
            dwell = np.where(inside, dwell + 1, 0)
            reached |= dwell >= 10
            if bool(np.all(done_flags)):
                break

        return {
            "observations": np.concatenate(obs_rows).astype(np.float32),
            "actions": np.concatenate(action_rows).astype(np.float32),
            "seeds": np.concatenate(seed_rows),
            "frames": np.concatenate(frame_rows),
        }
    finally:
        env.close()


def collect_dataset(seeds: Iterable[int], label: str) -> dict[str, np.ndarray]:
    chunks = []
    for index, seed in enumerate(seeds, start=1):
        print(f"[{label}] collecting seed {seed} ({index})", flush=True)
        chunks.append(collect_seed(int(seed)))
    return {
        key: np.concatenate([chunk[key] for chunk in chunks])
        for key in chunks[0]
    }


def save_dataset(path: Path, dataset: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def load_source_config() -> Namespace:
    with (SOURCE_RUN / "config.json").open(encoding="utf-8") as handle:
        return Namespace(**json.load(handle))


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def batch_loss(
    actor: R_Actor,
    observations: torch.Tensor,
    actions: torch.Tensor,
    config: Namespace,
) -> torch.Tensor:
    rnn_states = torch.zeros(
        (len(observations), config.recurrent_N, config.hidden_size),
        dtype=torch.float32,
        device=observations.device,
    )
    masks = torch.ones(
        (len(observations), 1),
        dtype=torch.float32,
        device=observations.device,
    )
    predicted, _log_probs, _rnn = actor(
        observations,
        rnn_states,
        masks,
        deterministic=True,
    )
    return torch.mean((predicted - actions) ** 2)


def validation_metrics(
    actor: R_Actor,
    dataset: dict[str, np.ndarray],
    config: Namespace,
    device: torch.device,
    batch_size: int,
) -> Tuple[float, float, float]:
    squared: list[np.ndarray] = []
    absolute: list[np.ndarray] = []
    actor.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset["observations"]), batch_size):
            stop = min(start + batch_size, len(dataset["observations"]))
            observations = torch.as_tensor(
                dataset["observations"][start:stop],
                dtype=torch.float32,
                device=device,
            )
            rnn_states = torch.zeros(
                (len(observations), config.recurrent_N, config.hidden_size),
                dtype=torch.float32,
                device=device,
            )
            masks = torch.ones(
                (len(observations), 1),
                dtype=torch.float32,
                device=device,
            )
            predicted, _log_probs, _rnn = actor(
                observations,
                rnn_states,
                masks,
                deterministic=True,
            )
            errors = (
                predicted.cpu().numpy()
                - dataset["actions"][start:stop]
            )
            squared.append(errors ** 2)
            absolute.append(np.abs(errors))
    squared_values = np.concatenate(squared).reshape(-1)
    absolute_values = np.concatenate(absolute).reshape(-1)
    return (
        float(np.mean(squared_values)),
        float(np.mean(absolute_values)),
        float(np.quantile(absolute_values, 0.95)),
    )


def train_student(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    out_root: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Path:
    config = load_source_config()
    env = make_env(TRAIN_SEEDS[0])
    try:
        actor = R_Actor(
            config,
            env.observation_space[0],
            env.action_space[0],
            device=device,
        )
    finally:
        env.close()
    source_actor = SOURCE_RUN / "models/actor.pt"
    actor.load_state_dict(torch_load(source_actor, device))
    optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=learning_rate,
        eps=config.opti_eps,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(170001)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        actor.train()
        permutation = rng.permutation(len(train["observations"]))
        train_loss_sum = 0.0
        train_samples = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            observations = torch.as_tensor(
                train["observations"][indices],
                dtype=torch.float32,
                device=device,
            )
            actions = torch.as_tensor(
                train["actions"][indices],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(actor, observations, actions, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(indices)
            train_samples += len(indices)

        val_mse, val_mae, val_p95 = validation_metrics(
            actor,
            validation,
            config,
            device,
            batch_size,
        )
        row = {
            "epoch": float(epoch),
            "train_mse": train_loss_sum / max(train_samples, 1),
            "validation_mse": val_mse,
            "validation_mae": val_mae,
            "validation_abs_error_p95": val_p95,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if val_mse < best_loss:
            best_loss = val_mse
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in actor.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("No distilled checkpoint was selected")
    model_dir = out_root / "run1/models"
    model_dir.mkdir(parents=True, exist_ok=True)
    actor_path = model_dir / "actor.pt"
    torch.save(best_state, actor_path)

    with (out_root / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    config_payload = vars(config).copy()
    config_payload.update(
        {
            "experiment_name": "distilled_sync_stage_waypoint_ippo",
            "model_dir": str(model_dir),
            "distillation_source_actor": str(source_actor),
            "distillation_source_actor_sha256": sha256(source_actor),
            "distillation_train_seeds": list(TRAIN_SEEDS),
            "distillation_validation_seeds": list(VALIDATION_SEEDS),
            "distillation_epochs": epochs,
            "distillation_batch_size": batch_size,
            "distillation_learning_rate": learning_rate,
            "distillation_best_validation_mse": best_loss,
            "shared_goal_slot_radius": 0.45,
        }
    )
    (out_root / "run1/config.json").write_text(
        json.dumps(config_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return actor_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--force-collect", action="store_true")
    args = parser.parse_args()

    if args.epochs != 12 or args.batch_size != 4096 or args.learning_rate != 3e-4:
        raise ValueError("Stage 1i hyperparameters are locked to the preregistration")
    if not (SOURCE_RUN / "models/actor.pt").is_file():
        raise FileNotFoundError(SOURCE_RUN / "models/actor.pt")
    random.seed(170001)
    np.random.seed(170001)
    torch.manual_seed(170001)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Stage 1i requires the configured CUDA device")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(170001)

    args.out_root.mkdir(parents=True, exist_ok=True)
    data_dir = args.out_root / "datasets"
    train_path = data_dir / "teacher_train_160000_160031.npz"
    validation_path = data_dir / "teacher_validation_161000_161003.npz"
    if args.force_collect or not train_path.is_file():
        save_dataset(train_path, collect_dataset(TRAIN_SEEDS, "train"))
    if args.force_collect or not validation_path.is_file():
        save_dataset(
            validation_path,
            collect_dataset(VALIDATION_SEEDS, "validation"),
        )
    train = load_dataset(train_path)
    validation = load_dataset(validation_path)
    actor_path = train_student(
        train,
        validation,
        args.out_root,
        device,
        args.epochs,
        args.batch_size,
        args.learning_rate,
    )
    manifest = {
        "train_dataset": str(train_path),
        "train_dataset_sha256": sha256(train_path),
        "train_samples": int(len(train["observations"])),
        "validation_dataset": str(validation_path),
        "validation_dataset_sha256": sha256(validation_path),
        "validation_samples": int(len(validation["observations"])),
        "actor": str(actor_path),
        "actor_sha256": sha256(actor_path),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    (args.out_root / "distillation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
