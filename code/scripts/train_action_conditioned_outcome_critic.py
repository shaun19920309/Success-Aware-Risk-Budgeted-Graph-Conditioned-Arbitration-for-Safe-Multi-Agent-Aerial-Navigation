#!/usr/bin/env python3
"""Train a v5 action-conditioned outcome critic for graph arbitration.

The critic receives graph-risk features and a candidate safety-expert weight
alpha, then predicts rollout-level reward, success, risk, collision, and final
goal outcomes.  It is intended as the learned model behind a later constrained
selector over candidate alphas.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from graph_gate_model import augment_feature_matrix  # noqa: E402


HEADS = [
    "reward",
    "final_goal",
    "score",
    "success",
    "risk_lt_1_0",
    "risk_lt_0_65",
    "collision",
    "deadlock",
]


def select_device(text: str) -> torch.device:
    requested = text.strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested for outcome critic training, but torch.cuda.is_available() is false.")
    return torch.device(requested)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ActionConditionedOutcomeCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            *[ResidualBlock(hidden_dim, dropout) for _ in range(max(1, num_layers))],
            nn.LayerNorm(hidden_dim),
        )
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_dim, 1) for name in HEADS})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {name: head(h).squeeze(-1) for name, head in self.heads.items()}


def split_by_source(source_index: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique_sources = np.unique(source_index.astype(np.int64))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_sources)
    n_val = max(1, int(round(len(unique_sources) * val_fraction)))
    val_sources = set(int(v) for v in unique_sources[:n_val])
    val_mask = np.asarray([int(idx) in val_sources for idx in source_index], dtype=bool)
    train_mask = ~val_mask
    return np.flatnonzero(train_mask), np.flatnonzero(val_mask)


def finite_std(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    return mean, max(std, 1e-6)


def binary_pos_weight(values: np.ndarray) -> float:
    values = np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)
    pos = float(np.sum(values > 0.5))
    neg = float(len(values) - pos)
    if pos <= 0.0 or neg <= 0.0:
        return 1.0
    return float(np.clip(neg / pos, 0.25, 20.0))


def build_features(data: np.lib.npyio.NpzFile, metadata: dict[str, object], augment_mode: str) -> tuple[np.ndarray, list[str]]:
    feature_names = list(metadata["feature_names"])
    features = np.asarray(data["features"], dtype=np.float32)
    if "source_features" in data and "source_index" in data:
        source = np.asarray(data["source_features"], dtype=np.float32)
        source_meta = np.asarray(data["source_meta"], dtype=np.int32) if "source_meta" in data else None
        if augment_mode != "none":
            source, feature_names = augment_feature_matrix(
                source,
                feature_names,
                meta=source_meta,
                include_pairwise=augment_mode in {"pairwise", "full"},
                include_temporal=augment_mode in {"temporal", "full"},
            )
        features = source[np.asarray(data["source_index"], dtype=np.int64)]
    elif augment_mode != "none":
        meta = np.asarray(data["meta"], dtype=np.int32) if "meta" in data else None
        features, feature_names = augment_feature_matrix(
            features,
            feature_names,
            meta=meta[:, :4] if meta is not None and meta.ndim == 2 and meta.shape[1] >= 4 else meta,
            include_pairwise=augment_mode in {"pairwise", "full"},
            include_temporal=augment_mode in {"temporal", "full"},
        )
    return features.astype(np.float32), feature_names


def make_action_conditioned_input(features: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, list[str]]:
    alpha = np.asarray(alpha, dtype=np.float32).reshape(-1, 1)
    alpha_terms = np.concatenate([alpha, alpha**2, alpha * (1.0 - alpha)], axis=1)
    interactions = features * alpha
    x = np.concatenate([features, alpha_terms, interactions], axis=1).astype(np.float32)
    names = (
        [f"x{i}" for i in range(features.shape[1])]
        + ["alpha", "alpha_sq", "alpha_balance"]
        + [f"alpha_x{i}" for i in range(features.shape[1])]
    )
    return x, names


def standardize_targets(data: np.lib.npyio.NpzFile) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    raw = {
        "reward": np.asarray(data["target_reward"], dtype=np.float32).reshape(-1),
        "final_goal": np.asarray(data["target_final_goal"], dtype=np.float32).reshape(-1),
        "score": np.asarray(data["target_score"], dtype=np.float32).reshape(-1),
        "success": np.asarray(data["target_success"], dtype=np.float32).reshape(-1),
        "risk_lt_1_0": np.asarray(data["target_risk_lt_1_0"], dtype=np.float32).reshape(-1),
        "risk_lt_0_65": np.asarray(data["target_risk_lt_0_65"], dtype=np.float32).reshape(-1),
        "collision": np.asarray(data["target_collision"], dtype=np.float32).reshape(-1),
        "deadlock": np.asarray(data["target_deadlock"], dtype=np.float32).reshape(-1),
    }
    stats: dict[str, dict[str, float]] = {}
    out: dict[str, np.ndarray] = {}
    for name in ["reward", "final_goal", "score"]:
        mean, std = finite_std(raw[name])
        stats[name] = {"mean": mean, "std": std}
        out[name] = ((np.nan_to_num(raw[name], nan=mean) - mean) / std).astype(np.float32)
    for name in ["success", "risk_lt_1_0", "risk_lt_0_65", "collision", "deadlock"]:
        values = np.clip(np.nan_to_num(raw[name], nan=0.0), 0.0, 1.0).astype(np.float32)
        stats[name] = {"pos_weight": binary_pos_weight(values), "mean": float(np.mean(values))}
        out[name] = values
    return out, stats


def make_target_matrix(targets: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([targets[name] for name in HEADS], axis=1).astype(np.float32)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    mse: nn.Module,
    bce_losses: dict[str, nn.Module],
    device: torch.device,
    loss_weights: dict[str, float],
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "count": 0.0}
    for name in HEADS:
        totals[f"{name}_loss"] = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            batch_loss = torch.zeros((), dtype=torch.float32, device=device)
            count = float(len(xb))
            for col, name in enumerate(HEADS):
                target = yb[:, col]
                if name in {"reward", "final_goal", "score"}:
                    loss = mse(pred[name], target)
                else:
                    loss = bce_losses[name](pred[name], target)
                batch_loss = batch_loss + loss_weights[name] * loss
                totals[f"{name}_loss"] += float(loss.item()) * count
            totals["loss"] += float(batch_loss.item()) * count
            totals["count"] += count
    count = max(totals.pop("count"), 1.0)
    return {name: value / count for name, value in totals.items()}


def compute_calibration(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    quantile: float,
) -> dict[str, object]:
    model.eval()
    preds: dict[str, list[np.ndarray]] = {name: [] for name in HEADS}
    targets: dict[str, list[np.ndarray]] = {name: [] for name in HEADS}
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            raw = model(xb)
            y_np = yb.detach().cpu().numpy()
            for col, name in enumerate(HEADS):
                value = raw[name].detach().float().cpu().numpy()
                if name not in {"reward", "final_goal", "score"}:
                    value = 1.0 / (1.0 + np.exp(-np.clip(value, -20.0, 20.0)))
                preds[name].append(value.astype(np.float32))
                targets[name].append(y_np[:, col].astype(np.float32))

    q = float(np.clip(quantile, 0.5, 0.99))
    out: dict[str, object] = {"quantile": q, "heads": {}}
    for name in HEADS:
        pred = np.concatenate(preds[name], axis=0)
        target = np.concatenate(targets[name], axis=0)
        err = pred - target
        head = {
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "abs_q": float(np.quantile(np.abs(err), q)),
        }
        if name not in {"reward", "final_goal", "score"}:
            # LCB margin protects against over-prediction, UCB against under-prediction.
            head["lcb_margin"] = float(np.quantile(np.maximum(err, 0.0), q))
            head["ucb_margin"] = float(np.quantile(np.maximum(-err, 0.0), q))
            head["target_mean"] = float(np.mean(target))
            head["pred_mean"] = float(np.mean(pred))
        out["heads"][name] = head
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--augment-graph-features", choices=["none", "pairwise", "temporal", "full"], default="full")
    parser.add_argument("--reward-weight", type=float, default=1.0)
    parser.add_argument("--final-goal-weight", type=float, default=0.4)
    parser.add_argument("--score-weight", type=float, default=0.4)
    parser.add_argument("--success-weight", type=float, default=2.0)
    parser.add_argument("--risk-weight", type=float, default=1.5)
    parser.add_argument("--critical-risk-weight", type=float, default=2.5)
    parser.add_argument("--collision-weight", type=float, default=2.0)
    parser.add_argument("--deadlock-weight", type=float, default=0.5)
    parser.add_argument("--calibration-quantile", type=float, default=0.9)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)

    data = np.load(args.dataset, allow_pickle=False)
    metadata = json.loads(str(data["metadata"]))
    features, feature_names = build_features(data, metadata, args.augment_graph_features)
    alpha = np.asarray(data["alpha"], dtype=np.float32).reshape(-1)
    if len(features) != len(alpha):
        raise SystemExit(f"Feature/alpha length mismatch: {len(features)} vs {len(alpha)}")

    train_idx, val_idx = split_by_source(np.asarray(data["source_index"], dtype=np.int32), args.val_fraction, args.seed)
    train_features = features[train_idx]
    feature_mean = np.mean(train_features, axis=0).astype(np.float32)
    feature_std = np.maximum(np.std(train_features, axis=0), 1e-6).astype(np.float32)
    features = ((features - feature_mean) / feature_std).astype(np.float32)
    x, input_names = make_action_conditioned_input(features, alpha)
    targets, target_stats = standardize_targets(data)
    y = make_target_matrix(targets)

    train_ds = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]))
    val_ds = TensorDataset(torch.from_numpy(x[val_idx]), torch.from_numpy(y[val_idx]))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)

    model = ActionConditionedOutcomeCritic(
        input_dim=x.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse = nn.MSELoss()
    bce_losses = {
        name: nn.BCEWithLogitsLoss(pos_weight=torch.tensor(target_stats[name]["pos_weight"], device=device))
        for name in ["success", "risk_lt_1_0", "risk_lt_0_65", "collision", "deadlock"]
    }
    loss_weights = {
        "reward": args.reward_weight,
        "final_goal": args.final_goal_weight,
        "score": args.score_weight,
        "success": args.success_weight,
        "risk_lt_1_0": args.risk_weight,
        "risk_lt_0_65": args.critical_risk_weight,
        "collision": args.collision_weight,
        "deadlock": args.deadlock_weight,
    }
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else out.with_suffix(".metrics.csv")
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    best_val = math.inf
    best_state = None
    rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                pred = model(xb)
                loss = torch.zeros((), dtype=torch.float32, device=device)
                for col, name in enumerate(HEADS):
                    target = yb[:, col]
                    if name in {"reward", "final_goal", "score"}:
                        head_loss = mse(pred[name], target)
                    else:
                        head_loss = bce_losses[name](pred[name], target)
                    loss = loss + loss_weights[name] * head_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.detach().cpu().item()) * len(xb)
            train_count += len(xb)

        val = evaluate(model, val_loader, mse, bce_losses, device, loss_weights)
        train_loss = train_loss_sum / max(train_count, 1)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}}
        rows.append(row)
        print(
            "epoch={epoch} train={train:.5f} val={val:.5f} success={success:.5f} risk={risk:.5f}".format(
                epoch=epoch,
                train=train_loss,
                val=val["loss"],
                success=val["success_loss"],
                risk=val["risk_lt_1_0_loss"],
            ),
            flush=True,
        )
        if val["loss"] < best_val:
            best_val = val["loss"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if best_state is not None:
        model.load_state_dict(best_state)
    calibration = compute_calibration(model, val_loader, device, args.calibration_quantile)
    payload = {
        "model_type": "action_conditioned_outcome_critic",
        "state_dict": model.state_dict(),
        "feature_names": feature_names,
        "input_names": input_names,
        "input_dim": x.shape[1],
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_stats": target_stats,
        "heads": HEADS,
        "dataset": str(args.dataset),
        "dataset_metadata": metadata,
        "args": vars(args),
        "best_val_loss": best_val,
        "calibration": calibration,
    }
    torch.save(payload, out)
    print(f"Wrote {out} best_val_loss={best_val:.6f} metrics={metrics_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
