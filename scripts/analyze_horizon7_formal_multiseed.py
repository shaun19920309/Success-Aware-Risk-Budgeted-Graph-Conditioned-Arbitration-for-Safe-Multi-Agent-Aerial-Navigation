#!/usr/bin/env python3
"""Analyze model-seed and environment-seed uncertainty for formal 7 s runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from formal_row_normalization import (
    normalize_pool_row,
    normalize_student_row,
    read_csv,
)
from final_revision_statistics import holm_adjust, monte_carlo_sign_flip_pvalue


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = ROOT / "results/final_formal_multiseed"
DISPLAY = {
    "proposed": "Proposed",
    "mappo": "MAPPO",
    "ippo": "IPPO",
    "lagrangian": "MAPPO-Lagrangian",
    "mat": "MAT",
    "hatrpo": "HATRPO",
}
SCENARIO_DISPLAY = {
    "obstacle4_nominal": "Obstacle-4 nominal",
    "obstacle8_dense_large": "Obstacle-8 dense/large",
    "obstacle8_sparse_small": "Obstacle-8 sparse/small",
}
METRICS = (
    ("success", "Success", "higher", "rate"),
    ("collision", "Collision", "lower", "rate"),
    ("deadlock", "Deadlock", "lower", "rate"),
    ("progress", "Goal progress", "higher", "value"),
    ("objective_s", "Objective/s", "higher", "value"),
)
DESCRIPTIVE_METRICS = METRICS + (
    ("risk_065", "Risk <0.65 m", "lower", "rate"),
    ("risk_100", "Risk <1.0 m", "lower", "rate"),
    ("moving", "Moving-frame ratio", "higher", "rate"),
    ("path_length", "Path length", "descriptive", "value"),
    ("final_goal_distance", "Final goal distance", "lower", "value"),
    ("transit_risk_065", "Transit risk <0.65 m", "lower", "rate"),
    ("transit_risk_100", "Transit risk <1.0 m", "lower", "rate"),
)
RUNTIME_METRICS = (
    "policy_ms_per_frame",
    "coordination_ms_per_frame",
    "end_to_end_ms_per_frame",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames = list(rows[0])
    seen = set(fieldnames)
    for row in rows[1:]:
        for field in row:
            if field not in seen:
                fieldnames.append(field)
                seen.add(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def one_csv_row(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, found {len(rows)}")
    return rows[0]


def load_matrix(
    result_root: Path,
    protocol: dict[str, object],
) -> tuple[
    dict[str, np.ndarray],
    tuple[int, ...],
    dict[str, tuple[int, ...]],
]:
    eval_seeds = tuple(int(seed) for seed in protocol["evaluation_seeds"])
    baseline_seeds = tuple(int(seed) for seed in protocol["baseline_training_seeds"])
    proposed_seeds = tuple(int(seed) for seed in protocol["proposed_training_seeds"])
    model_seeds = {"proposed": proposed_seeds}
    model_seeds.update({method: baseline_seeds for method in DISPLAY if method != "proposed"})

    rows: dict[str, list[list[dict[str, float | str | int]]]] = {method: [] for method in DISPLAY}
    for seed in proposed_seeds:
        path = (
            result_root
            / "evaluation/proposed"
            / f"train_seed{seed}/distilled_student_seed_rows.csv"
        )
        normalized = [normalize_student_row(row) for row in read_csv(path)]
        keyed = {int(row["seed"]): row for row in normalized}
        if tuple(sorted(keyed)) != eval_seeds:
            raise ValueError(f"Unexpected proposed eval seeds in {path}")
        rows["proposed"].append([keyed[eval_seed] for eval_seed in eval_seeds])

    for method in DISPLAY:
        if method == "proposed":
            continue
        for train_seed in baseline_seeds:
            variant = f"{method}_train{train_seed}"
            normalized = []
            for eval_seed in eval_seeds:
                path = (
                    result_root
                    / "evaluation/baselines"
                    / variant
                    / f"quad_eval_seed{eval_seed}/eval_summary.csv"
                )
                normalized.append(normalize_pool_row(one_csv_row(path)))
            rows[method].append(normalized)

    reference_hashes = [str(row["physical_hash"]) for row in rows["proposed"][0]]
    for method, method_models in rows.items():
        for model_index, model_rows in enumerate(method_models):
            if any(int(row["frames"]) != 701 for row in model_rows):
                raise ValueError(f"Noncanonical frame count: {method} model {model_index}")
            hashes = [str(row["physical_hash"]) for row in model_rows]
            if hashes != reference_hashes:
                raise ValueError(f"Physical-state hash mismatch: {method} model {model_index}")

    matrices = {}
    for method, method_models in rows.items():
        matrices[method] = np.asarray(
            [
                [[float(row[metric]) for metric, *_rest in DESCRIPTIVE_METRICS] for row in model_rows]
                for model_rows in method_models
            ],
            dtype=np.float64,
        )
    return matrices, eval_seeds, model_seeds


def hierarchical_ci(
    proposed: np.ndarray,
    baseline: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    remaining = n_boot
    chunk_size = 1000
    while remaining:
        count = min(chunk_size, remaining)
        p_index = rng.integers(0, proposed.shape[0], size=(count, proposed.shape[0]))
        b_index = rng.integers(0, baseline.shape[0], size=(count, baseline.shape[0]))
        e_index = rng.integers(0, proposed.shape[1], size=(count, proposed.shape[1]))
        p_sample = proposed[p_index[:, :, None], e_index[:, None, :]].mean(axis=(1, 2))
        b_sample = baseline[b_index[:, :, None], e_index[:, None, :]].mean(axis=(1, 2))
        values.append(p_sample - b_sample)
        remaining -= count
    samples = np.concatenate(values)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def hierarchical_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = []
    remaining = n_boot
    chunk_size = 1000
    while remaining:
        count = min(chunk_size, remaining)
        model_index = rng.integers(0, values.shape[0], size=(count, values.shape[0]))
        env_index = rng.integers(0, values.shape[1], size=(count, values.shape[1]))
        sampled = values[model_index[:, :, None], env_index[:, None, :]].mean(
            axis=(1, 2)
        )
        samples.append(sampled)
        remaining -= count
    low, high = np.quantile(np.concatenate(samples), [0.025, 0.975])
    return float(low), float(high)


def load_proposed_generalization(
    result_root: Path,
    protocol: dict[str, object],
) -> dict[str, tuple[np.ndarray, tuple[int, ...]]]:
    proposed_seeds = tuple(int(seed) for seed in protocol["proposed_training_seeds"])
    scenarios = protocol.get("generalization_scenarios", {})
    if not isinstance(scenarios, dict) or set(scenarios) != set(SCENARIO_DISPLAY):
        raise ValueError("Missing or unexpected formal generalization scenarios")

    result = {}
    for scenario, config_object in scenarios.items():
        if not isinstance(config_object, dict):
            raise TypeError(f"Invalid scenario config for {scenario}")
        eval_seeds = tuple(int(seed) for seed in config_object["evaluation_seeds"])
        model_rows = []
        for train_seed in proposed_seeds:
            path = (
                result_root
                / "evaluation/generalization"
                / scenario
                / "proposed"
                / f"train_seed{train_seed}/distilled_student_seed_rows.csv"
            )
            normalized = [normalize_student_row(row) for row in read_csv(path)]
            keyed = {int(row["seed"]): row for row in normalized}
            if tuple(sorted(keyed)) != eval_seeds:
                raise ValueError(f"Unexpected generalization eval seeds in {path}")
            model_rows.append([keyed[seed] for seed in eval_seeds])

        reference_hashes = [str(row["physical_hash"]) for row in model_rows[0]]
        for model_index, rows in enumerate(model_rows):
            if any(int(row["frames"]) != 701 for row in rows):
                raise ValueError(
                    f"Noncanonical frame count: {scenario} model {model_index}"
                )
            if [str(row["physical_hash"]) for row in rows] != reference_hashes:
                raise ValueError(
                    f"Physical-state hash mismatch: {scenario} model {model_index}"
                )
        matrix = np.asarray(
            [
                [
                    [float(row[metric]) for metric, *_rest in DESCRIPTIVE_METRICS]
                    for row in rows
                ]
                for rows in model_rows
            ],
            dtype=np.float64,
        )
        result[scenario] = (matrix, eval_seeds)
    return result


def load_runtime_rows(
    result_root: Path,
    protocol: dict[str, object],
) -> list[dict[str, object]]:
    runtime_protocol_path = result_root / "formal_runtime_protocol.json"
    runtime_protocol = json.loads(runtime_protocol_path.read_text(encoding="utf-8"))
    effect_protocol_path = result_root / "formal_multiseed_protocol.json"
    if runtime_protocol.get("formal_effect_protocol_sha256") != sha256(effect_protocol_path):
        raise ValueError("Runtime protocol does not match the formal effect protocol")
    runtime_seeds = tuple(int(seed) for seed in runtime_protocol["runtime_environment_seeds"])
    model_seeds = {
        "proposed": tuple(int(seed) for seed in protocol["proposed_training_seeds"])
    }
    baseline_seeds = tuple(int(seed) for seed in protocol["baseline_training_seeds"])
    model_seeds.update({method: baseline_seeds for method in DISPLAY if method != "proposed"})

    raw: dict[str, list[list[dict[str, str]]]] = {method: [] for method in DISPLAY}
    for train_seed in model_seeds["proposed"]:
        path = (
            result_root
            / "runtime/proposed"
            / f"train_seed{train_seed}/distilled_student_seed_rows.csv"
        )
        keyed = {int(row["seed"]): row for row in read_csv(path)}
        if tuple(sorted(keyed)) != runtime_seeds:
            raise ValueError(f"Unexpected proposed runtime seeds in {path}")
        raw["proposed"].append([keyed[seed] for seed in runtime_seeds])

    for method in DISPLAY:
        if method == "proposed":
            continue
        for train_seed in model_seeds[method]:
            variant = f"{method}_train{train_seed}"
            rows = []
            for seed in runtime_seeds:
                path = (
                    result_root
                    / "runtime/baselines"
                    / variant
                    / f"quad_eval_seed{seed}/eval_summary.csv"
                )
                rows.append(one_csv_row(path))
            raw[method].append(rows)

    reference_hashes = [
        str(row["initial_physical_state_sha256"]) for row in raw["proposed"][0]
    ]
    for method, models in raw.items():
        for model_index, rows in enumerate(models):
            if any(int(row["frames"]) != 701 for row in rows):
                raise ValueError(f"Noncanonical runtime frames: {method} model {model_index}")
            hashes = [str(row["initial_physical_state_sha256"]) for row in rows]
            if hashes != reference_hashes:
                raise ValueError(f"Runtime physical hash mismatch: {method} model {model_index}")

    output = []
    for method, models in raw.items():
        model_metric_rows = []
        for model_index, (train_seed, rows) in enumerate(zip(model_seeds[method], models)):
            if method == "proposed":
                policy = np.asarray(
                    [float(row["policy_wall_ms_per_frame"]) for row in rows]
                )
                coordination = np.asarray(
                    [float(row["coordinator_wall_ms_per_frame"]) for row in rows]
                )
                end_to_end = np.asarray(
                    [
                        1000.0 * float(row["episode_wall_seconds"]) / float(row["frames"])
                        for row in rows
                    ]
                )
            else:
                policy = np.asarray(
                    [float(row["expert_inference_ms_per_frame"]) for row in rows]
                )
                coordination = np.asarray(
                    [float(row["gate_and_mix_ms_per_frame"]) for row in rows]
                )
                end_to_end = np.asarray(
                    [float(row["end_to_end_ms_per_frame"]) for row in rows]
                )
            metric_values = {
                "policy_ms_per_frame": float(policy.mean()),
                "coordination_ms_per_frame": float(coordination.mean()),
                "end_to_end_ms_per_frame": float(end_to_end.mean()),
            }
            model_metric_rows.append(metric_values)
            output.append(
                {
                    "method": method,
                    "display": DISPLAY[method],
                    "training_seed": train_seed,
                    "runtime_environment_seed_count": len(runtime_seeds),
                    **metric_values,
                }
            )
        aggregate: dict[str, object] = {
            "method": method,
            "display": DISPLAY[method],
            "training_seed": "all",
            "runtime_environment_seed_count": len(runtime_seeds),
        }
        for metric in RUNTIME_METRICS:
            values = np.asarray([row[metric] for row in model_metric_rows])
            aggregate[metric] = float(values.mean())
            aggregate[f"{metric}_training_seed_sd"] = float(values.std(ddof=1))
        output.append(aggregate)
    return output


def fmt(value: float, kind: str, signed: bool = False) -> str:
    prefix = "+" if signed else ""
    if kind == "rate":
        return f"{value * 100:{prefix}.2f}%"
    return f"{value:{prefix}.4f}"


def verify_training_integrity(
    result_root: Path, protocol: dict[str, object]
) -> None:
    methods = [method for method in DISPLAY if method != "proposed"]
    command = [
        sys.executable,
        str(ROOT / "scripts/verify_horizon7_formal_training.py"),
        "--result-root",
        str(result_root / "training"),
        "--seeds",
        *(str(seed) for seed in protocol["baseline_training_seeds"]),
        "--methods",
        *methods,
    ]
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--n-bootstrap", type=int, default=100_000)
    parser.add_argument("--n-sign-flips", type=int, default=200_000)
    parser.add_argument("--analysis-seed", type=int, default=20260827)
    parser.add_argument(
        "--skip-training-integrity",
        action="store_true",
        help=(
            "Reanalyze the compact public evidence package without large baseline "
            "checkpoint trees. Package SHA-256 verification remains mandatory."
        ),
    )
    args = parser.parse_args()

    protocol_path = args.result_root / "formal_multiseed_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not args.skip_training_integrity:
        verify_training_integrity(args.result_root, protocol)
    matrices, eval_seeds, model_seeds = load_matrix(args.result_root, protocol)
    generalization = load_proposed_generalization(args.result_root, protocol)
    runtime_rows = load_runtime_rows(args.result_root, protocol)
    output_root = args.result_root / "analysis"

    mean_rows = []
    metric_index = {name: index for index, (name, *_rest) in enumerate(DESCRIPTIVE_METRICS)}
    for method, matrix in matrices.items():
        for model_index, train_seed in enumerate(model_seeds[method]):
            row: dict[str, object] = {
                "method": method,
                "display": DISPLAY[method],
                "training_seed": train_seed,
                "environment_seed_count": len(eval_seeds),
            }
            for metric, *_rest in DESCRIPTIVE_METRICS:
                row[metric] = float(matrix[model_index, :, metric_index[metric]].mean())
            mean_rows.append(row)
        row = {
            "method": method,
            "display": DISPLAY[method],
            "training_seed": "all",
            "environment_seed_count": len(eval_seeds),
        }
        for metric, *_rest in DESCRIPTIVE_METRICS:
            model_means = matrix[:, :, metric_index[metric]].mean(axis=1)
            row[metric] = float(model_means.mean())
            row[f"{metric}_training_seed_sd"] = float(model_means.std(ddof=1))
        mean_rows.append(row)

    effect_rows = []
    raw_p_values = []
    proposed = matrices["proposed"]
    for method_index, method in enumerate(DISPLAY):
        if method == "proposed":
            continue
        baseline = matrices[method]
        for index, (metric, label, direction, kind) in enumerate(METRICS):
            column = metric_index[metric]
            p_values = proposed[:, :, column]
            b_values = baseline[:, :, column]
            env_deltas = p_values.mean(axis=0) - b_values.mean(axis=0)
            raw_p = monte_carlo_sign_flip_pvalue(
                env_deltas,
                n_draws=args.n_sign_flips,
                seed=args.analysis_seed + method_index * 100 + index,
            )
            low, high = hierarchical_ci(
                p_values,
                b_values,
                n_boot=args.n_bootstrap,
                seed=args.analysis_seed + method_index * 1000 + index,
            )
            cross_effects = (
                p_values.mean(axis=1)[:, None] - b_values.mean(axis=1)[None, :]
            )
            favorable = cross_effects > 0 if direction == "higher" else cross_effects < 0
            row = {
                "baseline": method,
                "baseline_display": DISPLAY[method],
                "metric": metric,
                "metric_label": label,
                "direction": direction,
                "kind": kind,
                "delta": float(env_deltas.mean()),
                "hierarchical_ci_low": low,
                "hierarchical_ci_high": high,
                "conditional_environment_p": raw_p,
                "holm_p": math.nan,
                "favorable_training_seed_pairs": int(favorable.sum()),
                "training_seed_pair_count": int(favorable.size),
            }
            effect_rows.append(row)
            raw_p_values.append(raw_p)

    generalization_rows = []
    proposed_seeds = tuple(int(seed) for seed in protocol["proposed_training_seeds"])
    for scenario_index, scenario in enumerate(SCENARIO_DISPLAY):
        matrix, scenario_eval_seeds = generalization[scenario]
        for model_index, train_seed in enumerate(proposed_seeds):
            row: dict[str, object] = {
                "scenario": scenario,
                "scenario_display": SCENARIO_DISPLAY[scenario],
                "training_seed": train_seed,
                "environment_seed_count": len(scenario_eval_seeds),
            }
            for metric, *_rest in DESCRIPTIVE_METRICS:
                row[metric] = float(matrix[model_index, :, metric_index[metric]].mean())
            generalization_rows.append(row)
        row = {
            "scenario": scenario,
            "scenario_display": SCENARIO_DISPLAY[scenario],
            "training_seed": "all",
            "environment_seed_count": len(scenario_eval_seeds),
        }
        for metric_offset, (metric, *_rest) in enumerate(DESCRIPTIVE_METRICS):
            values = matrix[:, :, metric_index[metric]]
            model_means = values.mean(axis=1)
            low, high = hierarchical_mean_ci(
                values,
                n_boot=args.n_bootstrap,
                seed=args.analysis_seed + 10_000 + scenario_index * 100 + metric_offset,
            )
            row[metric] = float(model_means.mean())
            row[f"{metric}_training_seed_sd"] = float(model_means.std(ddof=1))
            row[f"{metric}_hierarchical_ci_low"] = low
            row[f"{metric}_hierarchical_ci_high"] = high
        generalization_rows.append(row)

    adjusted = holm_adjust(raw_p_values)
    for row, value in zip(effect_rows, adjusted):
        row["holm_p"] = value

    write_csv(output_root / "formal_multiseed_method_means.csv", mean_rows)
    write_csv(output_root / "formal_multiseed_effects.csv", effect_rows)
    write_csv(
        output_root / "formal_multiseed_proposed_generalization.csv",
        generalization_rows,
    )
    write_csv(output_root / "formal_multiseed_runtime.csv", runtime_rows)

    report = [
        "# Formal 7 s multiseed evidence",
        "",
        f"- Baseline training seeds: {list(protocol['baseline_training_seeds'])}",
        f"- Proposed training seeds: {list(protocol['proposed_training_seeds'])}",
        f"- Unseen evaluation seeds: {eval_seeds[0]}--{eval_seeds[-1]} (n={len(eval_seeds)})",
        "- Every rollout has 701 frames and matched physical-state hashes.",
        "- Confidence intervals resample training seeds and matched environment seeds hierarchically.",
        "- Sign-flip p-values operate on environment-seed deltas after averaging the three training seeds.",
        "",
        "## Grand means across training and environment seeds",
        "",
        "| Method | Success | Collision | Deadlock | Progress | Objective/s | Moving | Risk <0.65 | Risk <1.0 | Transit <0.65 | Transit <1.0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in DISPLAY:
        row = next(
            item for item in mean_rows if item["method"] == method and item["training_seed"] == "all"
        )
        report.append(
            "| "
            + " | ".join(
                [
                    DISPLAY[method],
                    fmt(float(row["success"]), "rate"),
                    fmt(float(row["collision"]), "rate"),
                    fmt(float(row["deadlock"]), "rate"),
                    fmt(float(row["progress"]), "value"),
                    fmt(float(row["objective_s"]), "value"),
                    fmt(float(row["moving"]), "rate"),
                    fmt(float(row["risk_065"]), "rate"),
                    fmt(float(row["risk_100"]), "rate"),
                    fmt(float(row["transit_risk_065"]), "rate"),
                    fmt(float(row["transit_risk_100"]), "rate"),
                ]
            )
            + " |"
        )

    report.extend(
        [
            "",
            "## Proposed generalization across training and environment seeds",
            "",
            "These suites quantify the proposed controller's robustness only; they do not reuse the old 1 s-trained baselines as comparators.",
            "",
            "| Scenario | Success (95% hierarchical CI) | Collision (95% hierarchical CI) | Deadlock | Progress | Objective/s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIO_DISPLAY:
        row = next(
            item
            for item in generalization_rows
            if item["scenario"] == scenario and item["training_seed"] == "all"
        )
        report.append(
            f"| {SCENARIO_DISPLAY[scenario]} | "
            f"{fmt(float(row['success']), 'rate')} "
            f"[{fmt(float(row['success_hierarchical_ci_low']), 'rate')}, "
            f"{fmt(float(row['success_hierarchical_ci_high']), 'rate')}] | "
            f"{fmt(float(row['collision']), 'rate')} "
            f"[{fmt(float(row['collision_hierarchical_ci_low']), 'rate')}, "
            f"{fmt(float(row['collision_hierarchical_ci_high']), 'rate')}] | "
            f"{fmt(float(row['deadlock']), 'rate')} | "
            f"{fmt(float(row['progress']), 'value')} | "
            f"{fmt(float(row['objective_s']), 'value')} |"
        )

    report.extend(
        [
            "",
            "## Isolated sequential RTX 5090 runtime",
            "",
            "| Method | Policy ms/frame | Coordination ms/frame | End-to-end ms/frame |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in DISPLAY:
        row = next(
            item
            for item in runtime_rows
            if item["method"] == method and item["training_seed"] == "all"
        )
        report.append(
            f"| {DISPLAY[method]} | "
            f"{float(row['policy_ms_per_frame']):.3f} | "
            f"{float(row['coordination_ms_per_frame']):.3f} | "
            f"{float(row['end_to_end_ms_per_frame']):.3f} |"
        )

    report.extend(
        [
            "",
            "## Proposed-minus-baseline effects",
            "",
            "| Baseline | Metric | Delta | Hierarchical 95% CI | Conditional Holm p | Favorable model-seed pairs |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in effect_rows:
        report.append(
            f"| {row['baseline_display']} | {row['metric_label']} | "
            f"{fmt(float(row['delta']), str(row['kind']), True)} | "
            f"[{fmt(float(row['hierarchical_ci_low']), str(row['kind']), True)}, "
            f"{fmt(float(row['hierarchical_ci_high']), str(row['kind']), True)}] | "
            f"{float(row['holm_p']):.6f} | "
            f"{row['favorable_training_seed_pairs']}/{row['training_seed_pair_count']} |"
        )

    report.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "A superiority claim requires the hierarchical interval to exclude zero in the favorable direction, a Holm-adjusted conditional p-value below 0.05, and favorable effects for all nine cross-training-seed pairs. Raw proximity exposure remains descriptive and is interpreted jointly with completion and progress.",
        ]
    )
    report_path = output_root / "formal_multiseed_evidence_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
