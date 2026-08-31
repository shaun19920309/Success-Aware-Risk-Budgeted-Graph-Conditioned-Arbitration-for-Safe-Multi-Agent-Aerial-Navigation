#!/usr/bin/env python3
"""Validate the compact formal release and all paper-facing values."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/final_formal_multiseed"
BASELINE_METHODS = ("mappo", "ippo", "lagrangian", "mat", "hatrpo")
BASELINE_SEEDS = (240000, 240001, 240002)
PROPOSED_SEEDS = (171001, 171002, 171003)
EVAL_SEEDS = tuple(range(250000, 250032))
RUNTIME_SEEDS = (254000, 254001, 254002)
GENERALIZATION = {
    "obstacle4_nominal": tuple(range(251000, 251016)),
    "obstacle8_dense_large": tuple(range(252000, 252016)),
    "obstacle8_sparse_small": tuple(range(253000, 253016)),
}
EXPECTED_TRAIN = "fa8fbd768041b0277dec563b60b212bfc01a9ed23b625e34c72d98c241a44ce4"
EXPECTED_VALIDATION = "5eae3c6e9da8adc0b389ed1169c36ff0ca2dfa0fd310030000428b12c5b578e7"


def fail(message: str) -> None:
    raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def one_row(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    if len(rows) != 1:
        fail(f"Expected one row in {path.relative_to(ROOT)}, found {len(rows)}")
    return rows[0]


def assert_close(actual: float, expected: float, label: str, tolerance: float = 5e-6) -> None:
    if abs(actual - expected) > tolerance:
        fail(f"{label}: expected {expected}, found {actual}")


def check_checksum_file(path: Path) -> None:
    checksum = path.with_suffix(".sha256")
    expected = checksum.read_text(encoding="ascii").split()[0]
    if sha256(path) != expected:
        fail(f"Checksum mismatch: {path.relative_to(ROOT)}")


def check_manifest() -> None:
    manifest = ROOT / "manifests/package_sha256.csv"
    if not manifest.is_file():
        fail("Package manifest is missing")
    rows = read_csv(manifest)
    if not rows:
        fail("Package manifest is empty")
    for row in rows:
        path = ROOT / row["relative_path"]
        if not path.is_file():
            fail(f"Manifest file is missing: {row['relative_path']}")
        if int(row["bytes"]) != path.stat().st_size:
            fail(f"Manifest size mismatch: {row['relative_path']}")
        if sha256(path) != row["sha256"]:
            fail(f"Manifest hash mismatch: {row['relative_path']}")


def check_protocols() -> dict[str, object]:
    prereg = RESULT_ROOT / "formal_multiseed_preregistered_protocol.json"
    correction = RESULT_ROOT / "formal_multiseed_correction_addendum_20260831.json"
    check_checksum_file(prereg)
    check_checksum_file(correction)
    frozen = json.loads(prereg.read_text(encoding="utf-8"))
    if tuple(frozen["training"]["baseline_training_seeds"]) != BASELINE_SEEDS:
        fail("Preregistered baseline training seeds differ")
    if tuple(frozen["training"]["proposed_training_seeds"]) != PROPOSED_SEEDS:
        fail("Preregistered proposed training seeds differ")
    if int(frozen["training"]["baseline_environment_steps"]) != 1_000_000:
        fail("Formal baseline training horizon is not 1,000,000 steps")
    if float(frozen["training"]["episode_duration_seconds"]) != 7.0:
        fail("Formal episode duration is not 7 seconds")
    if tuple(frozen["nominal_confirmation"]["evaluation_seeds"]) != EVAL_SEEDS:
        fail("Preregistered nominal evaluation seeds differ")

    addendum = json.loads(correction.read_text(encoding="utf-8"))
    expected_lagrangian = {
        "use_lagrangian": True,
        "lagrangian_cost_type": "hybrid",
        "lagrangian_cost_limit": 0.0,
        "lagrangian_lr": 0.05,
        "lagrangian_init": 1.0,
        "lagrangian_max": 20.0,
    }
    if addendum["corrected_formal_training"]["parameters"] != expected_lagrangian:
        fail("Corrected MAPPO-Lagrangian adapter parameters differ")
    if addendum["original_preregistered_protocol_sha256"] != sha256(prereg):
        fail("Correction addendum does not bind to the preregistration")

    protocol_path = RESULT_ROOT / "formal_multiseed_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if tuple(protocol["baseline_training_seeds"]) != BASELINE_SEEDS:
        fail("Formal protocol baseline seeds differ")
    if tuple(protocol["proposed_training_seeds"]) != PROPOSED_SEEDS:
        fail("Formal protocol proposed seeds differ")
    if tuple(protocol["evaluation_seeds"]) != EVAL_SEEDS:
        fail("Formal protocol evaluation seeds differ")
    runtime = json.loads((RESULT_ROOT / "formal_runtime_protocol.json").read_text(encoding="utf-8"))
    if tuple(runtime["runtime_environment_seeds"]) != RUNTIME_SEEDS:
        fail("Formal runtime seeds differ")
    if runtime["formal_effect_protocol_sha256"] != sha256(protocol_path):
        fail("Runtime protocol is not bound to the effect protocol")
    return protocol


def check_training_assets(protocol: dict[str, object]) -> None:
    train = ROOT / "data/training/teacher_train_160000_160031.npz"
    validation = ROOT / "data/training/teacher_validation_161000_161003.npz"
    if sha256(train) != EXPECTED_TRAIN or sha256(validation) != EXPECTED_VALIDATION:
        fail("Teacher dataset SHA-256 mismatch")
    with np.load(train, allow_pickle=False) as payload:
        if len(payload["observations"]) != 179_456:
            fail("Unexpected teacher training sample count")
    with np.load(validation, allow_pickle=False) as payload:
        if len(payload["observations"]) != 22_432:
            fail("Unexpected teacher validation sample count")

    for seed in PROPOSED_SEEDS:
        run = RESULT_ROOT / f"training/proposed_bc/seed{seed}"
        checkpoint = run / "models/student.pt"
        manifest = json.loads((run / "final_bc_manifest.json").read_text(encoding="utf-8"))
        digest = sha256(checkpoint)
        if int(manifest["training_seed"]) != seed or manifest["checkpoint_sha256"] != digest:
            fail(f"Proposed checkpoint manifest mismatch for seed {seed}")
        if protocol["proposed_runs"][str(seed)]["checkpoint_sha256"] != digest:
            fail(f"Formal protocol checkpoint mismatch for seed {seed}")


def proposed_rows(path: Path, seeds: tuple[int, ...]) -> list[dict[str, str]]:
    rows = read_csv(path)
    if tuple(sorted(int(row["seed"]) for row in rows)) != seeds:
        fail(f"Seed rows mismatch: {path.relative_to(ROOT)}")
    if any(int(row["frames"]) != 701 for row in rows):
        fail(f"Noncanonical frame count: {path.relative_to(ROOT)}")
    return rows


def check_nominal_matrix() -> None:
    reference: dict[int, str] = {}
    for seed in PROPOSED_SEEDS:
        rows = proposed_rows(
            RESULT_ROOT / f"evaluation/proposed/train_seed{seed}/distilled_student_seed_rows.csv",
            EVAL_SEEDS,
        )
        current = {int(row["seed"]): row["initial_physical_state_sha256"] for row in rows}
        if not reference:
            reference = current
        elif current != reference:
            fail("Proposed training seeds did not use matched physical initial states")

    count = 0
    for method in BASELINE_METHODS:
        for train_seed in BASELINE_SEEDS:
            for eval_seed in EVAL_SEEDS:
                row = one_row(
                    RESULT_ROOT
                    / f"evaluation/baselines/{method}_train{train_seed}"
                    / f"quad_eval_seed{eval_seed}/eval_summary.csv"
                )
                if int(row["seed"]) != eval_seed or int(row["frames"]) != 701:
                    fail(f"Invalid {method} row for train {train_seed}, eval {eval_seed}")
                if row["initial_physical_state_sha256"] != reference[eval_seed]:
                    fail(f"Physical-state mismatch for {method}, train {train_seed}, eval {eval_seed}")
                count += 1
    if count != 480:
        fail(f"Expected 480 baseline nominal rows, found {count}")


def check_generalization_and_runtime() -> None:
    for scenario, seeds in GENERALIZATION.items():
        reference: dict[int, str] = {}
        for train_seed in PROPOSED_SEEDS:
            rows = proposed_rows(
                RESULT_ROOT
                / f"evaluation/generalization/{scenario}/proposed/train_seed{train_seed}"
                / "distilled_student_seed_rows.csv",
                seeds,
            )
            current = {int(row["seed"]): row["initial_physical_state_sha256"] for row in rows}
            if not reference:
                reference = current
            elif current != reference:
                fail(f"Generalization physical-state mismatch: {scenario}")

    reference: dict[int, str] = {}
    for train_seed in PROPOSED_SEEDS:
        rows = proposed_rows(
            RESULT_ROOT / f"runtime/proposed/train_seed{train_seed}/distilled_student_seed_rows.csv",
            RUNTIME_SEEDS,
        )
        current = {int(row["seed"]): row["initial_physical_state_sha256"] for row in rows}
        if not reference:
            reference = current
        elif current != reference:
            fail("Proposed runtime physical-state mismatch")
    for method in BASELINE_METHODS:
        for train_seed in BASELINE_SEEDS:
            for eval_seed in RUNTIME_SEEDS:
                row = one_row(
                    RESULT_ROOT
                    / f"runtime/baselines/{method}_train{train_seed}"
                    / f"quad_eval_seed{eval_seed}/eval_summary.csv"
                )
                if int(row["frames"]) != 701:
                    fail(f"Noncanonical runtime frames for {method}")
                if row["initial_physical_state_sha256"] != reference[eval_seed]:
                    fail(f"Runtime physical-state mismatch for {method}")


def check_paper_values() -> None:
    means = read_csv(RESULT_ROOT / "analysis/formal_multiseed_method_means.csv")
    grand = {row["method"]: row for row in means if row["training_seed"] == "all"}
    expected = {
        "proposed": (0.9127604167, 0.07421875, 0.0130208333, 2.3595063317, -0.5637748353),
        "mappo": (0.0013020833, 0.6927083333, 0.3059895833, -1.2893179005, -4.0646713726),
        "ippo": (0.0026041667, 0.4791666667, 0.5182291667, -1.1764425505, -3.1618495574),
        "lagrangian": (0.0013020833, 0.6028645833, 0.3958333333, -1.5334082990, -3.9950825870),
        "mat": (0.0026041667, 0.5833333333, 0.4140625, -1.5678240173, -4.2211178857),
        "hatrpo": (0.0, 0.7721354167, 0.2278645833, -1.3040315795, -4.7276377253),
    }
    for method, values in expected.items():
        for key, value in zip(("success", "collision", "deadlock", "progress", "objective_s"), values):
            assert_close(float(grand[method][key]), value, f"{method} {key}")

    effects = read_csv(RESULT_ROOT / "analysis/formal_multiseed_effects.csv")
    if len(effects) != 25:
        fail(f"Expected 25 primary effects, found {len(effects)}")
    for row in effects:
        low, high = float(row["hierarchical_ci_low"]), float(row["hierarchical_ci_high"])
        favorable = low > 0 if row["direction"] == "higher" else high < 0
        if not favorable or float(row["holm_p"]) >= 0.05:
            fail(f"Primary claim gate failed: {row['baseline']} {row['metric']}")
        if (int(row["favorable_training_seed_pairs"]), int(row["training_seed_pair_count"])) != (9, 9):
            fail(f"Cross-training-seed gate failed: {row['baseline']} {row['metric']}")
    gate = json.loads((RESULT_ROOT / "analysis/formal_multiseed_claim_gate.json").read_text(encoding="utf-8"))
    if not gate["all_25_primary_claims_accepted"] or len(gate["claims"]) != 25:
        fail("Machine-readable claim gate is not fully accepted")

    generalization = read_csv(RESULT_ROOT / "analysis/formal_multiseed_proposed_generalization.csv")
    ggrand = {row["scenario"]: row for row in generalization if row["training_seed"] == "all"}
    for scenario, success in {
        "obstacle4_nominal": 0.8802083333,
        "obstacle8_dense_large": 0.6979166667,
        "obstacle8_sparse_small": 0.9635416667,
    }.items():
        assert_close(float(ggrand[scenario]["success"]), success, f"{scenario} success")

    runtime = read_csv(RESULT_ROOT / "analysis/formal_multiseed_runtime.csv")
    rgrand = {row["method"]: row for row in runtime if row["training_seed"] == "all"}
    assert_close(float(rgrand["proposed"]["policy_ms_per_frame"]), 0.9036280264, "policy runtime")
    assert_close(float(rgrand["proposed"]["coordination_ms_per_frame"]), 2.4047172073, "coordination runtime")
    assert_close(float(rgrand["proposed"]["end_to_end_ms_per_frame"]), 9.0197186496, "end-to-end runtime")


def main() -> int:
    required = (
        ROOT / "README.md",
        ROOT / "docs/METHOD.md",
        ROOT / "docs/EXPERIMENTS.md",
        ROOT / "docs/RESULTS.md",
        ROOT / "scripts/analyze_horizon7_formal_multiseed.py",
        ROOT / "scripts/launch_horizon7_formal_multiseed.sh",
        ROOT / "results/final_component_ablation/component_ablation_means.csv",
    )
    for path in required:
        if not path.is_file():
            fail(f"Required file is missing: {path.relative_to(ROOT)}")
    if any(ROOT.rglob("*.tex")) or any(ROOT.rglob("main.pdf")):
        fail("The public code package must not contain manuscript source or PDF")
    check_manifest()
    protocol = check_protocols()
    check_training_assets(protocol)
    check_nominal_matrix()
    check_generalization_and_runtime()
    check_paper_values()
    print("Package verification passed")
    print("  formal models: 18/18")
    print("  nominal matrix: 6 methods x 3 training seeds x 32 matched environment seeds")
    print("  primary claims: 25/25 accepted under hierarchical CI + Holm + 9/9 gate")
    print("  manuscript files: excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
