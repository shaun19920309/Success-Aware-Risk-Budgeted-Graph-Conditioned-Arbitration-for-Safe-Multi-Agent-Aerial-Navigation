#!/usr/bin/env python3
"""Regression tests for the formal 7 s multiseed workflow."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from analyze_horizon7_formal_multiseed import (
    hierarchical_ci,
    hierarchical_mean_ci,
    write_csv,
)
from evaluate_horizon7_formal_multiseed import (
    DEFAULT_BC_SEEDS,
    DEFAULT_EVAL_SEEDS,
    DEFAULT_RESULT_ROOT,
    DEFAULT_TRAIN_SEEDS,
    verify_preregistered_protocol,
)
from report_horizon7_formal_training_status import latest_progress
from verify_horizon7_formal_training import (
    METHODS,
    checkpoint_manifest,
    duplicate_policy_conflicts,
    lagrangian_correction_errors,
)


class HierarchicalBootstrapTest(unittest.TestCase):
    def test_constant_positive_effect_has_point_interval(self) -> None:
        proposed = np.full((3, 8), 2.0, dtype=np.float64)
        baseline = np.full((3, 8), 0.5, dtype=np.float64)

        low, high = hierarchical_ci(proposed, baseline, n_boot=500, seed=17)

        self.assertAlmostEqual(low, 1.5)
        self.assertAlmostEqual(high, 1.5)

    def test_bootstrap_is_reproducible_and_preserves_direction(self) -> None:
        # Keep every proposed training seed above every baseline training seed.
        # Training seeds are independent replicates and are resampled separately.
        proposed = np.arange(24, dtype=np.float64).reshape(3, 8) + 30.0
        baseline = np.arange(24, dtype=np.float64).reshape(3, 8)

        first = hierarchical_ci(proposed, baseline, n_boot=1000, seed=23)
        second = hierarchical_ci(proposed, baseline, n_boot=1000, seed=23)

        self.assertEqual(first, second)
        self.assertGreater(first[0], 0.0)

    def test_hierarchical_mean_interval_resamples_both_axes(self) -> None:
        values = np.full((3, 16), 0.75, dtype=np.float64)

        low, high = hierarchical_mean_ci(values, n_boot=500, seed=29)

        self.assertAlmostEqual(low, 0.75)
        self.assertAlmostEqual(high, 0.75)


class TrainingProgressParserTest(unittest.TestCase):
    def test_latest_seed_and_step_are_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.log"
            path.write_text(
                "seed240000 updates 1/10 episodes, total num timesteps 128/1000000\n"
                "seed240001 updates 2/10 episodes, total num timesteps 256/1000000\n",
                encoding="utf-8",
            )

            self.assertEqual(latest_progress(path), (240001, 256, 1_000_000))


class CsvWriterTest(unittest.TestCase):
    def test_union_of_row_fields_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"

            write_csv(path, [{"method": "a"}, {"method": "all", "sd": 0.1}])

            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                ["method,sd", "a,", "all,0.1"],
            )


class PreregisteredProtocolTest(unittest.TestCase):
    def test_frozen_checksum_and_splits_match_code(self) -> None:
        path, payload = verify_preregistered_protocol(
            DEFAULT_RESULT_ROOT,
            list(METHODS),
            DEFAULT_TRAIN_SEEDS,
            DEFAULT_BC_SEEDS,
            DEFAULT_EVAL_SEEDS,
        )

        self.assertTrue(path.is_file())
        self.assertTrue(payload["frozen_before_effect_evaluation"])


class CheckpointIntegrityTest(unittest.TestCase):
    def test_duplicate_policy_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mappo_config = root / "mappo/run1/config.json"
            lagrangian_config = root / "lagrangian/run1/config.json"
            for config in (mappo_config, lagrangian_config):
                model_dir = config.parent / "models"
                model_dir.mkdir(parents=True)
                (model_dir / "actor.pt").write_bytes(b"same policy")
                (model_dir / "critic.pt").write_bytes(config.as_posix().encode())

            manifests = {
                ("mappo", 7): checkpoint_manifest(mappo_config, "onpolicy", "mappo"),
                ("lagrangian", 7): checkpoint_manifest(
                    lagrangian_config, "onpolicy", "mappo_lagrangian"
                ),
            }

            conflicts = duplicate_policy_conflicts(manifests)

            self.assertIn(("mappo", 7), conflicts)
            self.assertIn(("lagrangian", 7), conflicts)
            self.assertIn("duplicate policy checkpoint", conflicts[("mappo", 7)][0])

    def test_partial_corrected_lagrangian_run_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            formal_root = Path(directory)
            training_root = formal_root / "training"
            config = training_root / "lagrangian/run2/config.json"
            model_dir = config.parent / "models"
            model_dir.mkdir(parents=True)
            addendum = formal_root / "formal_multiseed_correction_addendum_20260831.json"
            addendum.write_text(
                json.dumps(
                    {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "corrected_formal_training": {
                            "method": "lagrangian",
                            "training_seeds": [240000],
                        },
                    }
                ),
                encoding="utf-8",
            )
            config.write_text("{}", encoding="utf-8")
            (model_dir / "actor.pt").write_bytes(b"partial actor")
            (model_dir / "critic.pt").write_bytes(b"partial critic")
            now = addendum.stat().st_mtime + 1.0
            for path in (config, model_dir / "actor.pt", model_dir / "critic.pt"):
                os.utime(path, (now, now))

            errors = lagrangian_correction_errors(training_root, config, 240000)

            self.assertTrue(any("EXIT_STATUS:0" in error for error in errors))

            log_dir = formal_root / "logs"
            log_dir.mkdir()
            log = log_dir / "lagrangian_corr_20260831_s240000.log"
            log.write_text("training complete\nEXIT_STATUS:0\n", encoding="utf-8")
            os.utime(log, (now + 1.0, now + 1.0))

            self.assertEqual(
                lagrangian_correction_errors(training_root, config, 240000), []
            )


if __name__ == "__main__":
    unittest.main()
