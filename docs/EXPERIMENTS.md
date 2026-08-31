# Formal Experiment Protocol

## Environment

- WSL2 Ubuntu 24.04.2 LTS
- Python 3.9.25
- PyTorch 2.8.0+cu128
- NumPy 1.24.4
- NVIDIA GeForce RTX 5090, 32 GB
- 7.0 s episodes, exactly 701 simulator frames
- one rollout thread and one episode per physical seed

The source of truth is `results/final_formal_multiseed/formal_multiseed_preregistered_protocol.json`. Its SHA-256 sidecar freezes the protocol. The MAPPO-Lagrangian correction is separately frozen in `formal_multiseed_correction_addendum_20260831.json` and bound to the original preregistration by checksum.

## Training Matrix

Six methods are evaluated:

- Proposed bounded BC controller
- MAPPO
- IPPO
- MAPPO-Lagrangian
- MAT
- HATRPO

Each method has three independent training seeds. Proposed seeds are `171001..171003`; baseline seeds are `240000..240002`. Every learned baseline is trained for 1,000,000 environment steps using the same 7 s task horizon.

Shared baseline settings include:

```text
num_agents = 8
quads_mode = o_static_same_goal
use_obstacles = true
visible_neighbors = 2
episode_duration = 7.0 s
obstacle_density = 0.2
obstacle_size = 0.6
shared_goal_slot_radius = 0.45
rollout_threads = 1
episode_length = 128
```

The corrected MAPPO-Lagrangian configuration is:

```text
use_lagrangian = true
cost_type = hybrid
cost_limit = 0.0
lagrangian_lr = 0.05
lagrangian_init = 1.0
lagrangian_max = 20.0
```

An earlier adapter mapped the safety cost incorrectly and yielded duplicate MAPPO/Lagrangian behavior. Those outputs are invalid and are not in this package. The correction addendum was frozen before corrected training, and all three corrected runs are used here.

## Nominal Matched Test

Nominal evaluation seeds are `250000..250031`. Every one of the 18 trained models is evaluated on all 32 seeds, yielding:

```text
6 methods x 3 training seeds x 32 environment seeds = 576 model-seed evaluations.
```

The physical initial-state SHA-256 must match across all methods and training replicates for a given environment seed. `reproduce/verify_package.py` enforces this invariant.

## Proposed-Only Generalization

The three frozen proposed checkpoints are evaluated without retraining:

| Scenario | Agents | Density | Size | Seeds |
|---|---:|---:|---:|---|
| Obstacle-4 nominal | 4 | 0.2 | 0.6 | `251000..251015` |
| Obstacle-8 dense/large | 8 | 0.3 | 0.8 | `252000..252015` |
| Obstacle-8 sparse/small | 8 | 0.1 | 0.4 | `253000..253015` |

These suites quantify robustness of the proposed method only. Because baselines were not rerun in these shifted settings under the final multi-training-seed protocol, they must not be used for cross-method superiority claims.

## Isolated Runtime

Runtime seeds are `254000..254002`. Methods run serially with one worker and synchronized CUDA timing. Policy inference, coordination/gate overhead, and measured end-to-end frame time are recorded separately. Each reported method mean averages 3 training seeds x 3 runtime seeds.

## Metrics

For `N` agents, terminal success, collision, and deadlock are mutually exclusive indicators:

```text
Success = (1/N) sum_i 1[i reaches and dwells in the goal],
Collision = (1/N) sum_i 1[i has a canonical collision],
Deadlock = 1 - Success - Collision.
```

Goal progress is

```text
Progress = (1/N) sum_i (d_i(0) - d_i(T)),
```

where `d_i(t)` is Euclidean distance from agent `i` to its assigned physical goal. Positive progress means net motion toward the goal.

The simulator-native objective per second is

```text
Objective/s = (1/T) sum_t r_true(t),
```

where `r_true` is computed from the physical trajectory, not the shaped training objective. Higher values are better.

Moving-frame ratio, path length, final goal distance, and pairwise proximity exposure below `0.65 m` and `1.0 m` are descriptive diagnostics. Proximity is not a primary universal safety claim because a failed policy can avoid pairwise exposure by never entering the task region.

## Hierarchical Statistical Analysis

For each baseline and each primary metric, the analysis forms proposed-minus-baseline effects while preserving the two uncertainty axes: training seed and matched environment seed.

The hierarchical bootstrap independently resamples proposed training seeds, baseline training seeds, and matched environment seeds, then computes the grand-mean difference. Each 95% interval uses 100,000 resamples.

A conditional two-sided sign-flip/randomization test uses 200,000 draws. Holm correction controls family-wise error over the 25 primary comparisons (5 baselines x 5 metrics).

A paper claim is accepted only when all conditions hold:

1. the hierarchical 95% interval excludes zero in the favorable direction;
2. Holm-adjusted `p < 0.05`; and
3. all nine proposed-training-seed/baseline-training-seed pair effects have the favorable sign.

This gate is machine-readable in `results/final_formal_multiseed/analysis/formal_multiseed_claim_gate.json`.

## Commands

Train the three proposed controllers:

```bash
PYTHON=/home/xzl/miniconda3/envs/sci1-rl/bin/python \
  bash reproduce/train_final_bc.sh
```

Train one baseline family for all three formal seeds:

```bash
PY=/home/xzl/miniconda3/envs/sci1-rl/bin/python \
  bash scripts/launch_horizon7_formal_multiseed.sh METHOD
```

Recompute compact-release statistics:

```bash
PYTHON=/home/xzl/miniconda3/envs/sci1-rl/bin/python \
  bash reproduce/reanalyze.sh
```

Run package and statistical unit tests:

```bash
PYTHON=/home/xzl/miniconda3/envs/sci1-rl/bin/python \
  bash reproduce/run_tests.sh
```

## Compact Release Policy

The package includes all seed-level source rows needed to reproduce paper statistics, three proposed checkpoints, protocols, and checksums. It excludes large baseline checkpoint trees, duplicate per-frame trajectory CSVs, and logs. Full rollout regeneration requires upstream QuadSwarm, On-Policy, and HARL repositories plus `third_party_patches/`; statistical audit does not.
