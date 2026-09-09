# Final Experiment Protocol

## Primary Fair-Budget Study

Source: `results/revision_horizon7_fair_waypoint_budget_20260901/fair_waypoint_budget_preregistered_protocol.json`, with SHA-256 sidecar. This supersedes the earlier unequal-hierarchy 1M comparison as the main baseline evidence.

- WSL2 Ubuntu; sci1-rl environment; Python 3.9.25, PyTorch 2.8.0+cu128, NumPy 1.24.4; RTX 5090.
- MAPPO, IPPO, MAPPO-Lagrangian, MAT, HATRPO; training seeds 260000/260001/260002.
- Cumulative checkpoints at 1M/3M/5M; final actual steps 4,999,936 due to 128-step rollout alignment.
- 45 immutable checkpoints x 32 layouts (250000..250031) = 1440 baseline evaluations.
- Three proposed models 171001..171003 x the same 32 layouts = 96 reference evaluations.
- Eight agents, shared-goal slot radius 0.45 m, two observed neighbors, static cylindrical obstacles at density 0.2 and size 0.6 m; nominal 7.0 s and exactly 701 frames.
- Same synchronized stage-enter-egress targets and A* waypoints during baseline training and testing. Same active-target/progress/arrival/collision reward for all five.
- No action mixture or shield: each raw row must record synchronized_waypoint_expert_enabled=1, one loaded/active expert, fixed_0 mode, and a matching initial physical-state SHA-256.

Lagrangian uses corrected hybrid cost with limit=0, learning rate=0.05, initial multiplier=1, maximum=20. Invalid duplicate MAPPO/Lagrangian outputs from the earlier cost adapter are excluded.

## Supervision and Continuity Caveats

BC uses 179,456 analytic-teacher training labels from seeds 160000..160031, 22,432 validation labels from 161000..161003, 60 epochs, and independent initialization seeds 171001..171003. RL interaction budgets are not equivalent to teacher labels or teacher-design effort.

The September 3 interruption was recovered without optimizer/process RNG state and without on-policy ValueNorm. September 8 MAT/HATRPO recovery restored available weights, optimizers, ValueNorm and process RNG, but not simulator episodes or rollout buffers. These are cumulative budgets, not uninterrupted or bitwise-continuous runs. See `verification/recovery_20260908_status.md` and per-run `training_provenance/` manifests. Fresh uninterrupted retraining may produce different policies.

Two MAT 5M policies have distinct weight hashes but identical executed-action hash sequences and saturated actions in every evaluation. IPPO has large between-training-seed variance. Neither 5M convergence nor optimal tuning is established.

## Metrics and Analysis

Success means canonical goal reach/dwell **without** a canonical post-grace collision. Collision and deadlock (neither success nor collision) form a mutually exclusive partition, S+C+D=1. Goal reach requires distance <=0.5 m, speed <=0.5 m/s, for ten consecutive frames. Progress is mean initial-minus-final distance to the assigned physical goal.

The final fair analysis defines objective/s as `avg_true_objective / 7.0` for every raw row. The legacy exported per-second field used 7.0 for BC and 7.01 for baselines; it is not reused. This post-results reporting correction preserves raw CSVs and is documented in `verification/objective_duration_correction_20260909.md`.

Raw proximity is the fraction of frames with closest-pair distance below 0.65 or 1.0 m; it is not collision probability. Exposure is secondary and interpreted with completion, signed progress, and collision.

Hierarchical bootstrap: 100,000 draws, independently resampling three trained models within each method and jointly resampling the 32 matched environments. Conditional two-sided sign flips: 200,000 draws over model-averaged environment differences. Holm family: 25 primary 5M contrasts (five baselines x success/collision/deadlock/progress/objective).

A superiority gate additionally requires all nine cross-model differences to point favorably. These pairings reuse three models per method and are not nine independent seeds. 1M/3M and exposure contrasts are secondary, not test-selected primary endpoints. Rates in effect tables are percentage-point differences.

## Supporting Experiments

- Component ablation: 228000..228031; direct teacher, waypoint teacher, coordinated teacher, bounded BC, DAgger. The last has no verified outcome advantage and is not deployed.
- Proposed-only generalization: 251000..251015 (four agents), 252000..252015 (dense/large), 253000..253015 (sparse/small), three frozen BC models each.
- Earlier isolated timing: 254000..254002, three models per method, serial CUDA-synchronized calls. Proposed timing remains relevant; baseline timing uses earlier unadapted 1M models, not new fair 5M policies.
- `results/final_formal_multiseed/` contains the proposed reference and these supporting suites, plus earlier unequal-hierarchy 1M diagnostics. The primary new baseline evidence lives in the fair-budget result root.

## Commands

After installing the environment and external adapters:

```bash
PYTHON=python TRAIN_DEVICE=cuda bash reproduce/train_final_bc.sh
OUT_ROOT="$PWD/results/regenerated_fair_waypoint_budget" PY=python \
  bash scripts/run_horizon7_fair_waypoint_budget_logged.sh ippo 260000
```

Repeat baseline training for methods mappo/ippo/lagrangian/mat/hatrpo and seeds 260000/260001/260002. Use a new output root to protect archived evidence. The logged launcher writes the completion marker required by the verifier.

```bash
python scripts/verify_horizon7_fair_waypoint_budget.py \
  --result-root results/regenerated_fair_waypoint_budget
python scripts/evaluate_horizon7_fair_waypoint_budget.py \
  --result-root results/regenerated_fair_waypoint_budget \
  --reference-root results/final_formal_multiseed
python scripts/analyze_horizon7_fair_waypoint_budget.py \
  --result-root results/regenerated_fair_waypoint_budget \
  --reference-root results/final_formal_multiseed
```

For statistics on the included raw rows, without retraining:

```bash
python reproduce/verify_package.py
PYTHON=python bash reproduce/reanalyze.sh
PYTHON=python bash reproduce/run_tests.sh
```

Large baseline weights and per-frame trajectories are omitted; their manifests and compact raw summaries are included. Training metadata is archived outside runnable training trees. Upstream sources are not commit-pinned, which limits bitwise/full-environment reproducibility. Statistical reconstruction from released rows is independently audited.
