# Reproduction Guide

## Runtime Assumption

The original experiments were run in WSL Ubuntu with:

```text
/home/xzl/miniconda3/envs/sci1-rl/bin/python
```

For the original workstation layout, the repository can be located inside WSL without passing the Chinese path through PowerShell:

```bash
BASE=$(find /mnt/h/xzl/sci1 -maxdepth 1 -type d -name '*graph_conditioned_lagrangian_shield' ! -name '*line1*' -print -quit)
cd "$BASE"
PY=/home/xzl/miniconda3/envs/sci1-rl/bin/python
```

## Verify This Package

From the package root:

```bash
python reproduce/verify_package.py
```

## Reproduce the Reported Numerical Evidence

The manuscript is intentionally excluded from this public repository. The retained data are sufficient to verify every reported aggregate and the paired-bootstrap statement without baseline retraining:

    data/tables/final_results_long.csv
    data/tables/paper_final_results_compact.csv
    data/tables/paper_main_vs_best_baseline_by_scenario.csv
    data/statistics/obstacle8_sa_rb_gca_vs_mappo_bootstrap_ci.csv
    data/statistics/obstacle8_sa_rb_gca_vs_mappo_per_seed_deltas.csv

The six canonical Obstacle-8 group summaries under `data/final_group_summaries/` contain the secondary safety and progress diagnostics. The 96 files under `data/final_seed_summaries/` provide the complete retained per-seed evidence.

No baseline retraining is required to verify these packaged results.

## Re-run Final SA-RB-GCA Evaluation

Full re-evaluation requires the original trained checkpoints in `results/`. The complex 8-agent obstacle case can be rerun with:

```bash
BASE=$(find /mnt/h/xzl/sci1 -maxdepth 1 -type d -name '*graph_conditioned_lagrangian_shield' ! -name '*line1*' -print -quit)
cd "$BASE"
PY=/home/xzl/miniconda3/envs/sci1-rl/bin/python
PYTHON=$PY PY=$PY SEEDS='0 1111 2222 3333' EVAL_DEVICE=cuda \
  bash code/scripts/run_sa_rb_gca_expert_pool_static_obstacle8.sh
```

The expected output root is:

```text
results/sa_rb_gca_expert_pool/o_static_same_goal_8agents_obstacle_1000000steps/
```

## Re-run Paired Bootstrap CI

```bash
BASE=$(find /mnt/h/xzl/sci1 -maxdepth 1 -type d -name '*graph_conditioned_lagrangian_shield' ! -name '*line1*' -print -quit)
cd "$BASE"
PY=/home/xzl/miniconda3/envs/sci1-rl/bin/python
$PY code/scripts/analyze_sa_rb_gca_paired_bootstrap_ci.py --base "$BASE" --n-boot 50000
```

## Re-train Baselines

Formal baseline launchers are included in `code/scripts/`:

```text
run_mappo_lagrangian_formal_matrix.sh
run_ippo_quad_swarm_formal_baselines.sh
run_mat_quad_swarm_formal_baselines.sh
run_hatrpo_quad_swarm_formal_baselines.sh
```

The baseline runners are designed to skip only when both a model and the corresponding eval summary exist. This avoids treating partial checkpoints as complete 1M-step formal results.

## Important Size Note

This package intentionally does not copy model checkpoints. It is GitHub-ready as a reproducible evidence bundle. Full retraining or re-evaluation from scratch requires the original workspace or rerunning the training launchers to recreate checkpoints.
