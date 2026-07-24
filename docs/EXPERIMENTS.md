# Experiment Protocol

## Formal Scenarios

| Short name | Formal baseline path name | SA-RB-GCA path name |
|---|---|---|
| `static_4` | `static_same_goal_4agents_1000000steps` | `static_same_goal_4agents_no_obstacle_1000000steps` |
| `static_8` | `static_same_goal_8agents_1000000steps` | `static_same_goal_8agents_no_obstacle_1000000steps` |
| `obstacle_4` | `o_static_same_goal_4agents_1000000steps` | `o_static_same_goal_4agents_obstacle_1000000steps` |
| `obstacle_8` | `o_static_same_goal_8agents_1000000steps` | `o_static_same_goal_8agents_obstacle_1000000steps` |

## Methods

| Method | Group summary source |
|---|---|
| MAPPO | `results/onpolicy_quad_swarm/<scenario>/onpolicy_eval_group_summary.csv` |
| MAPPO-Lagrangian | `results/onpolicy_lagrangian_quad_swarm/<scenario>/onpolicy_eval_group_summary.csv` |
| IPPO | `results/ippo_quad_swarm/<scenario>/onpolicy_eval_group_summary.csv` |
| MAT | `results/mat_quad_swarm/<scenario>/onpolicy_eval_group_summary.csv` |
| HATRPO | `results/hatrpo_quad_swarm/<scenario>/harl_eval_group_summary.csv` |
| SA-RB-GCA | `results/sa_rb_gca_expert_pool/<scenario>/sa_rb_gca_expert_pool_group_summary.csv` |

## Seeds and Episodes

- Training/evaluation seeds included in this package: `0`, `1111`, `2222`, `3333`.
- Evaluation episodes per seed: `200`.
- Training horizon for formal baseline checkpoints: `1,000,000` steps.

## Statistical Test

The paper-facing inferential test is paired by seed in the complex 8-agent obstacle scenario:

```text
SA-RB-GCA expert-pool full_equal minus MAPPO
```

For each seed, the paired delta is computed first. Bootstrap confidence intervals are then computed by resampling those seed-level paired deltas. This is intentionally stricter than comparing unpaired aggregate means because it controls for seed-level variation.

The final report is:

```text
docs/source_reports/SA_RB_GCA_paired_bootstrap_CI_4seeds_final_2026-07-09_130736.md
```

The machine-readable CI table is:

```text
data/statistics/obstacle8_sa_rb_gca_vs_mappo_bootstrap_ci.csv
```

