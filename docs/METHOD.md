# Final Method: SA-RB-GCA Expert Pool

## Method Identity

The final paper method is still SA-RB-GCA: a success-aware, risk-budgeted, graph-conditioned arbitration policy. The additional baseline policies are not a new method. They are integrated as frozen experts inside the same SA-RB-GCA action-selection framework.

## Core Idea

SA-RB-GCA separates policy behavior into two expert groups:

- Efficiency group: policies that tend to preserve progress and task completion.
- Safety group: policies that tend to reduce close-distance risk or stabilize difficult states.

At every step, a learned graph-conditioned gate predicts the safety mass. The runtime controller then blends the frozen experts under a risk budget:

```text
action =
  (1 - alpha_safe) * weighted_mean(efficiency_expert_actions)
  + alpha_safe   * weighted_mean(safety_expert_actions)
```

The gate is trained once and reused at evaluation time. Expert policies are frozen.

## Final Expert Pool

Final paper configuration:

| Group | Experts | Within-group weights |
|---|---|---:|
| Efficiency | MAPPO, IPPO | equal |
| Safety | MAPPO-Lagrangian, MAT, HATRPO | equal |

The final evaluation script is:

```text
code/scripts/evaluate_sa_rb_gca_expert_pool.py
```

The final WSL launcher for the complex obstacle-8 case is:

```text
code/scripts/run_sa_rb_gca_expert_pool_static_obstacle8.sh
```

## Gate Checkpoint

The final gate checkpoint path in the full workspace is:

```text
results/trainable_graph_gate/o_static_same_goal_8agents_obstacle_1000000steps_rb_gca_v4_success_pareto_strong/rb_gca_v4_success_pareto_full_h1024_l4/graph_gate.pt
```

The default gate mode is:

```text
learned_graph_gate_shielded_rb_gca_v4_success_pareto_full_ff1.0_fc0.25_ft0.5_fo0.2_fmax0.25
```

## Metrics

The evaluation scripts report:

- `agent_success_rate`: fraction of agents that reach the task goal.
- `agent_deadlock_rate`: fraction of agents that enter deadlock.
- `risk_rate_dist_lt_0_65`: fraction of finite minimum pair-distance samples below 0.65m.
- `risk_rate_dist_lt_1_0`: fraction of finite minimum pair-distance samples below 1.0m.
- `avg_true_objective`: environment-provided `true_objective`, averaged over completed agents; higher is better.

The objective is not a post-hoc score created by the analysis script. It is read from the environment `info` field when available and summarized by the evaluator.

## Paper Positioning

The paper should present SA-RB-GCA as a risk-aware arbitration layer over frozen MARL policies. The expert pool is an implementation of the method's expert set, not a separate algorithmic contribution.

