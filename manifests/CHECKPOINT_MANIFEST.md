# Checkpoint Manifest

Large checkpoints are not included in this GitHub package. They are required only to rerun policy evaluation or training.

## Final Gate Checkpoint

```text
results/trainable_graph_gate/o_static_same_goal_8agents_obstacle_1000000steps_rb_gca_v4_success_pareto_strong/rb_gca_v4_success_pareto_full_h1024_l4/graph_gate.pt
```

## Expert Checkpoint Families

The final expert pool expects these checkpoint families in the original workspace:

```text
results/onpolicy_quad_swarm/
results/onpolicy_lagrangian_quad_swarm/
results/ippo_quad_swarm/
results/mat_quad_swarm/
results/hatrpo_quad_swarm/
```

For each formal seed, the launcher finds the latest `run*` or `seed-*` directory under the relevant seed parent and loads the frozen policy.

## Included Instead Of Checkpoints

This package includes:

- Per-seed final evaluation summaries.
- Final group summaries.
- Paper-ready aggregate result tables.
- Paired bootstrap CI tables.

That is sufficient for paper writing and result auditing without storing multi-GB checkpoints in GitHub.

