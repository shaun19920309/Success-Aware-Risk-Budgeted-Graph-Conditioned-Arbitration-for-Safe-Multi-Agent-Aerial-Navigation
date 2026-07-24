# Final Results

## Aggregate Results

The complete long table is in:

```text
data/tables/final_results_long.csv
```

The compact paper table is in:

```text
data/tables/paper_final_results_compact.csv
```

## Key Scenario: 8-Agent Obstacles

| Method | Success % | Risk < 0.65m % | Risk < 1.0m % | Deadlock % | Objective |
|---|---:|---:|---:|---:|---:|
| MAPPO | 5.984 | 3.306 | 27.130 | 94.016 | -2.473 |
| MAPPO-Lagrangian | 4.906 | 3.709 | 27.748 | 95.094 | -2.241 |
| IPPO | 5.344 | 1.587 | 28.527 | 94.656 | -1.958 |
| MAT | 4.453 | 2.584 | 28.649 | 95.547 | -2.687 |
| HATRPO | 4.828 | 5.156 | 29.823 | 95.172 | -2.710 |
| SA-RB-GCA | 6.016 | 1.911 | 24.408 | 93.984 | -2.157 |

In this scenario, SA-RB-GCA has the highest success rate by a small margin and the lowest `risk < 1.0m` rate among the compared methods.

## Paired Bootstrap CI: SA-RB-GCA vs MAPPO

| Metric | Mean paired delta | 95% CI | Interpretation |
|---|---:|---:|---|
| Success rate | +0.031 pp | [-0.641, +0.469] | No material success change |
| Risk rate < 0.65m | -1.395 pp | [-2.108, -0.827] | Stable risk decrease |
| Risk rate < 1.0m | -2.722 pp | [-5.028, -0.735] | Stable risk decrease |
| Deadlock rate | -0.031 pp | [-0.469, +0.641] | No material deadlock change |
| Objective | +0.3153 | [+0.2781, +0.3526] | Stable objective increase |

## Claim Boundary

The result does not support a blanket claim that SA-RB-GCA dominates all baselines in every simple scenario. In easier static scenarios, IPPO or MAT can have better single metrics. The defensible paper claim is more specific:

> In the high-density 8-agent obstacle scenario, SA-RB-GCA preserves task success while significantly reducing near-collision risk and improving the environment true objective.

That is also the most meaningful setting for the method, because graph-conditioned risk arbitration is designed for interaction-heavy and safety-constrained states.

