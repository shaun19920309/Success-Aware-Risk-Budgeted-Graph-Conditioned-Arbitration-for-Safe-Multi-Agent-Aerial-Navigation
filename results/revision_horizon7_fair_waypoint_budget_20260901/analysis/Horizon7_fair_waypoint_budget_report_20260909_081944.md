# Horizon-7 fair waypoint and training-budget comparison
Objective normalization correction (2026-09-09): all objective/s values are recomputed from raw avg_true_objective using the frozen 7.0-second horizon. Legacy BC exports divided by 7.0 while baseline exports divided by 701*dt=7.01. Raw CSVs are unchanged. Earlier reports using mixed denominators are superseded; success, collision, deadlock, progress, and exposure are unchanged.

Generated: 2026-09-09T08:19:44+08:00

## Protocol and integrity

All five baselines received the same synchronized stage targets, inflated-grid A* waypoints, active-target reward, collision penalties, and canonical arrival definition. Each method used 3 training seeds, each with a cumulative 5M-step training budget and immutable 1M/3M/5M checkpoints. Evaluation used 32 matched seeds, 701 frames, no action ensemble, and no safety shield. Every initial physical-state hash matched the proposed controller reference.

Training-continuity limitation: the archived runs were interrupted and resumed in their original directories; they were not uninterrupted 5M-step trajectories. The 2026-09-03 legacy recovery lacked optimizer and process RNG state, and the on-policy checkpoints also lacked ValueNorm state. The 2026-09-08 MAT/HATRPO recovery restored weights, optimizers, ValueNorm, and process RNG, but not simulator state or rollout buffers; episodes restarted, so this was not a bitwise-exact continuation. Original 1M/3M checkpoints were preserved. See verification/recovery_20260908_status.md and the per-run recovery manifests for provenance. These limitations must accompany comparisons across training budgets and methods.

## Primary 5M comparison

| Method | Success | Collision | Deadlock | Risk <0.65 m | Risk <1.0 m | Progress | Objective/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Proposed | 91.28% | 7.42% | 1.30% | 21.79% | 49.04% | 2.3595 | -0.5638 |
| MAPPO | 0.26% | 58.72% | 41.02% | 8.81% | 25.09% | -1.3810 | -3.4208 |
| IPPO | 8.72% | 28.12% | 63.15% | 31.73% | 60.87% | 0.5766 | -2.0436 |
| MAPPO-Lagrangian | 0.52% | 57.68% | 41.80% | 12.25% | 33.15% | -1.4065 | -3.5037 |
| MAT | 0.39% | 61.59% | 38.02% | 8.74% | 24.06% | -1.8634 | -4.5498 |
| HATRPO | 0.00% | 85.29% | 14.71% | 29.95% | 56.60% | -1.5793 | -5.2023 |

The following deltas are proposed minus baseline. Positive is favorable for success, progress, and objective; negative is favorable for collision, deadlock, and risk.

The hierarchical confidence intervals independently resample trained models within each method and jointly resample matched environments. Holm-adjusted p-values come from conditional sign-flip tests of environment-level differences averaged over the available trained models, not tests that resample training runs. The nine cross-seed comparisons are the 3-by-3 combinations of trained models, not nine independent training seeds.

| Baseline | Metric | Delta | Hierarchical 95% CI | Holm p | Cross-seed direction | Strict rule |
|---|---|---:|---:|---:|---:|---:|
| MAPPO | Success | +91.02 pp | [+87.89 pp, +94.01 pp] | 0.000125 | 9/9 | PASS |
| MAPPO | Collision | -51.30 pp | [-61.07 pp, -41.54 pp] | 0.000125 | 9/9 | PASS |
| MAPPO | Deadlock | -39.71 pp | [-49.35 pp, -30.47 pp] | 0.000125 | 9/9 | PASS |
| MAPPO | Goal progress | +3.7405 | [+3.5092, +3.9801] | 0.000125 | 9/9 | PASS |
| MAPPO | Objective/s | +2.8571 | [+2.3965, +3.2179] | 0.000125 | 9/9 | PASS |
| IPPO | Success | +82.55 pp | [+67.84 pp, +92.71 pp] | 0.000125 | 9/9 | PASS |
| IPPO | Collision | -20.70 pp | [-36.33 pp, -8.07 pp] | 0.000125 | 9/9 | PASS |
| IPPO | Deadlock | -61.85 pp | [-73.44 pp, -51.56 pp] | 0.000125 | 9/9 | PASS |
| IPPO | Goal progress | +1.7829 | [+0.2565, +3.2190] | 0.000125 | 9/9 | PASS |
| IPPO | Objective/s | +1.4798 | [+0.6351, +2.4208] | 0.000125 | 9/9 | PASS |
| MAPPO-Lagrangian | Success | +90.76 pp | [+87.50 pp, +93.88 pp] | 0.000125 | 9/9 | PASS |
| MAPPO-Lagrangian | Collision | -50.26 pp | [-56.38 pp, -43.75 pp] | 0.000125 | 9/9 | PASS |
| MAPPO-Lagrangian | Deadlock | -40.49 pp | [-46.74 pp, -34.77 pp] | 0.000125 | 9/9 | PASS |
| MAPPO-Lagrangian | Goal progress | +3.7660 | [+3.3623, +4.1422] | 0.000125 | 9/9 | PASS |
| MAPPO-Lagrangian | Objective/s | +2.9399 | [+2.5835, +3.2426] | 0.000125 | 9/9 | PASS |
| MAT | Success | +90.89 pp | [+87.50 pp, +94.14 pp] | 0.000125 | 9/9 | PASS |
| MAT | Collision | -54.17 pp | [-59.51 pp, -49.09 pp] | 0.000125 | 9/9 | PASS |
| MAT | Deadlock | -36.72 pp | [-41.54 pp, -32.03 pp] | 0.000125 | 9/9 | PASS |
| MAT | Goal progress | +4.2229 | [+4.0210, +4.4232] | 0.000125 | 9/9 | PASS |
| MAT | Objective/s | +3.9860 | [+3.8414, +4.1270] | 0.000125 | 9/9 | PASS |
| HATRPO | Success | +91.28 pp | [+88.28 pp, +94.14 pp] | 0.000125 | 9/9 | PASS |
| HATRPO | Collision | -77.86 pp | [-85.29 pp, -69.53 pp] | 0.000125 | 9/9 | PASS |
| HATRPO | Deadlock | -13.41 pp | [-21.22 pp, -6.25 pp] | 0.000125 | 9/9 | PASS |
| HATRPO | Goal progress | +3.9388 | [+3.7318, +4.1416] | 0.000125 | 9/9 | PASS |
| HATRPO | Objective/s | +4.6385 | [+4.2883, +5.0115] | 0.000125 | 9/9 | PASS |

## Training-budget trajectory

| Method | Budget | Success | Risk <0.65 m | Progress | Objective/s |
|---|---:|---:|---:|---:|---:|
| MAPPO | 1M | 0.26% | 10.44% | -1.5598 | -4.2757 |
| MAPPO | 3M | 0.13% | 10.16% | -1.4328 | -3.7113 |
| MAPPO | 5M | 0.26% | 8.81% | -1.3810 | -3.4208 |
| IPPO | 1M | 0.39% | 23.50% | -0.5603 | -2.8691 |
| IPPO | 3M | 3.26% | 29.45% | 0.1940 | -2.1990 |
| IPPO | 5M | 8.72% | 31.73% | 0.5766 | -2.0436 |
| MAPPO-Lagrangian | 1M | 0.26% | 10.15% | -1.4877 | -4.0049 |
| MAPPO-Lagrangian | 3M | 0.39% | 12.72% | -1.2308 | -3.8016 |
| MAPPO-Lagrangian | 5M | 0.52% | 12.25% | -1.4065 | -3.5037 |
| MAT | 1M | 0.39% | 28.93% | -1.2199 | -4.2798 |
| MAT | 3M | 0.39% | 8.59% | -1.9047 | -4.5151 |
| MAT | 5M | 0.39% | 8.74% | -1.8634 | -4.5498 |
| HATRPO | 1M | 0.00% | 29.93% | -1.3420 | -4.8053 |
| HATRPO | 3M | 0.00% | 30.39% | -1.5114 | -5.0899 |
| HATRPO | 5M | 0.00% | 29.95% | -1.5793 | -5.2023 |

## Interpretation

The strict preregistered superiority rule passed for 25/25 primary method-metric comparisons. A failed strict rule is not automatically evidence of equivalence: inspect the signed effect and interval. The 1M and 3M checkpoints and both risk thresholds are secondary diagnostics; only the 5M five-metric family receives Holm correction.

This report replaces any earlier fairness interpretation that compared the proposed hierarchy against baselines trained without the same waypoint and stage-target information.
