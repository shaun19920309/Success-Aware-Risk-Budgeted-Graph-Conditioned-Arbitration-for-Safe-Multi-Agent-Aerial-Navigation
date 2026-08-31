# Formal 7 s multiseed evidence

- Baseline training seeds: [240000, 240001, 240002]
- Proposed training seeds: [171001, 171002, 171003]
- Unseen evaluation seeds: 250000--250031 (n=32)
- Every rollout has 701 frames and matched physical-state hashes.
- Confidence intervals resample training seeds and matched environment seeds hierarchically.
- Sign-flip p-values operate on environment-seed deltas after averaging the three training seeds.

## Grand means across training and environment seeds

| Method | Success | Collision | Deadlock | Progress | Objective/s | Moving | Risk <0.65 | Risk <1.0 | Transit <0.65 | Transit <1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Proposed | 91.28% | 7.42% | 1.30% | 2.3595 | -0.5638 | 99.09% | 21.79% | 49.04% | 2.64% | 10.63% |
| MAPPO | 0.13% | 69.27% | 30.60% | -1.2893 | -4.0647 | 99.43% | 11.64% | 30.84% | 0.44% | 1.29% |
| IPPO | 0.26% | 47.92% | 51.82% | -1.1764 | -3.1618 | 99.68% | 21.85% | 49.35% | 0.88% | 2.36% |
| MAPPO-Lagrangian | 0.13% | 60.29% | 39.58% | -1.5334 | -3.9951 | 99.49% | 13.14% | 33.21% | 0.51% | 1.41% |
| MAT | 0.26% | 58.33% | 41.41% | -1.5678 | -4.2211 | 99.49% | 17.32% | 37.54% | 0.70% | 1.72% |
| HATRPO | 0.00% | 77.21% | 22.79% | -1.3040 | -4.7276 | 99.71% | 26.79% | 54.54% | 1.11% | 2.81% |

## Proposed generalization across training and environment seeds

These suites quantify the proposed controller's robustness only; they do not reuse the old 1 s-trained baselines as comparators.

| Scenario | Success (95% hierarchical CI) | Collision (95% hierarchical CI) | Deadlock | Progress | Objective/s |
|---|---:|---:|---:|---:|---:|
| Obstacle-4 nominal | 88.02% [77.60%, 96.88%] | 11.98% [3.12%, 22.40%] | 0.00% | 2.0649 | -0.5773 |
| Obstacle-8 dense/large | 69.79% [60.16%, 78.39%] | 26.04% [18.75%, 34.11%] | 4.17% | 2.3449 | -1.0163 |
| Obstacle-8 sparse/small | 96.35% [92.45%, 99.22%] | 2.86% [0.26%, 6.51%] | 0.78% | 1.9828 | -0.3499 |

## Isolated sequential RTX 5090 runtime

| Method | Policy ms/frame | Coordination ms/frame | End-to-end ms/frame |
|---|---:|---:|---:|
| Proposed | 0.904 | 2.405 | 9.020 |
| MAPPO | 2.376 | 0.073 | 11.319 |
| IPPO | 2.256 | 0.073 | 11.141 |
| MAPPO-Lagrangian | 2.199 | 0.073 | 11.326 |
| MAT | 45.506 | 0.098 | 54.591 |
| HATRPO | 12.826 | 0.080 | 23.276 |

## Proposed-minus-baseline effects

| Baseline | Metric | Delta | Hierarchical 95% CI | Conditional Holm p | Favorable model-seed pairs |
|---|---|---:|---:|---:|---:|
| MAPPO | Success | +91.15% | [+88.02%, +94.14%] | 0.000125 | 9/9 |
| MAPPO | Collision | -61.85% | [-67.97%, -55.73%] | 0.000125 | 9/9 |
| MAPPO | Deadlock | -29.30% | [-34.64%, -23.83%] | 0.000125 | 9/9 |
| MAPPO | Goal progress | +3.6488 | [+3.3703, +3.9265] | 0.000125 | 9/9 |
| MAPPO | Objective/s | +3.5009 | [+3.3214, +3.7023] | 0.000125 | 9/9 |
| IPPO | Success | +91.02% | [+88.02%, +93.88%] | 0.000125 | 9/9 |
| IPPO | Collision | -40.49% | [-46.22%, -34.77%] | 0.000125 | 9/9 |
| IPPO | Deadlock | -50.52% | [-55.99%, -44.92%] | 0.000125 | 9/9 |
| IPPO | Goal progress | +3.5359 | [+3.1156, +4.0407] | 0.000125 | 9/9 |
| IPPO | Objective/s | +2.5981 | [+2.2704, +2.9793] | 0.000125 | 9/9 |
| MAPPO-Lagrangian | Success | +91.15% | [+88.02%, +94.14%] | 0.000125 | 9/9 |
| MAPPO-Lagrangian | Collision | -52.86% | [-63.02%, -43.36%] | 0.000125 | 9/9 |
| MAPPO-Lagrangian | Deadlock | -38.28% | [-47.40%, -28.12%] | 0.000125 | 9/9 |
| MAPPO-Lagrangian | Goal progress | +3.8929 | [+3.5721, +4.2453] | 0.000125 | 9/9 |
| MAPPO-Lagrangian | Objective/s | +3.4313 | [+3.2665, +3.6002] | 0.000125 | 9/9 |
| MAT | Success | +91.02% | [+87.89%, +94.01%] | 0.000125 | 9/9 |
| MAT | Collision | -50.91% | [-57.29%, -43.62%] | 0.000125 | 9/9 |
| MAT | Deadlock | -40.10% | [-47.27%, -34.11%] | 0.000125 | 9/9 |
| MAT | Goal progress | +3.9273 | [+3.3425, +4.3735] | 0.000125 | 9/9 |
| MAT | Objective/s | +3.6573 | [+3.0898, +4.0769] | 0.000125 | 9/9 |
| HATRPO | Success | +91.28% | [+88.28%, +94.14%] | 0.000125 | 9/9 |
| HATRPO | Collision | -69.79% | [-84.11%, -54.43%] | 0.000125 | 9/9 |
| HATRPO | Deadlock | -21.48% | [-36.59%, -7.16%] | 0.000125 | 9/9 |
| HATRPO | Goal progress | +3.6635 | [+3.3706, +3.9505] | 0.000125 | 9/9 |
| HATRPO | Objective/s | +4.1639 | [+3.5212, +4.9991] | 0.000125 | 9/9 |

## Interpretation rule

A superiority claim requires the hierarchical interval to exclude zero in the favorable direction, a Holm-adjusted conditional p-value below 0.05, and favorable effects for all nine cross-training-seed pairs. Raw proximity exposure remains descriptive and is interpreted jointly with completion and progress.
