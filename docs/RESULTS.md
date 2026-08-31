# Final Results and Interpretation

Machine-readable source of truth: `results/final_formal_multiseed/analysis/`.

## Nominal Multi-Training-Seed Comparison

| Method | Success | Collision | Deadlock | Progress | Objective/s |
|---|---:|---:|---:|---:|---:|
| Proposed | **91.28%** | **7.42%** | **1.30%** | **2.3595** | **-0.5638** |
| MAPPO | 0.13% | 69.27% | 30.60% | -1.2893 | -4.0647 |
| IPPO | 0.26% | 47.92% | 51.82% | -1.1764 | -3.1618 |
| MAPPO-Lagrangian | 0.13% | 60.29% | 39.58% | -1.5334 | -3.9951 |
| MAT | 0.26% | 58.33% | 41.41% | -1.5678 | -4.2211 |
| HATRPO | 0.00% | 77.21% | 22.79% | -1.3040 | -4.7276 |

The proposed method is stable across its three independently trained checkpoints: success ranges from 90.23% to 91.80%, and collision from 6.64% to 8.20%. Every individual baseline checkpoint has success at or below 0.39%. Thus the result is not driven by one favorable proposed seed or one failed baseline seed.

All 25 primary comparisons pass the hierarchical interval, Holm correction, and `9/9` cross-training-seed consistency gate. Effect ranges are:

| Metric | Proposed-minus-baseline range | Direction |
|---|---:|---|
| Success | +91.02 to +91.28 percentage points | higher is better |
| Collision | -40.49 to -69.79 percentage points | lower is better |
| Deadlock | -21.48 to -50.52 percentage points | lower is better |
| Goal progress | +3.5359 to +3.9273 m | higher is better |
| Objective/s | +2.5981 to +4.1639 | higher is better |

The joint interpretation matters. Baselines remain active but generally move away from the goal and terminate in collision or deadlock. The proposed policy combines high completion with positive progress and a less negative native objective. This supports a liveness-and-safety result under the matched simulation protocol, not merely a low-risk-exposure result.

## MAPPO-Lagrangian Correction

The corrected MAPPO-Lagrangian grand mean is 0.13% success, 60.29% collision, and 39.58% deadlock. Its collision rate varies from 53.12% to 70.70% across training seeds. These values replace all outputs from the invalid original adapter, which had accidentally reproduced MAPPO behavior. The correction does not change the paper conclusion: every primary proposed-minus-Lagrangian effect remains favorable under the full claim gate.

## Generalization Boundary

| Scenario | Success | 95% hierarchical CI | Collision | Deadlock | Progress | Objective/s |
|---|---:|---:|---:|---:|---:|---:|
| Obstacle-4 nominal | 88.02% | [77.60, 96.88] | 11.98% | 0.00% | 2.0649 | -0.5773 |
| Obstacle-8 dense/large | 69.79% | [60.16, 78.39] | 26.04% | 4.17% | 2.3449 | -1.0163 |
| Obstacle-8 sparse/small | 96.35% | [92.45, 99.22] | 2.86% | 0.78% | 1.9828 | -0.3499 |

The sparse/small and four-agent settings retain strong completion. Dense/large obstacles are the clearest remaining boundary: success drops by about 21.5 percentage points relative to nominal and collision rises to 26.04%. The method therefore generalizes meaningfully but is not insensitive to obstacle difficulty.

These experiments are proposed-only. They support robustness of the frozen method and do not support claims that every baseline is inferior in every shifted scenario.

## Runtime

| Method | Policy ms/frame | Coordination ms/frame | End-to-end ms/frame |
|---|---:|---:|---:|
| Proposed | **0.904** | 2.405 | **9.020** |
| MAPPO | 2.376 | 0.073 | 11.319 |
| IPPO | 2.256 | 0.073 | 11.141 |
| MAPPO-Lagrangian | 2.199 | 0.073 | 11.326 |
| MAT | 45.506 | 0.098 | 54.591 |
| HATRPO | 12.826 | 0.080 | 23.276 |

The deterministic route/coordinator adds more non-policy overhead than the learned baselines, but the small single controller offsets this cost. Its measured end-to-end frame time is lower than every comparator in the isolated serial RTX 5090 benchmark.

## Component Evidence

The final component study in `results/final_component_ablation/` shows the design sequence:

1. direct analytic control establishes motion but collides frequently;
2. obstacle waypoints deliver the largest collision reduction;
3. synchronized stage-enter-egress coordination resolves shared-goal blocking;
4. bounded BC retains the coordinated behavior with fast inference;
5. the tested DAgger extension changes no outcome rate and is excluded.

This supports the implemented module roles without presenting failed routing or ensemble variants as contributions.

## What the Evidence Supports

The formal evidence supports the following simulator-scoped conclusion:

> Under matched 7 s shared-goal obstacle navigation, the final route-coordination-plus-bounded-control method substantially improves success, collision, deadlock, goal progress, and simulator-native objective relative to MAPPO, IPPO, corrected MAPPO-Lagrangian, MAT, and HATRPO, with effects consistent across independent training and environment seeds.

## What It Does Not Support

- No formal collision-avoidance guarantee.
- No claim of universal raw-proximity dominance.
- No baseline-superiority claim in proposed-only generalization suites.
- No robustness guarantee for arbitrary dynamics mismatch.
- No real-world transfer claim.

The machine-readable claim gate and report should be cited instead of older single-checkpoint result files.
