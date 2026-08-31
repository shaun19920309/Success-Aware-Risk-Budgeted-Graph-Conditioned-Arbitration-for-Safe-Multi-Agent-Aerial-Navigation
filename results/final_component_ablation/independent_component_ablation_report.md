# Independent 32-seed component ablation

The protocol and checkpoints were frozen before any 228xxx rollout. Each row uses the same 32 physical initial states and 701-frame corrected horizon.

| Method | Success | Collision | Deadlock | Progress (m) | Objective/s | Risk <0.65 | Risk <1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct teacher | 83.98% | 16.02% | 0.00% | 3.1028 | -0.6854 | 84.72% | 90.26% |
| Waypoint teacher | 91.41% | 6.64% | 1.95% | 3.0592 | -0.3840 | 83.74% | 89.73% |
| Coordinated teacher | 92.58% | 6.25% | 1.17% | 2.4002 | -0.5383 | 22.66% | 49.14% |
| Initial bounded BC | 92.97% | 5.47% | 1.56% | 2.3926 | -0.5329 | 23.32% | 47.42% |
| Final bounded DAgger | 92.97% | 5.47% | 1.56% | 2.3856 | -0.5533 | 23.22% | 48.09% |

## Paired confirmatory contrasts

### Final bounded DAgger versus Direct teacher

| Metric | Delta | 95% paired bootstrap CI | Raw p | Holm p |
|---|---:|---:|---:|---:|
| Success | +8.984 pp | [+2.344, +16.016] pp | 0.019810 | 0.316958 |
| Collision | -10.547 pp | [-17.969, -3.906] pp | 0.007545 | 0.128264 |
| Deadlock | +1.562 pp | [+0.391, +3.125] pp | 0.125604 | 1.000000 |
| Goal progress (m) | -0.717 | [-0.749, -0.688] | 0.000005 | 0.000100 |
| Objective per second | +0.132 | [-0.038, +0.316] | 0.161809 | 1.000000 |

### Final bounded DAgger versus Waypoint teacher

| Metric | Delta | 95% paired bootstrap CI | Raw p | Holm p |
|---|---:|---:|---:|---:|
| Success | +1.562 pp | [-1.953, +5.469] pp | 0.560367 | 1.000000 |
| Collision | -1.172 pp | [-5.078, +2.344] pp | 0.704941 | 1.000000 |
| Deadlock | -0.391 pp | [-1.953, +1.172] pp | 1.000000 | 1.000000 |
| Goal progress (m) | -0.674 | [-0.720, -0.620] | 0.000005 | 0.000100 |
| Objective per second | -0.169 | [-0.207, -0.130] | 0.000005 | 0.000100 |

### Final bounded DAgger versus Coordinated teacher

| Metric | Delta | 95% paired bootstrap CI | Raw p | Holm p |
|---|---:|---:|---:|---:|
| Success | +0.391 pp | [-1.953, +3.125] pp | 1.000000 | 1.000000 |
| Collision | -0.781 pp | [-3.125, +1.172] pp | 0.749651 | 1.000000 |
| Deadlock | +0.391 pp | [-1.172, +1.953] pp | 1.000000 | 1.000000 |
| Goal progress (m) | -0.015 | [-0.057, +0.024] | 0.486528 | 1.000000 |
| Objective per second | -0.015 | [-0.040, +0.009] | 0.248844 | 1.000000 |

### Final bounded DAgger versus Initial bounded BC

| Metric | Delta | 95% paired bootstrap CI | Raw p | Holm p |
|---|---:|---:|---:|---:|
| Success | +0.000 pp | [-2.344, +2.344] pp | 1.000000 | 1.000000 |
| Collision | +0.000 pp | [-1.953, +1.953] pp | 1.000000 | 1.000000 |
| Deadlock | +0.000 pp | [-1.953, +1.953] pp | 1.000000 | 1.000000 |
| Goal progress (m) | -0.007 | [-0.037, +0.019] | 0.646527 | 1.000000 |
| Objective per second | -0.020 | [-0.050, +0.008] | 0.191614 | 1.000000 |

## Integrity

- Exact seed set: PASS.
- 701 frames for every rollout: PASS.
- Matched physical initial-state hash for every seed: PASS.
- Frozen student checkpoint hashes: PASS.

Raw proximity exposure is descriptive and must be interpreted jointly with success, collision, deadlock, and progress.
