# Final Results and Interpretation

The primary result is the **same-waypoint 5M comparison**, not the earlier unadapted 1M screen. See the root README for the complete main table, and `results/revision_horizon7_fair_waypoint_budget_20260901/analysis/` for all 105 method/budget/metric contrasts, model means, budget effects, learning curves and the latest report.

## Main Conclusion

Proposed success is 91.28%, collision 7.42%, deadlock 1.30%, progress 2.3595 m, and objective/s -0.5638. The strongest 5M baseline by success, progress and objective is IPPO: 8.72%, 0.5766 m, and -2.0436 objective/s.

Against IPPO, paired success gain is +82.55 percentage points, hierarchical 95% CI [67.84,92.71]; collision change is -20.70 points, CI [-36.33,-8.07]; deadlock -61.85 points, CI [-73.44,-51.56]; progress +1.7829 m, CI [0.2565,3.2190]; objective/s +1.4798, CI [0.6351,2.4208].

All 25 primary 5M tests pass the favorable hierarchical interval, conditional Holm p=0.000125, and 9/9 model-pair direction gates. Across baselines, success improves 82.55--91.28 points, collision declines 20.70--77.86 points, deadlock declines 13.41--61.85 points, progress improves 1.7829--4.2229 m, and objective/s improves 1.4798--4.6385.

Objective figures use a unified 7.0-second denominator, rebuilt from raw terminal objectives after discovering that legacy baseline exports used 7.01 seconds. This correction leaves all other metrics unchanged.

## Why the Gap Is Not a Pure Coordination Effect

Providing identical stage targets and A* waypoints controls the earlier planning-assistance imbalance. It does not equalize analytic-teacher information or optimizer behavior. BC is trained on privileged teacher actions; the RL policies must learn motor tracking from interaction. The experiment compares these complete systems under matched high-level assistance, not equally informed optimizers.

IPPO improves from 0.39% to 3.26% to 8.72% success at 1M/3M/5M. Its 5M seeds achieve 0.78%, 24.61%, and 0.78%, revealing meaningful training variability. Two distinct MAT checkpoints have identical executed-action hash sequences and fully saturated actions in all 32 layouts. This is not checkpoint duplication, but its optimization cause is not yet isolated.

Host interruptions and partial-state recovery further limit learning-curve interpretation. Five million cumulative steps do not establish convergence, and no comprehensive hyperparameter search or clean uninterrupted replication is available.

## Risk Is Not Uniformly Improved

Proposed mean risk below 0.65 m is 21.79%, above MAPPO 8.81%, Lagrangian 12.25%, and MAT 8.74%. Their low exposure coexists with negative progress and frequent collision. IPPO has higher mean risk but its paired intervals cross zero at both thresholds. HATRPO's exposure differences favor proposed and exclude zero, but are exploratory, outside the 25 primary tests.

The supported claim is improved completion and collision outcomes in this simulator, **not universal risk minimization**.

## Supporting Evidence

| Proposed-only suite | Success | Hierarchical 95% CI | Collision |
|---|---:|---:|---:|
| Four-agent nominal | 88.02% | [77.60,96.88] | 11.98% |
| Eight-agent sparse/small | 96.35% | [92.45,99.22] | 2.86% |
| Eight-agent dense/large | 69.79% | [60.16,78.39] | 26.04% |

These suites establish a measured dense-obstacle boundary, not superiority over baselines in shifted scenarios. Component ablations support the implemented waypoint/phase decomposition and no verified DAgger outcome gain.

The earlier isolated proposed runtime is 0.904 ms neural inference, 2.405 ms coordination, and 9.020 ms end to end. Baseline timing rows are earlier unadapted 1M measurements and must not be presented as a fair 5M latency comparison.

No formal collision-avoidance proof, real-world flight validation, universal RL superiority, or convergence claim is supported.
