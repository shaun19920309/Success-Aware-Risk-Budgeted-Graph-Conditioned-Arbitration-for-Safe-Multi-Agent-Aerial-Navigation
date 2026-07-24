# SA-RB-GCA paired bootstrap CI 4-seed final report

Generated: 2026-07-09 13:07:36

## Scope

- Scenario: `o_static_same_goal_8agents_obstacle_1000000steps`
- Comparison: `SA-RB-GCA expert-pool full_equal` minus `MAPPO`
- Paired seeds: `0, 1111, 2222, 3333`
- Bootstrap: 50,000 paired resamples over seed-level deltas
- Rate metrics are reported as percentage-point deltas.

## Results

| Metric | Mean delta | 95% bootstrap CI | Per-seed deltas | Direction |
|---|---:|---:|---|---|
| success rate | +0.031 | [-0.641, +0.469] | +0.438, +0.500, +0.187, -1.000 | increase trend |
| risk rate < 0.65m | -1.395 | [-2.108, -0.827] | -1.213, -2.406, -1.287, -0.673 | stable decrease |
| risk rate < 1.0m | -2.722 | [-5.028, -0.735] | -2.703, -2.099, -6.005, -0.079 | stable decrease |
| deadlock rate | -0.031 | [-0.469, +0.641] | -0.437, -0.500, -0.187, +1.000 | decrease trend |
| objective | +0.3153 | [+0.2781, +0.3526] | +0.2735, +0.3631, +0.2828, +0.3420 | stable increase |

## Interpretation

The paper-facing claim should focus on the complex 8-agent obstacle setting. If both risk CIs remain below zero while success and deadlock CIs include zero, the clean interpretation is that the method reduces near-collision risk without materially changing task success or deadlock.

This 4-seed paired analysis is sufficient for the current paper framing as supporting evidence: the complex obstacle scene shows consistent risk reduction across all paired seeds, while success and deadlock remain effectively unchanged. The claim should still be phrased as scenario-specific evidence rather than a broad all-scenario dominance claim.
