# Objective normalization correction

The final artifact audit found mixed denominators in the exported
`avg_true_objective_per_second` column: the BC evaluator used the configured
7.0 s, whereas the baseline evaluator used 701 frames times 0.01 s = 7.01 s.

The fair-budget analysis now uses `avg_true_objective / 7.0` for every raw row,
with 7.0 taken from the original frozen protocol. No rollout, checkpoint,
raw CSV, seed split, outcome definition, or reward is changed. All bootstrap,
sign-flip, Holm, and learning-curve calculations are rebuilt from these values.
This is a post-results reporting correction, not a new preregistration.

Earlier analysis is preserved locally in
`analysis_pre_objective_duration_correction_20260909/` for provenance and is
excluded from the final release. The correction does not affect success,
collision, deadlock, progress, or proximity. It makes baseline objectives
slightly more negative and cannot explain the large success gap.
