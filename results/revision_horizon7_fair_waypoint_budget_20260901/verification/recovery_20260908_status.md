# Fair-waypoint training recovery, 2026-09-08

Six MAT/HATRPO runs resumed in their original run directories at approximately
13:43 China time on 2026-09-08, following another WSL interruption. MAPPO, IPPO,
and MAPPO-Lagrangian already have nine verified complete runs. No new run2 was
created. The RTX 5090 is used by all six resumed jobs without GPU locks.

| Method | Training seed | Saved resume step |
|---|---:|---:|
| MAT | 260000 | 4,352,128 |
| MAT | 260001 | 4,352,128 |
| MAT | 260002 | 4,352,128 |
| HATRPO | 260000 | 4,352,000 |
| HATRPO | 260001 | 4,480,000 |
| HATRPO | 260002 | 4,352,000 |

Each original run has an immutable source snapshot and a hash manifest in
`recovery/20260908_full_state_resume/`. The audit validates method, seed, step,
optimizer metadata, finite tensors, model/state save times, and HATRPO's two
copies of ValueNorm. It records configuration, source-code, and existing
milestone hashes. All 1M/3M milestone checks passed before recovery.

Weights, optimizers, ValueNorm, and process RNG were restored. The simulator
state and rollout buffer were not serialized: recovery resets the episode and
is not a bitwise continuation. Save-time consistency is evidence of a matching
model/state pair, not a pre-existing cryptographic binding. The earlier
20260903 legacy-recovery limitations remain applicable to these runs.

The CUDA RNG restoration path in both runners required converting saved RNG
tensors back to CPU. `tests/test_training_resume_cuda_rng.py` passed for both
frameworks on the RTX 5090, covering optimizer and ValueNorm restoration, RNG
equality, and rejection of an inconsistent resume step.

Launch command: `python scripts/resume_horizon7_fair_training_state.py --launch`.
The default command without `--launch` audits only. Repeating `--launch` while
these sessions are active reported `already_running` for all six and started
no duplicates. A prepared recovery tag cannot be silently reused after exit.

Training logs use `<method>_seed<seed>_20260908_full_state_resume.log` and
`.err.log` under the result root's `logs/`. The postprocess session is
`sci1_fair_postprocess`, with logs `postprocess_20260908_full_state_resume.log`
and `.err.log`. It waits for 15/15 training verification, then runs the 1,440
paired evaluations and statistical analysis. Paper and public-package updates
remain pending those results; no new performance conclusion is asserted here.

Initial resumed throughput was approximately 10-11 environment steps per second
per job. A provisional estimate is 18-22 hours of remaining concurrent training,
followed by evaluation and artifact work, approximately 20-28 hours in total
from resumption if the host remains running. This is not a completion guarantee.
