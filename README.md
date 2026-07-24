# Success-Aware Risk-Budgeted Graph-Conditioned Arbitration of Frozen Experts for Safe Multi-Agent Aerial Navigation

This repository contains the final SA-RB-GCA expert-pool method, formal baseline results, four-seed statistics, and the scripts needed to reproduce the reported numerical results. The manuscript source and compiled paper are intentionally not distributed in this repository.

## Scope

Included:

- Final method only: SA-RB-GCA with a frozen expert pool.
- Formal baselines: MAPPO, MAPPO-Lagrangian, IPPO, MAT, HATRPO.
- Four formal scenarios and four paired seeds: `0`, `1111`, `2222`, `3333`.
- Raw group summaries, per-seed evaluation summaries, publication-ready tables, paired bootstrap CI, and secondary safety/progress diagnostics.

Excluded:

- Intermediate branch versions and exploratory sweeps.
- Exploratory checks, branch sweeps, weighted variants, and incomplete extended-seed experiments.
- Large model checkpoints and raw training logs.
- Manuscript source, IEEE template files, author materials, and compiled paper PDFs.

## Final Claim

The strongest evidence is in the complex 8-agent obstacle scenario. Compared with MAPPO, the final method keeps success/deadlock essentially unchanged while reducing near-collision risk:

| Metric | SA-RB-GCA minus MAPPO | 95% paired bootstrap CI |
|---|---:|---:|
| Success rate | +0.031 pp | [-0.641, +0.469] |
| Risk rate, distance < 0.65m | -1.395 pp | [-2.108, -0.827] |
| Risk rate, distance < 1.0m | -2.722 pp | [-5.028, -0.735] |
| Deadlock rate | -0.031 pp | [-0.469, +0.641] |
| Environment true objective | +0.3153 | [+0.2781, +0.3526] |

This supports the paper claim that the method is most useful under high interaction density and obstacles: it preserves task completion while reducing risk.

## Package Layout

- `code/scripts/`: final evaluation, baseline, summary, and statistics scripts.
- `data/final_group_summaries/`: copied final group-level CSVs.
- `data/final_seed_summaries/`: per-seed final evaluation summaries for seeds `0,1111,2222,3333`.
- `data/tables/`: publication-ready result tables.
- `data/statistics/`: paired per-seed deltas and bootstrap CI.
- `docs/`: method, experiment protocol, result interpretation, and source report.
- `reproduce/`: exact reproduction workflow and package verifier.
- `environment/`: WSL/conda environment notes.
- `manifests/`: data and checkpoint manifests.

## Publication

The associated paper is:

> Success-Aware Risk-Budgeted Graph-Conditioned Arbitration of Frozen Experts for Safe Multi-Agent Aerial Navigation

The repository accounts for all retained final-result families: 24 formal method-scenario means, five paired-bootstrap metrics, 20 per-seed paired deltas, 36 secondary mean/seed-SD pairs, and 96 raw per-seed summaries. Intermediate branch experiments are intentionally excluded.

## Quick Verification

From the package root:

```bash
python reproduce/verify_package.py
```

Expected output:

```text
Package verification passed.
```

For full experiment reproduction in WSL, see `reproduce/REPRODUCE.md`.
