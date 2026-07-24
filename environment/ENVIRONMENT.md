# Environment Notes

The experiments were executed from WSL Ubuntu using the conda environment:

```text
/home/xzl/miniconda3/envs/sci1-rl
```

Recommended invocation pattern:

```bash
BASE=$(find /mnt/h/xzl/sci1 -maxdepth 1 -type d -name '*graph_conditioned_lagrangian_shield' ! -name '*line1*' -print -quit)
cd "$BASE"
PY=/home/xzl/miniconda3/envs/sci1-rl/bin/python
```

The code depends on the repository-local third-party baseline sources under:

```text
repos/baseline_candidates/on-policy
repos/baseline_candidates/HARL
```

GPU evaluation uses:

```bash
EVAL_DEVICE=cuda
```

CPU evaluation is supported for script debugging but is not recommended for full formal evaluation.

## Validated Final Workstation

The retained final workflow was revalidated with:

    WSL2 Ubuntu 24.04.2 LTS
    Python 3.9.25
    PyTorch 2.8.0+cu128
    CUDA toolkit reported by PyTorch: 12.8
    NumPy 1.24.4
    NVIDIA GeForce RTX 5090, 32607 MiB
    Intel Core i9-10900K

The public repository intentionally excludes manuscript sources and therefore does not require a TeX installation for package verification.
