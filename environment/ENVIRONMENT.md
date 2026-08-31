# Validated Environment

The final experiments ran in WSL2 Ubuntu 24.04.2 with `/home/xzl/miniconda3/envs/sci1-rl/bin/python`.

Core versions:

```text
Python 3.9.25
PyTorch 2.8.0+cu128
CUDA 12.8 (PyTorch runtime)
NumPy 1.24.4
Matplotlib 3.8.4
Gym 0.25.2
sample-factory 2.1.1
NVIDIA GeForce RTX 5090, 32607 MiB
```

Create an environment with the matching major versions, then install the simulator and baseline repositories in editable mode:

```bash
pip install -r environment/requirements-analysis.txt
pip install -r environment/requirements-training.txt
pip install -e repos/quad-swarm-rl
pip install -e repos/baseline_candidates/on-policy
pip install -e repos/baseline_candidates/HARL
```

`pip_freeze.txt` records the complete validated environment. Its three local editable paths describe the original workstation and must be replaced by paths in the new checkout.
