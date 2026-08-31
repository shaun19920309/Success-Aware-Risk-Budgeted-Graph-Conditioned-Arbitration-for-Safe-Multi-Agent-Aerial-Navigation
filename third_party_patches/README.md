# Third-Party Sources and QuadSwarm Adapters

Full simulator rollout requires these upstream projects:

```bash
git clone https://github.com/Zhehui-Huang/quad-swarm-rl.git repos/quad-swarm-rl
git clone https://github.com/marlbenchmark/on-policy.git repos/baseline_candidates/on-policy
git clone https://github.com/PKU-MARL/HARL.git repos/baseline_candidates/HARL
```

The original local copies no longer contain Git metadata, so this package does not claim unverifiable upstream commit hashes. The exact local QuadSwarm integration files used for the experiments are retained under `third_party_patches/on-policy/` and `third_party_patches/HARL/`. Overlay them after cloning:

```bash
cp -a third_party_patches/on-policy/. repos/baseline_candidates/on-policy/
cp -a third_party_patches/HARL/. repos/baseline_candidates/HARL/
```

Install each project in editable mode as described in `environment/ENVIRONMENT.md`. Their upstream licenses remain applicable; the adapter files do not relicense those projects.
