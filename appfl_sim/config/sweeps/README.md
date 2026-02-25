# WandB Sweeps

## Files
- `cifar10_iid.yaml`: CIFAR10 IID sweep.
- `cifar10_non_iid.yaml`: CIFAR10 non-IID (Dirichlet) sweep.
- `mnist_iid.yaml`: MNIST IID sweep.
- `mnist_non_iid.yaml`: MNIST non-IID (pathological) sweep.
- `tools/wandb_sweep_bandit.py`: runner called by `wandb agent`.

## Quick start
```bash
cd /home/hahns/workspace/GenFL
source .venv/bin/activate
wandb login

# 1) Create sweep (example: CIFAR10 non-IID, prints sweep id)
wandb sweep appfl_sim/config/sweeps/cifar10_non_iid.yaml

# 2) Start agent (replace entity/project/sweep_id)
wandb agent <entity>/<project>/<sweep_id>
```

## Notes
- `posterior_variance` is mapped to `algorithm.scheduler_kwargs.likelihood_variance` for SWTS.
- For SWUCB runs, TS-specific params are ignored.
- For SWTS runs, UCB-specific params are ignored.
- Each sweep YAML pins scheduler-specific base configs via `config_path_swucb` and `config_path_swts`.
