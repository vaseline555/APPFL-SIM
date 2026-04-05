# WandB Sweeps

## Directories
- `dc/`: discounted sweep configs (`dsucb`, `dsts`, `dslinucb_r`, `dslints_r`, `dslinucb_c`, `dslints_c`).
- `tools/wandb_sweep_bandit.py`: runner called by `wandb agent`.

## Quick start
```bash
cd <repo-root>
source .venv/bin/activate
wandb login

# 1) Create one sweep (example: CIFAR10 non-IID + DSTS, prints sweep id)
wandb sweep appfl_sim/config/sweeps/dc/cifar10_non_iid_dsts.yaml

# 2) Start agent (replace entity/project/sweep_id)
wandb agent <entity>/<project>/<sweep_id>
```

## Notes
- `posterior_variance` is mapped to `algorithm.scheduler_kwargs.likelihood_variance` for TS variants (`dsts`, `dslints_r`, `dslints_c`).
- DC files expose `discount_gamma` + family-specific hyperparameters.
