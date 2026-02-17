# APPFL[sim] Quick Guide

This spinoff is simulation-focused and MPI-backed.

## 1) Environment

Run from:

```bash
cd /Users/vaseline555/Desktop/workspace/APPFL_SIM/APPFL-SIM
```

Install package in editable mode (optional but recommended):

```bash
.venv/bin/pip install -e .
```

## 2) Simple FL Scenario (MNIST, 3 clients, 2 rounds, IID)

### Serial run

```bash
PYTHONPATH=. .venv/bin/python -m appfl_sim.runner \
  --config appfl_sim/config/examples/split/mnist_iid.yaml \
  backend=serial
```

### MPI parallel run

```bash
PYTHONPATH=. .venv/bin/python -m appfl_sim.runner \
  --config appfl_sim/config/examples/split/mnist_iid.yaml \
  backend=mpi
```

Notes:
- With `backend=mpi`, runner auto-attaches MPI launch and auto-sizes worker ranks.
- You can pin worker count with `mpi_num_workers=<N>`.
- If already running inside `mpiexec`/`mpirun`, auto-launch is skipped.
- Logs are written under `./logs`.
- Default/base config is `appfl_sim/config/examples/simulation.yaml`.
- Split examples live in `appfl_sim/config/examples/split/`.
- Logging examples live in `appfl_sim/config/examples/logging/`.
- Metrics examples live in `appfl_sim/config/examples/metrics/`.
- Algorithm placeholders live in `appfl_sim/config/algorithms/`.
- Model backend can be set with `model.source=appfl|timm|hf|auto`.
- For multi-node PoC:
  - `mpi_dataset_download_mode=rank0` for shared FS, or `local_rank0` for node-local FS.
  - keep `mpi_use_local_rank_device=true` to map GPUs by local rank.

## 3) Fast sanity run (MNIST)

```bash
PYTHONPATH=. .venv/bin/python -m appfl_sim.runner \
  backend=mpi \
  dataset=MNIST \
  num_clients=3 \
  num_rounds=2 \
  client_fraction=1.0 \
  split_type=iid \
  download=true \
  batch_size=16 \
  local_epochs=1
```

## 4) HF / timm examples

HF (runs without downloading pretrained weights):

```bash
PYTHONPATH=. .venv/bin/python -m appfl_sim.runner \
  --config appfl_sim/config/examples/external_datasets/hf/template.yaml
```

timm:

```bash
PYTHONPATH=. .venv/bin/python -m appfl_sim.runner \
  --config appfl_sim/config/examples/external_datasets/timm/template.yaml
```

## 5) Logging backend

- `logging_backend=file` for default file logging.
- `logging_backend=tensorboard` writes to `log_dir/project_name`.
- `logging_backend=wandb` uses `project_name` as WandB project and `experiment_name` as run name.
- For WandB online mode, authenticate first:
  `wandb login` or set `WANDB_API_KEY`.
